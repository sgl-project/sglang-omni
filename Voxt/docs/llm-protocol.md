# LLM protocol 1

How VoxtNew (Go) talks to `voxtnew-llm` (Swift, [llm/](../llm/README.md)), the process that runs the language models. Go types: `internal/llmrt/wire`. Swift types: `llm/Sources/VoxtNewLLMKit/Wire*.swift`. Golden transcripts: `llm/Fixtures/*.jsonl`, decoded by both sides' tests. Supervisor and backend: `internal/llmrt`.

It has the helper protocol's envelope, handshake and error shape ([helper-protocol.md](helper-protocol.md)), with methods of its own. **Change it in all places at once**: the Go types, the Swift types, the fixtures, the fake process (`internal/llmrt/llmrttest`) and this page. `go test ./internal/llmrt/...` and `cd llm && swift test` fail when they disagree.

## Why stdin and stdout, not loopback HTTP

The plan first said loopback HTTP. JSON lines over the process's own pipes stream and cancel more simply:

- A stream is notifications naming the request (`chat.delta`); cancelling is one request (`job.cancel`). Over HTTP the Swift side would need a server (Network.framework, chunked or SSE framing) and would see a cancel only as a closed connection.
- No port: nothing to choose, announce or leak, and no other process on the Mac can talk to the model.
- The process's end is the pipe's: when the app dies, stdin closes and the process exits; when the process dies, Go's reads end at once.
- The helper's supervisor shape and its fixture tests carry over.

## 1. Transport

- Go starts the process with no arguments; its working directory is the executable's. It is `VoxtNew.app/Contents/Resources/llm/voxtnew-llm` (with `mlx.metallib` and `llama.framework` beside it), or `$VOXTNEW_LLM`.
- One JSON message per line (UTF-8, `\n`), at most 4 MiB; Go → process on stdin, process → Go on stdout. stderr is free-form text: the supervisor writes it to `~/Library/Logs/VoxtNew/llm.log` and to the logs panel (source `llm`).
- The process exits when stdin closes or on `llm.shutdown`. Go starts it on first use and stops it after a while unused (`llmrt.Config.IdleTimeout`): exiting is the one sure way to give a model's memory back.

## 2. Messages

| Shape | Members | Direction |
|---|---|---|
| request | `id`, `method`, `params`? | Go → process |
| notification | `method`, `params`? (no `id`) | process → Go |
| response | `id` + `result` or `id` + `error` | process → Go |

Ids are strings (`c1`…). Unknown members are ignored. A member Go marks `omitempty` is absent when zero (Swift: an optional left nil); the others are always present.

Errors are `{"code": "…", "message": "…", "data"?: {…}}`:

| Code | When |
|---|---|
| `bad_request` | malformed params (an unknown format, no messages) |
| `method_not_found` | unknown method |
| `not_initialized` | a request before `llm.hello` |
| `protocol_mismatch` | hello with another protocol; data `{"supported":[1]}` |
| `cancelled` | the chat was cancelled (`job.cancel`, `llm.shutdown`) |
| `load_failed` | the model's files are missing or broken, or its architecture is unknown |
| `context_length` | the prompt does not fit the model's context |
| `internal` | anything else |

## 3. Sequencing

- `llm.hello` comes first; until it is answered every other request fails with `not_initialized`. Go refuses a process whose `protocol` is not 1.
- One model is loaded at a time. `model.load`, `model.unload` and `chat` run one after another, in the order they arrive (one GPU); `model.status`, `job.cancel` and `llm.shutdown` are answered at once, even while a chat runs.
- `chat` is a **job**: it sends `chat.delta` notifications naming it in `job` before its response. `job.cancel` makes a running or queued chat answer `cancelled` within a token; `job.cancel` is answered after the chat it ends.
- A chat names its model (`model`: id, format and path). The process loads it first when another (or none) is loaded, so a chat after a restart needs no separate load; `model.load` warms a model up ahead of its first chat.
- If the process exits, Go fails its calls and starts a new one on the next call.

## 4. Methods

| Method | Params → result |
|---|---|
| `llm.hello` | `{protocol, app, logLevel?}` → `{protocol, version, pid, methods, device{name, os, memoryBytes}}` |
| `llm.shutdown` | → `{}`, then exit 0 (a chat is cancelled) |
| `job.cancel` | `{job}` → `{cancelled}` |
| `model.load` | `{model{id, format, path}, cacheLimitMB?}` → `{model, cached?, loadMs, memory}` |
| `model.unload` | → `{unloaded, memory}` |
| `model.status` | → `{model?, memory}` |
| `chat` (job) | `{model, messages[{role, content}], maxTokens, sampling{temperature, topP?, topK?, minP?, repetitionPenalty?, repetitionContextSize?, seed?}, prefillStepSize?, template?}` → `{text, finish, usage{promptTokens, completionTokens}, timings{loadMs?, promptMs, firstTokenMs, totalMs, tokensPerSecond, cache?{hit, prefixTokens}}, memory}` |
| `chat.delta` (notification) | `{job, text}`: the next piece of the answer, in order |
| `log` (notification) | `{level, message, job?}` |

- `format` is `mlx` (a directory with `config.json`, safetensors and a tokenizer) or `gguf` (the path of one `.gguf` file).
- `sampling.temperature` 0 is greedy. Absent fields are the engine's neutral values (top-p 1, no top-k, no min-p, no repetition penalty). `prefillStepSize` is MLX's prompt chunk.
- `template` is passed to the model's chat template as variables (`{"enable_thinking": false}` for Qwen3.5); MLX only, a GGUF model's template takes none.
- `finish` is `stop` (the model ended, or the repetition guard cut a loop) or `length` (`maxTokens` reached). `text` is the deltas joined, but when the repetition guard (Voxt's: a phrase repeated six times or more) cut a loop: then it keeps one copy and is shorter than the deltas.
- `memory` is `{footprintBytes, peakFootprintBytes, residentBytes, mlxActiveBytes?, mlxCacheBytes?, mlxPeakBytes?}`: the process's physical footprint now and at its peak (which counts Metal buffers, so an MLX model's weights), its resident size (which counts mapped files, so a GGUF model's weights, which llama.cpp maps from the file and the footprint leaves out), and MLX's own counters for an MLX model.
- `cacheLimitMB` bounds MLX's buffer cache (0: MLX's default).
- `timings`: `loadMs` when the chat loaded its model, `promptMs` the prompt's evaluation, `firstTokenMs` from the request to the first piece of text, `totalMs` the whole chat, `tokensPerSecond` the generation's rate after the first token, `cache` the prompt cache's use (below; MLX models only).
- `maxTokens` 0 is a **prefill**: the process evaluates the prompt's shared prefix into its prompt cache and answers `text` "", `finish` `length`, no deltas. The last message's text does not matter (a GGUF model does nothing).

Policy is Go's: which model, its sampling, its token budget, its prompt shape (a translation model's own prompt, internal/llmrt) and when to unload. The process keeps no settings.

### The prompt cache

An MLX model keeps its caches (attention KV and Qwen3.5's linear-attention state) after the **shared prefix** of a prompt: the tokens before the last message's text, which for compose's prompts is the system prompt and the clean-up's example turn (≈ 520–580 tokens, the same for one feature, scene, language and dictionary). A later chat whose prompt starts with all of a kept prefix's tokens evaluates only the rest, and its first token comes sooner. It is the process's own business: nothing in the requests changes, and the answers are the same (greedy: token for token).

- The shared prefix is found by rendering the chat template a second time with the last message's text replaced: where the two token lists part. With compose's prompts it is the system prompt and the example turn.
- A snapshot is reused whole, never trimmed (linear-attention state cannot be rewound): it serves a prompt that has all of its tokens and at least one more.
- The shared prefix is evaluated in chunks ending at every multiple of 256 tokens and at its end, whatever `prefillStepSize` says and whatever the cache holds (the cache off too); snapshots are kept and reused only at those points. A prompt's evaluation is then the same chunks cold or from a snapshot, and a greedy answer does not depend on the cache, token for token. (A prompt evaluated in one piece, as before M2b, can differ in its last bits, which flips near ties: 感谢 / 谢谢.)
- A prompt that parts from a kept one inside its shared prefix (a glossary that lists the text's terms first, a scene) also keeps a **branch point**: the last multiple of 256 before they part, which the next such prompt starts from. A point under 64 tokens is not kept.
- At most 6 snapshots and 512 MB are kept, the least recently used dropped first; all go with the model (`model.unload`, another model's load, the process's exit).
- `timings.cache` says what a chat did: `hit` when it started from a snapshot; `prefixTokens` the tokens it reused (a hit) or kept (a miss), 0 when it did neither (no shared prefix, or a short one). It is absent for a GGUF model, and when `VOXTNEW_LLM_PROMPT_CACHE=0` in the process's environment turns the cache off (tests compare answers with and without it).
- Go warms it up: while the user speaks, a session sends its prompt with an empty text as a prefill (`maxTokens` 0, `llmrt.Backend.Prefill`), so that the chat after the recording is a hit ([llm-models.md](llm-models.md#prompt-cache-m2b)).

## 5. Fixtures

`llm/Fixtures/*.jsonl` are golden transcripts, one `{"to":"llm"|"app","msg":{…}}` per line. `go test ./internal/llmrt/wire` decodes each line strictly with the Go types (unknown fields fail), checks that it encodes back to the same JSON and checks the sequencing (one response per request, a job's deltas before its response, `job.cancel` answered after its job). `swift test` (FixtureTests) decodes and re-encodes each line with the Swift types, and replays the requests through the server with a fake engine.
