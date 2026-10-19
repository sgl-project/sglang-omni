# Native Chat Stage

Understanding models that native SGLang serves through its OpenAI chat API, such as the Cosmos3 Reasoner, run in Omni as a native chat stage. The stage starts a native SGLang Engine and its chat service in the stage process and sends every Omni request through them, so the chat template, multimodal processing, sampling and KV cache are those of native SGLang. `--text-only` selects the Cosmos3 Reasoner.

```bash
sgl-omni serve --model-path /path/to/Cosmos3-Nano --text-only --port 8000
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": [{"type": "text", "text": "What is in this image?"}, {"type": "image_url", "image_url": {"url": "https://example.com/scene.png"}}]}], "max_tokens": 128}'
```

## Requests

- Messages carry text, `image_url`, `video_url` and `audio_url` content parts. The top level `images` and `videos` lists are added to the last user message. Top level `audios` and the video processing options such as `video_fps` are refused with 400, since native SGLang reads media only from content parts.
- Sampling fields that a request leaves out come from the checkpoint generation config, as with `sglang serve`. For Cosmos3 Nano that is temperature 0.7, top_p 0.8 and top_k 20.
- Other native chat request fields go under the stage name in `stage_params`, for example `{"stage_params": {"reasoner": {"ignore_eos": true}}}`. Native SGLang validates them, and an invalid one returns 400.
- The stage returns one text completion per request, so `stage_params` that ask for native tools, `n` above 1 or log probabilities on a stream are refused with 400. Run `sglang serve` on the same checkpoint when you need native tool calls, a reasoning parser, several choices or streamed log probabilities.

## Media policy

`--allowed-local-media-path` and `--allowed-media-domain` apply as on the other chat routes. File URLs need `--allowed-local-media-path` and must resolve inside it, and local paths are confined to it once it is set. While `--allowed-media-domain` is set, the stage fetches remote media itself, checking the domain on every redirect, keeps native SGLang's `media_url_max_file_size_mb` limit, and hands native SGLang a temporary file that it removes when the request ends. A refused reference returns 400 before native SGLang loads anything. A fetch that fails returns the status the other chat routes return: 400 when the media host answers 404, 500 when it fails, times out or sends more than the limit.

## Sizing and admission

- Native SGLang derives the context length from the checkpoint and the memory fraction from the GPU, as `sglang serve` does. `stages.reasoner.gpu_memory_fraction` turns the placement budget into the memory fraction, and `--mem-fraction-static` or any native server argument under `stages.reasoner.engine`, for example `--reasoner.engine.context_length 32768`, overrides native SGLang. An explicit memory fraction wins over the placement budget with a warning.
- Admission is native SGLang's by default: up to its `max_running_requests` run at once and further requests wait in its queue, as with `sglang serve`. Set `stages.reasoner.engine.max_running_requests` and `stages.reasoner.engine.max_queued_requests` to bound it. Omni then answers 503 to requests beyond their sum before any media is loaded.
- Before the stage reports ready it sends the same warmup request as `sglang serve`. `stages.reasoner.engine.skip_server_warmup` turns it off.

## Lifecycle

- A cancelled request or a closed stream aborts the native request at once.
- Native SGLang processes exit with the stage process. If a native scheduler or detokenizer process fails, native SGLang stops the stage and the server stops.
