# Full Duplex Interactive Model

SGLang-Omni serves a live voice conversation on `/v1/realtime` in two ways. A native full-duplex model such as MiniCPM-o 4.5 listens and speaks at the same time and decides itself when to talk. The streaming cascade composes independent speech recognition, language model and speech synthesis services into a turn-based conversation, and the server adds voice activity detection, the conversation history, turn taking and interruption. Both run as a Session on the server, a stateful conversation described under [Session Graph](#session-graph).

| Model | Interaction | Launch |
| --- | --- | --- |
| MiniCPM-o 4.5 | Native full duplex, audio and camera input | [Native Full Duplex: MiniCPM-o](#native-full-duplex-minicpm-o) |
| Streaming cascade | Turn-based, with interruption | [Streaming Cascade](#streaming-cascade) |

## Native Full Duplex: MiniCPM-o

[MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) is served for full duplex with its own config; [MiniCPM-o](../cookbook/minicpm_o.md) covers its installation and chat usage.

```bash
python -m sglang_omni.cli serve \
  --config examples/full_duplex/minicpmo.yaml \
  --model-path openbmb/MiniCPM-o-4_5 \
  --enable-realtime --port 8000
```

We provide two demonstrative config files.

| Config | Use it for |
|---|---|
| `examples/full_duplex/minicpmo.yaml` | Normal serving. Sampling matches the MiniCPM-o demo |
| `examples/full_duplex/minicpmo-parity.yaml` | Repeatable output for regression and parity recordings. Differs in greedy sampling, `top_k: 100`, and running the thinker and talker without CUDA graphs |

| Setting | Default | Meaning |
|---|---|---|
| `max_sessions` | 2 | Conversations at the same time. Further connections get HTTP 503. The thinker and talker reserve GPU memory for this many full-length conversations, so raise it only as far as the GPU has room. Startup warms up perception at each batch size up to this value |
| `reference_audio` | checkpoint default | Voice used when a session sends no reference |
| `speech_state_bytes_per_session` | 2 GiB | Memory the speech stage may hold per conversation. A conversation that needs more is closed and the others keep running |
| `speech.dtype` | `float32` | Precision of the voice decoder's flow model: `float32`, `float16` or `bfloat16`; the lower precisions change the voice slightly |
| `speech.enable_dit_torch_compile` | `false` | Compile the voice decoder's flow model with `torch.compile` while the server starts; CUDA only |
| `speech.n_timesteps` | 10 | Flow-matching steps per audio chunk; fewer steps decode faster at some cost in voice quality |
| `stages.thinker/talker.engine.enable_torch_compile` | `false` | Compiles every decode graph batch size; adds minutes to startup |
| `sampling` | see the config | Default sampling when a session does not set its own |
| `vision` | see the config | Camera-frame limits per unit (1 s of audio) |

While the server starts, the speech stage records the voice decoder as CUDA graphs for up to 8 conversations decoded together, or `max_sessions` if lower; the graphs hold GPU memory and add to startup time, and a voice whose reference audio is longer than the default voice's is decoded without them.

Speech settings also accept dotted CLI flags, such as `--speech.dtype float16 --speech.n_timesteps 10`. Stage settings use `--speech.factory.<field>` or `--speech.process`. The automatic thinker KV budget follows `thinker.engine.dtype` and `thinker.engine.kv_cache_dtype`; block-quantized KV caches require an explicit `thinker.engine.kv_cache_bytes` budget.

A session holds at most 8192 tokens of history, which is the model's trained context length. When that fills, the server sends `context_exhausted` and closes the session.

The full-duplex server is ready when this returns JSON containing `"native_full_duplex":true`:

```bash
curl --fail http://localhost:8000/v1/realtime/capabilities
```

The results should be as follows:

```json
{"model":"openbmb/MiniCPM-o-4_5","interaction":"native","native_full_duplex":true,"proactive_output":false,"turn_control":[null],"client_commit":false,"input_modalities":["audio","image"],"output_modalities":["audio","text"],"input_audio_format":{"type":"audio/pcm","rate":16000},"output_audio_format":{"type":"audio/pcm","rate":24000},"native_unit_ms":1000,"first_unit_ms":1000,"microturn_ms":"variable","tail_policy":"pad","supports_server_interrupt":false,"supports_truncate":false,"supports_resume":false,"partial_style":"append_only","pressure_policy":"reject","strict_order":true,"sampling_parameters":["greedy","temperature","top_k","top_p","repetition_penalty","listen_prob_scale","force_listen_count","max_new_tokens_per_unit","repetition_window_size","talker_temperature","talker_repetition_penalty"],"supports_reference_audio":true,"input_image_format":{"types":["image/jpeg","image/png"],"max_bytes":524288,"max_frames_per_unit":4,"max_slice_nums":9},"limits":{"max_input_bytes":1920000,"max_output_bytes":33554432,"max_history_chars":65536,"cleanup_timeout_s":30}}#
```

The duplex server only accepts `/v1/realtime` requests; other chat completions or voice cloning against it fail.

### Browser demo

With the full-duplex server running, start the demo page in another terminal:

```bash
python playground/realtime/app.py --api-base http://127.0.0.1:8000 --port 8080
```

Thanks to the OpenMoss team for the demo page.

Once the terminal prints `Running on`, open <http://localhost:8080> and click **Start talking** to talk through your microphone; the camera can be turned on during the call. If the server is remote, first run `ssh -N -L 8080:127.0.0.1:8080 USER@SERVER` on your own machine. Open the page through `localhost`, or the browser will not grant microphone access.

The settings panel switches between English and Chinese, changes the voice and adjusts sampling. The download button in the top bar saves the conversation trace; attach it when reporting a problem.


### Full-duplex protocol

Clients connect to `/v1/realtime` over WebSocket. They send 16 kHz mono PCM16 audio in `input_audio_buffer.append`; the server returns 24 kHz audio in `response.output_audio.delta` and text in `response.output_audio_transcript.delta`. The model decides once per second of audio whether to keep listening or to speak.

1. After `session.created`, send `session.update` and wait for `session.updated`.
2. Send audio with `input_audio_buffer.append` at the pace it is captured; replies arrive while you send.
3. When the audio ends, send `sglang.input_audio.end`, wait for `sglang.input_audio.drained`, then send `session.close`.

For a Python client, see `warmup` in `playground/realtime/app.py`.

Session settings go in the `sglang` field of `session.update`, before the first audio packet, and stay fixed for the session:

| Setting | Field | Notes |
|---|---|---|
| Voice | `reference_audio` | `{"media_type": "audio/wav", "data": "<base64>"}`, a PCM16 WAV of at most 30 s and 1 MiB; `tts_reference_audio` changes only the output voice |
| Sampling | `sampling` | For example `temperature`, `top_p` and `listen_prob_scale`; unset fields keep the defaults in `examples/full_duplex/minicpmo.yaml` |
| Image detail | `max_slice_nums` | Higher is sharper but accepts fewer frames per second |

Send camera frames with `sglang.input_image.append`: a base64 JPEG or PNG in `image`, and its position on the audio timeline in `sglang.t_ms`. By default up to 4 frames per second are accepted; `session.updated` reports the actual limit.

[Full-Duplex-Bench](full_duplex_bench.md) measures a native full-duplex deployment.

## Streaming Cascade

A profile names the services the cascade calls; one command starts them and the cascade, and the conversation is served on `/v1/realtime` and `/v1/session`.

### Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then install the network clients the nodes use:

```bash
python -m pip install websockets httpx
```

The profile `qwen3-asr-qwen3-tts-base-clone` runs three Qwen models on one GPU: [Qwen3-ASR](../cookbook/qwen3_asr.md) for recognition, Qwen3-4B for the answers and [Qwen3-TTS](../cookbook/qwen3_tts.md) Base speaking in a voice cloned from `reference_zh.wav` in the profile's directory, with Qwen3-ForcedAligner finding where each word of the reply ends in its audio and [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3) deciding when the user has finished speaking. Install the two speech models' dependencies as their cookbooks say, then download the checkpoints (all public, no token):

```bash
hf download Qwen/Qwen3-ASR-1.7B
hf download Qwen/Qwen3-4B-Instruct-2507
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base
hf download Qwen/Qwen3-ForcedAligner-0.6B
hf download pipecat-ai/smart-turn-v3 smart-turn-v3.2-gpu.onnx
```

### Launching

The profile's `aligner` node asks a forced-alignment worker where each word of the reply ends in its audio, so an interrupted reply can be cut at the last word heard. Start the worker first, from a separate virtual environment set up as [The forced-alignment worker](#the-forced-alignment-worker) describes:

```bash
PYTHONPATH=. python examples/streaming_cascade/alignment_worker.py --model Qwen/Qwen3-ForcedAligner-0.6B --device cuda:0 --language English --port 8010
```

Then start the cascade:

```bash
sgl-omni serve --config examples/streaming_cascade/profiles/qwen3-asr-qwen3-tts-base-clone/launch.yaml --enable-realtime --port 8765
```

Then open the browser page and talk, as [Voice Conversation](#voice-conversation) describes:

```bash
python playground/cascade/app.py --api-base http://127.0.0.1:8765 --port 8090
```

One command starts everything: the profile's `services` are started in order and waited for, a service already answering on its port is reused, and the cascade starts once every service and the worker answer; stopping the server stops the services it started. The profile's `turn_detection` node ends a turn only when Smart Turn takes a pause for its end, so a pause in the middle of a sentence does not cut it, as [Turn Detection](#turn-detection) describes. When an interrupted reply is cut, the profile's aligner keeps the words the listener heard: the forced aligner's word ends for the sentences whose audio was complete, and its own recognizer's transcript of the delivered audio for the sentence it was cut in. The page sends the position it played with `conversation.item.truncate` and marks the heard part of the reply. [Profiles](#profiles) lists the other bundled profiles, such as `qwen3-asr-qwen3.8-27b-customvoice` with Qwen3.8-27B on two GPUs.

To use services you already run, drop their `services` entries and point the nodes' `base_url` at them; a command-line override such as `--session_nodes.llm.base_url http://127.0.0.1:9000/v1` changes one field without editing the file. The rest of this section lists every launch mode and what each profile needs started beforehand; [Configuration](#configuration), [Node Reference](#node-reference) and [Compatibility and Constraints](#compatibility-and-constraints) list every setting and the settings that cannot be combined.

### One command with a profile

The default CustomVoice cascade uses Qwen3.8-27B with thinking off on GPU 0 and Qwen3-ASR/Qwen3-TTS on GPU 1. The audio-file example in `examples/streaming_cascade/run.py` uses this profile when `--config` is omitted.

```bash
sgl-omni serve --config examples/streaming_cascade/profiles/qwen3-asr-qwen3-tts-customvoice/launch.yaml --enable-realtime --port 8765
```

`--enable-realtime` mounts `/v1/realtime`, `/v1/realtime/capabilities` and `/v1/session`; without it the server starts the cascade but serves none of them.

The server starts the profile's `services` one after another, in the order they are listed, before it starts the cascade. For each entry it first requests `http://127.0.0.1:<port>/v1/models`: a service that answers 200 is reused and left running when the server stops, and the log shows `service 'tts' already serving on :8005`. Otherwise it runs `command` with its own environment plus the entry's `env`, in the directory `sgl-omni serve` was started from, and polls the port every second for up to 600 s. A service that exits first fails the start with `service '<name>' exited with code <n> before :<port> answered`; one that does not answer in time fails it with `service '<name>' did not answer on :<port> within 600 s`. The cascade itself must then start within `SGLANG_OMNI_STARTUP_TIMEOUT` seconds (default 600).

While the server runs, a service it started that exits stops the server with `service '<name>' exited with code <n>`. Stopping the server sends SIGTERM to every service it started and SIGKILL to any still running 15 s later.

Relative paths in a service `command` and in a node's `reference_audio_path` resolve against the directory `sgl-omni serve` was started from; the bundled profiles assume the repository root.

A service's port appears in three places that must agree: the `--port` inside its `command`, its `port`, and the `base_url` of the node that calls it. To move a service, change all three in the profile. If the server's own `--port` is taken, the server logs `Port <n> is already in use` and serves on a free port it names in `Using port <m> instead`; set `SGLANG_OMNI_STRICT_PORT=1` to fail instead.

### Services you run elsewhere

Remove the service's entry from `services` and set the node's `base_url` and `model` to the running service. `openai_transcription`, `openai_chat` and `openai_speech` work with any OpenAI-compatible transcription, chat completions and speech endpoint; `api_key_env` names the environment variable holding its bearer token. A `services` entry is only for a service on this machine, since its readiness check requests `127.0.0.1`. The node still probes the service at startup, so a wrong `base_url` fails the start (see [Verifying a Deployment](#verifying-a-deployment)).

### A model inside the server as a stage

A node written `stages: [...]` runs a model inside this server instead of calling a service. Two stage routes exist: `vad`, the voice activity detector, and `asr`, Nemotron 3.5 streaming ASR. Any other stage name fails the load with `no stage group serves the stages [...]`.

`nemotron-stage-qwen3-tts` runs Nemotron 3.5 ASR as the `asr` stage node reading `input:pcm16`, and a `running_transcript` node turns its running text into utterances:

```yaml
stages:
  asr:
    gpu: 0

session_nodes:
  asr:
    stages: [asr]
    inputs: ["input:pcm16"]
  transcript:
    kind: running_transcript
    inputs: [input, "asr:text", "vad:session"]
    tail_ms: 400
```

`stages.asr.gpu` (or `--asr.gpu 1` on the command line) picks the GPU. Install the Nemotron dependencies as [Nemotron 3.5 ASR](../cookbook/nemotron3_5_asr.md) describes.

### Several GPUs

Each service picks its GPUs through `env`, as `qwen3-asr-qwen3.8-27b-customvoice` does with `env: {CUDA_VISIBLE_DEVICES: "0"}` for the language model and `"1"` for the two speech models; a stage picks its GPU with `gpu`. A service's memory share is the `--mem-fraction-static` inside its own `command`. The cascade server has no SGLang engine stage, so `sgl-omni serve --mem-fraction-static` on a cascade profile fails with `--mem-fraction-static requires a pipeline with at least one SGLang engine stage`.

### The forced-alignment worker

An `alignment` node with `method: forced` sends each finished sentence and its audio to a worker that returns word boundaries. The worker cannot be a `services` entry: it answers `/health` and `/align` but not the `/v1/models` request the server waits for, and it needs the `qwen-asr` package, which pins a `transformers` release of its own, so it runs from a separate virtual environment. In that environment install `qwen-asr`, `fastapi` and `uvicorn`, plus `httpx`, `msgpack`, `msgspec`, `numpy` and `pydantic` for the modules it shares with the server, then start it from the repository root before `sgl-omni serve`:

```bash
PYTHONPATH=. python examples/streaming_cascade/alignment_worker.py --model Qwen/Qwen3-ForcedAligner-0.6B --device cuda:0 --language English --port 8010
```

| Flag | Default | Effect |
| --- | --- | --- |
| `--model` | required | The forced-aligner checkpoint. |
| `--device` | `cpu` | The device the aligner runs on. |
| `--language` | `Chinese` | The language of a sentence sent without one; a sentence carries the language of its reply once the recognizer has named the user's language. |
| `--host` | `127.0.0.1` | The bind address. |
| `--port` | required | The port; the node's `alignment_url` must point at it. |

`curl http://127.0.0.1:8010/health` answers `{"status":"ready"}` once the model is loaded. To run a Qwen3-ASR profile without the worker, switch its aligner to the asr method on the command line: `--session_nodes.aligner.method asr --session_nodes.aligner.alignment_url ""`. The Nemotron profiles have no transcription endpoint and switch to the sentence method: `--session_nodes.aligner.method sentence --session_nodes.aligner.alignment_url ""`.

### Browser playground

```bash
python playground/cascade/app.py --api-base http://127.0.0.1:8765 --port 8090
```

It needs `aiohttp`, serves the page on `--host` (default `127.0.0.1`) and `--port` (default `8080`), and forwards `/v1/realtime` and `/v1/realtime/capabilities` to `--api-base` (default `http://127.0.0.1:8000`). The page sends only `output_modalities` in its `session.update`, so the profile's prompt and turn detection apply; when a reply is interrupted it sends `conversation.item.truncate` with the audio it played, and it marks the heard part of the reply when the server sends `sglang.response.progress`. Browsers grant the microphone only to `localhost` or HTTPS pages, so open the page on the same machine or forward its port.

### An OpenAI Realtime client or the Session API

Any OpenAI Realtime client connects to `/v1/realtime` with the session flow in [Voice Conversation](#voice-conversation). `/v1/session` serves the same Session without the realtime protocol. `examples/streaming_cascade/run.py` runs a profile in its own process instead of connecting to a server; it does not start the profile's `services`, so start them first.

### Profiles

Every profile is a `launch.yaml` under `examples/streaming_cascade/profiles/`.

| Profile | Recognizer | Language model | Speech | GPUs | Nodes | Start beforehand |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3-asr-qwen3.8-27b-customvoice` | Qwen3-ASR-1.7B service on 8002 (`openai_transcription`) | Qwen3.8-27B on 8003, thinking off | Qwen3-TTS-12Hz-1.7B-CustomVoice on 8005, speaker `Vivian` | 2: the language model on GPU 0, both speech models on GPU 1 | `vad`, `asr`, `dialog`, `llm`, `tts`, `aligner` (forced, recognizing the sentence at the cut) | Alignment worker on 8010 |
| `qwen3-asr-qwen3-tts-customvoice` (default) | Qwen3-ASR-1.7B service on 8002 | Qwen3.8-27B on 8000, thinking off | Qwen3-TTS-12Hz-1.7B-CustomVoice on 8005, speaker `Vivian` | 2: the language model on GPU 0, both speech models on GPU 1 | `vad`, `asr`, `dialog`, `llm`, `tts`, `aligner` (forced, recognizing the sentence at the cut, published) | Alignment worker on 8010 |
| `qwen3-asr-qwen3-tts-base-clone` | Qwen3-ASR-1.7B service on 8002 | Qwen3-4B-Instruct-2507 on 8000 | Qwen3-TTS-12Hz-1.7B-Base on 8005, cloning `reference_zh.wav` in the profile directory | 1 | `vad`, `turn_detection` (published), `asr`, `dialog`, `llm`, `tts`, `aligner` (forced, recognizing the sentence at the cut, published) | Alignment worker on 8010 |
| `nemotron-qwen3-tts` | Nemotron 3.5 streaming ASR service on 8001 (`realtime_transcription`) | Qwen3-4B-Instruct-2507 on 8000 | Qwen3-TTS-12Hz-0.6B-Base on 8004, cloning `reference.wav` | 1 | `vad`, `asr`, `dialog`, `llm`, `tts`, `aligner` (forced) | Alignment worker on 8010; a recording at `reference.wav`, with `reference_text` set to its transcript |
| `nemotron-stage-qwen3-tts` | Nemotron 3.5 ASR as the `asr` stage on GPU 0, with `running_transcript` | Qwen3-4B-Instruct-2507 on 8000 | Qwen3-TTS-12Hz-0.6B-Base on 8004, cloning `reference.wav` | 1 | `vad`, `asr` (stage), `transcript`, `dialog`, `llm`, `tts`, `aligner` (forced) | Alignment worker on 8010; a recording at `reference.wav`, with `reference_text` set to its transcript |

`qwen3-asr-qwen3-tts-customvoice` and `qwen3-asr-qwen3-tts-base-clone` publish their aligner's `session`, so their clients receive `sglang.response.progress`. Every profile cuts an interrupted reply at the word ends of the sentences whose audio was complete; the Qwen3-ASR profiles also send the delivered audio of the sentence at the cut to their own recognizer, so a sentence cut before its audio was complete keeps the part the listener heard.

### Voice Conversation

With the page from [Launching](#launching) running, open `http://127.0.0.1:8090`, speak, and the reply plays as it is produced. Speaking while the reply plays interrupts it.

Any OpenAI Realtime client works the same way. Open a WebSocket to `/v1/realtime`, wait for `session.created`, and send the session settings:

```json
{"type": "session.update", "event_id": "cfg-1", "session": {"instructions": "Answer in one short English sentence.", "output_modalities": ["text", "audio"], "audio": {"input": {"turn_detection": {"type": "server_vad", "silence_duration_ms": 800}}}}}
```

Send 16 kHz mono PCM16 audio at real-time pace as `{"type": "input_audio_buffer.append", "event_id": "...", "audio": "<base64>", "sglang": {"seq": 0}}`, with `seq` counting up from 0 for the whole session. The server answers with these events:

| Event | Meaning |
| --- | --- |
| `input_audio_buffer.speech_started`, `input_audio_buffer.speech_stopped` | The start and the end of your utterance. |
| `conversation.item.input_audio_transcription.delta`, `.completed` | Partial and final transcript of what you said. |
| `response.created` | A reply begins. |
| `response.output_text.delta`, `response.output_audio.delta` | Reply text and reply audio, base64 PCM16 at 24 kHz mono. |
| `sglang.response.progress` | In a profile that publishes its aligner's `session`: `text_end`, the length of the reply text whose audio has been produced, and `audio_ms`, where that audio ends in the reply's audio, so a client can mark the text as it plays. |
| `response.done` | The reply is complete; `response.status_details.reason` is `stop`, `empty_turn` when the utterance had no words, or `interrupted` when you spoke over it. |
| `error` | The request or the session failed. |

`response.cancel` interrupts the reply from the client side. An interrupted reply stays in the history only up to the last boundary whose audio the server had sent before the interruption: the last word with a forced aligner, the part of the sentence at the cut that its transcript says with `heard_asr_url`, the last sentence otherwise; a reply interrupted before its first boundary leaves an empty answer. `conversation.item.truncate` with `audio_end_ms` tells the server how much of the reply was heard; the history then keeps the reply up to the last boundary whose audio ended by then, never past what the server sent.

To end turns yourself instead of by silence, set `"turn_detection": null` in the first `session.update`; `input_audio_buffer.commit` then ends the utterance after the audio sent before it.

Instructions and turn detection can change with a later `session.update` and apply from the next turn; sampling (`sglang.sampling.temperature`, `max_new_tokens_per_unit`) is set by the first one. [Session settings](#session-settings) lists every field.

### Turn Detection

Without it an utterance ends when the `vad` node hears `silence_duration_ms` of silence, so a pause to think in the middle of a sentence ends the turn and the reply answers half a sentence. A profile with a `turn_detection` node, such as `qwen3-asr-qwen3-tts-base-clone`, ends the turn only when the [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3) end-of-turn model, scoring the utterance so far at each pause the `vad` node reports, finds it complete, or when the pause outlasts the longest pause the eagerness allows. The node reads the Session input and `vad:session` and publishes `session`: its `speech.started` opens the turn and its `speech.stopped` ends it, so the recognizer, the dialog and the client read `turn_detection:session` where they read `vad:session` before, and speech that resumes inside a held pause continues the utterance instead of interrupting a reply.

The stage loads the BSD-2 licensed `smart-turn-v3.2-gpu.onnx` from the Hugging Face repository `pipecat-ai/smart-turn-v3` through the Hugging Face cache, or from the file or directory `stages.turn_detection.factory.model_path` names, and verifies its SHA-256 at startup. The stage's `factory.eagerness` (`low`, `medium` or `high`, default `medium`) sets how readily a pause ends the turn: `high` takes a shorter silence on a confident score and gives up a held pause sooner, `low` waits longer. The profile sets the `vad` node's `silence_duration_ms` to 160, the pause the detector scores as a candidate end; a client that selects `{"type": "server_vad"}` on this profile gets the VAD's boundaries unchanged, with that pause.

A client selects the behaviour per session with the standard field, `{"turn_detection": {"type": "semantic_vad", "eagerness": "high"}}`; `session.created` lists `semantic_vad` under `turn_control` when the profile has the node. The Qwen3-Omni Session graph takes the node the same way: declare it in `examples/qwen3_omni/session_h200.yaml` beside `vad`, let `dialog` read `turn_detection:session` in place of `vad:session`, and publish `turn_detection: [session]` in place of `vad: [session]`.

### Scripted Conversation

The same conversation runs without the realtime protocol through the Session API:

```bash
PYTHONPATH=. python examples/streaming_cascade/run.py question.wav followup.wav --output response.wav
```

Each input is a 16 kHz mono signed 16-bit PCM WAV. The command runs the cascade in its own process without starting the profile's `services`, so start them first. It streams the files back to back through one Session of the profile named by `--config` and writes the reply audio to `response.wav`; the second file is answered with the first turn in the history. `--config` defaults to `nemotron-qwen3-tts`. Pass `--commit` for a profile without the VAD node. The `/v1/session` protocol is described in [Session API](#session-api).

### Benchmarking

The turn latency benchmark streams one WAV for several turns and prints, per turn, the time from speech end to first text, first audio, final transcript and completion:

```bash
python benchmarks/realtime_turn_latency.py --url ws://127.0.0.1:8765/v1/realtime --audio question.wav --turns 5
```

### Configuration

A setting comes from one of three places: the profile file, read at launch; command-line overrides of the profile, also read at launch; and the settings a client sends for its own session.

### Profile file

| Field | Default | Effect |
| --- | --- | --- |
| `config_cls` | required | `StreamingCascadePipelineConfig` selects the cascade. |
| `model_path` | required | The deployment's name: the id `/v1/models` reports and the realtime model name, unless `sgl-omni serve --model-name` sets another. |
| `services` | `[]` | The services to start, in order (see below). |
| `session_nodes` | required | The nodes of the conversation, by name. |
| `session_publish` | required | Per node, the formats that reach the client. |
| `stages` | none | Settings of stage routes by stage name, such as `asr: {gpu: 0}` or `vad: {factory: {silence_duration_ms: 800}}`. |
| `timeout_s` | `120` | How long one input or control operation on a cascade node may take, including the wait for the last replies once the input has ended; past it the session fails with `component '<name>' append timed out`. |
| `unit_ms` | `200` | The length of the input audio units the server forwards into the session; reported as `native_unit_ms`. |
| `max_output_lead_ms` | `1000` | How far ahead of real time reply audio is delivered; audio already delivered cannot be taken back by an interruption. |
| `max_pending_output_chunks` | `1024` | Reply chunks held for a client that has stopped reading before the session fails. |
| `max_pending_output_bytes` | `33554432` | The same bound in bytes. |

A `services` entry has `name` (required, unique), `command` (required, a list), `port` (required, 1 to 65535, unique) and `env` (a map, default empty).

A `session_nodes` entry is either a node of a `kind`, with that kind's options beside `kind`, or a stage route written `stages: [...]`, which takes no options. Both take `inputs`, a non-empty list of:

| Input | Takes |
| --- | --- |
| `input` | Every format of the client input: its 16 kHz `pcm16` audio and its `session` controls (`commit`, `configure`, `truncate`). |
| `input:pcm16`, `input:session` | Only the client audio, or only its controls. |
| `<node>` | Every format `<node>` produces that this node's kind consumes. |
| `<node>:<format>` | Only that format. |
| `{source: <input>, capacity: <n>, overflow: block \| drop_oldest, feedback: true \| false}` | The same, with a bound on the edge: at most `capacity` chunks wait (default 64); when full, `block` (default) holds the producer and `drop_oldest` drops the oldest chunk. `feedback: true` marks the edge that closes a cycle and must not block. |
| `{kind: control, source: <node or input>, event: <control type>, preempt: false}` | The source's control event, delivered to this node as a control instead of as data; `preempt: true` cancels the node's operations in flight first. |

The formats each kind consumes and produces:

| Kind | Consumes | Produces |
| --- | --- | --- |
| `realtime_transcription`, `openai_transcription` | `pcm16`, `session` | `transcript` |
| `running_transcript` | `pcm16`, `session`, `text` | `transcript` |
| `dialog` | `transcript`, `text_delta`, `session` | `messages`, `session` |
| `openai_chat` | `messages` | `text_delta`, `session` |
| `openai_speech` | `text_delta`, `session` | `pcm_f32le` at 24 kHz, `session` |
| `alignment` | `text_delta`, `pcm_f32le`, `session` | `session` |
| stage `vad` | `input` | `session` |
| stage `asr` | `input:pcm16` | `text` |

`session_publish` maps a node to the formats the client receives: `transcript` becomes `conversation.item.input_audio_transcription.*`, `text_delta` becomes `response.output_text.delta`, `pcm_f32le` becomes `response.output_audio.delta` (PCM16), and `session` carries speech boundaries, the response lifecycle and `sglang.response.progress`.

### Command-line overrides

Any field of the profile can be overridden after `--config`, as `--<path> <value>` or `<path>=<value>`, one leaf value per flag. Paths start at the top of the profile, except stage settings, which start at the stage name:

```bash
sgl-omni serve --config examples/streaming_cascade/profiles/qwen3-asr-qwen3-tts-customvoice/launch.yaml --enable-realtime --port 8765 \
  --session_nodes.llm.base_url http://127.0.0.1:9000/v1 \
  --session_nodes.llm.temperature 0.3 \
  --session_nodes.tts.language English \
  --services.tts.env.CUDA_VISIBLE_DEVICES 1 \
  --vad.factory.silence_duration_ms 800 \
  --timeout_s 60
```

Services are addressed by `name` (`--services.tts.port`), not by position. An override may add an option the file leaves out, and is validated like the file: an option the node's kind does not have fails with `Extra inputs are not permitted`, and a whole group such as `--session_nodes.aligner x` fails with `addresses a settings group, not a single value`. Lists such as a service `command` or a node's `inputs`, and removing a node or a service, need an edit of the file.

### Session settings

A realtime client sends its settings in `session.update`. The first `session.update` opens the session; later ones may change only the fields marked hot below, and any other change is refused with `session field is frozen`.

| Field | Reaches | When it can change |
| --- | --- | --- |
| `instructions` | `system_prompt` of every node that has it (the `dialog`) | Any time; applies to the next prompt. At most 65536 characters, else `instructions exceed context or have invalid type`. |
| `sglang.sampling.temperature` | `temperature` of every node that has it (`openai_chat`), 0 to 2 | First `session.update` only. |
| `sglang.sampling.max_new_tokens_per_unit` | `max_tokens` of `openai_chat`, at least 1 | First `session.update` only. |
| `audio.input.turn_detection` | `type: server_vad` reaches the session's `vad` stage: `threshold`, `prefix_padding_ms`, `silence_duration_ms`; with a `turn_detection` node, `type: semantic_vad` and its `eagerness` reach that node, and `server_vad` makes it pass the VAD's boundaries through | Any time. `null` turns detection off and utterances end only at `input_audio_buffer.commit`; send it in the first `session.update` (see [Allowed, with consequences](#allowed-with-consequences)). `interrupt_response` is accepted and has no effect. |
| `output_modalities` | `["text", "audio"]` or `["text"]` | Any time. |

The realtime endpoint takes no per-node options. A `/v1/session` client sets them, together with the standard fields, as the `params` of its open frame and later as the fields of a `configure` control:

```json
{"instructions": "Answer in one short sentence.", "sampling": {"temperature": 0.2, "max_new_tokens_per_unit": 128}, "turn_detection": {"silence_duration_ms": 800}, "nodes": {"tts": {"language": "English"}, "llm": {"max_tokens": 256}}}
```

`nodes` maps a node name to options. An entry without `kind`, or with the node's current kind, changes only the listed options; an entry with another kind replaces the node with a new one built from that entry alone, so it must list that kind's required options. The new node takes over the conversation state of the old one. A `configure` reaches only the nodes that read `input` or `input:session`; a node that reads neither, such as `aligner` in the profiles, takes options only from the open frame. A node in the middle of a reply is replaced once the reply ends, and every node the `configure` reaches answers with a `configured` control naming its kind.

What needs a restart and what does not:

| Setting | Changes |
| --- | --- |
| `services`, the node set, `inputs`, `session_publish`, `stages`, `timeout_s`, `unit_ms`, `max_output_lead_ms`, `max_pending_output_*` | Restart the server. |
| Default node options and VAD defaults | Restart, or per session as above. |
| Node options through `nodes` | When a `/v1/session` session opens, and later through `configure` for nodes that read `input`. |
| `sglang.sampling` | When a realtime session opens. |
| `instructions`, `turn_detection`, `output_modalities` | Any time in a realtime session. |

### Node Reference

Every node of a kind has `kind` (required) and `timeout_s`. Options marked required have no default.

### `realtime_transcription`

Streams each utterance over its own WebSocket to `<base_url>/v1/realtime?model=<model>` of an SGLang-Omni transcription server started with `--enable-realtime`, and reports partial and final transcripts. Its transcripts carry no language.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `base_url` | string, `ws://` or `wss://` | required | The transcription server. |
| `model` | string | required | Sent as the `model` query parameter. |
| `api_key_env` | string | `OPENAI_API_KEY` | Environment variable whose value is sent as a bearer token; nothing is sent when it is unset. |
| `timeout_s` | number > 0 | `120` | The limit for connecting and for one utterance's whole exchange. |

### `openai_transcription`

Sends each finished utterance as a 16 kHz WAV to `<base_url>/audio/transcriptions` and reports one final transcript.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `base_url` | string, `http://` or `https://` | required | The service, including `/v1`. |
| `model` | string | required | The `model` form field. |
| `language` | string | `""` | Sent as the `language` hint when set. |
| `response_format` | `verbose_json` or `json` | `verbose_json` | `verbose_json` reports the language heard in each utterance; `json` suits a service without it and reports none. |
| `api_key_env` | string | `OPENAI_API_KEY` | As above. |
| `timeout_s` | number > 0 | `120` | The request timeout. |

### `running_transcript`

Turns the running text of the `asr` stage into utterances: partial transcripts while it grows, and a final one once `tail_ms` of further audio has arrived after `speech.stopped` or a `commit`. Its transcripts carry no language.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `tail_ms` | number ≥ 0 | `400` | Audio after the end of speech that still belongs to the utterance. |

### `dialog`

Keeps the conversation history, opens a turn for each final transcript and sends the history to the language model. An utterance that arrives during a reply waits for it to end; speech that starts during a reply interrupts it.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `system_prompt` | string | `You are a helpful voice assistant. Answer concisely.` | The system message; realtime `instructions` set it. |
| `max_history_chars` | integer > 0 | `65536` | The bound on the history: a turn whose history plus utterance plus `max_text_chars` would exceed it fails the session. |
| `max_text_chars` | integer > 0 | `16000` | The longest utterance transcript and the longest reply. |

### `openai_chat`

Streams one chat completion from `<base_url>/chat/completions` per turn.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `base_url` | string, `http://` or `https://` | required | The service, including `/v1`. |
| `model` | string | required | The `model` field. |
| `max_tokens` | integer > 0 | `512` | The `max_tokens` field; realtime `max_new_tokens_per_unit` sets it. |
| `temperature` | number 0 to 2, or null | null | The `temperature` field, left out when null; realtime `sampling.temperature` sets it. |
| `extra_body` | object | `{}` | Further request fields, such as `chat_template_kwargs`. |
| `api_key_env` | string | `OPENAI_API_KEY` | As above. |
| `timeout_s` | number > 0 | `120` | The request timeout. |

### `openai_speech`

Cuts the reply text into sentences and sends one streaming request per sentence to `<base_url>/audio/speech` with `model`, `voice`, `input`, `response_format: pcm`, `stream: true`, `language` when `language` is set, and `references: [{audio_path, text}]` when `reference_audio_path` is set. A sentence ends after a word ending in `.`, `!`, `?`, `;` or `…` (closing quotes and brackets included), after `max_sentence_words` words, after Chinese end punctuation followed by more text, or at a line end.

`openai_speech` sends each sentence without its emoji, asterisks and backticks, and a markdown link as its text alone, because a speech service reads them as stray sounds or a spelled-out URL. A sentence with no word left, such as one of only emoji or punctuation, is not sent because the speech service answers it with an error, and its end still counts in `reply.progress`. A sentence tagged `English` without Chinese characters is sent with its numbers written out as words, the way the forced-alignment worker reads them: a year as two pairs, spaced digit groups such as a phone number digit by digit, a decimal with `point`, and an ordinal such as `21st` as `twenty first`. The reply text and the `reply.progress` offsets keep everything the model wrote. The profiles' system prompts also tell the language model to use neither markdown nor emoji.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `base_url` | string, `http://` or `https://` | required | The service, including `/v1`. |
| `model` | string | required | The `model` field. |
| `voice` | string | `default` | The `voice` field; for Qwen3-TTS CustomVoice a built-in speaker, `default` meaning `Vivian`. |
| `reference_audio_path` | string | `""` | A WAV the speech service reads, for voice cloning. |
| `reference_text` | string | `""` | The transcript of the reference, sent with it when set. |
| `language` | string | `""` | The language tag: empty sends none; `auto` tags every sentence with the language the recognizer last heard in an answered turn; any other name tags every sentence with that name. With a fixed name, and with `auto` before the recognizer has named a language, a sentence containing Chinese characters is tagged `Chinese`; `auto` tags the others `English` until then. |
| `max_sentence_words` | integer > 0 | `24` | The longest sentence sent in one request. |
| `api_key_env` | string | `OPENAI_API_KEY` | As above. |
| `timeout_s` | number > 0 | `120` | The request timeout. |

### `alignment`

Reports how far into the reply text the reply audio has reached, as `reply.progress`, so the dialog can cut an interrupted reply where the listener stopped hearing it. It reads `["llm:text_delta", "tts:pcm_f32le", "tts:session", "dialog:session"]` in the profiles, and `input:session` too in the Qwen3-ASR profiles.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `method` | `sentence`, `forced` or `asr` | `sentence` | `sentence` reports the end of each spoken sentence; `forced` also reports the word ends the worker finds in each finished sentence; `asr` reports the sentence ends and, on a cut, the part of the sentence at the cut its transcript says. |
| `alignment_url` | string, `http://` or `https://`, or empty | `""` | The worker, for `forced`. |
| `heard_asr_url` | string, `http://` or `https://`, or empty | `""` | An OpenAI-compatible transcription service, including `/v1`, that recognizes the sentence at the cut; required for `asr`, optional for `forced`. |
| `heard_asr_model` | string | `""` | The `model` form field of those requests; set together with `heard_asr_url`. |
| `timeout_s` | number > 0 | `2` | The limit for one worker or transcription request; a sentence whose request fails or times out keeps only its sentence end. |

The dialog cuts an interrupted reply at the points `reply.progress` reports, up to the audio the graph sent to the client before the interruption, which the graph reports to the readers of the `tts` node's `session` output as `reply.delivered`. Without an `alignment` node the dialog takes `reply.progress` from the `tts` node, which reports the end of each sentence. With one, as in `qwen3-asr-qwen3-tts-customvoice`, the `aligner` node reads `["llm:text_delta", "tts:pcm_f32le", "tts:session", "dialog:session", "input:session"]` and the dialog's `{kind: control, source: aligner, event: reply.progress}` input replaces the one from `tts`; a profile names exactly one source of `reply.progress`.

`method: sentence` keeps the sentence ends, so a truncated reply keeps the sentences whose audio was played to the end and drops the sentence the listener was hearing. `method: forced` also sends each finished sentence and its audio to the worker at `alignment_url` and reports the word ends it returns, so the cut can fall inside a sentence that finished; when the worker fails or exceeds `timeout_s`, that sentence keeps its sentence end. A sentence that an interruption cuts off before its audio is complete gets no word ends and counts as unheard in both methods.

`method: asr` keeps the sentence ends and, when a reply is cut, sends the audio of the sentence at the cut, from the sentence's start to the cut, as a WAV file in the reply's language to the `/audio/transcriptions` endpoint of the OpenAI-compatible service at `heard_asr_url` with `heard_asr_model`; the reply keeps the sentences before it and the longest prefix of that sentence's text the transcript says, with the punctuation that follows it. The cut is the client's `truncate` position and never past `reply.delivered`, the same cut the dialog makes, so the aligner reads `input:session` for the client's `truncate`. The match ignores case and punctuation, reads digits, English numbers up to twenty and ordinary Chinese numbers as the same number, and stops at the first word the transcript leaves out, repeats or says differently. When the request fails, exceeds `timeout_s` or the first words differ, the sentence counts as unheard. It needs no model beyond the profile's own recognizer. `method: forced` takes `heard_asr_url` and `heard_asr_model` too, and then recognizes the sentence at the cut only when no word end at or after the cut covers it, such as a sentence cut off before its audio was complete; `method: sentence` takes neither.

### Stage `vad`

Declared as `vad: {stages: [vad], inputs: [input]}`; reports `speech.started` and `speech.stopped`. Its defaults apply to every session and a session's `turn_detection` overrides them; set them under `stages.vad.factory` or as `--vad.factory.<option>`.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `threshold` | number 0 to 1 | `0.5` | Speech probability above which a frame is speech. |
| `prefix_padding_ms` | integer ≥ 0 | `300` | Audio before the first speech frame that belongs to the utterance. |
| `silence_duration_ms` | integer ≥ 0 | `500` | Silence that ends an utterance. |
| `max_open_sessions` | integer ≥ 1 | `16` | Sessions the detector holds at once. |
| `max_concurrency` | integer ≥ 1 | `4` | Sessions it processes at once. |

### Stage `turn_detection`

Declared as `turn_detection: {stages: [turn_detection], inputs: ["input:pcm16", "input:session", "vad:session"]}`; reports the turn's `speech.started` and `speech.stopped`, which the recognizer and the dialog read in place of the VAD's. A session's `turn_detection` overrides `eagerness`; set the options under `stages.turn_detection.factory` or as `--turn_detection.factory.<option>`.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `model_path` | string | `pipecat-ai/smart-turn-v3` | The Smart Turn ONNX file, its directory, or the Hugging Face repository holding `smart-turn-v3.2-gpu.onnx`. |
| `eagerness` | `low`, `medium` or `high` | `medium` | How readily a pause ends the turn: the silence a confident, a likely and an unlikely end of turn needs. |
| `max_open_sessions` | integer ≥ 1 | `16` | Sessions the node holds at once. |
| `max_concurrency` | integer ≥ 1 | `4` | Sessions it processes at once. |

### Stage `asr`

Declared as `asr: {stages: [asr], inputs: ["input:pcm16"]}`; runs Nemotron 3.5 streaming ASR on the GPU set by `stages.asr.gpu` (default 0) and produces the running `text` a `running_transcript` node reads.

### Compatibility and Constraints

### Rejected when the profile loads

Each rule is followed by the error it raises.

- A node names both `stages` and a `kind`, or neither — `a session node names either stages or a kind`.
- A stage route with options beside `stages` — `options [...] belong to a node of a kind`.
- An unknown `kind` — `unknown component kind '<kind>'`.
- An option the kind does not have — `Extra inputs are not permitted`.
- A missing required option (`base_url`, `model`) — `Field required`.
- A node name of a kind that is not lowercase letters, digits and `_` starting with a letter — `String should match pattern '^[a-z][a-z0-9_]*$'`.
- A node named `input` — `'input' names the session input, not a node`.
- No node reads `input` — `no node reads 'input'`.
- A node reads itself — `node '<name>' reads itself`.
- An input names a node that does not exist — `node '<name>' reads unknown inputs [...]`.
- A `<node>:<format>` input that the source does not produce or the reader does not consume — `node '<name>' cannot take [...]`.
- A required format no input produces (every consumed format except `session`, such as `messages` for `openai_chat`) — `no input of node '<name>' produces [...]`.
- A cycle of data edges without a `feedback: true` input — `nodes [...] form a cycle of data edges; declare the input that closes it with feedback: true`.
- A `feedback` edge with `overflow: block` — `the feedback edge from '<source>' into '<name>' must not block`.
- Two different bounds on inputs from one source — `node '<name>' bounds its edge from '<source>' twice`.
- A node takes `reply.progress` on control edges from two sources — `node '<name>' takes reply.progress from [...]; one source reports it`.
- A node reads `reply.progress` of a source as data (the `alignment` kind does) while a control edge takes the same event from that source — `node '<name>' reads ['<source>:reply.progress'] as data, but a control edge takes them`.
- `session_publish` names an unknown node — `publish names unknown nodes [...]`.
- `session_publish` lists a format the node does not produce — `node '<name>' does not produce [...]`.
- No published `session` of `openai_chat` or `openai_speech` while a `dialog` exists — `no published node ends turns`.
- Both `openai_chat` and `openai_speech` publish `session` — `turns end twice: [...] all publish turn.done`.
- A `dialog` whose `session` is not published — `nodes [...] open turns and must publish 'session'`.
- A stage name other than `vad` or `asr` — `no stage group serves the stages [...]`.
- `base_url` of `realtime_transcription` not `ws://` or `wss://`, or of the other service kinds not `http://` or `https://` — `String should match pattern ...`.
- `method: forced` without `alignment_url` — `the forced method needs alignment_url`.
- `alignment_url` with a method other than `forced` — `alignment_url belongs to the forced method`.
- Only one of `heard_asr_url` and `heard_asr_model` — `heard_asr_url and heard_asr_model go together`.
- `method: asr` without them — `the asr method needs heard_asr_url and heard_asr_model`.
- `heard_asr_url` with `method: sentence` — `heard_asr_url belongs to the asr and forced methods`.
- `extra_body` sets `model`, `messages`, `stream`, `max_tokens` or `temperature` — `extra_body cannot set [...]`.
- `temperature` outside 0 to 2, a non-positive `max_tokens`, `max_sentence_words`, `max_history_chars`, `max_text_chars` or `timeout_s`, or a negative `tail_ms` — a pydantic range error naming the field.
- Two `services` with one name or one port — `services must have distinct names and ports`.
- An empty `model_path` — `Model path is required`.

### Rejected on the command line or at startup

- `--stages.<...>` — `the stages. prefix is implied on the command line; write --<...>`.
- `--services.0.<...>` — `services is addressed by name, not by index`.
- A flag naming a group instead of a value — `addresses a settings group, not a single value`.
- `--mem-fraction-static` — `--mem-fraction-static requires a pipeline with at least one SGLang engine stage`.
- A service that exits or does not answer `/v1/models` on its port — the service errors under [One command with a profile](#one-command-with-a-profile).
- A node whose service does not answer its probe: `openai_transcription`, `openai_chat` and `openai_speech` request `<base_url>/models`, `openai_chat` then streams one token sampled with the node's settings, `realtime_transcription` requests `/v1/models` on the `http` form of its `base_url`, a forced `alignment` requests `<alignment_url>/health` and then aligns one second of silence, and an `alignment` with `heard_asr_url` requests `<heard_asr_url>/models` — `cannot reach <url>: ...` or `<url> answered <status>`, or the error of the chat token's completion, and the server stops.

### Rejected in a realtime session

- A sampling parameter other than `temperature` and `max_new_tokens_per_unit` — `unsupported sampling parameters: ...`.
- `sglang.sampling` or another non-hot field changed after the first `session.update` — `session field is frozen`.
- `instructions` longer than 65536 characters — `instructions exceed context or have invalid type`.
- A `model` other than the deployment's — `model differs from deployment`.
- An audio format other than 16 kHz input or 24 kHz output PCM — `unsupported PCM format or sample rate`.
- A `turn_detection` with a `type` other than `server_vad`, or other than `server_vad` and `semantic_vad` in a profile with a `turn_detection` node — `turn detection type is unavailable`; any non-null `turn_detection` in a profile without a `vad` node — `VAD is only available for turn-based sessions`.
- `nodes` in `session.update` — `invalid session configuration`.
- `response.create` — `manual turn commands are not granted`.
- `response.cancel` in a profile where no node takes `interrupt` from `input` on a control edge — `no node takes the control 'interrupt' from 'input'`.
- A new session beyond the 16 that every cascade node holds at once — the first `session.update` is refused with `admission_rejected`.
- A `session.update` that changes `instructions`, or a `configure` that changes a node, while the user is speaking, between `speech_started` and `speech_stopped` — the session fails with `cannot replace '<node>' while speech is in progress`.

### Rejected by the speech service

The speech service validates each sentence request, and a rejected request fails the session.

- Qwen3-TTS CustomVoice with `reference_audio_path` — `references are not supported by this model`.
- Qwen3-TTS CustomVoice with a `voice` that is not one of its speakers — `Unknown voice '<voice>'. Supported voices: default, ...`.
- Qwen3-TTS Base without `reference_audio_path` — `Qwen3-TTS Base requires reference audio via ref_audio or references[0].audio_path`. With a reference but no `reference_text`, Base clones from the speaker embedding of the recording alone.
- A `language` tag the service does not speak, including a language name the recognizer reported under `language: auto` — `language must be one of: ...`. Qwen3-TTS speaks Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish and Italian.
- A speech service that does not stream 24 kHz PCM with an `X-Sample-Rate` header — `speech service sent no X-Sample-Rate header, so its PCM rate is unknown` or `speech service streams <rate> Hz, expected 24000`.

### Runtime limits

- A conversation whose history, next utterance and `max_text_chars` together exceed `max_history_chars` — `conversation exceeds max_history_chars; open a new session`, and the session fails. With the defaults this happens once the history passes about 49500 characters.
- An utterance transcript or a reply longer than `max_text_chars` — `transcript exceeds max_text_chars` or `reply exceeds max_text_chars`.
- A language model stream that stops for a reason other than `stop` or `length` — `LLM stopped with <reason>`; one that ends without a finish reason — `LLM stream ended before completion`.
- A client that stops reading while `max_pending_output_chunks` or `max_pending_output_bytes` of output wait — the session fails naming the limit.

Every failure ends the realtime session with an `error` event.

### Allowed, with consequences

- Without an `alignment` node, the dialog's `reply.progress` control edge comes from `tts`, which reports sentence ends, so an interrupted reply is cut at the end of the last sentence heard in full.
- `method: sentence`: the same sentence-level cut. With `forced`, the cut can fall between words of a finished sentence; a sentence cut off before its audio is complete counts as unheard in both methods unless `heard_asr_url` is set.
- With `heard_asr_url`, a transcript that leaves out, repeats or changes the first words of the sentence at the cut keeps none of it.
- No `reply.progress` input on the dialog at all: the profile loads, and an interrupted reply leaves an empty assistant entry in the history.
- An aligner that does not read `tts:session` loads but receives no sentence ends and reports nothing.
- `sglang.response.progress` reaches the client only from a node that publishes `session` and whose `session` no other node reads: the aligner when it is published, never `tts` while the dialog or an aligner reads it.
- A client that sends no `conversation.item.truncate`: the history keeps an interrupted reply up to the audio the server had delivered, which is ahead of what was played by up to `max_output_lead_ms`. With `truncate`, the cut moves back to the client's `audio_end_ms`, never past the delivered audio.
- `language: auto` with `realtime_transcription`, `running_transcript` or `response_format: json`: no language is ever reported, so every sentence is tagged by its script, `Chinese` with Chinese characters and `English` otherwise. A fixed `language` tags every sentence without Chinese characters with that name whatever the user speaks.
- `turn_detection: null` sent after the VAD has already reported speech in the session: the recognizer keeps waiting for speech boundaries, and each `input_audio_buffer.commit` then ends an empty turn (`empty_turn`). Turn detection off from the first `session.update` works as intended; `running_transcript` ends utterances at a commit either way.
- `input_audio_buffer.commit` is granted in every profile (`client_commit: true`); with the VAD on, it also ends an open utterance.
- `max_text_chars` at or above `max_history_chars` loads, and then every turn fails with `conversation exceeds max_history_chars`.
- A reasoning language model that thinks by default speaks its reasoning unless `extra_body` turns thinking off, as `chat_template_kwargs: {enable_thinking: false}` does in `qwen3-asr-qwen3.8-27b-customvoice`.
- A reply cut by `max_tokens` (finish reason `length`) is spoken and kept as it is.
- Speech always interrupts a running reply; `interrupt_response` does not change that.

### Verifying a Deployment

A healthy start logs each service as `starting service '<name>': ...` and `service '<name>' serving on :<port>` (or `already serving`), then `Pipeline '<model_path>' started (<n> GPU(s))`, then the HTTP server listening on its port. A failed node probe logs `Scheduler thread for stage <node> crashed` with the probe error, and the server stops.

`curl http://127.0.0.1:8765/health` answers 200 with `"status": "healthy"` while the cascade runs, and 503 when it does not.

`curl http://127.0.0.1:8765/v1/models` lists one model whose `id` is the profile's `model_path`, the name a realtime client must use if it sends `model`.

`curl http://127.0.0.1:8765/v1/realtime/capabilities` answers 503 `instance is not ready` when the cascade is not running, and 404 without `--enable-realtime`. For the bundled profiles it reports:

```json
{
  "model": "examples/streaming_cascade/profiles/qwen3-asr-qwen3-tts-customvoice",
  "interaction": "turn_based",
  "turn_control": ["server_vad", null],
  "client_commit": true,
  "input_audio_format": {"type": "audio/pcm", "rate": 16000},
  "output_audio_format": {"type": "audio/pcm", "rate": 24000},
  "output_modalities": ["text", "audio"],
  "native_unit_ms": 200,
  "supports_truncate": true,
  "sampling_parameters": ["temperature", "max_new_tokens_per_unit"],
  "limits": {"max_input_bytes": 1920000, "max_output_bytes": 33554432, "max_history_chars": 65536, "cleanup_timeout_s": 30}
}
```

with further fields omitted here. `turn_control` is `[null]` for a profile without a `vad` node, `output_modalities` is `["text"]` for one that publishes no audio, and `sampling_parameters` is empty when no node takes `temperature` or `max_tokens`. A first conversation turn through the playground or the [turn latency benchmark](#benchmarking) confirms the whole chain.

### Known Limitations

- **Reasoning models must have thinking off.** The cascade speaks everything the language model returns as content, so a model that thinks by default needs `enable_thinking: false` in the node's `extra_body` (as in `qwen3-asr-qwen3.8-27b-customvoice`), or its reasoning is synthesized.
- **History is bounded.** A conversation whose history exceeds `max_history_chars` fails its next turn; open a new session.
- **Audio only.** The cascade takes audio input and produces text and audio; image input is not supported.

## Session Graph

A Session is one stateful conversation on the server: the client opens it, appends input for its whole lifetime, reads what it publishes and closes it. A pipeline declares the shape of its Session in its config, and the coordinator opens it; the serving layer appends input and reads the published formats without knowing the nodes behind them. A Session is either the one stage route from the pipeline's entry stage or a graph of nodes declared in `session_nodes`.

### Nodes

Every node of the Session is an entry of `session_nodes`, and `session_publish` lists, per node, the formats that reach the client. A node with `stages` is a route through those stages of the server.

`inputs` lists the nodes a node reads; `input` is the Session's own input, its audio, its `commit` and other controls, and camera frames when a node of the Session takes them. `node:format` takes only that format from that node; two nodes never send the same format to one reader, and the audio format of `input` is the Session input's alone. A node that reads every format of `input` while the Session takes camera frames must take them too, or name the formats it reads, such as `input:pcm16` and `input:session`.

An input written `{kind: control, source, event, preempt}` is a control edge. Its `event` is a control type of the source's `session` chunks or the format of its other chunks, such as `tool_response`; the event is handed to the node as a `control` operation, outside its input queue, to every stage of its route in route order, and travels on no other edge and to no client. With `preempt` the stages cancel the node's in-flight appends first. The client's `interrupt` reaches a node through a control edge from `input`.

An input of a stage node written `{source, capacity, overflow}` bounds its edge. The inputs naming one source are one edge into the node; it holds at most `capacity` chunks the node's own session has not accepted yet, by default the Session's `max_pending_chunks`. When the edge is full, `overflow: block`, the default, holds the producing node until the node takes a chunk, and rejects the client's append as a full queue; `overflow: drop_oldest` drops the edge's oldest chunk and lets the producer go on; on the client's edge the dropped input is confirmed to the client like a taken one, so a real-time node such as a full-duplex model hears the newest audio instead of a backlog. Two inputs of one source that declare different bounds fail the declaration.

The Session opens its nodes upstream first and closes them downstream first. A cycle of data edges between nodes must name the input that closes it with `feedback: true`, and that input must not block: two nodes each held on the other's full edge never wake. An undeclared cycle fails with `form a cycle of data edges`. A control edge holds no queue, so a cycle it closes needs no mark.

### Stream vocabulary

| Format | Modality | Payload |
| --- | --- | --- |
| `pcm16` | `audio` | 16 kHz mono signed 16-bit PCM |
| `pcm_f32le` | `audio` | 24 kHz mono float32 PCM |
| `image_frame` | `image` | One JPEG or PNG camera frame, at the time of the audio unit it came with |
| `transcript` | `transcript` | `{"text", "is_final"}`; one final transcript per utterance |
| `messages` | `text` | `{"messages": [{"role", "content"}, ...]}` |
| `text_delta` | `text` | `{"text"}`, one increment of the reply |
| `tool_call` | `tool` | `{"calls": [{"id", "name", "arguments"}, ...]}`; the generator asks for these tools |
| `tool_response` | `tool` | `{"responses": [{"id", "name", "response"}, ...]}`; one response per call, in call order |
| `session` | `control` | `{"type", ...}` |

Control types sent into a Session are `commit` (the utterance ends here) and `configure` (settings change). Control types a Session emits are `speech.started` and `speech.stopped` with `t_ms` on the input timeline, `turn.queued` when an utterance waits for its reply, `turn.started` when the reply begins, `turn.done` after the reply's audio, `turn.empty` for an utterance without words, and `configured` with `{"nodes": {name: kind}}`. Every chunk of a reply carries its turn number in the chunk's `turn` field, numbered from 1 per Session in utterance order: the turn controls, the `messages`, `text_delta`, `tool_call` and `tool_response` chunks and the reply audio; every turn opens with `turn.queued` and closes with `turn.done` or `turn.empty`.

### VAD node

Voice activity detection is a stage node: `sglang_omni.models.vad` hosts one Silero detector per Session in a CPU stage of its own, `vad`, declared with `stages: [vad]`, reading the Session input's audio and controls and publishing `session`. It emits `speech.started` and `speech.stopped` controls whose `t_ms` is the boundary on the input audio timeline, `speech.started` reaching `prefix_padding_ms` before the first speech frame; `commit` resets the detector. Nodes that place utterances read it through a `vad:session` edge. Its detector defaults are the stage's `factory` options `threshold`, `prefix_padding_ms` and `silence_duration_ms`; a Session's `turn_detection` overrides them for that Session alone, at open and through `configure`.

### Turn detection node

End-of-turn detection is a stage node beside the VAD: `sglang_omni.models.turn_detection` runs the Smart Turn v3.2 model in a CPU stage of its own, `turn_detection`, declared with `stages: [turn_detection]`, reading the Session input's audio and controls and `vad:session`, and publishing `session`. It emits the same `speech.started` and `speech.stopped` controls as the VAD, at the turn's boundaries: the VAD's `speech.started` opens a turn, and at each `speech.stopped` of the VAD the node scores the turn's audio, ending the turn after the silence that score requires, up to the longest pause of the eagerness preset; speech that resumes before then continues the turn, and `commit` forgets it. Nodes that place utterances or interrupt replies on speech read `turn_detection:session` in place of `vad:session`. The stage's `factory` options are `model_path` (`SGLANG_OMNI_SMART_TURN_MODEL_PATH` when unset) and `eagerness`; a Session's `turn_detection` selects `semantic_vad` with an `eagerness`, or `server_vad`, which passes the VAD's boundaries through, at open and through `configure`. `SessionPorts.turn_detection_types` names the `turn_detection.type` values a Session's nodes implement, and the realtime bridge offers them as `turn_control`.

### Interruption

An `interrupt` naming a turn that reaches a node on a control edge first drops that turn's chunks from other nodes still queued for the node, on its edges and among its unsent appends, and a chunk of that turn that a node emits later no longer reaches it; the client's own input is never dropped. The node's own output of that turn, whether already buffered or emitted later, reaches neither its readers nor the client, except the `turn.interrupted` that ends the turn. Control edges are delivered upstream first, so a node's stale output is fenced before its readers hear the event. An `interrupt` without a turn, such as the client's `response.cancel`, means the open reply, which only the node knows; the Session leaves its queues to the node.

### Delivery pacing

A Session delivers reply audio in real time. For every node of the Session that publishes audio, an audio chunk leaves at most `max_output_lead_ms` ahead of the wall clock since the graph's published `turn.started` for the reply left, cut at that horizon; the node's later audio and its `turn.done` wait behind it, and its other chunks, such as transcripts, reply text, speech boundaries and the confirmation of each input append, pass, so the client keeps appending while the reply plays. Once the input has ended everything leaves at once. Audio leaving together, at the input's end or when delivery catches up, is joined per turn into contiguous chunks of at most `max_output_lead_ms`, so a held 40 s reply reaches the client as about 40 chunks rather than one per generated chunk. A Session that is one stage route paces its output the same way. `max_output_lead_ms` is a field of the pipeline config (default 1000); an interruption cannot take back audio the client already holds, so it bounds how long the client may keep speaking after the user started.

When the Session releases a node's `turn.done`, it delivers it to the nodes reading it and returns it to the node itself as a `control`, so a node can keep its reply open, and interruptible, until the reply's audio has left. A `turn.interrupted` the node publishes, or an `interrupt` control naming a turn, drops what the Session still holds of that turn, its `turn.done` included. Held audio counts against the Session's `max_output_bytes`. A `reply.progress` reaches the client only from a node that publishes `session` and whose `session` no node reads, the end of the route that builds the progress: with an aligner reading the speech node's `session`, the client hears the aligner's progress and not the speech node's, and without one it hears none.

### Session API

A Session takes four operations: `open`, `append`, `control` and `close`. `Client.control_session(identity, event, stages=None, should_preempt=False)` delivers a `session` control chunk such as `interrupt` to every stage of the Session's route, or to the named stages, in the Session's arrival order with its appends; `should_preempt` cancels the stages' in-flight appends first. On a Session graph the event follows the control edges declared from `input`. A Session delivers its outputs as the nodes produce them, between appends as well as during one, so a client that has spoken may wait for the replies without appending.

A `/v1/session` client opens the Session the pipeline declares; the open frame carries `params`, and the Session's limits are the deployment's. An `append` frame carries a `chunk` and is answered with `accepted`; a `control` frame carries the event as its `chunk`, with optional `stages` and `should_preempt`, and is answered with `controlled`. Both answers carry the chunk's seq, and a full queue answers either with `rejected` and leaves the Session open.

A chunk's `turn` identifies a cancellable reply within the Session, not a tool
call or an input audio frame. All work belonging to that reply preserves the
same ID, including tool responses. Interrupted reply IDs remain fenced for the
lifetime of the graph, including tool results carried by control edges. Session
control commands such as interrupt and truncate remain deliverable. Nodes must discard any matching
work they have already buffered when they receive the interrupt.

A producer without reply boundaries leaves `turn` unset. VoiceChat's continuous
duplex stream does not currently allocate reply turns; tool call IDs identify its
individual calls. Passing through a supplied `turn` does not add native speech
interruption or reply-boundary detection to VoiceChat.


### Cascade components

The cascade config's `kind` shorthand expands each component into its own CPU stage and Session node, with the stages sharing the `cascade` process. Data and control edges use the same coordinator runtime as native model stages. Feedback edges must be declared explicitly; a native stage may read a component's output directly.

A node declares either a component `kind`, with its options beside it, or a `stages` route for a native model or the VAD. `inputs` lists the sources it reads: `input` is the client's audio and controls, `node` a node's every format, `node:format` one format, `input:session` the client's controls, and `{kind: control, source, event}` a control edge. A component accepts the formats its kind consumes; the declaration is validated when the config loads: names are unique, every input exists, every format a component requires other than `session` controls is produced by one of its inputs, exactly one reply-ending kind publishes `session`, and the kinds that open turns (`dialog`) and end them (`openai_chat` or `openai_speech`) are present once. At startup every service kind is probed once: `openai_transcription`, `openai_chat` and `openai_speech` request `{base_url}/models`, `openai_chat` also streams one token sampled with the node's settings so a kernel the service compiles on its first sampled request is built before the server is ready, `realtime_transcription` requests `/v1/models` on the HTTP form of its `base_url`, and a forced `alignment` requests `{alignment_url}/health` and aligns one second of silence; a failed probe fails the start with the node's name.

A kind is a config class with a default `kind`, a constructor and the formats it consumes and produces, registered through `COMPONENTS.register` in `sglang_omni.models.streaming_cascade.components`; `opens_turns` marks the kind that numbers turns, `ends_turns` the one whose `turn.done` ends a reply, and `optional` the consumed formats a configuration may leave unconnected. `running_transcript` is the kind that turns the one running text of a Nemotron 3.5 ASR stage served as a node of the same Session into utterance transcripts, cut at `speech.stopped` or at a commit plus `tail_ms`.

A `configure` control reaches every node reading `input`. Its standard fields reach every node with the matching option, `instructions` as `system_prompt` and `sampling` as `temperature`, `max_tokens`; `nodes: {name: {...}}` changes one node's options, and a different `kind` replaces its component, which adopts the previous one's state. Names and inputs cannot change; a node is not replaced while it is speaking, the change waits for the reply to end. Each node answers with `configured {"nodes": {name: kind}}` on its `session` output.

An EOS append ends the input readers in graph order, after their upstream input readers finish. Its confirmation waits for pending routed input and published reply turns to drain. Background component failures fail the output read without waiting for another append.

The cascade graph enables `shared_timeline`: its published reply chunks use one coordinator-owned media clock. A reply starts no earlier than the latest client audio or the previous reply's audio end; audio advances that clock, and interruption resets it to the client audio clock. Speech boundaries retain their original input timestamps. Other graphs preserve their nodes' timestamps by default.
