# Full-Duplex Guide

This page consolidates three parts of the SGLang-Omni developer reference on real-time voice conversation, covering three layers: how the system works (the Full Duplex Interactive Model), how to benchmark it (the Full-Duplex-Bench runbook), and how to interpret the benchmark numbers (the v1.0 unified metrics). Together they form the complete chain: **Service → Measurement → Interpretation**.

This page covers full-duplex serving and benchmarking in SGLang-Omni in three parts: the [Full Duplex Interactive Model](#full-duplex-interactive-model), the [Full-Duplex-Bench runbook](#full-duplex-bench-runbook), and the [v1.0 unified metrics](#full-duplex-bench-v10-unified-metrics) specification.

## Full Duplex Interactive Model


SGLang-Omni serves a live voice conversation on `/v1/realtime` in two ways. A native full-duplex model such as MiniCPM-o 4.5 listens and speaks at the same time and decides itself when to talk. The streaming cascade composes independent speech recognition, language model and speech synthesis services into a turn-based conversation, and the server adds voice activity detection, the conversation history, turn taking and interruption. Both run as a Session on the server, a stateful conversation described under [Session Graph](#session-graph).

| Model | Interaction | Launch |
| --- | --- | --- |
| MiniCPM-o 4.5 | Native full duplex, audio and camera input | [Native Full Duplex: MiniCPM-o](#native-full-duplex-minicpm-o) |
| PersonaPlex | Native full duplex, audio input | [Native Full Duplex: PersonaPlex](#native-full-duplex-personaplex) |
| Nemotron VoiceChat | Native full duplex, audio input | [Native Full Duplex: Nemotron VoiceChat](#native-full-duplex-nemotron-voicechat) |
| Streaming cascade | Turn-based, with interruption | [Streaming Cascade](#streaming-cascade) |

### Native Full Duplex: MiniCPM-o

[MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) is served for full duplex with its `session` variant; [MiniCPM-o](../cookbook/minicpm_o.md) covers its installation and chat usage.

```bash
python -m sglang_omni.cli serve \
  --model-path openbmb/MiniCPM-o-4_5 --variant session \
  --enable-realtime --port 8000
```

`--variant session` selects the full-duplex pipeline, which `serve` refuses to start without `--enable-realtime`. Every setting below is changed with a flag of the same name, for example `--max_sessions 4`, and `config resolve` with the same arguments prints the whole configuration:

```bash
sgl-omni config resolve --model-path openbmb/MiniCPM-o-4_5 --variant session
```

| Serving | Flags |
|---|---|
| Normal | None. The defaults match the MiniCPM-o demo |
| Repeatable output for regression and parity recordings | `--sampling.greedy true --sampling.top_k 100 --thinker.engine.disable_cuda_graph true --talker.engine.disable_cuda_graph true`: greedy sampling, `top_k` 100, and the thinker and talker without CUDA graphs |

| Setting | Default | Meaning |
|---|---|---|
| `max_sessions` | 2 | Conversations at the same time. Further connections get HTTP 503. The thinker and talker reserve GPU memory for this many full-length conversations, so raise it only as far as the GPU has room. Startup warms up perception at each batch size up to this value |
| `batch_sessions` | `true` | Perception and speech handle the ready second of every conversation in one call. `false` gives each conversation's second a call of its own, one call at a time; `max_sessions`, memory and startup stay the same, and the thinker and talker still batch conversations |
| `reference_audio` | checkpoint default | Voice used when a session sends no reference |
| `speech_state_bytes_per_session` | 2 GiB | Memory the speech stage may hold per conversation. A conversation that needs more is closed and the others keep running |
| `speech.dtype` | `float32` | Precision of the voice decoder's flow model: `float32`, `float16` or `bfloat16`; the lower precisions change the voice slightly |
| `speech.enable_dit_torch_compile` | `false` | Compile the voice decoder's flow model with `torch.compile` while the server starts; CUDA only |
| `speech.n_timesteps` | 5 | Flow-matching steps per audio chunk; fewer steps decode faster at some cost in voice quality |
| `speech.enable_chunk_cuda_graph` | `true` | Replay the voice decoder's flow from the CUDA graphs described below; `false` decodes every chunk eagerly and skips their capture at startup |
| `speech.gather_window_ms` | 22.5 | How long an idle speech stage waits for the other conversations' next second so it can decode them together; the wait ends once every open conversation has one. `0` decodes what is ready |
| `stages.thinker/talker.engine.enable_torch_compile` | `false` | Compiles every decode graph batch size; adds minutes to startup |
| `sampling` | `MiniCPMODuplexSampling` in `sglang_omni/models/minicpm_o/native_config.py` | Default sampling when a session does not set its own; `--sampling.<field>` changes one, such as `--sampling.temperature 0.5` |
| `vision` | `MiniCPMODuplexVision` in the same file | Camera-frame limits per unit (1 s of audio); `--vision.<field>` changes one, such as `--vision.max_frames_per_unit 2` |

While the server starts, the speech stage records the voice decoder as CUDA graphs for up to 8 conversations decoded together, or `max_sessions` if lower; the graphs hold GPU memory and add to startup time, and a voice whose reference audio is longer than the default voice's is decoded without them. `speech.enable_chunk_cuda_graph: false` turns them off.

Speech settings also accept dotted CLI flags, such as `--speech.dtype float16 --speech.n_timesteps 10`. Stage settings use `--speech.factory.<field>` or `--speech.process`. The automatic thinker KV budget follows `thinker.engine.dtype` and `thinker.engine.kv_cache_dtype`; block-quantized KV caches require an explicit `thinker.engine.kv_cache_bytes` budget.

A session holds at most 8192 tokens of history, which is the model's trained context length. When that fills, the server sends `context_exhausted` and closes the session.

The realtime activity timeout closes an inactive session with `idle_timeout`; the pipeline's separate idle fallback reports `session_idle_timeout`. A session whose pending server events outgrow the server's output buffers is closed with `output_budget_exhausted`; this happens when the client stops reading events. A single response whose transcript outgrows `max_history_chars` is closed with `context_limit`.

The full-duplex server is ready when this returns JSON containing `"native_full_duplex":true`:

```bash
curl --fail http://localhost:8000/v1/realtime/capabilities
```

The results should be as follows:

```json
{"model":"openbmb/MiniCPM-o-4_5","interaction":"native","native_full_duplex":true,"proactive_output":false,"turn_control":[null],"client_commit":false,"input_modalities":["audio","image"],"output_modalities":["audio","text"],"input_audio_format":{"type":"audio/pcm","rate":16000},"output_audio_format":{"type":"audio/pcm","rate":24000},"native_unit_ms":1000,"first_unit_ms":1000,"microturn_ms":"variable","tail_policy":"pad","supports_server_interrupt":false,"supports_truncate":true,"supports_cancel":true,"supports_resume":false,"partial_style":"append_only","pressure_policy":"reject","strict_order":true,"sampling_parameters":["greedy","temperature","top_k","top_p","repetition_penalty","listen_prob_scale","force_listen_count","max_new_tokens_per_unit","repetition_window_size","talker_temperature","talker_repetition_penalty"],"supports_reference_audio":true,"supports_instruction_updates":false,"supports_sampling_updates":false,"input_image_format":{"types":["image/jpeg","image/png"],"max_bytes":524288,"max_frames_per_unit":4,"max_slice_nums":9},"limits":{"max_input_bytes":1920000,"max_output_bytes":33554432,"max_history_chars":65536,"cleanup_timeout_s":30,"session_update_timeout_s":60,"idle_timeout_s":300}}#
```

The duplex server only accepts `/v1/realtime` requests; other chat completions or voice cloning against it fail.

(browser-demo)=

**Browser demo**


With the full-duplex server running, start the demo page in another terminal:

```bash
python playground/realtime/app.py --api-base http://127.0.0.1:8000 --port 8080
```

Thanks to the OpenMoss team for the demo page.

Once the terminal prints `Running on`, open <http://localhost:8080> and click **Start talking** to talk through your microphone; the camera can be turned on during the call. If the server is remote, first run `ssh -N -L 8080:127.0.0.1:8080 USER@SERVER` on your own machine. Open the page through `localhost`, or the browser will not grant microphone access.

The settings panel shows only the controls the deployment declares in `/v1/realtime/capabilities`: the system prompt, the presets and a reference voice when `supports_reference_audio` is true (a preset pairs a MiniCPM-o duplex prompt with a voice, so no other backend gets them), a named voice when `voices` is listed, turn detection when `turn_control` lists `server_vad` or `semantic_vad`, camera detail when `input_image_format` allows more than one slice, and one numeric field per key in `sampling_parameters`, labelled for that backend (`max_new_tokens_per_unit` is tokens per 1 s unit on a native model and the reply token limit on a turn-based one). An empty field is not sent, so the deployment's default applies. The page sends every setting in its first `session.update`, so a change applies to the next conversation. Everything under `sglang` (sampling, voice, reference audio, image detail) is fixed once the session is open. Turn detection and `output_modalities` may change in a later `session.update`, and `instructions` only on a deployment that supports instruction updates, which is the cascade with a `system_prompt` node; MiniCPM-o, PersonaPlex and Qwen3-Omni freeze them with the first `session.update`. Settings are kept in the browser per deployment model. The download button in the top bar saves the conversation trace; attach it when reporting a problem.

The stop button in the call dock follows `supports_cancel` from `session.updated`. Where it is granted, as on MiniCPM-o, the Qwen3-Omni session graph and the cascade, the button sends `response.cancel`, shows Stopping until the server's `response.done` for the reply arrives, and Stopped when that status is `cancelled`. A reply that ended on its own in the meantime shows no stop. This page sends no `conversation.item.truncate`, so a turn-based history keeps what the server delivered. Where it is not granted, as on PersonaPlex and Nemotron VoiceChat, the button is labelled as a local mute: it sends nothing and the reply is not cancelled on the server. In both cases the page then plays no model audio, under any response id, until you speak again: on a turn-based deployment the server's `input_audio_buffer.speech_started` lifts the mute, on a native one the microphone's own level does, since a stopped model may continue its answer under a new response and a new response id alone is no proof of a new answer. Speaking over the model is left to the model on a native deployment.

An empty field falls back to the deployment's default, which lives here:

| Backend | Sampling | Turn detection and voice |
| --- | --- | --- |
| MiniCPM-o | `MiniCPMODuplexSampling` in `sglang_omni/models/minicpm_o/native_config.py`; `--sampling.<field>` changes one | No turn detection. The voice is the deployment's `reference_audio` |
| PersonaPlex | `PersonaPlexSessionSampling` in `sglang_omni/models/personaplex/config.py` (`--sampling.<field>`); a field left unset there keeps the offline value in `sglang_omni/models/personaplex/architecture.py` and `request_builders.py` | No turn detection. Voice `NATF2` unless the deployment sets `voice` |
| Qwen3-Omni | `QwenSessionSampling` in `sglang_omni/models/qwen3_omni/session_preprocessing.py`; the deployment has no override, only a session changes them | `silence_duration_ms` 800 from the `vad` stage in `sglang_omni/models/vad/node.py`; `--vad.factory.silence_duration_ms` changes it |
| Streaming cascade | The profile's node options; `--session_nodes.llm.<option>` changes one. `max_tokens` is 512 in the bundled profiles, and a `temperature` the profile leaves unset is not sent, so the chat service applies its own | The `vad` stage as above unless the profile sets it: `qwen3-asr-qwen3-tts-base-clone` runs it at 160 ms as Smart Turn's candidate pause, so picking server VAD in the panel prefills 800 ms. `eagerness` `medium` from `sglang_omni/models/turn_detection/node.py`; `--turn_detection.factory.eagerness` changes it |


(full-duplex-protocol)=

**Full-duplex protocol**


Clients connect to `/v1/realtime` over WebSocket. They send 16 kHz mono PCM16 audio in `input_audio_buffer.append`; the server returns 24 kHz audio in `response.output_audio.delta` and text in `response.output_audio_transcript.delta`. The model decides once per second of audio whether to keep listening or to speak.

1. After `session.created`, send `session.update` and wait for `session.updated`.
2. Send audio with `input_audio_buffer.append` at the pace it is captured; replies arrive while you send.
3. When the audio ends, send `sglang.input_audio.end`, wait for `sglang.input_audio.drained`, then send `session.close`.

When the session ends, the server closes the WebSocket with one of these codes:

| Close code | Meaning | Close reason |
|---|---|---|
| `1000` | Normal close | `client_closed`, `disconnect` |
| `1008` | A server limit was reached | `session_update_timeout`, `idle_timeout`, `session_idle_timeout`, `context_exhausted`, `context_limit`, `output_budget_exhausted` |
| `1011` | Server error | For example `internal`, `session_open_timeout` or `cleanup_timeout` |

If a session has no input and no output for `realtime_idle_timeout_s`, the server closes it with `idle_timeout`. The pipeline configuration defaults to 300 seconds for native sessions and 6 hours for audio Session graphs, including the streaming cascade. WebSocket pings do not count as activity. Set `realtime_idle_timeout_s` in the pipeline YAML or override it with `--realtime-idle-timeout-s`. The server also waits for the first `session.update` up to `realtime_session_update_timeout_s` (default 60 seconds), which can be overridden with `--realtime-session-update-timeout-s`.

For a Python client, see `warmup` in `playground/realtime/app.py`.

Session settings go in the `sglang` field of `session.update`, before the first audio packet, and stay fixed for the session:

| Setting | Field | Notes |
|---|---|---|
| Voice | `reference_audio` | `{"media_type": "audio/wav", "data": "<base64>"}`, a PCM16 WAV of at most 30 s and 1 MiB; `tts_reference_audio` changes only the output voice |
| Sampling | `sampling` | For example `temperature`, `top_p` and `listen_prob_scale`; unset fields keep the server's defaults (`sampling` above) |
| Image detail | `max_slice_nums` | Higher is sharper but accepts fewer frames per second |

Send camera frames with `sglang.input_image.append`: a base64 JPEG or PNG in `image`, and its position on the audio timeline in `sglang.t_ms`. By default up to 4 frames per second are accepted; `session.updated` reports the actual limit.

`response.cancel` ends the reply in progress: the turn ends, the server drops the audio of that reply it has not sent yet, and `response.done` arrives with status `cancelled` and reason `interrupted` about one second later, once the unit that was running completes. The unit the client did not receive is dropped from the session history, and the thinker then listens on every unit until the user speaks again, so the model does not resume the stopped answer. A cancel with no reply in progress, or a second cancel of the same reply, does nothing. `conversation.item.truncate` with `audio_end_ms` cuts a reply the client stopped, or one the model ended, at the last unit whose audio fully played, never past what the client received.

[Full-Duplex-Bench](#what-is-measured) measures a native full-duplex deployment.

### Native Full Duplex: PersonaPlex

[PersonaPlex](https://huggingface.co/nvidia/personaplex-7b-v1) is served for full duplex with its `session` variant; [PersonaPlex](../cookbook/personaplex.md) covers its installation and offline usage.

```bash
python -m sglang_omni.cli serve \
  --model-path nvidia/personaplex-7b-v1 --variant session \
  --enable-realtime --port 8000
```

| Setting | Default | Meaning |
|---|---|---|
| `max_sessions` | 1 | Conversations at the same time. Further connections get HTTP 503. The LM keeps `max_sessions + 1` request slots, and a lower `--lm.engine.max_running_requests` is rejected at startup |
| `voice` | `NATF2` | Packaged voice name, or the path of a voice recording, used when a session names no voice |
| `instructions` | the offline role prompt | Role prompt used when a session sends no `instructions` |
| `sampling` | the offline defaults | Default sampling when a session does not set its own: `temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty`, `audio_temperature`, `audio_top_k`, `seed` |
| `--code2wav.factory.max_graph_frames` | 2 | Mimi decode replays CUDA graphs for units of up to this many frames, for up to `max_sessions` conversations decoded together; `0` turns the graphs off |
| `--mimi_encode.factory.max_concurrency` | 1 | Conversations whose audio Mimi encodes at the same time |
| `--preprocessing.factory.voice_prompt_cache_bytes` | 512 MiB | Memory for the voice prompts loaded so far |

Every 80 ms of input has to pass Mimi encode, the LM and Mimi decode within 80 ms, or the conversation falls behind; concurrent conversations share that budget, and the LM decodes synchronously. The first conversation after the server starts waits several seconds while the LM runs its first steps. A conversation holds at most the LM context, 8192 positions or about 10.8 minutes (`--lm.engine.context_length` changes it); the unit that would pass it ends the session with `context_exhausted`.

The full-duplex server is ready when this returns JSON containing `"native_full_duplex":true`, `"native_unit_ms":80` and the packaged voices under `voices`:

```bash
curl --fail http://localhost:8000/v1/realtime/capabilities
```

The protocol follows the [MiniCPM-o steps](#full-duplex-protocol), with these differences:

| Setting | Field | Notes |
|---|---|---|
| Audio | `input_audio_buffer.append` | 24 kHz mono PCM16 in, 24 kHz out; the server pads a trailing partial 80 ms unit |
| Role prompt | `instructions` | Replaces the deployment's role prompt |
| Voice | `sglang.voice` | One of the `voices` in the capabilities |
| Sampling | `sglang.sampling` | The fields listed under `sampling` above; unset fields keep the deployment defaults |

The whole session is one response: `response.created` arrives with the first output, each unit brings 80 ms of agent audio (160 ms for the first), and `response.done` follows the end of the input. The transcript is the words the model speaks, streamed word piece by word piece. `response.cancel` and `conversation.item.truncate` are answered with `not_supported`, and the capabilities grant neither `supports_cancel` nor `supports_truncate`: a client can only mute its own playback, and the model keeps talking. The session variant only accepts `/v1/realtime` requests; use the default pipeline for `/generate` and chat completions.

### Native Full Duplex: Nemotron VoiceChat

[NemotronLabs VoiceChat 11B](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-VoiceChat-11B) is served for full duplex with its `session` variant; [Nemotron VoiceChat](../cookbook/nemotron_voicechat.md) covers its installation and offline usage.

```bash
python -m sglang_omni.cli serve \
  --model-path nvidia/NVIDIA-NemotronLabs-VoiceChat-11B --variant session \
  --enable-realtime --port 8000
```

Before it reports ready, the server runs one silent conversation, so the first real conversation compiles no kernels. A failed warmup stops the server, `SGLANG_OMNI_WARMUP_TIMEOUT` (seconds, default 600) bounds it, and `--skip-server-warmup` skips it.

| Setting | Default | Meaning |
|---|---|---|
| `max_sessions` | 1 | Conversations at the same time. Further connections get HTTP 503. Each conversation keeps its own perception, thinker, talker and codec state, and the thinker and talker reserve KV for `max_sessions + 1` conversations |

Every 80 ms of input has to pass perception, the thinker, the talker and the codec within 80 ms, or the conversation falls behind; concurrent conversations share that budget, and perception and the codec handle one conversation at a time. A conversation holds at most 6144 positions in the thinker and in the talker, about eight minutes; the unit that would pass it ends the session with `context_exhausted`.

The full-duplex server is ready when this returns JSON containing `"native_full_duplex":true` and `"native_unit_ms":80`:

```bash
curl --fail http://localhost:8000/v1/realtime/capabilities
```

The protocol follows the [MiniCPM-o steps](#full-duplex-protocol), with these differences:

| Setting | Field | Notes |
|---|---|---|
| Audio | `input_audio_buffer.append` | 16 kHz mono PCM16 in, 22.05 kHz out; the server pads a trailing partial 80 ms unit |
| Prompt | `instructions` | Not accepted: the model uses its checkpoint's system prompt, and a session that sends `instructions` is refused |

The whole session is one response, and every unit brings 80 ms of agent audio, silent while the model listens. The session variant only accepts `/v1/realtime` requests; use the default pipeline for offline recordings.

### Streaming Cascade

A profile names the services the cascade calls; one command starts them and the cascade, and the conversation is served on `/v1/realtime` and `/v1/session`.

(prerequisites)=

**Prerequisites**


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

(launching)=

**Launching**


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

One command starts everything: the profile's `services` are started in order and waited for, a service already answering on its port is reused, and the cascade starts once every service and the worker answer; stopping the server stops the services it started. The profile's `turn_detection` node ends a turn only when Smart Turn takes a pause for its end, so a pause in the middle of a sentence does not cut it, as [Turn Detection](#turn-detection) describes. When an interrupted reply is cut, the profile's aligner keeps the words the listener heard: the forced aligner's word ends for the sentences whose audio was complete, and its own recognizer's transcript of the delivered audio for the sentence it was cut in. The page sends the position it played with `conversation.item.truncate` and marks the heard part of the reply. [Profiles](#profiles) lists the other bundled profiles, such as `qwen3-asr-qwen3.8-27b-vivian` with Qwen3.8-27B on two GPUs.

To use services you already run, drop their `services` entries and point the nodes' `base_url` at them; a command-line override such as `--session_nodes.llm.base_url http://127.0.0.1:9000/v1` changes one field without editing the file. The rest of this section lists every launch mode and what each profile needs started beforehand; [Configuration](#configuration), [Node Reference](#node-reference) and [Compatibility and Constraints](#compatibility-and-constraints) list every setting and the settings that cannot be combined.

(one-command-with-a-profile)=

**One command with a profile**


The default CustomVoice cascade uses Qwen3.8-27B with thinking off on GPU 0 and Qwen3-ASR/Qwen3-TTS on GPU 1. The audio-file example in `examples/streaming_cascade/run.py` uses this profile when `--config` is omitted.

```bash
sgl-omni serve --config examples/streaming_cascade/profiles/qwen3-asr-qwen3-tts-vivian/launch.yaml --enable-realtime --port 8765
```

`--enable-realtime` mounts `/v1/realtime`, `/v1/realtime/capabilities` and `/v1/session`; without it the server starts the cascade but serves none of them.

The server starts the profile's `services` one after another, in the order they are listed, before it starts the cascade. For each entry it first requests `http://127.0.0.1:<port>/v1/models`: a service that answers 200 is reused and left running when the server stops, and the log shows `service 'tts' already serving on :8005`. Otherwise it runs `command` with its own environment plus the entry's `env`, in the directory `sgl-omni serve` was started from, and polls the port every second for up to 600 s. A service that exits first fails the start with `service '<name>' exited with code <n> before :<port> answered`; one that does not answer in time fails it with `service '<name>' did not answer on :<port> within 600 s`. The cascade itself must then start within `SGLANG_OMNI_STARTUP_TIMEOUT` seconds (default 600).

While the server runs, a service it started that exits stops the server with `service '<name>' exited with code <n>`. Stopping the server sends SIGTERM to every service it started and SIGKILL to any still running 15 s later.

Relative paths in a service `command` and in a node's `reference_audio_path` resolve against the directory `sgl-omni serve` was started from; the bundled profiles assume the repository root.

A service's port appears in three places that must agree: the `--port` inside its `command`, its `port`, and the `base_url` of the node that calls it. To move a service, change all three in the profile. If the server's own `--port` is taken, the server logs `Port <n> is already in use` and serves on a free port it names in `Using port <m> instead`; set `SGLANG_OMNI_STRICT_PORT=1` to fail instead.

(services-you-run-elsewhere)=

**Services you run elsewhere**


Remove the service's entry from `services` and set the node's `base_url` and `model` to the running service. `openai_transcription`, `openai_chat` and `openai_speech` work with any OpenAI-compatible transcription, chat completions and speech endpoint; `api_key_env` names the environment variable holding its bearer token. A `services` entry is only for a service on this machine, since its readiness check requests `127.0.0.1`. The node still probes the service at startup, so a wrong `base_url` fails the start (see [Verifying a Deployment](#verifying-a-deployment)).

(a-model-inside-the-server-as-a-stage)=

**A model inside the server as a stage**


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

(several-gpus)=

**Several GPUs**


Each service picks its GPUs through `env`, as `qwen3-asr-qwen3.8-27b-vivian` does with `env: {CUDA_VISIBLE_DEVICES: "0"}` for the language model and `"1"` for the two speech models; a stage picks its GPU with `gpu`. A service's memory share is the `--mem-fraction-static` inside its own `command`. The cascade server has no SGLang engine stage, so `sgl-omni serve --mem-fraction-static` on a cascade profile fails with `--mem-fraction-static requires a pipeline with at least one SGLang engine stage`.

(the-forced-alignment-worker)=

**The forced-alignment worker**


Word boundaries use Unicode character offsets into the original reply, not into the aligner's normalized text. The worker maps removed punctuation and Japanese compatibility normalization back to the source and rejects missing or changed tokens instead of searching ahead. An expanded number or compatibility character becomes heard only after its last aligned token finishes. The worker's tests use the actual language-specific tokenizer with simulated timestamps; real-model playback remains a separate validation step.

The heard-ASR matcher accepts joined spellings such as `long-term` / `long term`, `one-time` / `onetime`, and `U.S.` / `US`, but commits only at complete units on both sides. Thus `can` does not match `cannot`, and `now here` does not match `nowhere`. English and Chinese requests also compare the existing spoken number readings, preserving original offsets: `350` matches `three hundred fifty`, but `21` does not commit after only `twenty`. Both the original-spelling and expanded-reading comparisons must match a strict prefix; the former preserves counting sequences that the TTS reader treats as grouped digits. This change does not extend the TTS reading rules for contractions, percentages, versions or decades.

An `alignment` node with `method: forced` sends each finished sentence and its audio to a worker that returns word boundaries. The worker cannot be a `services` entry: it answers `/health` and `/align` but not the `/v1/models` request the server waits for, and it needs the `qwen-asr` package, which pins a `transformers` release of its own, so it runs from a separate virtual environment. In that environment install `qwen-asr`, `fastapi`, `regex` and `uvicorn`, plus `httpx`, `msgpack`, `msgspec`, `numpy`, `pydantic` and `websockets` for the modules it shares with the server, then start it from the repository root before `sgl-omni serve`:

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

(browser-playground)=

**Browser playground**


```bash
python playground/cascade/app.py --api-base http://127.0.0.1:8765 --port 8090
```

It needs `aiohttp`, serves the page on `--host` (default `127.0.0.1`) and `--port` (default `8080`), and forwards `/v1/realtime` and `/v1/realtime/capabilities` to `--api-base` (default `http://127.0.0.1:8000`). Its settings panel is the [browser demo's](#browser-demo): the system prompt, turn detection (`server_vad` with its silence, `semantic_vad` with its eagerness, as `turn_control` lists them) and one field per key in `sampling_parameters`, labelled for the cascade (`max_new_tokens_per_unit` is the reply token limit). It offers no text-only replies, because its captions and `conversation.item.truncate` follow audio playback. The fields are empty until you fill them, kept per deployment model in this browser, and cleared by Reset to defaults. While they are empty the page sends only `output_modalities` in its `session.update`, so the profile's prompt and turn detection apply; a filled field is sent in the first `session.update` and applies to the next conversation, with sampling fixed for the session as [Session settings](#session-settings) describes. Its stop button sends `response.cancel`. When a reply is interrupted, by the button or by speaking over it, the page sends `conversation.item.truncate` with the audio it played, and it marks the heard part of the reply from `sglang.response.progress` and the audio it has played, word by word through the stretch being played, at an even pace between two progress points; a cut reply then shows the part the history kept, from `sglang.response.heard`. A long reply's caption grows up to the room under the orb and then scrolls to keep the line being heard in view; the transcript holds the whole reply. Browsers grant the microphone only to `localhost` or HTTPS pages, so open the page on the same machine or forward its port.

(an-openai-realtime-client-or-the-session-api)=

**An OpenAI Realtime client or the Session API**


Any OpenAI Realtime client connects to `/v1/realtime` with the session flow in [Voice Conversation](#voice-conversation). `/v1/session` serves the same Session without the realtime protocol. `examples/streaming_cascade/run.py` runs a profile in its own process instead of connecting to a server; it does not start the profile's `services`, so start them first.

(profiles)=

**Profiles**


Every profile is a `launch.yaml` under `examples/streaming_cascade/profiles/`.

| Profile | Recognizer | Language model | Speech | GPUs | Nodes | Start beforehand |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3-asr-qwen3.8-27b-vivian` | Qwen3-ASR-1.7B service on 8002 (`openai_transcription`) | Qwen3.8-27B on 8003, thinking off | Qwen3-TTS-12Hz-1.7B-Base on 8005, cloning `voices/vivian.wav` | 2: the language model on GPU 0, both speech models on GPU 1 | `vad`, `asr`, `dialog`, `llm`, `tts`, `aligner` (forced, recognizing the sentence at the cut) | Alignment worker on 8010 |
| `qwen3-asr-qwen3-tts-vivian` (default) | Qwen3-ASR-1.7B service on 8002 | Qwen3.8-27B on 8000, thinking off | Qwen3-TTS-12Hz-1.7B-Base on 8005, cloning `voices/vivian.wav` | 2: the language model on GPU 0, both speech models on GPU 1 | `vad`, `asr`, `dialog`, `llm`, `tts`, `aligner` (forced, recognizing the sentence at the cut, published) | Alignment worker on 8010 |
| `qwen3-asr-streaming-qwen3-tts-vivian` | Qwen3-ASR-1.7B service on 8002 started with `--enable-realtime` (`streaming_transcription`, a partial transcript every 1000 ms of speech) | Qwen3.8-27B on 8000, thinking off | Qwen3-TTS-12Hz-1.7B-Base on 8005, cloning `voices/vivian.wav` | 2: the language model on GPU 0, both speech models on GPU 1 | `vad`, `asr`, `dialog`, `llm`, `tts`, `aligner` (forced, recognizing the sentence at the cut, published) | Alignment worker on 8010 |
| `qwen3-asr-qwen3-tts-base-clone` | Qwen3-ASR-1.7B service on 8002 | Qwen3-4B-Instruct-2507 on 8000 | Qwen3-TTS-12Hz-1.7B-Base on 8005, cloning `reference_zh.wav` in the profile directory | 1 | `vad`, `turn_detection` (published), `asr`, `dialog`, `llm`, `tts`, `aligner` (forced, recognizing the sentence at the cut, published) | Alignment worker on 8010 |
| `nemotron-qwen3-tts` | Nemotron 3.5 streaming ASR service on 8001 (`realtime_transcription`) | Qwen3-4B-Instruct-2507 on 8000 | Qwen3-TTS-12Hz-0.6B-Base on 8004, cloning `reference.wav` | 1 | `vad`, `asr`, `dialog`, `llm`, `tts`, `aligner` (forced) | Alignment worker on 8010; a recording at `reference.wav`, with `reference_text` set to its transcript |
| `nemotron-stage-qwen3-tts` | Nemotron 3.5 ASR as the `asr` stage on GPU 0, with `running_transcript` | Qwen3-4B-Instruct-2507 on 8000 | Qwen3-TTS-12Hz-0.6B-Base on 8004, cloning `reference.wav` | 1 | `vad`, `asr` (stage), `transcript`, `dialog`, `llm`, `tts`, `aligner` (forced) | Alignment worker on 8010; a recording at `reference.wav`, with `reference_text` set to its transcript |

`qwen3-asr-qwen3-tts-vivian` and `qwen3-asr-qwen3-tts-base-clone` publish their aligner's `session`, so their clients receive `sglang.response.progress`. Every profile cuts an interrupted reply at the word ends of the sentences whose audio was complete; the Qwen3-ASR profiles also send the delivered audio of the sentence at the cut to their own recognizer, so a sentence cut before its audio was complete keeps the part the listener heard.

(voice-conversation)=

**Voice Conversation**


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
| `sglang.response.heard` | After a `conversation.item.truncate`, in a profile that publishes its dialog's `session`: `text_end`, the length of the cut reply's text the history keeps, sent again whenever the cut moves, so a client can show what the history kept. |
| `response.done` | The reply is complete; `response.status_details.reason` is `stop`, `empty_turn` when the utterance had no words, or `interrupted` when you spoke over it. |
| `error` | The request or the session failed. |

`response.cancel` interrupts the reply from the client side. The capabilities grant `supports_cancel` when a node takes `interrupt` from `input` on a control edge, as every bundled profile's dialog does, and `supports_truncate` always. An interrupted reply stays in the history only up to the last boundary whose audio the server had sent before the interruption: the last word with a forced aligner, the part of the sentence at the cut that its transcript says with `heard_asr_url`, the last sentence otherwise. `conversation.item.truncate` with `audio_end_ms` tells the server how much of the reply was heard; the history then keeps the reply up to the last boundary whose audio ended by then, never past what the server sent. A reply the history keeps nothing of, such as one interrupted before its first boundary, leaves no answer: your next utterance joins the one before it in a single user message, as if you had kept talking. An utterance you finished before the interruption is answered once the server knows how much of the interrupted reply it delivered, so its prompt keeps that part. The truncate cuts only the reply its `item_id` names, also after the next reply began, so the prompts after that keep what you heard; one for a reply the backend can no longer cut, such as an earlier reply after a later one ended, changes nothing, and an `item_id` the session never sent gets an `item_not_found` error. The language model sees a cut reply ending with …, and while the history holds one, the system message, including instructions a client sets, ends with a sentence saying such a reply was cut off by the user. A reply the model finished is shown without a trailing …, so that it does not read as cut.

To end turns yourself instead of by silence, set `"turn_detection": null` in the first `session.update`; `input_audio_buffer.commit` then ends the utterance after the audio sent before it.

Instructions and sampling (`sglang.sampling.temperature`, `max_new_tokens_per_unit`) can change with a later `session.update` and apply from the next reply; a reply in progress keeps the settings it started with. Turn detection can change the same way and applies from the next audio unit. `session.updated` reports `supports_instruction_updates` and `supports_sampling_updates`, false in a profile whose nodes have no `system_prompt` or no `temperature` and `max_tokens`, respectively. [Session settings](#session-settings) lists every field.

(turn-detection)=

**Turn Detection**


Without it an utterance ends when the `vad` node hears `silence_duration_ms` of silence, so a pause to think in the middle of a sentence ends the turn and the reply answers half a sentence. A profile with a `turn_detection` node, such as `qwen3-asr-qwen3-tts-base-clone`, ends the turn only when the [Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3) end-of-turn model, scoring the utterance so far at each pause the `vad` node reports, finds it complete, or when the pause outlasts the longest pause the eagerness allows. The node reads the Session input and `vad:session` and publishes `session`: its `speech.started` opens the turn and its `speech.stopped` ends it, so the recognizer, the dialog and the client read `turn_detection:session` where they read `vad:session` before, and speech that resumes inside a held pause continues the utterance instead of interrupting a reply.

The stage loads the BSD-2 licensed `smart-turn-v3.2-gpu.onnx` from the Hugging Face repository `pipecat-ai/smart-turn-v3` through the Hugging Face cache, or from the file or directory `stages.turn_detection.factory.model_path` names, and verifies its SHA-256 at startup. The stage's `factory.eagerness` (`low`, `medium` or `high`, default `medium`) sets how readily a pause ends the turn: `high` takes a shorter silence on a confident score and gives up a held pause sooner, `low` waits longer. The profile sets the `vad` node's `silence_duration_ms` to 160, the pause the detector scores as a candidate end; a client that selects `{"type": "server_vad"}` on this profile gets the VAD's boundaries unchanged, with that pause.

A client selects the behaviour per session with the standard field, `{"turn_detection": {"type": "semantic_vad", "eagerness": "high"}}`; `session.created` lists `semantic_vad` under `turn_control` when the profile has the node. The Qwen3-Omni Session graph takes the node the same way: declare it in `examples/qwen3_omni/session_h200.yaml` beside `vad`, let `dialog` read `turn_detection:session` in place of `vad:session`, and publish `turn_detection: [session]` in place of `vad: [session]`.

(scripted-conversation)=

**Scripted Conversation**


The same conversation runs without the realtime protocol through the Session API:

```bash
PYTHONPATH=. python examples/streaming_cascade/run.py question.wav followup.wav --output response.wav
```

Each input is a 16 kHz mono signed 16-bit PCM WAV. The command runs the cascade in its own process without starting the profile's `services`, so start them first. It streams the files back to back through one Session of the profile named by `--config` and writes the reply audio to `response.wav`; the second file is answered with the first turn in the history. `--config` defaults to `nemotron-qwen3-tts`. Pass `--commit` for a profile without the VAD node. The `/v1/session` protocol is described in [Session API](#session-api).

(benchmarking)=

**Benchmarking**


The turn latency benchmark streams one WAV for several turns and prints, per turn, the time from speech end to first text, first audio, final transcript and completion:

```bash
python benchmarks/realtime_turn_latency.py --url ws://127.0.0.1:8765/v1/realtime --audio question.wav --turns 5
```

(configuration)=

**Configuration**


A setting comes from one of three places: the profile file, read at launch; command-line overrides of the profile, also read at launch; and the settings a client sends for its own session.

(profile-file)=

**Profile file**


| Field | Default | Effect |
| --- | --- | --- |
| `config_cls` | required | `StreamingCascadePipelineConfig` selects the cascade. |
| `model_path` | required | The deployment's name: the id `/v1/models` reports and the realtime model name, unless `sgl-omni serve --model-name` sets another. |
| `services` | `[]` | The services to start, in order (see below). |
| `session_nodes` | required | The nodes of the conversation, by name. |
| `session_publish` | required | Per node, the formats that reach the client. |
| `stages` | none | Settings of stage routes by stage name, such as `asr: {gpu: 0}` or `vad: {factory: {silence_duration_ms: 800}}`. |
| `timeout_s` | `120` | How long one input or control operation on a cascade node may take, including the wait for the last replies once the input has ended; past it the session fails with `component '<name>' append timed out`. |
| `unit_ms` | `200` | The length of the input audio units the server acknowledges with `sglang.unit.done`; reported as `native_unit_ms`. Audio reaches the session as each append arrives, so the VAD node ends a turn without waiting for a unit to fill. |
| `max_sessions` | `16` | Sessions that every component node, the VAD node and the turn detection node hold at once. A session beyond them is refused when it opens; raise it, for example with `--max_sessions 32`, when the services have room for more. |
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

(command-line-overrides)=

**Command-line overrides**


Any field of the profile can be overridden after `--config`, as `--<path> <value>` or `<path>=<value>`, one leaf value per flag. Paths start at the top of the profile, except stage settings, which start at the stage name:

```bash
sgl-omni serve --config examples/streaming_cascade/profiles/qwen3-asr-qwen3-tts-vivian/launch.yaml --enable-realtime --port 8765 \
  --session_nodes.llm.base_url http://127.0.0.1:9000/v1 \
  --session_nodes.llm.temperature 0.3 \
  --session_nodes.tts.language English \
  --services.tts.env.CUDA_VISIBLE_DEVICES 1 \
  --vad.factory.silence_duration_ms 800 \
  --timeout_s 60
```

Services are addressed by `name` (`--services.tts.port`), not by position. An override may add an option the file leaves out, and is validated like the file: an option the node's kind does not have fails with `Extra inputs are not permitted`, and a whole group such as `--session_nodes.aligner x` fails with `addresses a settings group, not a single value`. Lists such as a service `command` or a node's `inputs`, and removing a node or a service, need an edit of the file.

(session-settings)=

**Session settings**


A realtime client sends its settings in `session.update`. The first `session.update` opens the session; later ones may change only the fields marked hot below, and any other change is refused with `session field is frozen`.

| Field | Reaches | When it can change |
| --- | --- | --- |
| `instructions` | `system_prompt` of every node that has it (the `dialog`) | Any time; applies to the next prompt. At most 65536 characters, else `instructions exceed context or have invalid type`. |
| `sglang.sampling.temperature` | `temperature` of every node that has it (`openai_chat`), 0 to 2 | Any time; applies to the next reply, a reply in progress keeps its value. |
| `sglang.sampling.max_new_tokens_per_unit` | `max_tokens` of `openai_chat`, at least 1 | Any time; applies to the next reply, a reply in progress keeps its value. |
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
| `services`, the node set, `inputs`, `session_publish`, `stages`, `timeout_s`, `unit_ms`, `max_sessions`, `max_output_lead_ms`, `max_pending_output_*` | Restart the server. |
| Default node options and VAD defaults | Restart, or per session as above. |
| Node options through `nodes` | When a `/v1/session` session opens, and later through `configure` for nodes that read `input`. |
| `sglang.sampling` | When a realtime session opens. |
| `instructions`, `turn_detection`, `output_modalities` | Any time in a realtime session. |

(node-reference)=

**Node Reference**


Every node of a kind has `kind` (required) and `timeout_s`. Options marked required have no default.

(realtime_transcription)=

**`realtime_transcription`**


Streams each utterance over its own WebSocket to `<base_url>/v1/realtime?model=<model>` of an SGLang-Omni transcription server started with `--enable-realtime`, and reports partial and final transcripts. Its transcripts carry no language.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `base_url` | string, `ws://` or `wss://` | required | The transcription server. |
| `model` | string | required | Sent as the `model` query parameter. |
| `api_key_env` | string | `OPENAI_API_KEY` | Environment variable whose value is sent as a bearer token; nothing is sent when it is unset. |
| `timeout_s` | number > 0 | `120` | The limit for connecting and for one utterance's whole exchange. |

(streaming_transcription)=

**`streaming_transcription`**


Streams each utterance over its own WebSocket to `<base_url>/v1/realtime?intent=transcription&model=<model>` of an SGLang-Omni ASR server that declares realtime transcription, such as Qwen3-ASR started with `--enable-realtime`. The node cuts the utterances, so each session runs without server VAD: the node sends the utterance's audio, then `transcription.done`. Every `transcription.segment` that is not final becomes a partial transcript, and `transcription.completed` becomes the final transcript with the language the service heard. A segment the service cuts at its maximum length (30 s for Qwen3-ASR) starts the partial text again; the final transcript joins every segment.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `base_url` | string, `ws://` or `wss://` | required | The ASR server. |
| `model` | string | required | Sent as the `model` query parameter. |
| `language` | string | `""` | Sent as the session's `language` hint when set. |
| `decode_interval_ms` | integer > 0, or null | null | Audio between partial transcripts, sent in the session's `session.update`; null keeps the server's default (2000 ms for Qwen3-ASR), and the server refuses less than 200 ms. |
| `api_key_env` | string | `OPENAI_API_KEY` | As above. |
| `timeout_s` | number > 0 | `120` | The limit for connecting and for one utterance's whole exchange. |

(openai_transcription)=

**`openai_transcription`**


Sends each finished utterance as a 16 kHz WAV to `<base_url>/audio/transcriptions` and reports one final transcript.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `base_url` | string, `http://` or `https://` | required | The service, including `/v1`. |
| `model` | string | required | The `model` form field. |
| `language` | string | `""` | Sent as the `language` hint when set. |
| `response_format` | `verbose_json` or `json` | `verbose_json` | `verbose_json` reports the language heard in each utterance; `json` suits a service without it and reports none. |
| `api_key_env` | string | `OPENAI_API_KEY` | As above. |
| `timeout_s` | number > 0 | `120` | The request timeout. |

(running_transcript)=

**`running_transcript`**


Turns the running text of the `asr` stage into utterances: partial transcripts while it grows, and a final one once `tail_ms` of further audio has arrived after `speech.stopped` or a `commit`. Its transcripts carry no language.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `tail_ms` | number ≥ 0 | `400` | Audio after the end of speech that still belongs to the utterance. |

(dialog)=

**`dialog`**


Keeps the conversation history, opens a turn for each final transcript and sends the history to the language model. An utterance that arrives during a reply waits for it to end; after an interruption it also waits until the audio the graph delivered (`reply.delivered`) or the client's truncate says how much of the reply the history keeps. Speech that starts during a reply interrupts it, and so does speech that started after the utterance ended and has not stopped when the reply begins, so that reply ends before any of it is heard.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `system_prompt` | string | `You are a helpful voice assistant. Answer concisely.` | The system message; realtime `instructions` set it. |
| `max_history_chars` | integer > 0 | `65536` | The bound on the history: a turn whose history plus utterance plus `max_text_chars` would exceed it fails the session. |
| `max_text_chars` | integer > 0 | `16000` | The longest utterance transcript and the longest reply. |

(openai_chat)=

**`openai_chat`**


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

(openai_speech)=

**`openai_speech`**


Cuts the reply text into sentences and sends one streaming request per sentence to `<base_url>/audio/speech` with `model`, `voice`, `input`, `response_format: pcm`, `stream: true`, `language` when `language` is set, and `references: [{audio_path, text}]` when `reference_audio_path` is set. A sentence ends after a word ending in `.`, `!`, `?`, `;` or `…` (closing quotes and brackets included), after Chinese end punctuation followed by more text, or at a line end; a sentence longer than `max_sentence_words` is cut as that setting describes.

Every sentence is its own request, and Qwen3-TTS samples each request with a fresh seed. A CustomVoice speaker given by `voice` is held only by its speaker token, so its timbre moves from sentence to sentence and from turn to turn; a Base model cloning one `reference_audio_path` for every request keeps one voice, which is why the English profiles clone `examples/streaming_cascade/voices/vivian.wav`, a recording of the CustomVoice speaker Vivian. They leave `reference_text` empty, so Base clones from the recording's speaker embedding alone: with the transcript it continues the recording itself and opens each sentence with a pause of about half a second, which a listener hears as a later reply.

`openai_speech` sends each sentence without its emoji, asterisks and backticks, a markdown link as its text alone, a bare URL as its host alone, and a markdown line without its heading, list or quote marker and with its table cells separated by commas, because a speech service reads them as stray sounds or a spelled-out URL, and can run on for 40 s or more after a marker or a table row. A sentence with no word left, such as one of only emoji or punctuation or a table's separator row, is not sent because the speech service answers it with an error, and its end still counts in `reply.progress`. A sentence tagged `English` without Chinese characters is sent with its numbers written out as words, the way the forced-alignment worker reads them: a year as two pairs, spaced digit groups such as a phone number digit by digit, a decimal with `point`, and an ordinal such as `21st` as `twenty first`. The reply text and the `reply.progress` offsets keep everything the model wrote. The profiles' system prompts also tell the language model to use neither markdown nor emoji.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `base_url` | string, `http://` or `https://` | required | The service, including `/v1`. |
| `model` | string | required | The `model` field. |
| `voice` | string | `default` | The `voice` field; for Qwen3-TTS CustomVoice a built-in speaker, `default` meaning `Vivian`. |
| `reference_audio_path` | string | `""` | A WAV the speech service reads, for voice cloning. |
| `reference_text` | string | `""` | The transcript of the reference, sent with it when set. |
| `language` | string | `""` | The language tag: empty sends none; `auto` tags every sentence with the language the recognizer last heard in an answered turn; any other name tags every sentence with that name. With a fixed name, and with `auto` before the recognizer has named a language, a sentence containing Chinese characters is tagged `Chinese`; `auto` tags the others `English` until then. |
| `max_sentence_words` | integer > 0 | `24` | The most words sent in one request. A longer sentence is cut into requests of at most this many words and at least half as many, at a comma, colon or dash when one falls in that range; the first request of such a sentence waits for up to half as many words more, so its rest is never a word or two on its own. |
| `api_key_env` | string | `OPENAI_API_KEY` | As above. |
| `timeout_s` | number > 0 | `120` | The request timeout. |

(alignment)=

**`alignment`**


Reports how far into the reply text the reply audio has reached, as `reply.progress`, so the dialog can cut an interrupted reply where the listener stopped hearing it. It reads `["llm:text_delta", "tts:pcm_f32le", "tts:session", "dialog:session"]` in the profiles, and `input:session` too in the Qwen3-ASR profiles.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `method` | `sentence`, `forced` or `asr` | `sentence` | `sentence` reports the end of each spoken sentence; `forced` also reports the word ends the worker finds in each finished sentence; `asr` reports the sentence ends and, on a cut, the part of the sentence at the cut its transcript says. |
| `alignment_url` | string, `http://` or `https://`, or empty | `""` | The worker, for `forced`. |
| `heard_asr_url` | string, `http://` or `https://`, or empty | `""` | An OpenAI-compatible transcription service, including `/v1`, that recognizes the sentence at the cut; required for `asr`, optional for `forced`. |
| `heard_asr_model` | string | `""` | The `model` form field of those requests; set together with `heard_asr_url`. |
| `timeout_s` | number > 0 | `2` | The limit for one worker or transcription request; a sentence whose request fails or times out keeps only its sentence end. |

The dialog cuts an interrupted reply at the points `reply.progress` reports, up to the audio the graph sent to the client before the interruption, which the graph reports to the readers of the `tts` node's `session` output as `reply.delivered`. Without an `alignment` node the dialog takes `reply.progress` from the `tts` node, which reports the end of each sentence. With one, as in `qwen3-asr-qwen3-tts-vivian`, the `aligner` node reads `["llm:text_delta", "tts:pcm_f32le", "tts:session", "dialog:session", "input:session"]` and the dialog's `{kind: control, source: aligner, event: reply.progress}` input replaces the one from `tts`; a profile names exactly one source of `reply.progress`. Once the client's `conversation.item.truncate` arrives, the dialog reports every cut of that reply as `reply.heard` with `text_end`, the characters the history keeps, which reaches the client as `sglang.response.heard`.

`method: sentence` keeps the sentence ends, so a truncated reply keeps the sentences whose audio was played to the end and drops the sentence the listener was hearing. `method: forced` also sends each finished sentence and its audio to the worker at `alignment_url` and reports the word ends it returns, so the cut can fall inside a sentence that finished; when the worker fails or exceeds `timeout_s`, that sentence keeps its sentence end. A sentence that an interruption cuts off before its audio is complete gets no word ends and counts as unheard in both methods.

`method: asr` keeps the sentence ends and, when a reply is cut, sends the audio of the sentence at the cut, from the sentence's start to the cut, as a WAV file in the reply's language to the `/audio/transcriptions` endpoint of the OpenAI-compatible service at `heard_asr_url` with `heard_asr_model`; the reply keeps the sentences before it and the longest prefix of that sentence's text the transcript says, with the punctuation that follows it. The cut is the client's `truncate` position and never past `reply.delivered`, the same cut the dialog makes, so the aligner reads `input:session` for the client's `truncate`. The match ignores case and punctuation, reads digits, English numbers up to twenty and ordinary Chinese numbers as the same number, and stops at the first word the transcript leaves out, repeats or says differently. When the request fails, exceeds `timeout_s` or the first words differ, the sentence counts as unheard. It needs no model beyond the profile's own recognizer. `method: forced` takes `heard_asr_url` and `heard_asr_model` too, and then recognizes the sentence at the cut only when no word end at or after the cut covers it, such as a sentence cut off before its audio was complete; `method: sentence` takes neither.

(stage-vad)=

**Stage `vad`**


Declared as `vad: {stages: [vad], inputs: [input]}`; reports `speech.started` and `speech.stopped`. Its defaults apply to every session and a session's `turn_detection` overrides them; set them under `stages.vad.factory` or as `--vad.factory.<option>`.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `threshold` | number 0 to 1 | `0.5` | Speech probability above which a frame is speech. |
| `prefix_padding_ms` | integer ≥ 0 | `300` | Audio before the first speech frame that belongs to the utterance. |
| `silence_duration_ms` | integer ≥ 0 | `500` | Silence that ends an utterance. |
| `background_margin_db` | number ≥ 0 or `null` | `6` | A voice this many dB quieter than the last utterance starts no speech, so it neither opens a turn nor interrupts the reply. `null` hears every voice. A session's `turn_detection` does not change it. |
| `max_open_sessions` | integer ≥ 1 | `16` | Sessions the detector holds at once. |
| `max_concurrency` | integer ≥ 1 | `4` | Sessions it processes at once. |

(stage-turn_detection)=

**Stage `turn_detection`**


Declared as `turn_detection: {stages: [turn_detection], inputs: ["input:pcm16", "input:session", "vad:session"]}`; reports the turn's `speech.started` and `speech.stopped`, which the recognizer and the dialog read in place of the VAD's. A session's `turn_detection` overrides `eagerness`; set the options under `stages.turn_detection.factory` or as `--turn_detection.factory.<option>`.

| Option | Type | Default | Effect |
| --- | --- | --- | --- |
| `model_path` | string | `pipecat-ai/smart-turn-v3` | The Smart Turn ONNX file, its directory, or the Hugging Face repository holding `smart-turn-v3.2-gpu.onnx`. |
| `eagerness` | `low`, `medium` or `high` | `medium` | How readily a pause ends the turn: the silence a confident, a likely and an unlikely end of turn needs. |
| `max_open_sessions` | integer ≥ 1 | `16` | Sessions the node holds at once. |
| `max_concurrency` | integer ≥ 1 | `4` | Sessions it processes at once. |

(stage-asr)=

**Stage `asr`**


Declared as `asr: {stages: [asr], inputs: ["input:pcm16"]}`; runs Nemotron 3.5 streaming ASR on the GPU set by `stages.asr.gpu` (default 0) and produces the running `text` a `running_transcript` node reads.

(compatibility-and-constraints)=

**Compatibility and Constraints**


(rejected-when-the-profile-loads)=

**Rejected when the profile loads**


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
- A `dialog` whose `interrupt` no node that publishes audio takes on a control edge with its `session` read by the dialog, in a profile that publishes audio or where some node takes that `interrupt` — `node '<name>' waits for reply.delivered after it interrupts a reply: ...`.
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

(rejected-on-the-command-line-or-at-startup)=

**Rejected on the command line or at startup**


- `--stages.<...>` — `the stages. prefix is implied on the command line; write --<...>`.
- `--services.0.<...>` — `services is addressed by name, not by index`.
- A flag naming a group instead of a value — `addresses a settings group, not a single value`.
- `--mem-fraction-static` — `--mem-fraction-static requires a pipeline with at least one SGLang engine stage`.
- A service that exits or does not answer `/v1/models` on its port — the service errors under [One command with a profile](#one-command-with-a-profile).
- A node whose service does not answer its probe: `openai_transcription`, `openai_chat` and `openai_speech` request `<base_url>/models`, `openai_chat` then streams one token sampled with the node's settings, `realtime_transcription` requests `/v1/models` on the `http` form of its `base_url`, a forced `alignment` requests `<alignment_url>/health` and then aligns one second of silence, and an `alignment` with `heard_asr_url` requests `<heard_asr_url>/models` — `cannot reach <url>: ...` or `<url> answered <status>`, or the error of the chat token's completion, and the server stops.

(rejected-in-a-realtime-session)=

**Rejected in a realtime session**

During initial session opening, only a capacity rejection is reported as `admission_rejected` with `sglang.fatal: false`; the client may retry `session.update` on the same WebSocket before sending audio, within the admission deadline. If opening exceeds its own deadline, the server sends fatal `session_open_timeout` and closes the session. This differs from `session_update_timeout`, which limits the wait for an initial update or retryable admission; an admission already in progress may finish after that deadline. The separate pipeline idle fallback reports `session_idle_timeout`. Other initialization failures close the session with their known error code, or `internal` when no specific code is available. Configuration negotiation errors below retain their existing request-error behavior.

- A sampling parameter other than `temperature` and `max_new_tokens_per_unit` — `unsupported sampling parameters: ...`.
- A field other than the hot ones in [Session settings](#session-settings) changed after the first `session.update`, or `instructions` or `sglang.sampling` in a profile whose capabilities report `supports_instruction_updates` or `supports_sampling_updates` as false — `session field is frozen`.
- `instructions` longer than 65536 characters — `instructions exceed context or have invalid type`.
- A `model` other than the deployment's — `model differs from deployment`.
- An audio format other than 16 kHz input or 24 kHz output PCM — `unsupported PCM format or sample rate`.
- A `turn_detection` with a `type` other than `server_vad`, or other than `server_vad` and `semantic_vad` in a profile with a `turn_detection` node — `turn detection type is unavailable`; any non-null `turn_detection` in a profile without a `vad` node — `VAD is only available for turn-based sessions`.
- `nodes` in `session.update` — `invalid session configuration`.
- `response.create` — `manual turn commands are not granted`.
- `response.cancel` in a profile where no node takes `interrupt` from `input` on a control edge — `no node takes the control 'interrupt' from 'input'`.
- A new session beyond the `max_sessions` (16 unless the profile or `--max_sessions` sets another) that every cascade node holds at once — the first `session.update` is refused with `admission_rejected`.
- A `session.update` that changes `instructions`, or a `configure` that changes a node, while the user is speaking, between `speech_started` and `speech_stopped` — the session fails with `cannot replace '<node>' while speech is in progress`.

(rejected-by-the-speech-service)=

**Rejected by the speech service**


The speech service validates each sentence request, and a rejected request fails the session.

- Qwen3-TTS CustomVoice with `reference_audio_path` — `references are not supported by this model`.
- Qwen3-TTS CustomVoice with a `voice` that is not one of its speakers — `Unknown voice '<voice>'. Supported voices: default, ...`.
- Qwen3-TTS Base without `reference_audio_path` — `Qwen3-TTS Base requires reference audio via ref_audio or references[0].audio_path`. With a reference but no `reference_text`, Base clones from the speaker embedding of the recording alone.
- A `language` tag the service does not speak, including a language name the recognizer reported under `language: auto` — `language must be one of: ...`. Qwen3-TTS speaks Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish and Italian.
- A speech service that does not stream 24 kHz PCM with an `X-Sample-Rate` header — `speech service sent no X-Sample-Rate header, so its PCM rate is unknown` or `speech service streams <rate> Hz, expected 24000`.

(runtime-limits)=

**Runtime limits**


- A conversation whose history, next utterance and `max_text_chars` together exceed `max_history_chars` — `conversation exceeds max_history_chars; open a new session`, and the session fails. With the defaults this happens once the history passes about 49500 characters.
- An utterance transcript or a reply longer than `max_text_chars` — `transcript exceeds max_text_chars` or `reply exceeds max_text_chars`.
- A language model stream that stops for a reason other than `stop` or `length` — `LLM stopped with <reason>`; one that ends without a finish reason — `LLM stream ended before completion`.
- A client that stops reading while `max_pending_output_chunks` or `max_pending_output_bytes` of output wait — the session fails naming the limit.

Every failure ends the realtime session with an `error` event.

(allowed-with-consequences)=

**Allowed, with consequences**


- Without an `alignment` node, the dialog's `reply.progress` control edge comes from `tts`, which reports sentence ends, so an interrupted reply is cut at the end of the last sentence heard in full.
- `method: sentence`: the same sentence-level cut. With `forced`, the cut can fall between words of a finished sentence; a sentence cut off before its audio is complete counts as unheard in both methods unless `heard_asr_url` is set.
- With `heard_asr_url`, a transcript that leaves out, repeats or changes the first words of the sentence at the cut keeps none of it.
- No `reply.progress` input on the dialog at all: the profile loads, and an interrupted reply leaves no assistant entry in the history, so the next utterance joins the user message the reply answered.
- An aligner that does not read `tts:session` loads but receives no sentence ends and reports nothing.
- `sglang.response.progress` reaches the client only from a node that publishes `session` and whose `session` no other node reads: the aligner when it is published, never `tts` while the dialog or an aligner reads it.
- A client that sends no `conversation.item.truncate`: the history keeps an interrupted reply up to the audio the server had delivered, which is ahead of what was played by up to `max_output_lead_ms`. With `truncate`, the cut moves back to the client's `audio_end_ms`, never past the delivered audio.
- `language: auto` with `realtime_transcription`, `running_transcript` or `response_format: json`: no language is ever reported, so every sentence is tagged by its script, `Chinese` with Chinese characters and `English` otherwise. A fixed `language` tags every sentence without Chinese characters with that name whatever the user speaks.
- `turn_detection: null` sent after the VAD has already reported speech in the session: the recognizer keeps waiting for speech boundaries, and each `input_audio_buffer.commit` then ends an empty turn (`empty_turn`). Turn detection off from the first `session.update` works as intended; `running_transcript` ends utterances at a commit either way.
- `input_audio_buffer.commit` is granted in every profile (`client_commit: true`); with the VAD on, it also ends an open utterance.
- `max_text_chars` at or above `max_history_chars` loads, and then every turn fails with `conversation exceeds max_history_chars`.
- A reasoning language model can still open a think block of its own with `chat_template_kwargs: {enable_thinking: false}`. With `--reasoning-parser` on its server the block goes to `reasoning_content`, which the cascade never reads; without it the block is spoken and kept in the history.
- A reply cut by `max_tokens` (finish reason `length`) is spoken and kept as it is.
- Speech always interrupts a running reply; `interrupt_response` does not change that.

(verifying-a-deployment)=

**Verifying a Deployment**


A healthy start logs each service as `starting service '<name>': ...` and `service '<name>' serving on :<port>` (or `already serving`), then `Pipeline '<model_path>' started (<n> GPU(s))`, then the HTTP server listening on its port. A failed node probe logs `Scheduler thread for stage <node> crashed` with the probe error, and the server stops.

`curl http://127.0.0.1:8765/health` answers 200 with `"status": "healthy"` while the cascade runs, and 503 when it does not.

`curl http://127.0.0.1:8765/v1/models` lists one model whose `id` is the profile's `model_path`, the name a realtime client must use if it sends `model`.

`curl http://127.0.0.1:8765/v1/realtime/capabilities` answers 503 `instance is not ready` when the cascade is not running, and 404 without `--enable-realtime`. For the bundled profiles it reports:

```json
{
  "model": "examples/streaming_cascade/profiles/qwen3-asr-qwen3-tts-vivian",
  "interaction": "turn_based",
  "turn_control": ["server_vad", null],
  "client_commit": true,
  "input_audio_format": {"type": "audio/pcm", "rate": 16000},
  "output_audio_format": {"type": "audio/pcm", "rate": 24000},
  "output_modalities": ["text", "audio"],
  "native_unit_ms": 200,
  "supports_truncate": true,
  "supports_cancel": true,
  "sampling_parameters": ["temperature", "max_new_tokens_per_unit"],
  "supports_instruction_updates": true,
  "supports_sampling_updates": true,
  "limits": {"max_input_bytes": 1920000, "max_output_bytes": 33554432, "max_history_chars": 65536, "cleanup_timeout_s": 30}
}
```

with further fields omitted here. `turn_control` is `[null]` for a profile without a `vad` node, `output_modalities` is `["text"]` for one that publishes no audio, `sampling_parameters` is empty and `supports_sampling_updates` false when no node takes `temperature` or `max_tokens`, and `supports_instruction_updates` is false when no node has a `system_prompt`. A first conversation turn through the playground or the [turn latency benchmark](#benchmarking) confirms the whole chain.

(known-limitations)=

**Known Limitations**


- **Reasoning models must have thinking off.** The cascade speaks everything the language model returns as content. A model that thinks by default needs `enable_thinking: false` in the node's `extra_body` and a reasoning parser on its server (`--reasoning-parser qwen3` for Qwen3.8, as in `qwen3-asr-qwen3.8-27b-vivian`), or its reasoning is synthesized.
- **History is bounded.** A conversation whose history exceeds `max_history_chars` fails its next turn; open a new session.
- **Audio only.** The cascade takes audio input and produces text and audio; image input is not supported.

### Session Graph

A Session is one stateful conversation on the server: the client opens it, appends input for its whole lifetime, reads what it publishes and closes it. A pipeline declares the shape of its Session in its config, and the coordinator opens it; the serving layer appends input and reads the published formats without knowing the nodes behind them. A Session is either the one stage route from the pipeline's entry stage or a graph of nodes declared in `session_nodes`.

(nodes)=

**Nodes**


Every node of the Session is an entry of `session_nodes`, and `session_publish` lists, per node, the formats that reach the client. A node with `stages` is a route through those stages of the server.

`inputs` lists the nodes a node reads; `input` is the Session's own input, its audio, its `commit` and other controls, and camera frames when a node of the Session takes them. `node:format` takes only that format from that node; two nodes never send the same format to one reader, and the audio format of `input` is the Session input's alone. A node that reads every format of `input` while the Session takes camera frames must take them too, or name the formats it reads, such as `input:pcm16` and `input:session`.

An input written `{kind: control, source, event, preempt}` is a control edge. Its `event` is a control type of the source's `session` chunks or the format of its other chunks, such as `tool_response`; the event is handed to the node as a `control` operation, outside its input queue, to every stage of its route in route order, and travels on no other edge and to no client. With `preempt` the stages cancel the node's in-flight appends first. The client's `interrupt` reaches a node through a control edge from `input`.

An input of a stage node written `{source, capacity, overflow}` bounds its edge. The inputs naming one source are one edge into the node; it holds at most `capacity` chunks the node's own session has not accepted yet, by default the Session's `max_pending_chunks`. When the edge is full, `overflow: block`, the default, holds the producing node until the node takes a chunk, and rejects the client's append as a full queue; `overflow: drop_oldest` drops the edge's oldest chunk and lets the producer go on; on the client's edge the dropped input is confirmed to the client like a taken one, so a real-time node such as a full-duplex model hears the newest audio instead of a backlog. Two inputs of one source that declare different bounds fail the declaration.

A chunk one node sends another may be larger than the Session's `max_chunk_bytes`, as the unit of a Qwen3-Omni turn longer than about 32 s is. It enters the reader in fragments within that bound: the first stage of the reader's route holds them and restores the chunk for a single append, so every stage of the route sees the chunk whole and once. The fragments of a turn interrupted on the way are dropped.

The Session opens its nodes upstream first and closes them downstream first. A cycle of data edges between nodes must name the input that closes it with `feedback: true`, and that input must not block: two nodes each held on the other's full edge never wake. An undeclared cycle fails with `form a cycle of data edges`. A control edge holds no queue, so a cycle it closes needs no mark.

(stream-vocabulary)=

**Stream vocabulary**


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

Control types sent into a Session are `commit` (the utterance ends here), `configure` (settings change) and `truncate` (the listener heard the reply of the chunk's turn up to `t_ms`; without a turn, the latest reply). Control types a Session emits are `speech.started` and `speech.stopped` with `t_ms` on the input timeline, `turn.queued` when an utterance waits for its reply, `turn.started` when the reply begins, `turn.done` after the reply's audio, `turn.empty` for an utterance without words, `reply.heard` with `text_end` when a truncate cuts the reply the history keeps, and `configured` with `{"nodes": {name: kind}}`. Every chunk of a reply carries its turn number in the chunk's `turn` field, numbered from 1 per Session in utterance order: the turn controls, the `messages`, `text_delta`, `tool_call` and `tool_response` chunks and the reply audio; every turn opens with `turn.queued` and closes with `turn.done` or `turn.empty`.

(vad-node)=

**VAD node**


Voice activity detection is a stage node: `sglang_omni.models.vad` hosts one Silero detector per Session in a CPU stage of its own, `vad`, declared with `stages: [vad]`, reading the Session input's audio and controls and publishing `session`. It emits `speech.started` and `speech.stopped` controls whose `t_ms` is the boundary on the input audio timeline, `speech.started` reaching `prefix_padding_ms` before the first speech frame; `commit` resets the detector. Nodes that place utterances read it through a `vad:session` edge. Its detector defaults are the stage's `factory` options `threshold`, `prefix_padding_ms` and `silence_duration_ms`; a Session's `turn_detection` overrides them for that Session alone, at open and through `configure`.

(turn-detection-node)=

**Turn detection node**


End-of-turn detection is a stage node beside the VAD: `sglang_omni.models.turn_detection` runs the Smart Turn v3.2 model in a CPU stage of its own, `turn_detection`, declared with `stages: [turn_detection]`, reading the Session input's audio and controls and `vad:session`, and publishing `session`. It emits the same `speech.started` and `speech.stopped` controls as the VAD, at the turn's boundaries: the VAD's `speech.started` opens a turn, and at each `speech.stopped` of the VAD the node scores the turn's audio, ending the turn after the silence that score requires, up to the longest pause of the eagerness preset; speech that resumes before then continues the turn, and `commit` forgets it. Nodes that place utterances or interrupt replies on speech read `turn_detection:session` in place of `vad:session`. The stage's `factory` options are `model_path` (`SGLANG_OMNI_SMART_TURN_MODEL_PATH` when unset) and `eagerness`; a Session's `turn_detection` selects `semantic_vad` with an `eagerness`, or `server_vad`, which passes the VAD's boundaries through, at open and through `configure`. `SessionPorts.turn_detection_types` names the `turn_detection.type` values a Session's nodes implement, and the realtime bridge offers them as `turn_control`.

(interruption)=

**Interruption**


An `interrupt` naming a turn that reaches a node on a control edge first drops that turn's chunks from other nodes still queued for the node, on its edges and among its unsent appends, and a chunk of that turn that a node emits later no longer reaches it; the client's own input is never dropped. The node's own output of that turn, whether already buffered or emitted later, reaches neither its readers nor the client. The controls about the interruption pass both ways: the node's `interrupt` and `turn.interrupted` of that turn, and the `reply.delivered` that reports its audio, so a node the client interrupts by turn still stops the nodes it controls and hears what was delivered. Control edges are delivered upstream first, so a node's stale output is fenced before its readers hear the event. An `interrupt` without a turn, such as the client's `response.cancel`, means the open reply, which only the node knows; the Session leaves its queues to the node.

(delivery-pacing)=

**Delivery pacing**


A Session delivers reply audio in real time. For every node of the Session that publishes audio, an audio chunk leaves at most `max_output_lead_ms` ahead of the wall clock since the reply's first audio left, cut at that horizon, so the client never holds more than `max_output_lead_ms` of a reply however long its first audio took; the node's later audio and its `turn.done` wait behind it, and its other chunks, such as transcripts, reply text, speech boundaries and the confirmation of each input append, pass, so the client keeps appending while the reply plays. Once the input has ended everything leaves at once. Audio leaving together, at the input's end or when delivery catches up, is joined per turn into contiguous chunks of at most `max_output_lead_ms`, so a held 40 s reply reaches the client as about 40 chunks rather than one per generated chunk. A Session that is one stage route paces its output the same way. `max_output_lead_ms` is a field of the pipeline config (default 1000); an interruption cannot take back audio the client already holds, so it bounds how long the client may keep speaking after the user started.

A node's `turn.done` leaves once the wall clock reaches the end of the turn's audio, when the client has played it, and not when its last audio left. When the Session releases a node's `turn.done`, it delivers it to the nodes reading it and returns it to the node itself as a `control`, so a node can keep its reply open, and interruptible, until the client has played the reply's audio: a user who starts speaking during its last `max_output_lead_ms` still interrupts it. A `turn.interrupted` the node publishes, or an `interrupt` control naming a turn, drops what the Session still holds of that turn, its `turn.done` included. Held audio counts against the Session's `max_output_bytes`. A `reply.progress` reaches the client only from a node that publishes `session` and whose `session` no node reads, the end of the route that builds the progress: with an aligner reading the speech node's `session`, the client hears the aligner's progress and not the speech node's, and without one it hears none.

(session-api)=

**Session API**


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


(cascade-components)=

**Cascade components**


The cascade config's `kind` shorthand expands each component into its own CPU stage and Session node, with the stages sharing the `cascade` process. Data and control edges use the same coordinator runtime as native model stages. Feedback edges must be declared explicitly; a native stage may read a component's output directly.

A node declares either a component `kind`, with its options beside it, or a `stages` route for a native model or the VAD. `inputs` lists the sources it reads: `input` is the client's audio and controls, `node` a node's every format, `node:format` one format, `input:session` the client's controls, and `{kind: control, source, event}` a control edge. A component accepts the formats its kind consumes; the declaration is validated when the config loads: names are unique, every input exists, every format a component requires other than `session` controls is produced by one of its inputs, exactly one reply-ending kind publishes `session`, and the kinds that open turns (`dialog`) and end them (`openai_chat` or `openai_speech`) are present once. At startup every service kind is probed once: `openai_transcription`, `openai_chat` and `openai_speech` request `{base_url}/models`, `openai_chat` also streams one token sampled with the node's settings so a kernel the service compiles on its first sampled request is built before the server is ready, `realtime_transcription` requests `/v1/models` on the HTTP form of its `base_url`, and a forced `alignment` requests `{alignment_url}/health` and aligns one second of silence; a failed probe fails the start with the node's name.

A kind is a config class with a default `kind`, a constructor and the formats it consumes and produces, registered through `COMPONENTS.register` in `sglang_omni.models.streaming_cascade.components`; `opens_turns` marks the kind that numbers turns, `ends_turns` the one whose `turn.done` ends a reply, and `optional` the consumed formats a configuration may leave unconnected. `running_transcript` is the kind that turns the one running text of a Nemotron 3.5 ASR stage served as a node of the same Session into utterance transcripts, cut at `speech.stopped` or at a commit plus `tail_ms`.

A `configure` control reaches every node reading `input`. Its standard fields reach every node with the matching option, `instructions` as `system_prompt` and `sampling` as `temperature`, `max_tokens`; `nodes: {name: {...}}` changes one node's options, and a different `kind` replaces its component, which adopts the previous one's state. Names and inputs cannot change; a node is not replaced while it is speaking, the change waits for the reply to end. Each node answers with `configured {"nodes": {name: kind}}` on its `session` output.

An EOS append ends the input readers in graph order, after their upstream input readers finish. Its confirmation waits for pending routed input and published reply turns to drain. Background component failures fail the output read without waiting for another append.

The cascade graph enables `shared_timeline`: its published reply chunks use one coordinator-owned media clock. A reply starts no earlier than the latest client audio or the previous reply's audio end; audio advances that clock, and interruption resets it to the client audio clock. Speech boundaries retain their original input timestamps. Other graphs preserve their nodes' timestamps by default.

## Full-Duplex-Bench runbook

This page runs [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) (FDB) v1.0 and v1.5 against a full-duplex model served by SGLang-Omni (MiniCPM-o 4.5 by default). Every step is a subcommand of `python -m benchmarks.duplex.fdb`; the code is in `benchmarks/duplex/fdb/`.

### What is measured

FDB has two datasets, and they test different behavior:

- **v1.0** tests turn-taking. The model should stay quiet through a pause in the user's turn, come in once the turn ends, stop and answer when the user interrupts it, and, while the user talks at length, give short backchannels such as "mm-hmm" without taking the turn.
- **v1.5** tests overlap handling. The user asks something, and while the model answers, more speech arrives: an interruption, a backchannel, speech to someone else, or background speech. The question is what the model does with it.

The two measure different things, so a run takes samples from both by default. Each dataset gets its own result table, and their numbers are never pooled.

(v10)=

**v1.0**


v1.0 has 727 samples in five subsets. A sample is one input audio file plus an annotation (the pause positions, the end of the user's turn, the interruption, or the human backchannel timing), streamed in one session.

| Subset | Samples | Task | Source | Reported numbers |
|---|---:|---|---|---|
| `synthetic_pause_handling` | 137 | Pause handling | Synthetic | Takeover rate (lower is better) |
| `candor_pause_handling` | 216 | Pause handling | Candor conversations | Takeover rate (lower is better) |
| `candor_turn_taking` | 119 | Turn-taking | Candor conversations | Takeover rate (higher), latency (lower) |
| `synthetic_user_interruption` | 200 | User interruption | Synthetic | Takeover rate (higher), latency (lower), relevance (higher) |
| `icc_backchannel` | 55 | Backchannel | ICC conversations | Takeover rate (lower), backchannel frequency, backchannel JSD (lower) |

The following official reference numbers come from the evaluator at the pinned revision, run unchanged on the transcript of the model's output. The runbook also reports [unified v1.0 metrics](#results) for pause handling, turn taking and interruption:

- **Takeover rate**: the share of samples in which the model took the turn, meaning its transcript spans at least 1 s or has more than 3 words.
- **Latency**: the time from the annotated end of the user's turn to the model's first word, averaged over the samples in which the model took the turn. A start before the end of the turn counts as 0.
- **Relevance**: a 0-5 rating of the model's reply to the interruption, given by the judge (see [Judges](#judges)).
- **Backchannel frequency** and **JSD**: how often the model backchannels, and how far its backchannel timing is from the human one.

The capture, VAD and judge can differ from the paper's, so compare these numbers between runs of this runbook, not with the published tables.

(v15)=

**v1.5**


v1.5 has 498 pairs in four categories: `user_interruption` (200), `user_backchannel` (98), `talking_to_other` (100) and `background_speech` (100). A pair is two input files with the same user query: `input.wav` contains the later overlapping speech and `clean_input.wav` does not. Each file is streamed in its own session.

- **Stop latency**: how long the model keeps talking while the other speech overlaps it. Lower is better on `user_interruption`, where the model should give way. In the other three categories the model should keep its turn, so the overlap mostly tracks how long the other speech lasts.
- **Response latency**: the time from the end of a user speech span to the start of the next model speech. Lower is better when a reply is due. Both latencies are whole-file intervals from the official timing code, not latencies around the overlap event.
- **Behavior labels**: the judge gives each pair one label for how the model handled the overlap. `C_RESPOND`: it addressed the overlap's content. `C_RESUME`: it ignored the overlap and continued. `C_UNCERTAIN_HANDLING`: it asked for a repeat or showed that it did not catch the overlap. `C_UNKNOWN`: its reply was off-target, or it said nothing. The share that should be high is `C_RESPOND` on `user_interruption` and `C_RESUME` in the other categories; `C_UNKNOWN` should be low.
- **Semantic quality** (`JUDGE=qwen` only, not part of the paper): a label says what the model did, not whether that was right, so a `C_RESUME` that ignored a real interruption looks the same as one that correctly ignored a backchannel. The semantic judge grades each pair on three axes, each `accept`, `fail` or `unresolved`: `interaction_handling` (the overlap was handled correctly for its category), `relevance` (the reply addressed the request that mattered) and `grounding` (no unsupported facts, claimed actions or false memories). `joint` accepts only when all three accept. A verdict must quote the transcript it relies on, and a verdict with a missing or wrong quote becomes `unresolved`. Each cell reports `quality = A/(A+F)` and `coverage = (A+F)/N` over all N selected pairs; read the two together, since a high quality over a low coverage judges few pairs.

(listening-quality)=

**Listening quality**


The timing, label and semantic numbers say what the model did and when, not how its replies sound. `asr` also scores every reply of both datasets on its own, and `RESULTS.md` reports the scores per v1.5 category and per v1.0 subset, never as one total. A reply is the audio of one response, its deltas back to back as a client with enough buffer plays them, up to where the simulated playout stopped when a protocol cuts replies at barge-in; gaps from deltas that arrive late are left out, because the [serving metrics](#serving-metrics) measure them as underruns.

- **Omission rate** (lower is better): the words of the transcript in `response.done` that are missing from the reply's Parakeet transcript, over the transcript's words, pooled over the completed replies of a category. Both texts go through Whisper's English normalizer before alignment. A cancelled reply is not compared: its transcript can run past the audio it played.
- **WER** (lower is better): substitutions, deletions and insertions of the same alignment. A swallowed syllable often shows up as a substitution rather than a deletion. Both rates include Parakeet's own errors, so compare them between systems rather than read them as absolute rates.
- **UTMOS** (1-5, higher is better): the UTMOS prediction of each reply of at least 3 s, the model the TTS benchmarks use; it drops for clicks, distortion and unnatural prosody. The table shows the mean and the 10th percentile.
- **Voice similarity** (higher is better): 100 times the cosine of the WavLM speaker embedding of a reply's middle 3 s with the mean embedding of the same window of every other reply in the repeat, so a reply whose voice drifts from the model's usual one scores low. The table shows the mean and the 10th percentile.
- **Short replies**: replies with audio shorter than 3 s, such as most backchannels. They are compared for omission but get no UTMOS or voice score.

The 3 s comes from MiniCPM-o replies. Middle crops of replies of 6 s or more score within 0.07 of the full reply's UTMOS from 2 s on and 0.38 below it at 1 s. A speaker embedding depends on length at every duration measured, so the voice score compares windows of one length; copies of the replies shifted up 4 semitones separate from the originals by d = 2.7 with 2 s windows, 4.0 with 3 s and 5.0 with 4 s, while the replies long enough to score drop from 44 of 50 at 3 s to 38 at 4 s.

The metrics move for their own defects. On 24 MiniCPM-o replies, removing the middle 40% of the audio raises the omission rate from 4.4% to 43.4%; adding 10 clicks per second drops UTMOS from 4.26 to 2.08; shifting up 4 semitones drops voice similarity from 72.5 to 52.5. The omission rate moves at most 0.6 points under the clicks and the pitch shift, and voice similarity at most 0.5 under the cut and the clicks. UTMOS also drops 0.22 for the cut, which leaves a splice, and to 1.41 for the pitch shift, whose phase vocoder adds artifacts.

These are reported only, with no pass mark yet. `listening/scores.json` keeps each reply's duration, transcript, Parakeet text and scores.

### Setup

You need one GPU with at least 80 GB (H100 or H200), and `git`, `uv` and the FFmpeg shared libraries (version 4 to 8) on the host. Run every command in this page from the repository root, with the sglang-omni virtualenv active (`which python` should print its interpreter).

```bash
python -m benchmarks.duplex.fdb setup
```

`setup` is safe to rerun and skips finished steps. It installs the `minicpm-o` extra of sglang-omni (`onnx` and `einops`) into the active venv and puts everything else under `FDB_WORK` (default `$HOME/fdb`):

| Path | Content |
|---|---|
| `Full-Duplex-Bench/` | Official scoring code at the pinned revision `3e799c4`; file hashes are checked |
| `dataset/v1.0/`, `dataset/v1.5/` | The two datasets from the official Google Drive; `dataset/v1.0.revision` and `dataset/v1.5.revision` hold the archive hashes |
| `scoring-venv/` | Python 3.12 with NeMo, torch, Silero VAD and torchcodec for ASR, timing and the official evaluators. torchcodec loads FFmpeg, and `setup` checks that it can |
| `models/parakeet-tdt-0.6b-v2/` | ASR checkpoint, SHA-256 checked |
| `models/MiniCPM-o-4_5/` | Model under test at revision `503e754`. To reuse an existing copy, symlink it here or set `MODEL_PATH` |
| `models/Qwen3.8-27B/` | Judge checkpoint; its commit is saved in `models/Qwen3.8-27B.revision` |

(judges)=

**Judges**


`JUDGE` selects the judge:

| `JUDGE` | v1.5 behavior labels | v1.5 semantic quality | v1.0 interruption relevance |
|---|---|---|---|
| `qwen` (default) | [Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B), served locally: non-thinking, greedy, seeds 1-3 | Qwen3.8-27B: thinking, temperature 0, seed 1 | Qwen3.8-27B with the official prompt: non-thinking, seed 0 |
| `gpt` | `gpt-4o-2024-08-06`, the paper's judge | not run | `gpt-4-turbo`, which the official evaluator requests |

`JUDGE=gpt` needs `OPENAI_API_KEY`. With `JUDGE=qwen` the labels and ratings are not the paper's, but the judge is local and pinned, so runs with the same settings stay comparable with each other. The semantic judge sends six control cases (`benchmarks/duplex/semantic/`) together with its first grading batches and keeps the grades only if all 18 control axis statuses match `control-expected.json`; otherwise it discards them and stops. The controls check the rubric, not the judge's accuracy.

A judge starts only when the step has a request for it: the v1.5 pairs and the v1.0 interruption subset need one, while the other v1.0 subsets are scored without an LLM. A run with only those subsets starts no judge server and needs no API key.

### Run the benchmark

A run has three steps, then an aggregation. Each step starts the server it needs and stops it when it exits. `generate --all-stages` runs the other three after recording, so one command goes from recording to `RESULTS.md`:

```bash
python -m benchmarks.duplex.fdb generate --run-name minicpmo-10 --v15-per-subset 10 --v10-per-subset 10 --all-stages
```

Without `--all-stages`, `generate` only records, and each later step is its own subcommand.

| Step | Subcommand | What it does |
|---|---|---|
| 1 | `generate` | Starts the model server, streams the selected inputs to `/v1/realtime` (two sessions per v1.5 pair, one per v1.0 sample), records the model's audio, stops the server, then cuts a fixed observation window from each session |
| 2 | `asr` | Transcribes the windows with word timestamps (Parakeet), computes v1.5 official speech intervals, writes v1.0 unified metrics from the saved transcript and annotations on CPU, then cuts every reply and scores its [listening quality](#listening-quality) |
| 3 | `judge` | Runs the official v1.0 evaluator on the v1.0 subsets that need no LLM, then judges the v1.5 pairs and the v1.0 interruption subset (with `JUDGE=qwen` it starts the judge server for them and stops it after), and writes the per-repeat summaries |
| | `aggregate` | Combines the repeats of a run into `RESULTS.md` |

Run a small preflight first. It takes one pair per v1.5 category and three samples per v1.0 subset (about 15 minutes) and catches environment problems before a long run. v1.0 needs three samples rather than one because the official turn-taking and interruption evaluators divide by the number of takeovers and fail when no selected sample takes the turn.

```bash
python -m benchmarks.duplex.fdb generate --run-name preflight --v15-per-subset 1 --v10-per-subset 3 --all-stages
```

The preflight passes when `generate` prints `{"pass": 8}` and `{"eligible_pairs": 4, ...}` for v1.5 and `{"pass": 15}` with 3 eligible samples per v1.0 subset; `asr` prints `{"ok": 16}`, `{"ok": 8}` and 3 transcripts per v1.0 subset; and `judge` prints `{"valid": 4}`, `controls: 18/18 axis statuses match` and the parsed v1.0 results of each subset as JSON (the evaluator's own `[Result]` block is in `repeat-N/v10/reference/logs/evaluate-<subset>.log`). If anything differs, see [Troubleshooting](#troubleshooting).

A run is one fixed selection of samples under `$FDB_WORK/runs/RUN_NAME`. Decoding uses sampling, so two generations of the same inputs differ; to measure that variance, run the steps several times under one run name with `--repeat 1`, `--repeat 2`, and so on. `--repeat N` only names the output directory `repeat-N/`; one command always runs once. `aggregate` presents unified v1.0 scores per repeat and means ± standard deviations in its v1.5 and official v1.0 reference tables.

```bash
for repeat in 1 2 3; do
    python -m benchmarks.duplex.fdb generate --run-name minicpmo-12 --repeat "$repeat" --v15-per-subset 12 --v10-per-subset 12 --all-stages
done
```

Each repeat ends with `aggregate`, so `RESULTS.md` covers every repeat finished so far.

`generate` refuses to write into a repeat that already holds a recording; to redo one, delete its `repeat-N/` directory and run its three steps again. When `asr` or `judge` stops a `--all-stages` run, its message names the command that continues it; run that step and the ones after it on their own. `asr` and `judge` resume finished work when rerun. For v1.0, `asr` runs unified scoring after a successful or already completed `reference-asr`; failed ASR does not trigger scoring. The unified scorer does not need `judge` or its `summary.json`.

Every subcommand takes `--log-level` (`info` by default). With `--log-level debug`, a step also prints the server and recorder logs it writes, each line prefixed with the log's name, such as `[judge-server]` or `[record-shard-0]`. The logs are written under `repeat-N/` at either level (see the files table in [Results](#results)).

(choosing-samples)=

**Choosing samples**


Only `generate` selects samples; `asr` and `judge` score whatever the repeat recorded. Selection works the same way for both datasets and is never random: each subset (v1.5 calls them categories) takes its first N samples in numeric ID order (1, 2, …, 10, 11, …). So every repeat with the same options gets the same samples, and a smaller N is always a prefix of a larger one.

The options come in two matching sets, `--v15-*` and `--v10-*`. Both default to 12 per subset. A count of 0 leaves that dataset out of the run; leaving both out is an error.

| `generate` option | Effect |
|---|---|
| `--v15-per-subset N`, `--v10-per-subset N` | N samples from every subset of that dataset (default 12); `all` takes the whole dataset; `0` leaves it out |
| `--v15-subset-count NAME=N`, `--v10-subset-count NAME=N` | Overrides the count for one subset; `0` skips it, `all` takes all of it. Repeatable |
| `--v15-sample-id NAME/ID`, `--v10-sample-id NAME/ID` | Exactly these samples. Repeatable |
| `--v15-sample-ids-file PATH`, `--v10-sample-ids-file PATH` | One `NAME/ID` per line, such as another run's `repeat-1/sample-ids.txt` or `repeat-1/v10/sample-ids.txt` |

```bash
# Both full datasets.
python -m benchmarks.duplex.fdb generate --run-name minicpmo-full --v15-per-subset all --v10-per-subset all
# v1.0 only; v1.5 only.
python -m benchmarks.duplex.fdb generate --run-name v10-only --v15-per-subset 0
python -m benchmarks.duplex.fdb generate --run-name v15-only --v10-per-subset 0
# 20 v1.5 interruptions, 10 v1.5 backchannels and every v1.0 backchannel sample.
python -m benchmarks.duplex.fdb generate --run-name small \
    --v15-per-subset 0 --v15-subset-count user_interruption=20 --v15-subset-count user_backchannel=10 \
    --v10-per-subset 0 --v10-subset-count icc_backchannel=all
# Three specific samples.
python -m benchmarks.duplex.fdb generate --run-name debug \
    --v15-sample-id user_interruption/1 --v15-sample-id background_speech/7 \
    --v10-sample-id candor_turn_taking/37
# Another run's selection.
python -m benchmarks.duplex.fdb generate --run-name replay \
    --v15-sample-ids-file "$FDB_WORK/runs/minicpmo-12/repeat-1/sample-ids.txt" \
    --v10-sample-ids-file "$FDB_WORK/runs/minicpmo-12/repeat-1/v10/sample-ids.txt"
```

`generate` prints how many samples it selected per subset and writes their IDs to `repeat-N/sample-ids.txt` (v1.5) and `repeat-N/v10/sample-ids.txt` (v1.0). All repeats of a run must select the same samples, so pass the same options every time; a repeat with a different selection stops with an error.

Sessions run in real time, so `generate` dominates the wall time. Approximate times per repeat on one H200, one session at a time, without the 1 minute of model server startup per `generate` and 2 minutes of judge server startup per `judge`:

| Selection | `generate` | `asr` | `judge` |
|---|---|---|---|
| 48 v1.5 pairs (12 per category) | ~25 min | ~1 min | ~6 min |
| 498 v1.5 pairs | ~4 h | ~10 min | ~45 min |
| 60 v1.0 samples (12 per subset) | ~20 min | ~1 min | ~1 min |
| 727 v1.0 samples | ~3.6 h | ~10 min | ~5 min |

A v1.5 session lasts about 15 s and a v1.0 sample about 18 s on average. Most of a v1.5 `judge` is the semantic judge, which runs 8 thinking-mode batches of up to 8 pairs concurrently at about 5 minutes per batch; the Qwen behavior judge sends up to 16 requests at once. `generate --num-shards 2` roughly halves the generation time, but read the note on it in [Notes](#notes) first; `--gpus 0,1 --num-shards 2` halves it without sharing a GPU.

### Results

`aggregate` prints its tables and writes them to `$FDB_WORK/runs/RUN_NAME/RESULTS.md`, with one section per dataset the run selected. The v1.0 unified section shows each repeat separately, without averaging its rates, quantiles or confidence intervals. In the v1.5 and official v1.0 reference tables, a numeric cell is the mean ± sample standard deviation over repeats with available values; a count cell shows one value or a range when repeats differ.

The v1.5 section has one row per category and one for all of them. This example is a validation run (MiniCPM-o 4.5, 3 pairs per category, two repeats, Qwen judge), so the samples are small:

| Category | Pairs | Timed overlap sessions | Stop latency (s) | Response latency (s) | C_RESPOND | C_RESUME | C_UNCERTAIN_HANDLING | C_UNKNOWN | Judged pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 12 | 12 | 1.483 ± 0.070 | 2.698 ± 0.075 | 25.0 ± 11.8% | 66.7 ± 11.8% | 8.3 ± 0.0% | 0.0 ± 0.0% | 12 |
| user_interruption | 3 | 3 | 2.161 ± 0.015 | 2.479 ± 0.128 | 50.0 ± 23.6% | 50.0 ± 23.6% | 0.0 ± 0.0% | 0.0 ± 0.0% | 3 |
| user_backchannel | 3 | 3 | 0.625 ± 0.000 | 2.308 ± 0.121 | 0.0 ± 0.0% | 100.0 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 3 |
| talking_to_other | 3 | 3 | 1.751 ± 0.083 | 2.742 ± 0.083 | 33.3 ± 0.0% | 66.7 ± 0.0% | 0.0 ± 0.0% | 0.0 ± 0.0% | 3 |
| background_speech | 3 | 3 | 1.393 ± 0.181 | 3.204 ± 0.034 | 16.7 ± 23.6% | 50.0 ± 23.6% | 33.3 ± 0.0% | 0.0 ± 0.0% | 3 |

`Pairs` counts every selected pair, failed or not. `Timed overlap sessions` counts the overlap sessions that produced timing intervals. `Judged pairs` counts the pairs with a valid label, and the label shares are over these pairs only.

With `JUDGE=qwen` the v1.5 section adds the semantic table. From the same validation run:

| Category | interaction_handling quality | interaction_handling coverage | relevance quality | relevance coverage | grounding quality | grounding coverage | joint quality | joint coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 73.9 ± 1.6% | 95.8 ± 5.9% | 85.9 ± 5.8% | 87.5 ± 5.9% | 52.2 ± 11.0% | 79.2 ± 5.9% | 32.2 ± 17.3% | 79.2 ± 5.9% |
| user_interruption | 66.7 ± 47.1% | 83.3 ± 23.6% | 66.7 ± 47.1% | 83.3 ± 23.6% | 58.3 ± 11.8% | 83.3 ± 23.6% | 25.0 ± 35.4% | 83.3 ± 23.6% |
| user_backchannel | 100.0 ± 0.0% | 100.0 ± 0.0% | 100.0 ± 0.0% | 100.0 ± 0.0% | 100.0 ± 0.0% | 50.0 ± 23.6% | 100.0 ± 0.0% | 50.0 ± 23.6% |
| talking_to_other | 50.0 ± 23.6% | 100.0 ± 0.0% | 75.0 ± 35.4% | 66.7 ± 0.0% | 58.3 ± 11.8% | 83.3 ± 23.6% | 16.7 ± 23.6% | 83.3 ± 23.6% |
| background_speech | 83.3 ± 23.6% | 100.0 ± 0.0% | 100.0 ± 0.0% | 100.0 ± 0.0% | 16.7 ± 23.6% | 100.0 ± 0.0% | 16.7 ± 23.6% | 100.0 ± 0.0% |

The v1.0 unified section first shows per-repeat coverage and metric tables for pause handling, turn taking and user interruption. Coverage lists declared, selected, eligible and ineligible samples, exclusion reasons and the minimum observable horizon after the reference point. A sample rate uses eligible samples as its denominator; its Wilson 95% interval describes sampling uncertainty, while the all-selected bounds allow each ineligible outcome to be either false or true. Per-pause and seconds-per-second ratios pool their respective denominators over eligible samples. Other intervals bootstrap whole samples. `n/a` with zero eligible samples means no data. `not reached` with a positive eligible count means the requested response-latency quantile was not observed, including nonresponses; a window shorter than a response threshold limits the corresponding rate. The report labels any missing, stale, unsupported or incomplete unified artifacts unavailable and gives a CPU-only backfill command. It does not display stale numeric scores. Backchannel remains official-reference-only.

The following v1.0 official reference table has one row per selected subset. `Selected` counts the recorded samples, `Eligible` the ones the official evaluator can score, and `Evaluated` the ones it scored; then come `Takeover rate`, `Latency (s)`, `Relevance (0-5)`, `Judge retries`, `Backchannel frequency` and `Backchannel JSD`, with `n/a` where the subset's task does not report a number. A subset whose sessions all failed still has a row, with `Eligible 0`. A subset that a repeat has not evaluated is listed under the table as not evaluated yet (`judge` has not reached it) or failed (with its evaluator log); its cells average the repeats that did evaluate it, and show `n/a` when none did. A nonzero `Judge retries` means the interruption takeover rate is inflated (see [v1.0 scoring](#v10-scoring)).

The listening-quality section comes last, with one table per dataset and the same mean ± deviation cells; see [Listening quality](#listening-quality) for what each column measures. Its v1.5 table from a validation run (MiniCPM-o 4.5, 4 pairs per category, one repeat):

| Category | Replies | Short replies | Compared replies | Omission rate | WER | UTMOS mean | UTMOS p10 | Voice similarity mean | Voice similarity p10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 35 | 1 | 35 | 4.2% | 7.3% | 4.271 | 3.893 | 74.412 | 67.846 |
| user_interruption | 9 | 0 | 9 | 3.9% | 7.0% | 4.358 | 4.163 | 76.408 | 68.144 |
| user_backchannel | 8 | 0 | 8 | 5.5% | 6.7% | 4.248 | 3.960 | 72.195 | 67.167 |
| talking_to_other | 10 | 0 | 10 | 3.1% | 8.7% | 4.226 | 3.868 | 75.081 | 70.286 |
| background_speech | 8 | 1 | 8 | 4.0% | 7.1% | 4.250 | 3.910 | 73.422 | 67.977 |

Each repeat writes these files under `$FDB_WORK/runs/RUN_NAME/repeat-N/`:

| Path | Content |
|---|---|
| `sample-ids.txt`, `v10/sample-ids.txt` | The selected v1.5 pairs (empty when v1.5 was left out) and v1.0 samples (`v10/` exists only when v1.0 was selected) |
| `recording/shard-*/`, `v10/recording/shard-*/` | Raw session traces, input and output audio, protocol reports |
| `logs/`, `v10/logs/` | `logs/` has the server logs and the v1.5 recorder logs, `v10/logs/` the v1.0 recorder logs; the v1.5 ASR and timing logs are in `scores/logs/` |
| `reference-audio/` | v1.5 fixed-window WAVs and `reference-manifest.json` with eligibility |
| `scores/summary.json` | v1.5 timing, ASR coverage and, with `JUDGE=gpt`, behavior labels |
| `judge-qwen/summary.json` | v1.5 behavior labels with `JUDGE=qwen` |
| `semantic-qwen/` | Semantic judge: control comparison, the exact packets sent, raw responses, per-pair outcomes and `summary.json` |
| `report.txt` | Readable v1.5 report: coverage, timing and semantic quality |
| `listening/` | One 16 kHz WAV per reply in `wav/` and its scored 3 s window in `voice-window/`, `replies.json` with each reply's status, transcript and Parakeet text, `scores.json` with each reply's scores, and the per-category `summary.json` with the checkpoint hashes |
| `v10/reference/` | v1.0 windows (`<subset>/<id>/output.wav`), `manifest.json`, the ASR receipt `asr.json`, per-repeat `unified.json` and `unified-samples.jsonl`, official `summary.json`, the judge ledger in `judge/` and evaluator logs in `logs/` |

With `JUDGE=qwen`, the "Official behavior label distribution" section of `report.txt` shows `0 / 0` and `not_prepared`, because that section counts GPT-4o labels only; the Qwen labels are in `judge-qwen/summary.json` and `RESULTS.md`.

### Settings

Per-run choices (`--run-name`, `--repeat`, sample selection, `--num-shards`, `--log-level`) are command options; `--help` on any subcommand lists them. Machine settings are environment variables, read in `benchmarks/duplex/fdb/common.py`:

| Variable | Default | Meaning |
|---|---|---|
| `FDB_WORK` | `$HOME/fdb` | Root for downloads, environments and results |
| `JUDGE` | `qwen` | `qwen` or `gpt` |
| `CUDA_VISIBLE_DEVICES` | unset | The GPU indices this job may use. The first is the job GPU: it runs the one model server of `generate`, `asr` and `judge`; `generate --gpus` may list any of the others. See [Running jobs in parallel](#running-jobs-in-parallel) |
| `GPU` | `0` | The GPU index when `CUDA_VISIBLE_DEVICES` is unset |
| `SERVER_PORT`, `JUDGE_PORT` | `8097 + 10×GPU`, `30000 + 10×GPU` | Local ports. The judge also uses the next port for NCCL. Avoid 29500-29899, which the model server uses for NCCL |
| `MODEL_PATH` | `$FDB_WORK/models/MiniCPM-o-4_5` | Checkpoint of the model under test |
| `MODEL_REVISION` | `503e754…` | Recorded in the run manifest; must match `MODEL_PATH` |
| `JUDGE_MODEL_PATH` | `$FDB_WORK/models/Qwen3.8-27B` | Checkpoint of the Qwen judge |
| `SESSION_TIMEOUT_S`, `V10_SESSION_TIMEOUT_S` | `90`, `230` | Per-session client deadline for v1.5 and v1.0. v1.5 inputs run up to about 18 s; v1.0 inputs average about 18 s and some run past 90 s. Neither deadline may exceed the recorder's 230 s |
| `SCORING_VENV` | `$FDB_WORK/scoring-venv` | Venv that runs ASR, timing, the evaluators and the judge clients |

### Running jobs in parallel

One job uses one GPU. To run several at once, open one terminal per GPU and give each its own GPU index and run name:

```bash
export FDB_WORK="${FDB_WORK:-$HOME/fdb}"
unset GPU SERVER_PORT JUDGE_PORT
export CUDA_VISIBLE_DEVICES=1    # 0 in the first terminal, 1 in the second, and so on
RUN=minicpmo-30-gpu1
python -m benchmarks.duplex.fdb generate --run-name "$RUN" --v15-per-subset 30 --v10-per-subset 30
python -m benchmarks.duplex.fdb asr --run-name "$RUN"
python -m benchmarks.duplex.fdb judge --run-name "$RUN"
python -m benchmarks.duplex.fdb aggregate --run-name "$RUN"
```

The ports, the judge config and the compile caches follow the GPU index: the model port is `8097 + 10×GPU`, the judge port `30000 + 10×GPU`, the judge config is in `$FDB_WORK/judge/port-<judge port>/` and the caches in `$FDB_WORK/cache/gpu-<index>/`. The first line `generate` prints confirms them, for example `Job GPU 1: model port 8107, judge port 30010 (nccl 30011), ...`. The `unset` line matters when a shell profile sets `GPU` or a port. Never point two terminals at the same run name.

### Serving metrics

Serving metrics answer how well a server keeps up when several sessions share it. They are computed only from what the client records, which means any server that speaks the same realtime session protocol or the legacy turn-based protocol can be put in the same table.

Start the server yourself, with `max_sessions` at least the largest level, then sweep:

```bash
python -m benchmarks.eval.benchmark_duplex_serving sweep \
  --dataset v1.5 --concurrency 1,2,4,8 --output "$FDB_WORK/serving/minicpmo" -- \
  --url ws://127.0.0.1:8097/v1/realtime --profile minicpmo-native-pr2377 \
  --dataset-root "$FDB_WORK/dataset/v1.5" --dataset-revision "$(cat $FDB_WORK/dataset/v1.5.revision)" \
  --server-revision "$(git rev-parse HEAD)" --model openbmb/MiniCPM-o-4_5 --model-revision 503e754 \
  --max-per-subset 4
```

The options after `--` go unchanged to the `record` command of `benchmark_duplex_v15` (or `benchmark_duplex_v10` with `--dataset v1.0`); the sweep adds `--concurrency N` and `--output <output>/concurrency-N` for each level. It writes `serving.json`, `SERVING.md` and one run directory per level, and keeps going when a level has failed sessions, since those are part of the result. `report --run <run> --run <run> ... --output serving.json` builds the same table from runs that are already recorded.

Each level is one row:

| Column | Meaning |
|---|---|
| `mean open` | Sessions open at once, averaged over the level's recording; the first and last sessions of a level have fewer neighbours |
| `sessions failed` | Sessions whose protocol replay failed or that broke off |
| `503`, `pacing violations` | Sessions denied admission, and sessions whose input drifted more than one 80 ms packet from its schedule. A pacing violation means the load generator fell behind, so the level does not measure the server |
| `unit miss` | Native profiles only. Units whose lag exceeds one nominal unit (1000 ms for MiniCPM-o), plus every unit of a failed session |
| `unit lag` | Native profiles only. Receipt of `sglang.unit.done` minus the time the client finished sending the packet holding the unit's last audio; the short last unit, which the server fills with silence, starts at the input end instead. On MiniCPM-o it matches the server-side `ready_to_finished_ms` of the request event recorder within a few milliseconds |
| `first audio` | Native: from the unit carrying a reply's first audio being ready to that audio's receipt. Turn-based: from `response.created`, which arrives when the server ends the user's turn, to the first audio |
| `buffer needed` | The smallest startup buffer with which a reply plays without a gap. It does not depend on `--playback-reserve-ms` |
| `underrun replies` | Replies that a player starting `--playback-reserve-ms` (80 ms by default) after their first audio would have to pause. This column depends on the reserve: at concurrency 8 on MiniCPM-o, 21 of 35 replies pause at 80 ms while `buffer needed` p90 is 251 ms |
| `RTF p90` | Turn-based only. The time from a reply's first to its last audio arrival over the audio before the last chunk. Native profiles show `-`, because a native reply comes out one unit per input unit and its factor stays near 1 whatever the load |
| `meets targets` | `yes`, or the targets the level misses |

A level meets the targets when no session failed, no session was denied admission, no input pacing was violated, and, for native profiles, the unit miss rate is at most 1%, or, for turn-based profiles, the reply RTF p90 is below 1. A turn-based server's unit receipts track input intake, so it is judged by how fast its replies stream. The capacity is the highest level below the first one that misses; when every recorded level meets the targets, the capacity is at least the highest one and a higher level has to be recorded.

These are client-side times on the client's host: they include the network to the server and leave out the player's device and the acoustics. Where the time goes inside SGLang-Omni comes from the request event recorder instead; see [Full Duplex session Unit metrics](profiler.md#full-duplex-session-unit-metrics).

(minicpm-o-45-on-one-h100)=

**MiniCPM-o 4.5 on one H100**


One H100 80 GB ran a short v1.5 audio cohort with the native session protocol. Each recording selected the first 16 naturally sorted pairs from each of the four v1.5 subsets, including both clean and overlap variants: 128 sessions and 1,744 nominal input units per recording. The server allowed 64 session slots, but its memory pools were sized for this short cohort. A warmup pair was excluded. Each recording used a fresh client process against the same continuously running server.

For this campaign, a recording passed only with **fewer than 1%** late or missing units, all 128 sessions successful, no admission failures, no input pacing violations, and the requested overlap of complete input streams. A unit missed when `sglang.unit.done` arrived more than 1,000 ms after the client had its input ready. This strict campaign gate differs from the general serving target above, which permits exactly 1% misses. `session.created` alone does not establish model admission.

| Concurrent sessions | Recorded result | Qualification |
| ---: | --- | --- |
| 1, 2 | One screening run each, with no unit misses | Both passed; neither was repeated. |
| 4 | Five passing runs, 640 successful sessions, 0 of 8,720 unit misses | Highest tested level confirmed by five runs. The median of the five per-run unit-lag p95 values was 421.4 ms, with a range of 406.5 to 433.1 ms. |
| 8 | Two valid runs: 3 of 1,744 misses (0.172%) and 25 of 1,744 (1.433%) | The first passed; the second failed the unit gate. |
| 16 | Two valid runs: 103 of 1,744 misses (5.906%) and 90 of 1,744 (5.161%) | Both failed the unit gate. |
| 32, 48, 64 | CUDA out-of-memory failures with failed sessions and invalid aggregate input pacing | Observed failures. Their latency and unit-miss aggregates are not qualified paced-load results. |

The C4 runs reached four overlapping complete input streams with no session, admission, or pacing failures. At the separate 80 ms playback reserve, simulated playback still underruns in 438 of 731 C4 replies. Each C4 run recorded audio in 126 or 127 of its 128 sessions. Protocol success does not establish response quality. Concurrencies 5 through 7 were not measured, so the result does not identify an exact maximum concurrency of four. The cohort also does not qualify long calls, video input, semantic quality, or an H100 versus H200 comparison.

The 128 inputs totaled 1,676.565 seconds; the longest was 15.763 seconds. The run used Python 3.12.3, torch 2.13.0+cu130, sglang 0.5.21, transformers 5.12.1, websockets 17.1, and soundfile 0.13.1, with 64 assigned CPU threads. The screen order was C8, C16, C32, C48, C64, C4, C2, then C1. After a valid C8 repeat failed the unit gate, repeats 3 through 5 at C8 and C16 were cancelled and four more C4 runs were recorded. The lower-concurrency cells therefore include the same server's earlier high-load history.

To reproduce the configuration, use server and serving-tool commit [`3ae4c2464eb0b44e0850a9f17da46461a688bc32`](https://github.com/sgl-project/sglang-omni/commit/3ae4c2464eb0b44e0850a9f17da46461a688bc32), the [client changes with the measured file bytes](https://github.com/edwingao28/sglang-omni/commit/0b2ca15afa211cd5c2212a8803a20b7c5db6d880), and model snapshot `503e754207c94da6bb26850b4469f367c9ea3582`. The client changes move per-variant report writes off the event loop and import only the selected recorder. The separate empty-EOS scoring fix in [PR #2868](https://github.com/sgl-project/sglang-omni/pull/2868) was not applied to these measurements; the selected inputs had no integer-second durations. Keep the 80 ms input pacing tolerance unchanged.

The `--dataset-revision` value below is the retained historical label, not a content pin for the upstream Drive archives. The campaign verified all 448 selected file hashes against its retained inputs. Those hashes remain with the private recording artifacts; a fresh dataset download alone does not establish identical inputs.

After [setup](#setup), point `MODEL_PATH` at that exact model snapshot and use the v1.5 dataset described above. Start the server in one terminal:

```bash
export FDB_WORK="${FDB_WORK:-$HOME/fdb}"
export MODEL_PATH="$FDB_WORK/models/MiniCPM-o-4_5"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
python -m sglang_omni.cli serve \
  --model-path "$MODEL_PATH" --variant session --max_sessions 64 \
  --thinker.engine.kv_cache_bytes 17179869184 \
  --perception.gpu_memory_fraction null \
  --perception.total_reserve_bytes 17179869184 \
  --talker.engine.max_total_tokens 65536 \
  --talker.gpu_memory_fraction 0.08 \
  --enable-realtime --host 127.0.0.1 --port 8099
```

Run each concurrency and repeat in a new client process and output directory. For example, set `CONCURRENCY=4` and vary `REPEAT` from 1 through 5:

```bash
export FDB_WORK="${FDB_WORK:-$HOME/fdb}"
CONCURRENCY=4
REPEAT=1
python -m benchmarks.eval.benchmark_duplex_serving sweep \
  --dataset v1.5 --concurrency "$CONCURRENCY" \
  --output "$FDB_WORK/serving/h100-short/c${CONCURRENCY}-r${REPEAT}" -- \
  --url ws://127.0.0.1:8099/v1/realtime \
  --profile minicpmo-native-pr2377 \
  --dataset-root "$FDB_WORK/dataset/v1.5" \
  --dataset-revision 3e799c45a045256f47d5f1c9cda90157e2d2ec9e \
  --server-revision 3ae4c2464eb0b44e0850a9f17da46461a688bc32 \
  --model openbmb/MiniCPM-o-4_5 \
  --model-revision 503e754207c94da6bb26850b4469f367c9ea3582 \
  --timeout 90 --max-per-subset 16
```

The reported results came from one C1 and C2 screen, five C4 runs, two C8 runs, and two C16 runs. C32, C48, and C64 were attempted once each; their failed recordings remain part of the observation. The complete raw traces remain in the original experiment storage, and the selected local audit copy cannot rescore every session. Neither raw tree is hosted with this runbook.

### Notes

- **Keep the settings fixed across the repeats of a run.** Mixing repeats with different selections, `--num-shards` or judges makes their mean meaningless; use a new run name instead.
- **`--num-shards` changes what you measure.** With `--num-shards 2`, two sessions share the GPU, so the latencies are measured under load and do not compare with one shard. The model server keeps `max_sessions` at 2, so `generate` refuses more than 2 shards per server up front.
- **`--gpus` spreads the shards over GPUs.** `--gpus 0,1` starts one model server on each listed GPU, each with its own port (`8097 + 10×GPU`) and caches, and sends shard i to the i-th server modulo their count. With as many shards as GPUs every session has a GPU to itself, so the timing compares with one shard. `asr` and `judge` still run on the job GPU, the first index of `CUDA_VISIBLE_DEVICES`, so `CUDA_VISIBLE_DEVICES=0,1` with `--gpus 0,1` records on both GPUs and transcribes on GPU 0.
- **The first N samples are not a random sample.** They give fast numbers that compare between runs, not estimates for the full dataset.
- **Failed sessions are never dropped.** They stay in the denominators as ineligible, and an empty set of intervals shows as `n/a`, not as zero.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `ERROR: the model server did not become ready` (or the judge server) | The message ends with the last lines of the server log; the full log is `repeat-N/logs/model-server-gpu<N>.log` or `judge-server.log` |
| `ERROR: the model server exited with code N during the recordings` | The model server died while recording; the message ends with the last lines of that GPU's `repeat-N/logs/model-server-gpu<N>.log`. Fix the cause, delete the repeat and generate it again |
| `ERROR: the judge server exited with code N before ...` or `during ...` | The judge server died; the message names the phase and ends with the last lines of `repeat-N/logs/judge-server.log`. Fix the cause and rerun `judge` with `--retry-failed`. If the phase is the v1.5 semantic judge, delete `repeat-N/semantic-qwen` first, because the batches it could not grade are saved as unresolved and a rerun reuses them |
| `ERROR: something already serves ...` | Another server holds the port, often left from an interrupted run or a second job on the same GPU. Stop it, or give this job another `CUDA_VISIBLE_DEVICES` |
| `ERROR: .../repeat-N already holds a recording` | That repeat was generated before; use the next `--repeat`, or delete the directory to redo it |
| `ERROR: ... selects different pairs` or `... selects different v1.0 samples` | An earlier repeat of this run used another selection; pass the same options, or use a new run name |
| `ERROR: .../repeat-N is a v1.5-only repeat` | The earlier repeats of this run left v1.0 out; continue with the option change the message names, or use a new run name |
| `ERROR: both datasets are left out` | Give at least one dataset a nonzero count |
| `ERROR: .../dataset/v1.0 lacks the v1.0 subset(s) ...` | The workspace was set up without v1.0; rerun `setup`, which adds it |
| `ERROR: .../reference-manifest.json is missing` or `.../v10/reference/manifest.json is missing` | `generate` recorded that dataset but did not finish its export; delete the repeat and generate it again |
| A shard log reports `fail` or `error` sessions | They stay in the denominators. Read `repeat-N/logs/record-shard-*.log` or `repeat-N/v10/logs/record-shard-*.log`; if most sessions fail, fix the server and redo the repeat |
| `ERROR: --num-shards N over G GPU(s) exceeds the model server's max_sessions` | Lower `--num-shards` or add GPUs with `--gpus`; `generate` refuses more shards than the servers admit before recording |
| `ERROR: --num-shards N leaves model servers idle` | Use at least as many shards as GPUs in `--gpus` |
| `ERROR: SERVER_PORT gives every model server the same port` | Unset `SERVER_PORT` when `--gpus` lists several GPUs; each server then derives its port from its GPU |
| HTTP 503 in a record log | Another client may hold the server's sessions; check for competing recordings |
| `ERROR: CUDA_VISIBLE_DEVICES must list distinct numeric GPU indices` | Export indices such as `0` or `0,1`, not GPU UUIDs |
| `ERROR: --gpus ... reaches outside CUDA_VISIBLE_DEVICES` | List only GPUs that `CUDA_VISIBLE_DEVICES` names, or widen it |
| `--device cuda needs exactly one visible GPU` (manual ASR command) | Export a single index for that command |
| `ERROR: GPU and CUDA_VISIBLE_DEVICES disagree` | Unset `GPU`, or set it to the first index |
| `WARNING: a v1.5 phase reported failures` | Rerun the command the warning prints; it adds `--retry-failed` |
| `WARNING: the v1.0 ASR failed` | `repeat-N/v10/reference/logs/asr.log` has the official script's output and the terminal has the traceback. A rerun of `asr` transcribes every v1.0 sample again |
| `WARNING: v1.0 unified scoring failed` | ASR completed, but the saved export, transcript or annotation did not score. Fix the stated cause and run the CPU-only backfill command printed in the warning; it does not repeat ASR or v1.5 work. Rerun `aggregate` to refresh `RESULTS.md` |
| `Unified metrics unavailable` in `RESULTS.md` | The report names a missing export/ASR file, missing output pair, stale hashes, unsupported version or incomplete subset. Repair the export or ASR if named, then use the per-repeat backfill command in the report and rerun `aggregate` |
| `WARNING: the listening-quality phase failed` | The terminal has the traceback. A `ModuleNotFoundError` for `s3prl` or `whisper` means the serving venv lacks the repository's dependencies; reinstall them. Rerun `asr`; it recomputes the listening metrics in full |
| `WARNING: the v1.0 evaluation failed for ...` | The traceback is at the end of `repeat-N/v10/reference/logs/evaluate-<subset>.log`. A `ZeroDivisionError` in `candor_turn_taking` or `synthetic_user_interruption` means no selected sample took the turn, so select more samples in a new run. After fixing another cause, rerun `judge`; it evaluates the missing subsets again |
| `judge request ... got no parsable rating in 3 attempts` | The v1.0 relevance judge answered without a rating; the ledger in `v10/reference/judge/` has each reply. The official parser needs `Analysis: <text>`, a line break, then `I would rate the AI's response as <digit>`. The attempts share a seed, so a local judge repeats the same reply and a rerun fails again |
| `custom judge identity changed; use a new --out` | The judge config changed since this repeat was judged, for example after an SGLang upgrade. Delete `repeat-N/judge-qwen` and rerun `judge` |
| `Control check failed: this judge configuration is not accepted.` | The judge missed a control case, listed in the output; nothing graded alongside the controls is kept. Check that the judge server runs the pinned Qwen3.8-27B revision; do not grade with a configuration that fails the controls |
| `... changed since this directory was created; use a new --out` | The semantic judge settings, prompt or inputs changed since this repeat was graded. Delete `repeat-N/semantic-qwen` and rerun `judge` |
| `model_mismatch` with `JUDGE=gpt` | The endpoint answered with a model other than `gpt-4o-2024-08-06`; use an endpoint that serves exactly that model |
| `No module named 'onnx'` (or `einops`) in `model-server-gpu<N>.log`, or `ModuleNotFoundError` in `asr` | Rerun `setup` |
| `setup` stops with `torchcodec cannot load its FFmpeg libraries` | Install the FFmpeg shared libraries (version 4 to 8) and rerun `setup` |

### Scoring CLI reference

The runbook wraps three CLIs, which also score a recording by hand: `benchmarks.eval.benchmark_duplex_v15` records v1.5 sessions, `benchmarks.eval.benchmark_duplex_reference` scores them with the official v1.5 ASR, timing and behavior code, and `benchmarks.eval.benchmark_duplex_v10` records and scores v1.0 sessions. Run `eval "$(python -m benchmarks.duplex.fdb env)"` first to set the variables the commands below use. Recording and export run in the sglang-omni venv (`python`); ASR, timing, judging and evaluation need the packages `setup` installs in the scoring venv, so those commands run with `"$SCORING_PYTHON"`. `record` needs the model server and `custom-judge` the judge server; `python -m benchmarks.duplex.fdb serve-model` and `serve-judge` start them in the foreground until Ctrl-C.

(dependencies-and-source)=

**Dependencies and source**


The official scoring scripts and the behavior prompt are not vendored. They are loaded from a [Full-Duplex-Bench checkout](https://github.com/DanielLin94144/Full-Duplex-Bench/tree/3e799c45a045256f47d5f1c9cda90157e2d2ec9e) at revision `3e799c45a045256f47d5f1c9cda90157e2d2ec9e`, and their SHA-256 is checked before every run. Their [CC BY-NC 4.0 license](https://github.com/DanielLin94144/Full-Duplex-Bench/blob/3e799c45a045256f47d5f1c9cda90157e2d2ec9e/LICENSE) applies.

The scoring venv uses Python 3.12, torch and torchaudio 2.11.0+cu130, NeMo 3.0.0, Silero VAD 6.2.1, numpy 2.5.3, scipy 1.18.1, soundfile 0.14.0, openai 3.28.0, pydantic, python-dotenv and torchcodec 0.11. Each phase records the package versions it ran with. No command downloads a model on its own. ASR uses a local `nvidia/parakeet-tdt-0.6b-v2` checkpoint at revision `ae9ad07059c7c739ffaf932226a8fe64ae2620b0`, `.nemo` SHA-256 `d99e39955c9d3d0350d8fb7c75e40c64a2b2eaeb003883d7c941fd2e8747b28c`.

(v15-scoring)=

**v1.5 scoring**


Record and export:

```bash
OUT=results/fdb-manual
python -m benchmarks.eval.benchmark_duplex_v15 record \
    --profile minicpmo-native-pr2377 \
    --dataset-root "$FDB_DATASET" --dataset-revision "$(cat "$FDB_WORK/dataset/v1.5.revision")" \
    --url "$REALTIME_URL" --model "$MODEL_ID" --model-revision "$MODEL_REVISION" \
    --server-revision "$(git rev-parse HEAD)" --timeout "$SESSION_TIMEOUT_S" \
    --output "$OUT/recording"

python -m benchmarks.eval.benchmark_duplex_reference export \
    --engine model --trace-format realtime-pcm16-v1 --run "$OUT/recording" \
    --dataset-root "$FDB_DATASET" --out "$OUT/reference-audio"
```

`record` takes the full dataset unless `--max-per-subset N` or repeated `--sample-id category/id` narrows it, and exits 0 only when every selected session passes. `export` cuts the observation windows into a new directory without touching the recording. With `--dataset-root`, every dataset pair that was not recorded counts as missing, so a partial run also passes `--only category/id` for each recorded pair. Repeated `--run` takes disjoint shards; `--engine` is a free label for the scored cohort.

A turn-based (half-duplex) endpoint speaks the conversation protocol instead of the native session protocol: it takes a `turn_detection` object in `session.update`, sends none of the `sglang.input_audio.*` receipts or `sglang.unit.done`, and streams `response.audio.delta` at 24 kHz. The profile `qwen3-omni-half-duplex` switches the recorder, the protocol oracle and the export to that protocol:

```bash
python -m benchmarks.eval.benchmark_duplex_v15 record \
    --profile qwen3-omni-half-duplex \
    --turn-detection '{"type": "server_vad", "silence_duration_ms": 500}' \
    --dataset-root "$FDB_DATASET" --dataset-revision "$(cat "$FDB_WORK/dataset/v1.5.revision")" \
    --url "$REALTIME_URL" --model "$MODEL_ID" --model-revision "$MODEL_REVISION" \
    --server-revision "$(git rev-parse HEAD)" --timeout "$SESSION_TIMEOUT_S" \
    --output "$OUT/recording"

python -m benchmarks.eval.benchmark_duplex_reference export \
    --engine model --trace-format realtime-legacy-pcm16-v1 --run "$OUT/recording" \
    --dataset-root "$FDB_DATASET" --out "$OUT/reference-audio"
```

- `--turn-detection` is required for this profile and rejected for native ones. It is sent once per session, so a detector sweep is one run per setting against the same server: `server_vad` takes `threshold`, `prefix_padding_ms` and `silence_duration_ms`, `semantic_vad` takes `eagerness`, and both take `interrupt_response`. The manifest records it under `config.legacy`. `benchmark_duplex_v10 record` takes the same options.
- A session fails unless `session.updated` echoes the requested `turn_detection.type`, so a detector that silently falls back is caught. There are no input or drain receipts: the session ends `--legacy-tail` seconds (default 1.0) after the input with a no-op `session.update`, whose `session.updated` proves the server is alive.
- Any `error` event fails the session. A response cancelled with reason `turn_detected` is accepted only after an `input_audio_buffer.speech_started` during it.
- Simulated playout follows the server's barge-in: it is cut at `output_audio_buffer.cleared`, or at `input_audio_buffer.speech_started` unless the session set `interrupt_response: false`. The export applies the same cut, records `barge_in_cuts`, and marks a session eligible only when its protocol verdict is `pass`.

A turn-based endpoint can also speak the session protocol. It takes `turn_detection` under `session.audio.input` in `session.update`, grants `interaction: turn_based` with 200 ms units and a flushed input tail, sends the `sglang.input_audio.*` receipts and `sglang.unit.done`, detects the turns itself and streams `response.output_audio.delta` at 24 kHz. The profile `turn-based` records it, and its export takes `--trace-format realtime-turn-based-pcm16-v1`:

```bash
python -m benchmarks.eval.benchmark_duplex_v15 record \
    --profile turn-based \
    --turn-detection '{"type": "server_vad", "silence_duration_ms": 500}' \
    --dataset-root "$FDB_DATASET" --dataset-revision "$(cat "$FDB_WORK/dataset/v1.5.revision")" \
    --url "$REALTIME_URL" --model "$MODEL_ID" --model-revision "$MODEL_REVISION" \
    --server-revision "$(git rev-parse HEAD)" --timeout "$SESSION_TIMEOUT_S" \
    --output "$OUT/recording"

python -m benchmarks.eval.benchmark_duplex_reference export \
    --engine model --trace-format realtime-turn-based-pcm16-v1 --run "$OUT/recording" \
    --dataset-root "$FDB_DATASET" --out "$OUT/reference-audio"
```

- `--turn-detection` is required. These endpoints accept only `server_vad`, configured by `threshold`, `prefix_padding_ms` and `silence_duration_ms`. A session fails unless `session.updated` echoes the type under `session.audio.input.turn_detection`. The manifest records the setting under `config.turn_based`.
- The session ends like a native one: `sglang.input_audio.end` at the input end, then `session.close` after `sglang.input_audio.drained`. The server first answers the utterance still open at the input end, so `--timeout` must cover the longest input plus that answer.
- A response may end `completed` with reason `stop` or `empty_turn`, or `cancelled` with reason `interrupted`. Each cancellation needs its own earlier `input_audio_buffer.speech_started`, but not one during the response, because the server can create the reply it cancels after that event. Any `error` fails the session. `supports_truncate` may take either value.
- The cut point differs from the legacy rule. These endpoints send no `output_audio_buffer.cleared`, so simulated playout and the export do not cut at `speech_started`; they drop queued audio when a `response.done` with status `cancelled` arrives. As for the legacy protocol, the export records `barge_in_cuts` and marks a session eligible only when its protocol verdict is `pass`.
- v1.0 recordings export the same way: `benchmark_duplex_v10 reference-export --trace-format realtime-turn-based-pcm16-v1` resamples the 24 kHz audio to 16 kHz and keeps only sessions whose protocol verdict is `pass`.

Score:

```bash
TREE=(--reference-source "$FDB_SOURCE" --tree "model=$OUT/reference-audio")
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_reference asr "${TREE[@]}" \
    --out "$OUT/scores" --nemo "$PARAKEET_NEMO" --nemo-sha256 "$PARAKEET_SHA256" --device cuda
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_reference timing "${TREE[@]}" \
    --out "$OUT/scores" --audio-loader soundfile
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_reference prepare-judge "${TREE[@]}" --out "$OUT/scores"
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_reference judge "${TREE[@]}" --out "$OUT/scores" \
    --judge gpt-4o-2024-08-06 --api-key-env OPENAI_API_KEY
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_reference summarize "${TREE[@]}" --out "$OUT/scores"
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_reference report --scores "$OUT/scores" --engine model
```

Run the phases in order on one output directory with the same `--tree`; repeat `--tree` to score several cohorts together. CUDA ASR needs exactly one visible GPU. A rerun resumes from the receipts; `--retry-failed` retries failed work units, and `--limit N` caps the units of one call. After changing the recording, the scoring configuration or the reference code, use a new output directory. A zero exit means the phase finished, not that every sample qualified.

`report` reads only `summary.json` and the manifest, so it needs no GPU or key. It prints the populations, protocol verdicts, ASR and timing coverage, interval statistics with confidence intervals, and the behavior label distribution. `--semantic-summary` adds the semantic judge's `summary.json`, and `--replay replay.json` adds offline replay agreement in the form `{"selected": N, "replayed": M, "samples": [{"sample": "category/id", "variant": "overlap", "recorded_status": "pass", "status": "match"}]}`.

The GPT judge reads its key from `OPENAI_API_KEY`, never from an argument or a manifest; `--base-url` selects a compatible endpoint. It sends the exact reference prompt with seed 1 and makes at most three attempts. A reply from a model other than `gpt-4o-2024-08-06`, a malformed label or a failed request leaves the label invalid, with no fallback judge. Without a key the labels stay pending and `summarize` still runs.

`custom-judge` sends the same prompt to a self-hosted model. Its labels never replace GPT-4o labels. It reads an existing scoring directory and writes to a separate new one:

```bash
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_reference custom-judge "${TREE[@]}" \
    --source-scores "$OUT/scores" --out "$OUT/judge-qwen" \
    --judge-config "$JUDGE_CONFIG" --base-url "$JUDGE_URL"
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_reference custom-summarize "${TREE[@]}" \
    --source-scores "$OUT/scores" --out "$OUT/judge-qwen" \
    --judge-config "$JUDGE_CONFIG"
```

`judge` and `serve-judge` write the judge config to `$JUDGE_CONFIG`, which is `$FDB_WORK/judge/port-$JUDGE_PORT/judge-config.json`:

```json
{
  "model_id": "Qwen/Qwen3.8-27B",
  "model_revision": "<40-hex commit>",
  "tokenizer_id": "Qwen/Qwen3.8-27B",
  "tokenizer_revision": "<40-hex commit>",
  "served_model": "qwen3.8-27b",
  "precision": "bf16",
  "enable_thinking": false,
  "decoding": {
    "temperature": 0.0, "top_p": 1.0, "top_k": -1,
    "min_p": 0.0, "repetition_penalty": 1.0, "max_tokens": 512
  },
  "seeds": [1, 2, 3],
  "server_launch_receipt": "launch-receipt.json",
  "server_launch_receipt_sha256": "<sha256 of the receipt>"
}
```

Both revisions must be full commit hashes. `temperature`, `top_p` and `max_tokens` are required; the other decoding fields and `enable_thinking` are sent only when present, so leave out any the endpoint does not support. The launch receipt records the server command, runtime version, checkpoint and tokenizer revisions and precision, and the client checks its hash, because the model name an endpoint returns does not prove which checkpoint it loaded. The endpoint key goes in `CUSTOM_JUDGE_API_KEY`, or a placeholder such as `EMPTY` without authentication. A rerun with the same config skips finished samples; a changed config or selection needs a new `--out`. `--concurrency` (default 16) caps the requests in flight and is recorded with the run. `--retry-failed` retries transport and parsing failures only; wrong models, invalid labels and non-`stop` finishes stay invalid. The per-category Wilson 95% intervals describe dataset sampling, not judge agreement or generation variance.

(v10-scoring)=

**v1.0 scoring**


v1.0 runs the official `get_transcript/asr.py` and `evaluation/evaluate.py` unchanged:

```bash
OUT=results/fdb10
python -m benchmarks.eval.benchmark_duplex_v10 record \
    --profile minicpmo-native-pr2377 \
    --dataset-root "$FDB10_DATASET" --dataset-revision "$(cat "$FDB_WORK/dataset/v1.0.revision")" \
    --url "$REALTIME_URL" --model "$MODEL_ID" --model-revision "$MODEL_REVISION" \
    --server-revision "$(git rev-parse HEAD)" --timeout "$V10_SESSION_TIMEOUT_S" \
    --output "$OUT/recording"
python -m benchmarks.eval.benchmark_duplex_v10 reference-export \
    --run "$OUT/recording" --dataset-root "$FDB10_DATASET" --out "$OUT/reference"
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_v10 reference-asr \
    --tree "$OUT/reference" --reference-source "$FDB_SOURCE" \
    --nemo "$PARAKEET_NEMO" --nemo-sha256 "$PARAKEET_SHA256" --device cuda
CUDA_VISIBLE_DEVICES="" "$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_v10 unified-score \
    --tree "$OUT/reference"
"$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_v10 reference-evaluate \
    --tree "$OUT/reference" --reference-source "$FDB_SOURCE"
```

- `reference-export` writes each sample's window as `<subset>/<id>/output.wav` with its annotation into a new directory, plus `manifest.json`. Repeated `--run` takes shards. An ineligible session gets no folder and keeps its reasons in the manifest. A recording made with a turn-based profile needs `--trace-format realtime-legacy-pcm16-v1`; as in the v1.5 export, the audio is resampled from the profile's output rate and a session is eligible only when its protocol verdict is `pass`.
- `reference-asr` transcribes every sample with the local Parakeet checkpoint. For user interruption it cuts the audio at the end of the interruption and shifts the word times back, so the transcript holds only the reply. It writes `output.json` per sample and the receipt `asr.json`, and runs once per export.
- `unified-score` reads the saved manifest, ASR transcript and annotations on CPU, then writes `unified.json` and `unified-samples.jsonl`. It is independent of the official evaluator and can be rerun for an existing export without repeating ASR. To backfill a runbook repeat, use `eval "$(python -m benchmarks.duplex.fdb env)"`, then `CUDA_VISIBLE_DEVICES="" "$SCORING_PYTHON" -m benchmarks.eval.benchmark_duplex_v10 unified-score --tree "$FDB_WORK/runs/RUN_NAME/repeat-N/v10/reference"`, replacing `RUN_NAME` and `N` with the recorded run and repeat; rerun `aggregate` afterward.
- `reference-evaluate` runs the evaluator per subset and stores the printed `[Result]` numbers, the counts and the log path in `summary.json`. `--subset` evaluates only the named subsets; the rest can follow later.

The interruption relevance is rated by `gpt-4-turbo` with seed 0, as the official evaluator requests, for every interruption the model took over. That needs `OPENAI_API_KEY` (or the variable named by `--api-key-env`); without a key, evaluate the other subsets with `--subset`. `--base-url URL --served-model NAME` sends the same prompts to a self-hosted judge, with thinking off and the reply capped at 512 tokens, since the evaluator sets neither and SGLang would stop at 128 tokens. `summary.json` records the endpoint actually used and marks the rating official only for OpenAI's endpoint without `--served-model`. Every exchange is appended to `judge/user-interruption-<UTC time>.jsonl`.

The official evaluator has a retry bug: when a rating does not parse, it resends the same request without limit, and each retry counts that sample's takeover again. The client stops after three identical requests, which fails the subset, and `summary.json` reports `retries`. A nonzero value means the interruption takeover rate is inflated.

(measurement-definitions)=

**Measurement definitions**


- **Input pacing.** Audio is sent in 80 ms packets. A packet that starts more than 80 ms off its source cadence makes the session ineligible. This bounds transport jitter only.
- **Observation window.** Both datasets score the output over `[0, T]`, where T is the input length, counted from the first input packet. Received audio plays first-in first-out, so initial delay, gaps and queueing stay in; audio still queued at T is cut, and the rest is silence. The window is mono 16 kHz PCM16 with exactly the input's sample count. An error inside the window makes the session ineligible; silence is valid output.
- **v1.5 intervals.** The official timing code merges VAD speech spans across gaps of 0.6 s (user) and 0.5 s (model). Stop intervals are overlaps of user and model spans; response intervals run from the end of a user span to the start of the next model span. `summary.json` reports their pooled means and medians with sample-cluster bootstrap intervals.
- **v1.0 takeover and latency.** The evaluator reads the Parakeet transcript of the window. A sample took the turn when the words span at least 1 s or number more than 3. The latency is the first word's start minus the annotated end of the user's turn, clamped at 0. Pause handling looks at the whole window, not only at the annotated pauses.
- **Missing values.** Empty interval sets and missing labels are null, never a zero latency or 0%. The counts keep every selected, eligible, invalid and unscored sample.
- **Limits.** Simulated playout is not acoustic latency, and the window can cut long replies. One generation per sample does not estimate generation variance. Prosody and MOS are not measured.

(verification)=

**Verification**


The unit tests run the official formulas on synthetic fixtures without a GPU or API key, and cover export, missing samples, valid silence, timing, ASR deduplication, bounded and concurrent judging, the semantic judge's control gate, resumption, and the runbook's selection, phase gating, server checks and aggregation for both datasets:

```bash
FDB_REFERENCE_SOURCE="$FDB_SOURCE" python -m pytest -q \
    tests/unit_test/benchmarks/test_duplex*.py tests/unit_test/benchmarks/test_fdb*.py
```

Export is deterministic: exporting a recording again reproduces every eligible WAV byte for byte, and summarizing saved transcripts and intervals again reproduces the same summaries.

## Full-Duplex-Bench v1.0 unified metrics


`unified-score` scores a Full-Duplex-Bench v1.0 reference tree with turn-taking metrics that do not depend on how a system decides to speak. It reads the same Parakeet transcripts and dataset annotations as the official evaluator, and for user interruption it also runs Silero VAD on the exported audio, so native full-duplex, turn-based and cascade systems are scored by the same rules. It needs no GPU, LLM or network access.

### Why the official numbers are only a reference

`reference-evaluate` runs the pinned official evaluator unchanged, and its numbers stay in the report as the reference column. Four of its rules measure something other than their names suggest:

1. **Talking over the user gives the best latency.** `eval_smooth_turn_taking.py` sets a negative latency, a reply that starts before the user's turn ends, to 0. On one 119-sample capture of a native full-duplex model, 39 latencies were clamped this way.
2. **Pause takeover does not look at the pauses.** `eval_pause_handling.py` never reads `pause.json`. It counts a takeover in every nonempty transcript of the whole window except one that both spans less than 1 s and has at most 3 words. Speech before the user's first pause counts, and an output of at most 3 words within 1 s, such as a takeover that the user's resumed speech cuts off, does not. On Candor pause handling, the official rule found a takeover in 30 samples of a turn-based system that answers after 500 ms of silence, while that system's transcript contained a word onset at or after the first annotated pause in 67.
3. **Non-responders leave the latency average.** Latency is averaged only over samples that took the turn, so a model too slow to answer inside the window drops out and can look faster.
4. **Not every interruption interrupts.** `eval_user_interruption.py` scores every sample alike, whether or not the model was speaking when the user cut in. On one 200-sample capture of a native full-duplex model, no speech was detected at the interruption onset in 87 samples, and a reply after R in those samples still enters the official takeover rate, latency and judge rating.

### Inputs and checks

Run it on a tree written by `reference-export` and transcribed by `reference-asr` (see the [runbook](#full-duplex-bench-runbook)). All times are seconds on the input window `[0, T]`, where T is the input length recorded in `manifest.json`. Each transcript word has a start and an end from `output.json`; a word with a null end counts as zero-length at its start. Times are rounded to the microsecond before any comparison. This removes float noise in the files (an annotation at 6.32000000000005 against a word at 6.32), not ASR alignment error.

Loading a sample checks its times and stops with the sample id and the problem if any check fails: every word starts in `[0, T]`; the reference times E, S and R defined below lie in `[0, T]` with `S <= R`; `pause.json` lists at least one pause, each `[a, b]` with `a <= b` inside `[0, T]`, sorted and not overlapping; and no interruption word starts before R.

These annotations do not reliably label backchannels versus aborted turn starts; we therefore count every word onset, including a one-word "yeah". The rules are a strict word-timing proxy that also penalizes acknowledgments a listener might accept.

### Metrics

(pause-handling)=

**Pause handling**


Subsets `synthetic_pause_handling` and `candor_pause_handling`, reported separately. The official evaluator scores the whole window as one user turn, and we operationally treat the window as a non-yielding user turn. From the start of the first annotated pause to T the user is pausing or speaking again, so a model word starting in that region takes the floor too early. Each pause owns the time from its start to the next pause start, and the last pause owns the time to T.

| Metric | Definition | Better |
|---|---|---|
| `pause_intrusion_rate` | Samples with a model word starting at or after the first pause start | Lower |
| `pause_intrusion_per_pause` | Pauses whose owned time contains a model word start, over all pauses | Lower |
| `intrusion_s_per_pause_s` | Annotated pause seconds covered by ASR word spans, over all annotated pause seconds | Lower |
| `inside_pause_onset_rate` | Samples with a model word starting inside an annotated pause `[a, b)` | Context |
| `pre_pause_speech_rate` | Samples with a model word starting before the first pause start | Context |
| `carry_in_rate` | Samples with a model word that starts before the first pause start and ends after it | Context |
| `any_speech_rate` | Samples with any model word in the window | Context |

`pause_intrusion_rate` is the headline. It counts onsets up to T, not only inside the pauses, because a model whose reaction time exceeds a pause replies after the user has resumed; this needs no added reaction-time parameter. `inside_pause_onset_rate` is the narrower count of onsets inside the pauses.

Onsets before the first pause are outside this pause-conditioned observation region: they cannot react to a pause that has not started. Their samples stay in the denominator, and a model still talking at the first pause counts once it starts another word. This is a scope restriction, not a judgment that early talk is harmless; `pre_pause_speech_rate` and `carry_in_rate` report it.

`intrusion_s_per_pause_s` measures coverage of annotated pause time by ASR word spans, not acoustic occupancy. Any word overlapping a pause adds that overlap, including one that started before the first pause, and a zero-length word adds nothing. `pause_intrusion_per_pause` is an owned-interval diagnostic, not a per-pause takeover probability: one continuing utterance can place words in several owned intervals, the intervals differ in length, and the last one runs to T. `intrusion_words` in `unified-samples.jsonl` counts the intruding word onsets, not utterances or seconds.

(smooth-turn-taking)=

**Smooth turn-taking**


Subset `candor_turn_taking`. The reference time E is `turn_taking.json[0].timestamp[0]`, the time the official evaluator uses as the end of the user's turn. Each sample has exactly one outcome:

- `early_start`: a model word starts before E.
- `response`: no word starts before E and at least one starts at or after it. The latency is the start of the first word minus E.
- `no_response`: no transcribed word in the window.

| Metric | Definition | Better |
|---|---|---|
| `early_start_rate` | `early_start` samples | Lower |
| `response_rate_1s`, `_2s`, `_5s` | `response` samples with a latency of at most 1, 2 or 5 s | Higher |
| `no_response_rate` | `no_response` samples | Lower |
| `response_latency_s` | Median with its interval, and p90, of the latency, where a sample without a `response` never responds | Lower |

An early start never counts as a response, so talking through the end of the user's turn cannot buy a short latency, and an early acknowledgment followed by a good answer is an early start. `no_response` means no transcribed word, not every failed sample, so a system can have no silent samples and still not reach its p90.

Let F(t) be the share of samples with a `response` latency of at most t. Then `response_rate_ks` is F(k), and a latency quantile is Q(q) = inf{t : F(t) >= q}, the ceil(qN)-th smallest latency with no interpolation. The quantiles and the response rates are one curve: Q(q) is at most k exactly when `response_rate_ks` is at least q, so the median is the time by which half of all samples have responded. A quantile that falls on a sample without a `response` is reported as not reached. Placing those samples at infinity is a scoring convention for a valid response within the window, not an estimate of when a silent system would answer.

(user-interruption)=

**User interruption**


Subset `synthetic_user_interruption`. `interrupt.json[0].timestamp` gives the onset S and the end R of the user's interruption. `reference-asr` cuts the audio at R before transcribing, so the transcript holds only words from R on and cannot show whether the model stopped. Silero VAD on the full `output.wav` supplies that (see [Speech detection](#speech-detection)). Each sample records whether detected speech covers S (`speech_at_onset`, a segment with start <= S < end) and whether speech that began before R carries over it (`speech_at_end`, start < R < end; a reply that begins exactly at R does not). The outcome is decided in this order:

- `not_exercised`: no detected speech at S, so the interruption did not cut into detected speech. The sample keeps `speech_at_end`, so a model that starts talking during the interruption and talks past R stays visible.
- `speaking_at_end`: speech at S and at R. The words after R may continue the interrupted answer, so the sample has no latency. A model that stopped and then started a reply before R lands here too, the interruption counterpart of an early start.
- `response`: speech at S, none carried over R, and a transcribed word starts at or after R. The latency is the start of the first word minus R.
- `no_response`: speech at S, none carried over R, and no transcribed word after R.

| Metric | Definition | Better |
|---|---|---|
| `exercised_rate` | Eligible samples with speech at S | Context |
| `speaking_at_end_rate` | Exercised samples that are `speaking_at_end` | Lower |
| `response_rate_1s`, `_2s`, `_5s` | Exercised samples with a `response` within 1, 2 or 5 s | Higher |
| `no_response_rate` | Exercised samples that are `no_response` | Lower |
| `response_latency_s` | Median with its interval, and p90, over exercised samples, as for turn-taking | Lower |
| `stop_latency_s` | Median with its interval, and p90, of the time from S to the end of the segment covering S; a segment that runs to T has no observed stop | Lower |
| `speech_s_per_interruption_s` | Detected speech seconds inside [S, R] over interruption seconds, pooled over exercised samples | Lower |
| `restart_rate` | Exercised samples with a segment starting inside (S, R) | Lower |

The outcome only checks the endpoint R. A model can keep talking through most of the interruption, stop just before R and still get a `response`; `stop_latency_s`, `speech_s_per_interruption_s` and `restart_rate` show that. None of them tells a reactive stop from an answer that ended on its own, which would need a matched run with the interruption muted.

Every metric except `exercised_rate` is over exercised samples. Whether a sample is exercised depends on the system as well as the dataset, because S is fixed in input time: a model that answers briefly or late is silent at S more often. A `not_exercised` sample therefore counts as an unknown outcome in the selected bounds of those metrics, like an ineligible one; for `exercised_rate` itself it is a known no. A model silent at every S gets bounds of `[0, 1]` and is visibly unmeasured, and a terse model's lower bound drops. The bounds are worst-case completions under this fixed-onset protocol, not a bound on how a model would handle an interruption timed while it speaks. Read `exercised_rate` and the bounds with every interruption number.

(backchannel)=

**Backchannel**


`icc_backchannel` has no unified metric; the official frequency and JSD are the reference. `eval_backchannel.py` has known problems. Fixing them needs new definitions with their own validation, so they are only listed here:

- It adds every VAD segment up to 3 s to the backchannel list, including the ones it classified as a takeover.
- Its takeover flag is overwritten by each segment, and the loop stops at the first segment over 3 s, which also drops the later segments from frequency and timing.
- The timing score is SciPy's Jensen-Shannon distance with the natural log, at most about 0.83, while a sample without backchannels is given 1. The average therefore mixes the share of samples without backchannels with the timing distance of the rest.

(speech-detection)=

**Speech detection**


The scorer runs the Silero model that `load_silero_vad()` loads, with Silero's default threshold 0.5, minimum speech 250 ms and minimum silence 100 ms, and no padding. The default 30 ms of padding widens each segment on both sides and can move a reply that starts just after R to before it. `output.wav` must hold exactly T seconds of samples. Segments come from the whole file, so near S they can depend on audio after S. A pause longer than 100 ms ends a segment, so a model pausing between sentences exactly at S counts as `not_exercised`. On one 200-sample capture of a native full-duplex model, thresholds from 0.3 to 0.7 and minimum silences from 50 to 500 ms moved `not_exercised` between 85 and 95 samples, `response_rate_2s` between 0.952 and 0.957 and the median `stop_latency_s` between 1.04 and 1.27 s, and left `speaking_at_end` at 4. `restart_rate` ran from 0.05 to 0.27, because a shorter minimum silence splits speech into more segments, so it is a diagnostic and does not rank systems.

The audio is the received output played in order. For the turn-based realtime protocol the playout drops queued audio when the server reports that the user started speaking while interruptions are on; native full-duplex traces have no such event, so audio a system has already sent plays out, and the outcome describes the system as delivered.

(reading-them-together)=

**Reading them together**


Silence scores perfectly on pause handling, and one canned word right after E counts as a response. Read pause restraint and response rates together, from the same system configuration; neither is enough alone.

### Denominators and intervals

- Sample shares and latency quantiles are over the subset's eligible samples, and for interruption over its exercised ones. Eligible samples are those whose capture passed the `reference-export` checks (input pacing, an error-free observation window, dataset hashes that match the run). The pooled ratios divide by the pauses and the pause seconds of the eligible samples. `declared` is the full subset size.
- Ineligible samples are counted under `counts`, never dropped silently. A sample can have several reasons, so the counts in `counts.ineligible_reasons` need not sum to `ineligible`.
- An ineligible sample's outcome is unknown, and a capture failure can depend on the system. Every sample share therefore also has `selected_bounds`, `[count / selected, (count + unknown) / selected]`: its value over all selected samples with each unknown outcome taken as no, then as yes. The unknown samples are the ineligible ones, plus the `not_exercised` ones for every interruption metric except `exercised_rate`. `coverage` is eligible over selected for each subset.
- A share of samples has a 95% Wilson interval, whose endpoints are exactly 0 at a count of 0 and exactly 1 at a count of n. The pooled ratios and the median latency have a 95% percentile bootstrap interval that resamples whole samples (2000 resamples, seed 0), because the pauses of one sample are not independent. Each resample's median and the interval endpoints use the same quantile rule as above. A resample with no pause time has no `intrusion_s_per_pause_s` and is skipped. The output is deterministic.
- `min_observable_s` is the shortest time from the reference to T. T is the dataset input length, so every system has the same horizon on a sample. A response after T is not observed, so `response_rate_ks` is a lower bound when `min_observable_s` is below k. Candor turn-taking windows end about 5 s after E, so there `response_rate_5s` is close to the share that responded inside the window.
- The intervals cover dataset sampling only, not repeated generations or ASR errors. To compare two systems, pair their rows in `unified-samples.jsonl` by `sample_id` over the samples eligible in both, and for interruption exercised in both. Pairing does not remove selection bias, so report coverage, `exercised_rate` and the selected bounds as well, and for interruption how many samples eligible in both were exercised in both, only in one system, or in neither.

### Run

```bash
python -m benchmarks.eval.benchmark_duplex_v10 unified-score --tree "$OUT/reference"
```

It refuses a tree without `asr.json`, stops if the transcript or annotation of an eligible sample, or the `output.wav` of an interruption sample, differs from `asr.json` or `manifest.json`, and stops on a time that fails the checks above. It writes:

- `unified.json`: per subset, the counts, `coverage` and every metric with its interval, plus `selected_bounds` for sample shares and `min_observable_s` for turn-taking and interruption. Only the median latency has an interval, not p90. At the top are the definitions version (`fdb-v1.0-unified-4`), the hashes of `manifest.json` and `asr.json`, and the silero-vad version, model file hash and settings.
- `unified-samples.jsonl`: one row per selected sample of the four scored subsets (`icc_backchannel` has no rows) with every per-sample quantity, including the pause flags `pause_intrusion`, `inside_pause_onset`, `pre_pause_speech`, `carry_in` and `any_speech`, and for interruption the outcome, S, `speech_at_onset`, `speech_at_end`, `speech_before_onset`, `stop_s`, `speech_in_interruption_s` and `restarts`. Ineligible rows keep their sample id, subset and task and add their reasons.

It prints each subset's metrics as value, interval and count, with the bounds over all selected samples for each share. When `reference-evaluate` has written `summary.json` from the same `manifest.json` and `asr.json`, its numbers are printed below them as the reference column; if either hash differs, the column is left out and a line says which input changed. Interruption numbers rated by a judge other than the official one are labeled with that judge's model instead of `official`. `summary.json` is never written or used for scoring. Rerunning on the same tree writes the same bytes.

### Not measured

- **Why the model stops after an interruption.** `stop_latency_s` says when the speech covering S ends, not whether the model stopped because of the interruption; that needs a matched run with the interruption muted.
- **The end of the user's speech in a pause sample.** The window is treated as a non-yielding user turn. The official evaluator reads output chunks, not a user-end annotation, so it does not establish the true turn end either. A reply that starts in a short trailing silence counts as an intrusion, so not every counted onset is premature.
- **Relevance of a reply.** Neither these metrics nor the official turn-taking score check that a reply answers the user. Only the official interruption judge rating covers it, and that rating depends on the judge model the run configured.
- **ASR errors.** A missed or invented word changes the metrics it falls in. All systems share the pinned Parakeet checkpoint, which does not mean they share the same recognition error.
- **Acoustic latency.** The output is a simulated playout of the received audio.
