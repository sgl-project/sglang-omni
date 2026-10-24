# Native Qwen3-TTS

The native implementation of [#1960](https://github.com/sgl-project/sglang-omni/pull/1960)
lives under `sglang_omni_mlx/native/src/qwen3_tts/`. Both the HTTP server and the
synthesis CLI run C++ MLX without Python, Torch, SGLang or mlx-audio serving dependencies.
Python is used for build provisioning and reference/CI tooling.

Status: implementation awaiting a native build, real audio validation and populated
Mac goldens. No runtime accuracy or performance result is claimed. The PR remains
a draft until that qualification is complete.

## Build and checkpoints

Use the existing runtime build script, which pins MLX 0.32.3:

```bash
sglang_omni_mlx/native/scripts/build_runtime.sh build/qwen3-tts-runtime build/qwen3-tts-work
```

Provision the official, unquantized 0.6B CustomVoice and Base artifacts. Immutable
revisions and SHA-256 hashes for every consumed file are in `ci/model_pins.json`.

```bash
python sglang_omni_mlx/native/ci/provision_models.py build/qwen3-tts-data \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base
```

The loader accepts original official checkpoint layouts. It converts convolution
weights and folds codec codebooks at load. Pre-converted MLX/quantized checkpoints
are not supported. Keep `config.json`, the text tokenizer files, the top-level
safetensors and `speech_tokenizer/{config.json,model.safetensors}` together.

## Serve preset voices

```bash
build/qwen3-tts-runtime/bin/qwen3_tts_server \
  --model-path build/qwen3-tts-data/models/Qwen_Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --model-name Qwen3-TTS-12Hz-0.6B-CustomVoice --port 8000

curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-TTS-12Hz-0.6B-CustomVoice","input":"Hello, welcome to the native speech server.","voice":"Ryan","language":"English","seed":42,"max_new_tokens":256}' \
  --output speech.wav
```

Output is mono 24 kHz PCM16. `response_format` accepts `wav` (default) or `pcm`.
Set `stream=true` and `response_format="pcm"` for chunked PCM. Streaming and
non-streaming use the same codec schedule: eight new frames per chunk, then the
remaining frames at completion. The request owns convolution tails, transposed
convolution overlap and the decoder's sliding-window KV state.

The synthesis worker and CivetWeb disconnect helper reuse the native TTS work
from [Ming #2870](https://github.com/sgl-project/sglang-omni/pull/2870), coordinated
with [AuK #2444](https://github.com/sgl-project/sglang-omni/pull/2444).

The shared supervisor also accepts `--supervised --model-kind qwen3_tts
--model-directory DIRECTORY`. `/health`, `/v1/models` and shutdown follow the other
native servers. One synthesis request runs at a time; a busy service returns 429
and a stopping service returns 503. Disconnect and shutdown cancel generation
between model computations. Failed streams close without a successful final chunk.

## Request fields and limits

| Field | Behavior |
| --- | --- |
| `input` | Required non-empty text, at most 16,384 UTF-8 bytes. Reserved token markers are rejected. |
| `voice` | CustomVoice preset, case insensitive; default `Vivian`. Names come from the checkpoint. |
| `language` | Case insensitive checkpoint language, or `auto` (default). Dialect voices supply their language in auto mode. |
| `task_type` | Optional `CustomVoice`, `Base` or `VoiceDesign`; must match the loaded checkpoint. |
| `seed` | Integer from 0 through 2,147,483,647; absent means a new random seed. Each request owns an MLX key sequence. |
| `max_new_tokens` | Codec-frame cap, 1–2,048; default 256. A frame contains 16 codes and produces 1,920 samples (12.5 frames/second). |
| `temperature`, `top_k`, `top_p`, `repetition_penalty` | Semantic sampling; defaults 0.9, 50, 1.0, 1.05. Temperature 0 is greedy. |
| `subtalker_dosample`, `subtalker_temperature`, `subtalker_top_k`, `subtalker_top_p` | Code-predictor sampling; defaults true, 0.9, 50, 1.0. |
| `non_streaming_mode` | Prompt-side text layout; default true, independently of output streaming. Base cloning always uses full text context. |
| `speed` | Only 1 is supported. |
| `instructions` | Style text for checkpoints supporting it; rejected by 0.6B checkpoints and required for VoiceDesign. |
| `ref_audio`, `ref_text` | Base in-context cloning only, described below. |

Unknown fields and unsupported formats are rejected. HTTP bodies are capped at
4 MiB. Prompt plus generated frames must fit the checkpoint context. The sampler
suppresses codec control tokens except EOS; EOS is never passed to the vocoder.
Immediate EOS returns a synthesis error rather than an empty successful WAV.

## Base voice cloning and VoiceDesign

Serve the pinned Base directory with the same binary. Send `input`, `ref_text`
and `ref_audio`, with no preset `voice`. Reference audio must be an inline base64
WAV or `data:audio/wav;base64,...`, mono 24 kHz PCM16 or float32. Its length must
be between 1,024 samples and 30 seconds; samples must be finite and in [-1,1].
The server does not fetch reference URLs or open caller-supplied filesystem paths.

The native ECAPA-TDNN, mel frontend and speech tokenizer encode the reference.
The prompt combines its text, speaker embedding and codec frames with the target
text. Reference frames warm up the waveform decoder, and their audio is excluded
from the output. There is no cross-request reference cache. X-vector-only cloning
and automatic reference transcription are not implemented.

VoiceDesign dispatch and instruction prompts are implemented for compatible
unquantized checkpoints; the 1.7B variants are not yet pinned or qualified here.
The initial pinned qualification targets are 0.6B CustomVoice and Base.

## Parity and CI

The synthesis CLI accepts a JSON request file using the HTTP fields and writes a
WAV, codec frames, finish reason and text/control-token diagnostics:

```bash
build/qwen3-tts-runtime/bin/qwen3_tts_synthesize \
  --model-path MODEL_DIRECTORY --request request.json \
  --output speech.wav --codes-output codes.json
```

`--decode-codes frames.json` replaces `--request` to decode a fixed `[frames,16]`
integer-code array. `--decode-chunk-frames N` exercises the same codes through
request-owned streaming state. Include a 96-frame case when comparing the waveform decoder;
short samples do not exercise the 72-frame attention boundary.

Use the original Python MLX components at #1960 commit
`f4ed5942633e790f59e57b3f150bbffaa21e395d` as the independent migration reference,
with identical official weights and MLX version. Compare prompts, prefill and
post-feedback logits, code-predictor groups, reference encodings and fixed-code
waveforms before recording regression outputs. Preserve the decoder's window-72
mask even when checkpoint JSON omits it. Source provenance and suggested seeded
requests are in `ci/qwen3_tts_reference/`.

`qwen3_tts_golden.py` plugs into the existing `check_model_golden.py` checker.
Its populated golden must identify `kind="qwen3_tts"`, the pinned model, named
request cases and per-chip `devices` outputs. It checks all prompt diagnostics,
codec frames, finish reasons, sample counts and waveform SHA-256 exactly. An
unrecorded chip or missing case fails explicitly. Import actual runner outputs
with the existing `--import` operation after independent parity is established.
No empty golden or invented model output is checked in.

Mac CI provisions the pinned artifacts and includes `test_qwen3_tts_api.py`.
The explicit reference-qualification gate remains failing until
`ci/model_golden/qwen3-tts-0.6b-customvoice.json` contains qualified outputs;
therefore this implementation is not merge-ready. Add Base goldens and fixed-code
long/streaming parity as part of qualification. Current local verification is
pre-commit only; no build, unit tests, audio inference or benchmark has been run.
