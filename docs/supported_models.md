# Supported models

This page answers two user-facing questions: which model families SGLang-Omni
supports, and what level of evidence exists for each accelerator backend. The
model matrix is family-level. Exact checkpoint IDs, revisions, and validated
configurations live in each cookbook and in
[model qualification evidence](./developer_reference/model_qualification.md).

## Model support matrix

Model status describes the maintained user contract, not CI coverage:

- **Supported**: the documented configuration is maintained and expected to
  work.
- **Experimental**: an implementation exists, but its documented behavior or
  supported scope may still change.

| Model | Task | Endpoint | Streaming | Status | Cookbook |
|---|---|---|---|---|---|
| Higgs Audio v3 | TTS | `/v1/audio/speech` | Audio output; see cookbook | Supported | [Higgs TTS](./cookbook/higgs_tts.md) |
| Fish Audio S2-Pro | TTS | `/v1/audio/speech` | Audio output; see cookbook | Supported | [Fish Audio S2-Pro](./cookbook/fishaudio_s2_pro.md) |
| Voxtral-4B-TTS | TTS | `/v1/audio/speech` | Audio output; see cookbook | Supported | [Voxtral TTS](./cookbook/voxtral_tts.md) |
| Qwen3-TTS | TTS | `/v1/audio/speech` | HTTP PCM or WebSocket output; Base checkpoints only | Supported | [Qwen3-TTS](./cookbook/qwen3_tts.md) |
| Fun-CosyVoice3 | TTS | `/v1/audio/speech` | No | Experimental | [Fun-CosyVoice3](./cookbook/fun_cosyvoice3.md) |
| MOSS-TTS v1.5 | TTS | `/v1/audio/speech` | Audio output; see cookbook | Supported | [MOSS-TTS](./cookbook/moss_tts.md) |
| MOSS-TTS Local v1.5 | TTS | `/v1/audio/speech` | Audio output; see cookbook | Supported | [MOSS-TTS Local](./cookbook/moss_tts_local.md) |
| Ming-Omni-TTS | TTS | `/v1/audio/speech` | No | Supported | [Ming-Omni-TTS](./cookbook/ming_tts.md) |
| dots.tts | TTS | `/v1/audio/speech` | Audio output; see cookbook | Supported | [dots.tts](./cookbook/dots_tts.md) |
| ZONOS2 | TTS | `/v1/audio/speech` | Audio output; see cookbook | Supported | [ZONOS2](./cookbook/zonos2.md) |
| Audar-TTS-V1 Turbo | TTS | `/v1/audio/speech` | No | Supported | [Audar-TTS](./cookbook/audar_tts.md) |
| MiniMax Music 3 | Music | `/v1/audio/speech` | No | Supported | [MiniMax Music 3](./cookbook/minimax_music3.md) |
| Qwen3-ASR | ASR | `/v1/audio/transcriptions` | SSE transcript output | Supported | [Qwen3-ASR](./cookbook/qwen3_asr.md) |
| Fun-ASR-Nano | ASR | `/v1/audio/transcriptions` | SSE transcript output | Supported | [Fun-ASR-Nano](./cookbook/fun_asr.md) |
| ARK-ASR-3B | ASR | `/v1/audio/transcriptions` | SSE transcript output | Supported | [ARK-ASR-3B](./cookbook/arkasr.md) |
| MOSS-Transcribe-Diarize | ASR + diarization | `/v1/audio/transcriptions` | SSE transcript output | Supported | [MOSS-Transcribe-Diarize](./cookbook/moss_transcribe_diarize.md) |
| Whisper | ASR / translation | `/v1/audio/transcriptions`, `/v1/audio/translations` | SSE transcript output | Experimental | [Whisper ASR](./cookbook/whisper_asr.md) |
| Qwen3-Omni | Omni | `/v1/chat/completions`, `/v1/realtime` | Chat SSE and realtime WebSocket | Supported | [Qwen3-Omni](./cookbook/qwen3_omni.md) |
| Ming-Omni | Omni | `/v1/chat/completions` | Model-dependent; see cookbook | Supported | [Ming-Omni](./cookbook/ming_omni.md) |
| LLaDA2.0-Uni | Multimodal generation | `/v1/chat/completions` | No | Experimental | [LLaDA2.0-Uni](./cookbook/llada2_uni.md) |

## Accelerator support matrix

Backend implementation, expected model scope, and runtime validation are
separate claims. The evidence status means:

- **CI tested**: recurring model-level CI runs on the named accelerator.
- **Manually validated**: current documentation records an end-to-end run, but
  there is no recurring model gate.
- **Experimental**: backend and model-specific implementation exists, but
  current main does not link recurring CI or a durable manual validation
  record.
- **Not recorded**: backend implementation exists, but no user-facing
  model/backend support set is recorded.
- **Unsupported**: end-to-end model serving is not supported on this backend.

| Accelerator | Backend implementation | Expected model scope | Validation | Documentation / evidence |
|---|---|---|---|---|
| NVIDIA CUDA | Yes; primary backend | Models in the support matrix unless their cookbook states otherwise | **CI tested** on H100 for eight models; the TTS and ASR sets rotate per run | [Installation](./get_started/installation.md), [Qualification evidence](./developer_reference/model_qualification.md#ci-coverage) |
| Apple Silicon | Yes; MLX and Torch MPS paths | Qwen3-ASR on macOS arm64 | **Experimental** | {ref}`Apple Silicon installation <macos-apple-silicon>`, [Qwen3-ASR cookbook](./cookbook/qwen3_asr.md#apple-silicon), [Apple platform](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/platforms/apple.py) |
| Intel XPU | Yes | Qwen3-ASR and Qwen3-TTS on one XPU; Qwen3-Omni text-only with multi-XPU tensor parallelism | **Manually validated** | [Intel XPU installation and serving guide](./get_started/installation_xpu.md) |
| AMD ROCm | Yes; standalone platform and image | Initial Qwen3-Omni, Qwen3-ASR, and Qwen3-TTS paths | **Experimental** | [ROCm platform](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/platforms/rocm.py), [ROCm image](https://github.com/sgl-project/sglang-omni/blob/main/docker/rocm.Dockerfile) |
| Ascend NPU | Yes | Documented install path only; no per-model validation record | **Not recorded** | [Ascend NPU installation guide](./get_started/installation_npu.md), [NPU platform](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/platforms/npu.py) |
| MUSA | Yes | No user-facing model/backend support set is recorded | **Not recorded** | [MUSA platform](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/platforms/musa.py) |
| CPU | Host-stage device support only | No end-to-end model-serving pipeline is documented | **Unsupported** | [CPU platform](https://github.com/sgl-project/sglang-omni/blob/main/sglang_omni/platforms/cpu.py) |

A checked-in backend, image, or model-specific code path shows implementation
scope; it does not by itself establish runtime validation. Add or upgrade a
validation claim only when current main contains a recurring model gate or an
explicit end-to-end validation record for that model/backend combination.
