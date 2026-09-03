<div align="center">

<img src="https://raw.githubusercontent.com/sgl-project/sglang-omni/main/docs/_static/image/sgl-omni-logo.svg" alt="SGLang-Omni logo" width="400"></img>

### High-performance serving for speech and omni models

<p>
<a href="https://pypi.org/project/sglang-omni/"><img src="https://img.shields.io/pypi/v/sglang-omni?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI" alt="PyPI"></a>
<a href="https://github.com/sgl-project/sglang-omni/stargazers"><img src="https://img.shields.io/github/stars/sgl-project/sglang-omni?style=for-the-badge&logo=github&label=stars" alt="GitHub stars"></a>
<a href="https://github.com/sgl-project/sglang-omni/blob/main/LICENSE"><img src="https://img.shields.io/github/license/sgl-project/sglang-omni?style=for-the-badge" alt="license"></a>
<a href="https://github.com/sgl-project/sglang-omni/issues"><img src="https://img.shields.io/github/issues-closed-raw/sgl-project/sglang-omni?style=for-the-badge&label=closed%20issues" alt="closed issues"></a>
<a href="https://github.com/sgl-project/sglang-omni/issues"><img src="https://img.shields.io/github/issues-raw/sgl-project/sglang-omni?style=for-the-badge&label=open%20issues" alt="open issues"></a>
<a href="https://deepwiki.com/sgl-project/sglang-omni"><img src="https://img.shields.io/badge/Ask-DeepWiki-087fca?style=for-the-badge" alt="Ask DeepWiki"></a>
</p>

<p>
<a href="https://sgl-project.github.io/sglang-omni/"><b>Documentation</b></a> |
<a href="#getting-started"><b>Quick Start</b></a> |
<a href="./docs/supported_models.md"><b>Models</b></a> |
<a href="https://lmsys.org/blog/"><b>Blog</b></a> |
<a href="https://slack.sglang.io"><b>Join Slack</b></a>
</p>

<p>
⭐ <b><a href="https://github.com/sgl-project/sglang-omni/stargazers">Star SGLang-Omni</a> to help more builders discover open infrastructure for multimodal and speech serving!</b>
</p>

</div>

--------------------------------------------------------------------------------

## News

- [2026/09] 🚀 SGLang-Omni **v0.1.4** is on [PyPI](https://pypi.org/project/sglang-omni/). Install with `uv pip install --prerelease=allow "sglang-omni==0.1.4"`. \[[Installation](https://sgl-project.github.io/sglang-omni/get_started/installation.html)\]
- [2026/08] 🎵 Day-0 support for [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3): lyrics + caption → 32 kHz stereo song on `/v1/audio/speech`. \[[Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/minimax_music3.html)\]
- [2026/08] 🚀 TTS architecture refactor: shared pipeline state, engine construction, reference encoding, capability metadata, and vocoder scheduling. \[[Roadmap](https://github.com/sgl-project/sglang-omni/issues/985)\] \[[Blog](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-omni/tts-refactor.md)\]

<details>
<summary>More news</summary>

- [2026/06] 🔥 MOSS-TTS Local Transformer v1.5 on SGLang-Omni with native-streaming 48 kHz speech. \[[Blog](https://lmsys.org/blog/2026-06-17-moss-tts-local-v15/)\] \[[Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts_local.html)\]
- [2026/06] 🔥 Higgs Audio v3 TTS for real-time, controllable speech. \[[Blog](https://lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/)\] \[[Cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)\]

</details>

## About

SGLang-Omni is a serving framework for speech, audio, and multimodal generative
models, built on [SGLang](https://github.com/sgl-project/sglang). It serves TTS,
ASR, speech translation, diarization, omni-modal chat, and music generation
through one multi-stage runtime.

<a id="getting-started"></a>

## Quick Start

Install SGLang-Omni in a Python 3.12 CUDA environment:

```bash
pip install --pre sglang-omni
```

Start Qwen3-ASR on one GPU:

```bash
sgl-omni serve --model-path Qwen/Qwen3-ASR-1.7B --port 8000
```

Send an audio file to the OpenAI-compatible transcription API:

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@audio.wav \
  -F response_format=json
```

For the CUDA image, reproducible version pinning, and source installation, see
[Installation](./docs/get_started/installation.md).

## Why SGLang-Omni

- **Multi-stage native:** Run preprocessing, encoders, autoregressive engines,
  talkers, codecs, vocoders, and aggregators as one request lifecycle.
- **SGLang execution:** Use continuous batching for autoregressive stages and
  workload-specific schedulers for other stage types.
- **Serving APIs:** Use OpenAI-compatible speech, transcription, translation,
  and chat endpoints, plus HTTP, SSE, and WebSocket streaming transports.
- **Flexible deployment:** Colocate or distribute stages, use tensor
  parallelism within supported stages, and scale with process replicas.

## Supported Models

Status is a maintained user contract, not a claim that every model/backend
combination runs in CI. See the canonical
[model support matrix](./docs/supported_models.md#model-support-matrix) for the
definition and endpoint details.

| Model | Task | Streaming | Status | Guide |
|---|---|---|---|---|
| Higgs Audio v3 | TTS | Audio output; see guide | Supported | [Cookbook](./docs/cookbook/higgs_tts.md) |
| Fish Audio S2-Pro | TTS | Audio output; see guide | Supported | [Cookbook](./docs/cookbook/fishaudio_s2_pro.md) |
| Voxtral-4B-TTS | TTS | Audio output; see guide | Supported | [Cookbook](./docs/cookbook/voxtral_tts.md) |
| Qwen3-TTS | TTS | HTTP PCM or WebSocket; Base checkpoints only | Supported | [Cookbook](./docs/cookbook/qwen3_tts.md) |
| Fun-CosyVoice3 | TTS | No | Experimental | [Cookbook](./docs/cookbook/fun_cosyvoice3.md) |
| MOSS-TTS v1.5 | TTS | Audio output; see guide | Supported | [Cookbook](./docs/cookbook/moss_tts.md) |
| MOSS-TTS Local v1.5 | TTS | Audio output; see guide | Supported | [Cookbook](./docs/cookbook/moss_tts_local.md) |
| Ming-Omni-TTS | TTS | No | Supported | [Cookbook](./docs/cookbook/ming_tts.md) |
| dots.tts | TTS | Audio output; see guide | Supported | [Cookbook](./docs/cookbook/dots_tts.md) |
| ZONOS2 | TTS | Audio output; see guide | Supported | [Cookbook](./docs/cookbook/zonos2.md) |
| Audar-TTS-V1 Turbo | TTS | No | Supported | [Cookbook](./docs/cookbook/audar_tts.md) |
| MiniMax Music 3 | Music | No | Supported | [Cookbook](./docs/cookbook/minimax_music3.md) |
| Qwen3-ASR | ASR | SSE transcript output | Supported | [Cookbook](./docs/cookbook/qwen3_asr.md) |
| Fun-ASR-Nano | ASR | SSE transcript output | Supported | [Cookbook](./docs/cookbook/fun_asr.md) |
| ARK-ASR-3B | ASR | SSE transcript output | Supported | [Cookbook](./docs/cookbook/arkasr.md) |
| MOSS-Transcribe-Diarize | ASR + diarization | SSE transcript output | Supported | [Cookbook](./docs/cookbook/moss_transcribe_diarize.md) |
| Whisper | ASR + translation | SSE transcript output | Experimental | [Cookbook](./docs/cookbook/whisper_asr.md) |
| Qwen3-Omni | Omni | Chat SSE + realtime WebSocket | Supported | [Cookbook](./docs/cookbook/qwen3_omni.md) |
| Ming-Omni | Omni | Model-dependent; see guide | Supported | [Cookbook](./docs/cookbook/ming_omni.md) |
| LLaDA2.0-Uni | Multimodal generation | No | Experimental | [Cookbook](./docs/cookbook/llada2_uni.md) |

## Runtime and Optimization Support

Availability is model-, stage-, and configuration-dependent. Follow the linked
guide before enabling a feature in production.

| Capability | Availability | Scope |
|---|---|---|
| Multi-stage pipeline serving | Core runtime | Encoder, thinker, talker, codec, vocoder, and other typed stages |
| Continuous batching | Available | Autoregressive stages backed by SGLang |
| Stage-specialized scheduling | Available | Scheduler selected per workload and stage |
| Streaming output | Model-dependent | HTTP body streaming, SSE, and WebSocket |
| CUDA Graph execution | Model/stage-dependent | Decode and selected non-AR paths |
| Asynchronous execution | Model/stage-dependent | Pipeline communication and selected decode paths |
| [Admission control](./docs/user_guide/advanced_features/admission_control.md) | Configurable | Pipeline and generation-stage limits |
| [Deterministic inference](./docs/user_guide/advanced_features/deterministic_inference.md) | Opt-in, model-specific | Qwen3-TTS Base deterministic mode; narrower seeded contracts elsewhere |
| [Stage placement](./docs/user_guide/deployment/stage_placement.md) | Available | Colocated, disaggregated, and tensor-parallel layouts |
| Tensor parallelism | Model-specific | Supported autoregressive stages |
| [Same-GPU data parallelism with CUDA MPS](./docs/basic_usage/mps_dp.md) | Validated configurations | Selected TTS and ASR pipelines |
| FP8 and AutoRound checkpoints | Qualified Qwen3-Omni paths | Native FP8 and AutoRound INT4 thinker configurations |
| Long-audio chunking | Model-specific | Supported ASR pipelines; limits remain model-owned |

## Hardware and Accelerator Support

Implementation, expected model scope, and validation are separate claims. A
backend existing in the repository does not establish model-level validation.
See the canonical
[accelerator matrix](./docs/supported_models.md#accelerator-support-matrix) for
definitions and evidence.

| Accelerator | Backend | Expected model scope | Validation |
|---|---|---|---|
| NVIDIA CUDA | Primary implementation | Models in the support matrix unless their guide states otherwise | CI tested on H100; eight models covered, see the [CI coverage table](./docs/developer_reference/model_qualification.md#ci-coverage) |
| Apple Silicon | MLX and Torch MPS implementations | Qwen3-ASR on macOS arm64 | Experimental |
| Intel XPU | Implemented | Qwen3-ASR and Qwen3-TTS on one XPU; Qwen3-Omni text-only with multi-XPU tensor parallelism | Manually validated |
| AMD ROCm | Implemented | Initial Qwen3-Omni, Qwen3-ASR, and Qwen3-TTS paths | Experimental |
| Ascend NPU | Implemented | Documented install path only; no per-model validation record | Not recorded |
| MUSA | Implemented | No user-facing model/backend set recorded | Not recorded |
| CPU | Host-stage support only | No documented end-to-end model-serving pipeline | Unsupported |

For installation instructions, see the
[NVIDIA CUDA guide](./docs/get_started/installation.md), the
[Apple Silicon installer](./docs/get_started/installation.md#macos-apple-silicon),
the [Intel XPU guide](./docs/get_started/installation_xpu.md), or the
[Ascend NPU guide](./docs/get_started/installation_npu.md). The canonical matrix
links the current implementation or evidence for the remaining backends.

## API and Serving Capabilities

| Use case | Endpoint | Transport / output |
|---|---|---|
| Speech and music generation | `POST /v1/audio/speech` | Encoded audio or incremental raw PCM |
| Stateful speech generation | `/v1/audio/speech/stream` | WebSocket control events and binary audio frames |
| Transcription | `POST /v1/audio/transcriptions` | Multipart upload; JSON, text, subtitles, or transcript SSE |
| Speech translation | `POST /v1/audio/translations` | Capability-gated multipart upload; response or transcript SSE |
| Multimodal chat | `POST /v1/chat/completions` | JSON response or chat SSE |
| Realtime conversation | `/v1/realtime` | Bidirectional WebSocket text, audio, VAD, and lifecycle events |

Transport and model coverage differ. See
[Streaming](./docs/user_guide/advanced_features/streaming.md),
[Speech API](./docs/user_guide/serving/speech_api.md), and
[Transcription API](./docs/user_guide/serving/transcription_api.md).

## Performance

SGLang-Omni optimizes the complete serving pipeline, not only its
autoregressive model stages:

- continuous and stage-local batching;
- CUDA Graph and asynchronous execution on qualified paths;
- streaming-first response paths;
- colocated and distributed multi-GPU placement;
- model-specific kernels and runtime integrations.

Results remain specific to the model, checkpoint, hardware, topology, workload,
and traffic shape. Use the [reproducible benchmark entry points](./benchmarks/README.md),
[benchmark methodology](./docs/benchmarks/methodology.md), and model cookbooks
for commands and evidence.

## Ecosystem

### Model Ecosystem

Qwen3-TTS · Higgs Audio · Fish Audio · Voxtral · MOSS-TTS · MOSS-TTS Local ·
Fun-CosyVoice · Ming-Omni-TTS · dots.tts · ZONOS · Audar-TTS · MiniMax Music ·
Qwen3-ASR · Fun-ASR · ARK-ASR · MOSS-Transcribe-Diarize · Whisper ·
Qwen3-Omni · Ming-Omni · LLaDA

### Accelerator Ecosystem

NVIDIA CUDA · Apple Silicon · Intel XPU · AMD ROCm · Ascend NPU · MUSA · CPU host stages

### Adoption and Sponsorship

Interested in deploying or sponsoring SGLang-Omni? Contact Chenyang Zhao at
[zhaochenyang@lmsys.org](mailto:zhaochenyang@lmsys.org). This README lists an
organization as an adopter or sponsor only when public evidence is available.

## Documentation

[Installation](./docs/get_started/installation.md) ·
[Supported models](./docs/supported_models.md) ·
[TTS](./docs/basic_usage/tts.md) ·
[Omni](./docs/basic_usage/qwen3_omni.md) ·
[Speech API](./docs/user_guide/serving/speech_api.md) ·
[Transcription API](./docs/user_guide/serving/transcription_api.md) ·
[Streaming](./docs/user_guide/advanced_features/streaming.md) ·
[Deployment](./docs/user_guide/deployment/stage_placement.md) ·
[Benchmarks](./docs/benchmarks/methodology.md) ·
[Developer guide](./docs/developer_reference/overview.md)

## Community

SGLang-Omni welcomes contributors working on inference systems, kernels,
scheduling, inter-stage communication, model integration, benchmarking, and
deployment. Join the [SGLang Slack](https://slack.sglang.io), read the
[project blog](https://lmsys.org/blog/), or open an
[issue](https://github.com/sgl-project/sglang-omni/issues).

## Acknowledgments

SGLang-Omni builds on the SGLang ecosystem and on open model work from the TTS,
speech, and omni-model communities. We thank the model teams, systems
contributors, and partner organizations helping make open multimodal serving
faster, more reliable, and easier to extend.
