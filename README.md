<div align="center">
<img src="https://raw.githubusercontent.com/sgl-project/sglang-omni/main/docs/_static/image/sgl-omni-logo.png" alt="logo" width="400"></img>

[![license](https://img.shields.io/github/license/sgl-project/sglang-omni.svg)](https://github.com/sgl-project/sglang-omni/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang-omni)](https://github.com/sgl-project/sglang-omni/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang-omni)](https://github.com/sgl-project/sglang-omni/issues)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang-omni)

</div>

--------------------------------------------------------------------------------

<p align="center">
<a href="https://sgl-project.github.io/sglang-omni/"><b>Documentation</b></a> |
<a href="https://slack.sglang.ai"><b>Join Slack</b></a>
</p>

## News
- [2026/05] 🔥 Higgs TTS streaming vocoder with batched decode for higher throughput ([PR #574](https://github.com/sgl-project/sglang-omni/pull/574)).
- [2026/05] 🔥 Ming-Omni streaming TTS support ([PR #506](https://github.com/sgl-project/sglang-omni/pull/506)).
- [2026/05] Qwen3-Omni talker partial-prefix startup for faster first-token latency ([PR #475](https://github.com/sgl-project/sglang-omni/pull/475)).
- [2026/05] Higgs Audio V3 TTS cookbook ([PR #560](https://github.com/sgl-project/sglang-omni/pull/560)).

## About

SGLang-Omni is a high-performance serving framework for omni and multimodal models, built on top of [SGLang](https://github.com/sgl-project/sglang). It is designed to orchestrate multi-stage pipelines with low latency and OpenAI-compatible APIs.

Core features:

- **Multi-Stage Pipeline**: Flexible framework for orchestrating preprocessing, AR engine, codec, and vocoder stages across processes and GPUs.
- **Native SGLang Integration**: Leverages SGLang's RadixAttention, continuous batching, and CUDA Graph optimizations for the AR backbone.
- **OpenAI-Compatible Server**: Drop-in `/v1/audio/speech` and `/v1/chat/completions` endpoints with real-time streaming support.
- **Broad Model Support**: Supports a growing set of TTS and omni models including Higgs Audio, Fish Audio S2-Pro, Voxtral TTS, Qwen3-Omni, and Ming-Omni.

## Supported Models

| Model | Type | Notes |
|-------|------|-------|
| [boson-sglang/higgs-audio-v3-tts-4b-base](https://huggingface.co/boson-sglang/higgs-audio-v3-tts-4b-base) | TTS | Voice cloning |
| [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) | TTS | Voice cloning, streaming |
| [mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) | TTS | Speaker presets |
| [Qwen/Qwen3-TTS-12Hz-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | TTS | Voice cloning, 0.6B / 1.7B |
| [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | Omni | Speech + text I/O |
| [inclusionAI/Ming-flash-omni-2.0](https://huggingface.co/inclusionAI/Ming-flash-omni-2.0) | Omni | Streaming TTS |
| [inclusionAI/LLaDA2.0-Uni](https://huggingface.co/inclusionAI/LLaDA2.0-Uni) | Omni | Text + image I/O |

## Get Started

- [Installation](./docs/get_started/installation.md)
- [Developer Reference](./docs/developer_reference/main.md)
- [Benchmarks](./benchmarks/README.md)
- [Examples](./examples/README.md)
