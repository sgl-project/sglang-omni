<div align="center">
<img src="https://raw.githubusercontent.com/sgl-project/sglang-omni/main/docs/_static/image/sgl-omni-logo.svg" alt="logo" width="400"></img>

[![license](https://img.shields.io/github/license/sgl-project/sglang-omni)](https://github.com/sgl-project/sglang-omni/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/sgl-project/sglang-omni)](https://github.com/sgl-project/sglang-omni/issues)
[![open issues](https://img.shields.io/github/issues-raw/sgl-project/sglang-omni)](https://github.com/sgl-project/sglang-omni/issues)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sgl-project/sglang-omni)

</div>

--------------------------------------------------------------------------------

<p align="center">
<a href="https://sgl-project.github.io/sglang-omni/"><b>Documentation</b></a> |
<a href="https://slack.sglang.ai"><b>Join Slack</b></a>
</p>

## About

SGLang-Omni is a high-performance serving framework for omni and multimodal models, built on top of [SGLang](https://github.com/sgl-project/sglang). It is designed to orchestrate multi-stage pipelines with low latency and OpenAI-compatible APIs.

Modern omni models — such as speech-output LLMs and multimodal generation systems — decompose into heterogeneous stages with fundamentally different computational profiles: a compute-bound thinker, a memory-bound talker, a latency-sensitive codec. SGLang-Omni is built around a **computation-centric design**: each stage runs its own independent scheduler tuned to its bottleneck, communicates through a shared inbox/outbox abstraction, and transfers tensors via zero-copy shared memory. This prevents any single stage from degrading the others and allows new models to plug into the framework by declaring a pipeline topology rather than building an inference system from scratch.

Core features:

- **Multi-Stage Pipeline**: Flexible framework for orchestrating preprocessing, AR engine, codec, and vocoder stages across processes and GPUs.
- **Native SGLang Integration**: Leverages SGLang's RadixAttention, continuous batching, and CUDA Graph optimizations for the AR backbone.
- **OpenAI-Compatible Server**: Drop-in `/v1/audio/speech` and `/v1/chat/completions` endpoints with real-time streaming support.
- **Broad Model Support**: Supports a growing set of TTS and omni models including Higgs Audio, Fish Audio S2-Pro, Voxtral TTS, Qwen3 TTS, MOSS-TTS, Qwen3-Omni, Ming-Omni, and LLaDA2.0-Uni.

## Supported Models

| Model | Type | Notes |
|-------|------|-------|
| [boson-sglang/higgs-audio-v3-tts-4b-base](https://huggingface.co/boson-sglang/higgs-audio-v3-tts-4b-base) | TTS | Voice cloning, streaming, 100+ languages |
| [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) | TTS | Voice cloning, streaming |
| [mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) | TTS | Named voices, streaming, 9 languages |
| [Qwen/Qwen3-TTS-12Hz-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | TTS | Voice cloning, streaming, 10 languages, 0.6B / 1.7B |
| [OpenMOSS-Team/MOSS-TTS-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5) | TTS | Voice cloning, streaming, 31 languages |
| [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | Omni | Text, image, audio, video → text + audio |
| [inclusionAI/Ming-flash-omni-2.0](https://huggingface.co/inclusionAI/Ming-flash-omni-2.0) | Omni | Streaming TTS |
| [inclusionAI/LLaDA2.0-Uni](https://huggingface.co/inclusionAI/LLaDA2.0-Uni) | Multimodal | Text + image understanding and generation |

## Get Started

- [Installation](./docs/get_started/installation.md)
- [Cookbook](./docs/cookbook/)
- [Developer Reference](./docs/developer_reference/main.md)
