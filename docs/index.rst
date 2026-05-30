SGLang-Omni
=======================

SGLang-Omni is a high-performance serving framework for omni and multimodal models, built on top of `SGLang <https://github.com/sgl-project/sglang>`_. It is designed to orchestrate multi-stage pipelines with low latency and OpenAI-compatible APIs.

Core features:

- **Multi-Stage Pipeline**: Flexible framework for orchestrating preprocessing, AR engine, codec, and vocoder stages across processes and GPUs.
- **Native SGLang Integration**: Leverages SGLang's RadixAttention, continuous batching, and CUDA Graph optimizations for the AR backbone.
- **OpenAI-Compatible Server**: Drop-in ``/v1/audio/speech`` and ``/v1/chat/completions`` endpoints with real-time streaming support.
- **Broad Model Support**: Supports a growing set of TTS and omni models including Higgs Audio, Fish Audio S2-Pro, Voxtral TTS, Qwen3 TTS, Qwen3-Omni, Ming-Omni, and LLaDA2.0-Uni.


.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/installation.md


.. toctree::
   :maxdepth: 1
   :caption: Cookbook

   cookbook/higgs_tts.md
   cookbook/voxtral_tts.md
   cookbook/qwen3_tts.md
   cookbook/qwen3_omni.md
   cookbook/llada2_uni.md

.. toctree::
   :maxdepth: 1
   :caption: Basic Usage

   basic_usage/qwen3_omni.md
   basic_usage/tts.md
   basic_usage/omni_router.md


.. toctree::
   :maxdepth: 1
   :caption: Benchmarks

   benchmarks/relay.md


.. toctree::
   :maxdepth: 1
   :caption: Developer Reference

   developer_reference/main.md
   developer_reference/apiserver_design.md
   developer_reference/pipeline.md
   developer_reference/config.md
   developer_reference/communication.md
   developer_reference/profiler.md
