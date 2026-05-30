SGLang-Omni
=======================

SGLang-Omni is a high-performance serving framework for omni and multimodal models, built on top of `SGLang <https://github.com/sgl-project/sglang>`_. It is designed to orchestrate multi-stage pipelines with low latency and OpenAI-compatible APIs.

News
----

- [2026/05] 🔥 Qwen3-Omni talker partial-start enabled by default (`PR #617 <https://github.com/sgl-project/sglang-omni/pull/617>`_).
- [2026/05] 🔥 Streaming TTS schedulers with framework-level support (`PR #614 <https://github.com/sgl-project/sglang-omni/pull/614>`_).
- [2026/05] 🔥 Higgs TTS streaming vocoder with batched decode for higher throughput (`PR #574 <https://github.com/sgl-project/sglang-omni/pull/574>`_).
- [2026/05] 🔥 Ming-Omni streaming TTS support (`PR #506 <https://github.com/sgl-project/sglang-omni/pull/506>`_).
- [2026/05] torch.compile + CUDA Graph for Qwen3-TTS and Voxtral-TTS AR backbones (`PR #527 <https://github.com/sgl-project/sglang-omni/pull/527>`_).
- [2026/05] Higgs TTS async decode (one-step lookahead) for lower latency (`PR #590 <https://github.com/sgl-project/sglang-omni/pull/590>`_).
- [2026/05] Qwen3-Omni talker partial-prefix startup for faster first-token latency (`PR #475 <https://github.com/sgl-project/sglang-omni/pull/475>`_).
- [2026/05] New cookbooks: LLaDA2.0-Uni (`PR #598 <https://github.com/sgl-project/sglang-omni/pull/598>`_), Voxtral TTS + Qwen3 TTS (`PR #585 <https://github.com/sgl-project/sglang-omni/pull/585>`_), Higgs Audio V3 TTS (`PR #560 <https://github.com/sgl-project/sglang-omni/pull/560>`_).

About
-----

Core features:

- **Multi-Stage Pipeline**: Flexible framework for orchestrating preprocessing, AR engine, codec, and vocoder stages across processes and GPUs.
- **Native SGLang Integration**: Leverages SGLang's RadixAttention, continuous batching, and CUDA Graph optimizations for the AR backbone.
- **OpenAI-Compatible Server**: Drop-in ``/v1/audio/speech`` and ``/v1/chat/completions`` endpoints with real-time streaming support.
- **Broad Model Support**: Supports a growing set of TTS and omni models including Higgs Audio, Fish Audio S2-Pro, Voxtral TTS, Qwen3 TTS, Qwen3-Omni, Ming-Omni, and LLaDA2.0-Uni.

Supported Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 45 15 40

   * - Model
     - Type
     - Notes
   * - `boson-sglang/higgs-audio-v3-tts-4b-base <https://huggingface.co/boson-sglang/higgs-audio-v3-tts-4b-base>`_
     - TTS
     - Voice cloning, streaming, 100+ languages
   * - `fishaudio/s2-pro <https://huggingface.co/fishaudio/s2-pro>`_
     - TTS
     - Voice cloning, streaming
   * - `mistralai/Voxtral-4B-TTS-2603 <https://huggingface.co/mistralai/Voxtral-4B-TTS-2603>`_
     - TTS
     - Named voices, streaming, 9 languages
   * - `Qwen/Qwen3-TTS-12Hz-Base <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base>`_
     - TTS
     - Voice cloning, streaming, 10 languages, 0.6B / 1.7B
   * - `Qwen/Qwen3-Omni-30B-A3B-Instruct <https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct>`_
     - Omni
     - Text, image, audio, video → text + audio
   * - `inclusionAI/Ming-flash-omni-2.0 <https://huggingface.co/inclusionAI/Ming-flash-omni-2.0>`_
     - Omni
     - Streaming TTS
   * - `inclusionAI/LLaDA2.0-Uni <https://huggingface.co/inclusionAI/LLaDA2.0-Uni>`_
     - Multimodal
     - Text + image understanding and generation


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
