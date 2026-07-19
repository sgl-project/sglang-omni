SGLang-Omni
=======================

SGLang-Omni is a high-performance serving framework for omni and multimodal models, built on top of `SGLang <https://github.com/sgl-project/sglang>`_. It is designed to orchestrate multi-stage pipelines with low latency and OpenAI-compatible APIs.

Modern omni models — such as speech-output LLMs and multimodal generation systems — decompose into heterogeneous stages with fundamentally different computational profiles: a compute-bound thinker, a memory-bound talker, a latency-sensitive codec. SGLang-Omni is built around a **computation-centric design**: each stage runs its own independent scheduler tuned to its bottleneck, communicates through a shared inbox/outbox abstraction, and transfers tensors via zero-copy shared memory. This prevents any single stage from degrading the others and allows new models to plug into the framework by declaring a pipeline topology rather than building an inference system from scratch.

About
-----

Core features:

- **Multi-Stage Pipeline**: Flexible framework for orchestrating preprocessing, AR engine, codec, and vocoder stages across processes and GPUs.
- **Native SGLang Integration**: Leverages SGLang's RadixAttention, continuous batching, and CUDA Graph optimizations for the AR backbone.
- **OpenAI-Compatible Server**: Drop-in ``/v1/audio/speech``, ``/v1/audio/transcriptions``, and ``/v1/chat/completions`` endpoints with real-time streaming support.
- **Broad Model Support**: Supports a growing set of TTS, ASR, and omni models including Higgs Audio, Fish Audio S2-Pro, Voxtral TTS, Qwen3 TTS, MOSS-TTS, Qwen3-ASR, Whisper ASR, Qwen3-Omni, Ming-Omni, and LLaDA2.0-Uni.

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
   * - `OpenMOSS-Team/MOSS-TTS-v1.5 <https://huggingface.co/OpenMOSS-Team/MOSS-TTS-v1.5>`_
     - TTS
     - Voice cloning, streaming, 31 languages
   * - `Qwen/Qwen3-ASR-1.7B <https://huggingface.co/Qwen/Qwen3-ASR-1.7B>`_
     - ASR
     - Audio transcription through ``/v1/audio/transcriptions``
   * - `openai/whisper-large-v3 <https://huggingface.co/openai/whisper-large-v3>`_
     - ASR
     - Experimental Whisper transcription route; response schema is served, correctness is not yet validated
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
   cookbook/fishaudio_s2_pro.md
   cookbook/qwen3_tts.md
   cookbook/moss_tts.md
   cookbook/moss_tts_local.md
   cookbook/qwen3_asr.md
   cookbook/fun_asr.md
   cookbook/moss_transcribe_diarize.md
   cookbook/whisper_asr.md
   cookbook/qwen3_omni.md
   cookbook/ming_omni.md
   cookbook/llada2_uni.md

.. toctree::
   :maxdepth: 1
   :caption: General Usage

   basic_usage/qwen3_omni.md
   basic_usage/tts.md
   basic_usage/omni_router.md
   basic_usage/mps_dp.md


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
   developer_reference/reference_encode_service.md
   developer_reference/profiler.md
   developer_reference/rl_admin_control.md
