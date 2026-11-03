.. _sglang-omni:

Welcome to SGLang-Omni
======================

Serve speech and multimodal models with SGLang-Omni.

- **Speech & Multimodal Models.** Transcribe audio, generate speech, and serve models that accept audio, images, and video through OpenAI-compatible APIs.
- **Streaming & Realtime.** Stream transcripts and speech as they are generated. Use realtime sessions for live audio input and interactive voice applications.
- **Multi-stage Runtime.** Run preprocessing, inference, and audio generation as stages with independent scheduling. The runtime is built on `SGLang <https://github.com/sgl-project/sglang>`_.
- **Hardware Support.** Choose the installation guide for your hardware: :doc:`CUDA <get_started/installation>`, :doc:`Apple Silicon <get_started/installation_apple_silicon>`, :doc:`Ascend NPU <get_started/installation_npu>`, :doc:`Intel XPU <get_started/installation_xpu>`, :doc:`Intel CPU <get_started/installation_cpu>`, or :doc:`MUSA <get_started/installation_musa>`. Model support varies by platform.

Get Started
-----------

- :doc:`Quickstart <get_started/apiserver_quickstart>` — Start a model server and transcribe one audio clip.
- :doc:`Installation <get_started/installation>` — Choose an installation method for your hardware.
- :doc:`Examples <get_started/examples>` — Stream a transcript, generate speech, or send an audio chat request.

About
-----

Modern omni models — such as speech-output LLMs and multimodal generation systems — decompose into heterogeneous stages with fundamentally different computational profiles: a compute-bound thinker, a memory-bound talker, a latency-sensitive codec. SGLang-Omni is built around a **computation-centric design**: each stage runs its own independent scheduler tuned to its bottleneck, communicates through a shared inbox/outbox abstraction, and transfers tensors via zero-copy shared memory. This prevents any single stage from degrading the others and allows new models to plug into the framework by declaring a pipeline topology rather than building an inference system from scratch.


Core features:

- **Multi-Stage Pipeline**: Flexible framework for orchestrating preprocessing, AR engine, codec, and vocoder stages across processes and GPUs.
- **Native SGLang Integration**: Leverages SGLang's RadixAttention, continuous batching, and CUDA Graph optimizations for the AR backbone.
- **OpenAI-Compatible Server**: Drop-in ``/v1/audio/speech``, ``/v1/audio/transcriptions``, ``/v1/audio/translations``, and ``/v1/chat/completions`` endpoints with real-time streaming support.
- **Broad Model Support**: TTS (Higgs, Fish S2-Pro, Voxtral, Qwen3-TTS, MOSS-TTS / Local, Ming-Omni-TTS, Ming-3.0-TTS, dots.tts, ZONOS2), Music (MiniMax Music 3), ASR (Qwen3-ASR, Fun-ASR, ARK-ASR, Nemotron 3.5 ASR, Whisper, MOSS-Transcribe-Diarize, Ming-3.0-ASR), Omni (Qwen3-Omni, MiniCPM-o, Ming-Omni, Ming-3.0-audio), and LLaDA2.0-Uni.

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
     - Delay-pattern MOSS-TTS; voice cloning, streaming, 31 languages
   * - `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 <https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5>`_
     - TTS
     - Local-transformer MOSS-TTS; 48 kHz stereo, streaming
   * - `inclusionAI/Ming-omni-tts-16.8B-A3B <https://huggingface.co/inclusionAI/Ming-omni-tts-16.8B-A3B>`_
     - TTS
     - Text-to-speech and zero-shot voice cloning
   * - `inclusionAI/Ming-3.0-tts-tiny <https://huggingface.co/inclusionAI/Ming-3.0-tts-tiny>`_
     - TTS
     - 44.1 kHz voice cloning from one reference recording
   * - `inclusionAI/Ming-3.0-tts-0.5B <https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B>`_
     - TTS
     - Dense 0.5B backbone, 44.1 kHz voice cloning from one reference recording
   * - `dots-studio/dots.tts-mf <https://huggingface.co/dots-studio/dots.tts-mf>`_
     - TTS
     - 48 kHz continuous-latent TTS; also ``dots.tts-soar`` / ``dots.tts-base``
   * - `Zyphra/zonos2 <https://huggingface.co/Zyphra/zonos2>`_
     - TTS
     - MoE TTS, 9 DAC codebooks, voice cloning
   * - `MiniMaxAI/MiniMax-Music3 <https://huggingface.co/MiniMaxAI/MiniMax-Music3>`_
     - Music
     - Text-to-music; lyrics + caption → 32 kHz stereo song
   * - `Qwen/Qwen3-ASR-1.7B <https://huggingface.co/Qwen/Qwen3-ASR-1.7B>`_
     - ASR
     - Multilingual transcription with 30 language hints
   * - `inclusionAI/Ming-3.0-asr-tiny <https://huggingface.co/inclusionAI/Ming-3.0-asr-tiny>`_
     - ASR
     - Chinese and English transcription
   * - `FunAudioLLM/Fun-ASR-Nano-2512-hf <https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-hf>`_
     - ASR
     - Multilingual Fun-ASR-Nano
   * - `AutoArk-AI/ARK-ASR-3B <https://huggingface.co/AutoArk-AI/ARK-ASR-3B>`_
     - ASR
     - Multilingual ARK-ASR
   * - `nvidia/nemotron-3.5-asr-streaming-0.6b <https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b>`_
     - ASR
     - Multilingual RNN-T transcription; internal native PCM streaming runtime (see `cookbook <cookbook/nemotron3_5_asr.html>`_)
   * - `nvidia/Nemotron-3.5-Transcribe-30B-A4B <https://huggingface.co/nvidia/Nemotron-3.5-Transcribe-30B-A4B>`_
     - ASR
     - Whole-file transcription with native speech and speaker-fusion encoders (see `cookbook <cookbook/nemotron_transcribe.html>`_)
   * - `OpenMOSS-Team/MOSS-Transcribe-Diarize <https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize>`_
     - ASR
     - Multi-speaker transcription + diarization + timestamps
   * - `nvidia/Nemotron-3-Diarization <https://huggingface.co/nvidia/Nemotron-3-Diarization>`_
     - Diarization
     - Up to eight overlapping speakers for uploads and live audio (see `cookbook <cookbook/nemotron_diarization.html>`_)
   * - `openai/whisper-large-v3 <https://huggingface.co/openai/whisper-large-v3>`_
     - ASR
     - Experimental transcription and speech-to-English translation routes; see the `audio translation support matrix <basic_usage/audio_translations.html>`_
   * - `Qwen/Qwen3-Omni-30B-A3B-Instruct <https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct>`_
     - Omni
     - Text, image, audio, video → text + audio
   * - `openbmb/MiniCPM-o-4_5 <https://huggingface.co/openbmb/MiniCPM-o-4_5>`_
     - Omni
     - Text, image, audio, video → text + audio
   * - `inclusionAI/Ming-flash-omni-2.0 <https://huggingface.co/inclusionAI/Ming-flash-omni-2.0>`_
     - Omni
     - Streaming TTS
   * - `inclusionAI/Ming-3.0-audio-flash <https://huggingface.co/inclusionAI/Ming-3.0-audio-flash>`_
     - Omni
     - Transcription, audio chat, and 44.1 kHz speech
   * - `inclusionAI/LLaDA2.0-Uni <https://huggingface.co/inclusionAI/LLaDA2.0-Uni>`_
     - Multimodal
     - Text + image understanding and generation


Releases & Blog
---------------

- `GitHub releases <https://github.com/sgl-project/sglang-omni/releases>`_ and `PyPI <https://pypi.org/project/sglang-omni/>`_.
- `The unified interactive runtime <https://www.sglang.io/blog/omni-interactive-runtime>`_ and `the Stage–Process–GPU topology <https://www.sglang.io/blog/spg-topology>`_.

Community & Resources
---------------------

- `SGLang Slack <https://slack.sglang.io>`_ — Ask questions and discuss development.
- `GitHub <https://github.com/sgl-project/sglang-omni>`_ and :doc:`Developer Reference <developer_reference/main>` — Find the source and contribution resources.


.. toctree::
   :maxdepth: 1
   :caption: Get Started

   Welcome to SGLang-Omni <self>
   Quickstart <get_started/apiserver_quickstart>
   Installation <get_started/installation>
   get_started/examples
   get_started/installation_apple_silicon.md
   get_started/installation_npu.md
   get_started/installation_xpu.md
   get_started/installation_cpu.md
   get_started/installation_musa.md
   get_started/troubleshooting.md


.. toctree::
   :maxdepth: 2
   :caption: Cookbook

   cookbook/companies/qwen.md
   cookbook/companies/openmoss.md
   cookbook/companies/openai.md
   cookbook/companies/nvidia.md
   cookbook/companies/fish_audio.md
   cookbook/companies/fun_audio_llm.md
   cookbook/companies/openbmb.md
   cookbook/companies/boson_ai.md
   cookbook/companies/minimax.md
   cookbook/companies/inclusionai.md
   cookbook/companies/tencent.md
   cookbook/companies/zyphra.md
   cookbook/companies/autoark_ai.md
   cookbook/companies/mistral_ai.md
   cookbook/companies/dots_studio.md

.. toctree::
   :maxdepth: 1
   :caption: General Usage

   basic_usage/qwen3_omni.md
   basic_usage/native_chat_stage.md
   basic_usage/audio_translations.md
   basic_usage/tts.md
   basic_usage/process_topology.md
   basic_usage/tts_process_topology.md
   basic_usage/process_topology_migration.md
   basic_usage/omni_router.md
   basic_usage/mps_dp.md
   basic_usage/native_media_routes.md


.. toctree::
   :maxdepth: 1
   :caption: MLX

   mlx/index.md


.. toctree::
   :maxdepth: 1
   :caption: Developer Reference

   developer_reference/main.md
   developer_reference/apiserver_design.md
   developer_reference/pipeline.md
   developer_reference/config.md
   developer_reference/adding_parameters.md
   developer_reference/communication.md
   developer_reference/reference_encode_service.md
   developer_reference/profiler.md
   developer_reference/qwen3_asr_concurrency_profile.md
   developer_reference/full_duplex_guide.md
   developer_reference/duplex_load_generator_pacing.md
   developer_reference/tts_bench.md
   developer_reference/tts_model_integration.md
   developer_reference/asr_bench.md
   developer_reference/omni_offline_bench.md
   developer_reference/rl_admin_control.md
   developer_reference/ci_labels.md
   developer_reference/bump_version.md
