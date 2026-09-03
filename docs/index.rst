SGLang-Omni
=======================

SGLang-Omni is a serving framework for speech, audio, and multimodal models,
built on `SGLang <https://github.com/sgl-project/sglang>`_. It coordinates
multi-stage inference pipelines and exposes OpenAI-compatible APIs.

Speech and multimodal pipelines often combine preprocessing, encoders,
autoregressive models, codecs, and vocoders. SGLang-Omni lets stages use
separate schedulers and places them across processes and devices. See the
:doc:`pipeline lifecycle <developer_reference/pipeline>` and
:doc:`communication design <developer_reference/communication>` for details.

About
-----

Capabilities:

- **Multi-stage pipelines** coordinate preprocessing, model, codec, and vocoder
  stages across processes and devices.
- **SGLang integration** provides SGLang scheduling and batching for
  autoregressive stages.
- **OpenAI-compatible APIs** include endpoints for speech generation,
  transcription, translation, chat completions, and model-dependent streaming.
- **Workload coverage** includes TTS, music generation, ASR, diarization, and
  multimodal models. See the supported-model matrix for exact scope.

Supported Models
----------------

See the :doc:`model and accelerator support matrices <supported_models>` for
tasks, endpoints, streaming behavior, status, and cookbook links.


.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/installation.md
   get_started/installation_npu.md
   get_started/installation_xpu.md
   get_started/installation_cpu.md


.. toctree::
   :maxdepth: 1
   :caption: Basic Usage

   Speech API <user_guide/serving/speech_api.md>
   Transcription API <user_guide/serving/transcription_api.md>
   Audio Translations <basic_usage/audio_translations.md>
   TTS <basic_usage/tts.md>
   Qwen3-Omni <basic_usage/qwen3_omni.md>
   Router <basic_usage/omni_router.md>


.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   user_guide/advanced_features/streaming.md
   user_guide/advanced_features/admission_control.md
   user_guide/advanced_features/deterministic_inference.md
   MPS / DP <basic_usage/mps_dp.md>


.. toctree::
   :maxdepth: 1
   :caption: Deployment

   user_guide/deployment/stage_placement.md
   basic_usage/tts_process_topology.md
   basic_usage/process_topology_migration.md


.. toctree::
   :maxdepth: 1
   :caption: Supported Models

   supported_models.md


.. toctree::
   :maxdepth: 1
   :caption: Benchmarks

   benchmarks/methodology.md
   benchmarks/relay.md


.. toctree::
   :maxdepth: 3
   :caption: Developer Guide

   Overview <developer_reference/overview.md>
   Architecture & Runtime <developer_reference/architecture_runtime.md>
   Development <developer_reference/development.md>
   Benchmarking & Validation <developer_reference/benchmarking_validation.md>
   Documentation Contribution Guide <STYLE_GUIDE.md>
   Bumping the SGLang Version <developer_reference/bump_version.md>


.. toctree::
   :maxdepth: 1
   :caption: Cookbook

   cookbook/higgs_tts.md
   cookbook/voxtral_tts.md
   cookbook/fishaudio_s2_pro.md
   cookbook/qwen3_tts.md
   cookbook/ming_tts.md
   cookbook/moss_tts.md
   cookbook/moss_tts_local.md
   cookbook/dots_tts.md
   cookbook/minimax_music3.md
   cookbook/zonos2.md
   cookbook/audar_tts.md
   cookbook/qwen3_asr.md
   cookbook/fun_asr.md
   cookbook/arkasr.md
   cookbook/moss_transcribe_diarize.md
   cookbook/whisper_asr.md
   cookbook/qwen3_omni.md
   cookbook/ming_omni.md
   cookbook/llada2_uni.md
