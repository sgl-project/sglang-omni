.. _sglang-omni:

Welcome to SGLang-Omni
======================

Serve speech and multimodal models with SGLang-Omni.

- **Speech & Multimodal Models.** Transcribe audio, generate speech, and serve models that accept audio, images, and video through OpenAI-compatible APIs.
- **Streaming & Realtime.** Stream transcripts and speech as they are generated. Use realtime sessions for live audio input and interactive voice applications.
- **Multi-stage Runtime.** Run preprocessing, inference, and audio generation as stages with independent scheduling. The runtime is built on `SGLang <https://github.com/sgl-project/sglang>`_.
- **Hardware Support.** Choose the installation guide for your hardware: :doc:`CUDA <get_started/installation>`, :doc:`Apple Silicon <other_platforms/installation_apple_silicon>`, :doc:`Ascend NPU <other_platforms/installation_npu>`, :doc:`Intel XPU <other_platforms/installation_xpu>`, :doc:`Intel CPU <other_platforms/installation_cpu>`, or :doc:`MUSA <other_platforms/installation_musa>`. Model support varies by platform.

Get Started
-----------

- :doc:`Quickstart <get_started/quickstart>` — Start a model server and transcribe one audio clip.
- :doc:`Installation <get_started/installation>` — Choose an installation method for your hardware.
- :doc:`Examples <get_started/examples>` — Stream a transcript, generate speech, or send an audio chat request.
- Browse supported models under **Cookbook** in the sidebar.

News and latest blogs
---------------------

Scroll sideways for more updates.

.. raw:: html

   <style>
     .omni-news {
       display: flex;
       gap: 1rem;
       overflow-x: auto;
       scroll-snap-type: x mandatory;
       list-style: none;
       margin: 0 0 0.75rem;
       padding: 0 0 0.75rem;
     }

     .omni-news > li {
       flex: 0 0 calc((100% - 2rem) / 2.12);
       min-width: 0;
       scroll-snap-align: start;
     }

     .omni-news a {
       display: flex;
       flex-direction: column;
       height: 100%;
       overflow: hidden;
       border: 1px solid var(--pst-color-border, #bbb);
       border-radius: 0.75rem;
       background: var(--pst-color-surface, transparent);
       color: var(--pst-color-text-base, inherit);
       text-decoration: none;
     }

     .omni-news a:hover strong {
       text-decoration: underline;
     }

     .omni-news:focus-visible,
     .omni-news a:focus-visible {
       outline: 3px solid var(--pst-color-primary, #176a96);
       outline-offset: -3px;
     }

     .omni-news img {
       display: block;
       width: 100%;
       aspect-ratio: 16 / 8;
       object-fit: contain;
       background: #fff;
     }

     .omni-news img.omni-news-logo {
       padding: 1.25rem;
       background: #fff6ef;
     }

     .omni-news strong {
       padding: 0.75rem 0.875rem 0.25rem;
       line-height: 1.35;
     }

     .omni-news time {
       margin-top: auto;
       padding: 0 0.875rem 0.75rem;
       color: var(--pst-color-text-muted, inherit);
       font-size: 0.85rem;
     }

     @media (max-width: 600px) {
       .omni-news > li {
         flex-basis: 85%;
       }
     }
   </style>

   <ul class="omni-news" role="list" tabindex="0" aria-label="News and latest blogs">
     <li>
       <a href="https://github.com/sgl-project/sglang-omni/blob/main/Voxt/README.md">
         <img class="omni-news-logo" src="https://raw.githubusercontent.com/sgl-project/sglang-omni/main/Voxt/resources/icon.png" alt="" loading="lazy">
         <strong>Voxt 2.0.3: Open-source voice input for Mac</strong>
         <time datetime="2026-11-10">November 10, 2026</time>
       </a>
     </li>
     <li>
       <a href="https://index.openrsi.foundation/blog/2026/openrsi-sglang-omni/">
         <img class="omni-news-logo" src="_static/image/sgl-omni-logo.svg" alt="" loading="lazy">
         <strong>OpenRSI × SGLang-Omni: Scaling RSI for AI Infra</strong>
         <time datetime="2026-11-09">November 9, 2026</time>
       </a>
     </li>
     <li>
       <a href="https://github.com/sgl-project/sglang-omni/releases/tag/v0.2.0">
         <img class="omni-news-logo" src="_static/image/sgl-omni-logo.svg" alt="" loading="lazy">
         <strong>SGLang-Omni v0.2.0</strong>
         <time datetime="2026-10-27">October 27, 2026</time>
       </a>
     </li>
     <li>
       <a href="https://www.sglang.io/blog/omni-interactive-runtime">
         <img src="https://www.sglang.io/images/omni-interactive-runtime/interaction-approaches.png" alt="" loading="lazy">
         <strong>A unified runtime for full-duplex interaction</strong>
         <time datetime="2026-10-20">October 20, 2026</time>
       </a>
     </li>
     <li>
       <a href="https://www.sglang.io/blog/spg-topology">
         <img src="https://www.sglang.io/images/spg-topology/spg-qwen3-omni-stage-graph.png" alt="" loading="lazy">
         <strong>Serving multi-stage omni models with SPG</strong>
         <time datetime="2026-10-13">October 13, 2026</time>
       </a>
     </li>
     <li>
       <a href="https://fish.audio/blog/fish-audio-drama-3/">
         <img src="https://public-platform.r2.fish.audio/media/blog-images/d42b7127d9f24eb3b222388bf81d6885" alt="" loading="lazy">
         <strong>Fish Audio Drama 3 launches with Omni support</strong>
         <time datetime="2026-10-13">October 13, 2026</time>
       </a>
     </li>
   </ul>

`View all releases <https://github.com/sgl-project/sglang-omni/releases>`_

Community & Resources
---------------------

- `SGLang Slack <https://slack.sglang.io>`_ — Ask questions and discuss development.
- `GitHub <https://github.com/sgl-project/sglang-omni>`_ and :doc:`Developer Reference <developer_reference/main>` — Find the source and contribution resources.


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Get Started

   Welcome to SGLang-Omni <self>
   Quickstart <get_started/quickstart>
   Installation <get_started/installation>
   get_started/examples
   get_started/troubleshooting.md


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Voxt

   Quickstart <get_started/voxt>


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Other Platforms

   Apple Silicon <other_platforms/installation_apple_silicon>
   Ascend NPU <other_platforms/installation_npu>
   Intel XPU <other_platforms/installation_xpu>
   Intel CPU <other_platforms/installation_cpu>
   Moore Threads MUSA <other_platforms/installation_musa>


.. toctree::
   :hidden:
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
   :hidden:
   :maxdepth: 1
   :caption: General Usage

   basic_usage/qwen3_omni.md
   basic_usage/native_chat_stage.md
   basic_usage/audio_translations.md
   basic_usage/tts.md
   basic_usage/asr.md
   basic_usage/omni.md
   basic_usage/process_topology.md
   basic_usage/tts_process_topology.md
   basic_usage/process_topology_migration.md
   basic_usage/omni_router.md
   basic_usage/mps_dp.md
   basic_usage/native_media_routes.md


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Developer Reference

   developer_reference/main.md
   developer_reference/contribution_guide.md
   Native MLX <developer_reference/mlx/index>
   developer_reference/apiserver_design.md
   developer_reference/pipeline.md
   developer_reference/config.md
   developer_reference/adding_parameters.md
   developer_reference/communication.md
   developer_reference/reference_encode_service.md
   developer_reference/profiler.md
   developer_reference/qwen3_asr_concurrency_profile.md
   developer_reference/full_duplex_guide.md
   developer_reference/tts_bench.md
   developer_reference/tts_model_integration.md
   developer_reference/asr_bench.md
   developer_reference/omni_offline_bench.md
   developer_reference/rl_admin_control.md
   developer_reference/ci_labels.md
   developer_reference/bump_version.md
