SGLang-Omni
=======================

SGLang-Omni is an ecosystem project for SGLang.
Omni models refer to models that have multi-modal inputs and multi-modal outputs.
These models typically consist of multiple stages, making SGLang's LLM-specific architecture no longer suitable.
Therefore, SGLang-Omni is designed to provide the ability to orchestrate multi-stage pipeline with high performance and real-time API support.
Our core features include:

- Native Integration with SGLang for performance
- Multi-Stage Pipeline Framework for Omni Models
- OpenAI-Compatible Server with Real-Time API support


.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/installation.md


.. toctree::
   :maxdepth: 1
   :caption: Basic Usage

   basic_usage/qwen3_omni.md
   basic_usage/tts_s2pro.md


.. toctree::
   :maxdepth: 1
   :caption: Benchmarks

   benchmarks/relay.md


.. toctree::
   :maxdepth: 1
   :caption: Developer Reference

   developer_reference/architecture.md
   developer_reference/relay_design.md
   developer_reference/talker_decode_parity.md
