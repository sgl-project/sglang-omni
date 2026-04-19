SGLang-Omni 中文文档
=======================

.. raw:: html

   <p>🌐 <strong>语言:</strong> <a href="../index.html">English</a> | 简体中文</p>

SGLang-Omni 是 SGLang 的生态扩展项目。
Omni 模型是指具有多模态输入和多模态输出的模型。
这些模型通常由多个阶段（stages）组成，使得 SGLang 原本专为大语言模型（LLM）设计的架构不再适用。
因此，SGLang-Omni 旨在提供编排多阶段流水线的能力，同时兼具高性能和对实时 API 的支持。
我们的核心特性包括：

- 与 SGLang 原生集成，保证卓越性能
- 专为 Omni 模型设计的多阶段流水线框架
- 支持实时 API 的 OpenAI 兼容服务器


.. toctree::
   :maxdepth: 1
   :caption: 快速入门 (Get Started)

   get_started/installation.md
   get_started/apiserver_quickstart.md


.. toctree::
   :maxdepth: 1
   :caption: 基础使用 (Basic Usage)

   basic_usage/qwen3_omni.md
   basic_usage/tts_s2pro.md


.. toctree::
   :maxdepth: 1
   :caption: 性能测试基准 (Benchmarks)

   benchmarks/relay.md


.. toctree::
   :maxdepth: 1
   :caption: 开发者参考 (Developer Reference)

   developer_reference/architecture.md
   developer_reference/relay_design.md
   developer_reference/apiserver_design.md
