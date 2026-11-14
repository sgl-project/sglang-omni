<div align="center">
<img src="https://raw.githubusercontent.com/sgl-project/sglang-omni/main/docs/_static/image/sgl-omni-logo.svg" alt="logo" width="400"></img>

<p>
<a href="https://pypi.org/project/sglang-omni/"><img src="https://img.shields.io/pypi/v/sglang-omni?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI" alt="PyPI"></a>
<a href="https://github.com/sgl-project/sglang-omni/stargazers"><img src="https://img.shields.io/github/stars/sgl-project/sglang-omni?style=for-the-badge&logo=github&label=stars" alt="GitHub stars"></a>
<a href="https://github.com/sgl-project/sglang-omni/blob/main/LICENSE"><img src="https://img.shields.io/github/license/sgl-project/sglang-omni?style=for-the-badge" alt="license"></a>
<a href="https://github.com/sgl-project/sglang-omni/issues"><img src="https://img.shields.io/github/issues-closed-raw/sgl-project/sglang-omni?style=for-the-badge&label=closed%20issues" alt="closed issues"></a>
<a href="https://github.com/sgl-project/sglang-omni/issues"><img src="https://img.shields.io/github/issues-raw/sgl-project/sglang-omni?style=for-the-badge&label=open%20issues" alt="open issues"></a>
<a href="https://deepwiki.com/sgl-project/sglang-omni"><img src="https://img.shields.io/badge/Ask-DeepWiki-087fca?style=for-the-badge" alt="Ask DeepWiki"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

</div>

--------------------------------------------------------------------------------

<p align="center">
<a href="https://lmsys.org/blog/"><b>技术博客</b></a> |
<a href="https://sgl-project.github.io/sglang-omni/"><b>官方文档</b></a> |
<a href="#快速上手"><b>快速上手</b></a> |
<a href="https://sgl-project.github.io/sglang-omni/index.html"><b>实战手册</b></a> |
<a href="https://github.com/sgl-project/sglang"><b>SGLang</b></a> |
<a href="https://slack.sglang.io"><b>加入 Slack</b></a>
</p>

<p align="center">
⭐ <b><a href="https://github.com/sgl-project/sglang-omni/stargazers">为 SGLang-Omni 点亮星标</a>，帮助更多开发者了解面向多模态与语音推理服务的开源基石架构！</b>
</p>

## 最新动态 (News)

- [2026/10] 🗺️ 面向交互式智能的统一运行时：通过可组合计算图与持久化会话（Persistent Sessions），协同调度原生全双工模型、轮次制多模态模型以及级联系统。\[[技术博客](https://www.sglang.io/blog/omni-interactive-runtime)\]
- [2026/10] 🚀 SGLang-Omni **v0.2.0** 已发布至 [PyPI](https://pypi.org/project/sglang-omni/)。可通过 `uv pip install --prerelease=allow "sglang-omni==0.2.0"` 进行安装。\[[安装指南](https://sgl-project.github.io/sglang-omni/get_started/installation.html)\]
- [2026/10] 🐟 零日首发支持 [Fish Audio Drama 3](https://huggingface.co/fishaudio/drama-3)：支持声音克隆与自然语言控制的多角色文本转语音（TTS），并在 `/v1/audio/speech` 上实现流式输出。\[[技术博客](https://www.sglang.io/blog/spg-topology)\] \[[实战手册](https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_drama3.html)\] \[[技术报告](https://github.com/fishaudio/fish-speech/tree/main/docs/assets/Fish_Audio_Drama_3_Technical_Report.pdf)\]
- [2026/10] 🎙️ 正式推出 [Voxt](https://github.com/sgl-project/sglang-omni/tree/main/Voxt)，一款面向 Mac 的开源语音输入工具，现已深度集成 SGLang-Omni，在 Apple Silicon 芯片上实现超高速本地语音识别。将 ASR 与基于大模型的文本润色及个性化 System Prompt 相结合，精准识别专业术语并支持个性化语音输入，提供全本地离线或远程推理选项。\[[下载体验](https://github.com/sgl-project/sglang-omni/releases#release-voxt-v1.16.0)\]\[[效果演示](https://x.com/GenAI_is_real/status/2109419016205017181)\]

<details>
<summary>历史动态</summary>

- [2026/09] 🐧 零日支持 [AuK](https://huggingface.co/tencent/AuK) 与 [AuK-Flash](https://huggingface.co/tencent/AuK-Flash)：文本 + 语音指令 → `/v1/audio/speech` 输出 24 kHz 语音；音频 + 编辑指令 → `/generate` 输出编辑后语音。\[[实战手册](https://sgl-project.github.io/sglang-omni/cookbook/auk.html)\]
- [2026/08] 🎵 零日支持 [MiniMax Music 3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)：歌词 + 风格描述 → `/v1/audio/speech` 输出 32 kHz 立体声歌曲。\[[实战手册](https://sgl-project.github.io/sglang-omni/cookbook/minimax_music3.html)\]
- [2026/08] 🚀 TTS 架构重大重构：共享流水线状态、引擎构建、参考音频编码、能力元数据与声码器（Vocoder）调度。\[[路线图](https://github.com/sgl-project/sglang-omni/issues/985)\] \[[技术博客](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/sglang/sglang-omni/tts-refactor.md)\]
- [2026/06] 🔥 MOSS-TTS Local Transformer v1.5 登陆 SGLang-Omni，支持原生流式输出 48 kHz 高清语音。\[[技术博客](https://lmsys.org/blog/2026-06-17-moss-tts-local-v15/)\] \[[实战手册](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts_local.html)\]
- [2026/06] 🔥 Higgs Audio v3 TTS 上线，提供实时可控语音生成。\[[技术博客](https://lmsys.org/blog/2026-06-04-higgs-audio-v3-tts/)\] \[[实战手册](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)\]

</details>

## 关于项目 (About)

SGLang-Omni 是一个专为全模态（Omni）、语音（Speech）与文本转语音（TTS）模型打造的多阶段推理服务运行时。其核心设计目标是**多阶段解码（Multi-stage decoding）**：将生成任务拆解至计算模式、依赖结构与资源需求各异的异构阶段中。SGLang-Omni 负责掌管流水线拓扑编排、阶段生命周期管理、跨阶段数据传输、模型族适配层以及 OpenAI 兼容的服务接口；并在适用场景下与 [SGLang](https://github.com/sgl-project/sglang) 深度协同，提供高性能的自回归调度与模型执行能力。

- **多阶段运行时 (Multi-stage runtime)**：SGLang-Omni 将生成流程抽象为协同运作的多个阶段：前处理、编码器、自回归引擎、发声模块（Talkers）、解码器、声码器（Vocoders）与聚合器。
- **阶段专用调度策略 (Stage-specialized scheduling)**：每个阶段运行在与其计算负载高度适配的调度器后，涵盖从 SGLang 支持的高性能自回归调度，到轻量级前处理与流式声码器循环。
- **传输感知的执行体系 (Transport-aware execution)**：控制平面统一协调请求调度，而中继数据平面（Relay data plane）则通过共享内存（Shared-memory）、NCCL、NIXL 以及 Mooncake 等后端高效传递张量数据。
- **标准 API 接口表面 (API surface)**：完全兼容 OpenAI 规范的 API 端点，支持多模态对话、语音生成、批量语音、流式语音、声音音色上传与音频转录。

## 支持的模型与能力矩阵 (What SGLang-Omni Serves)

- **全模态对话与语音 (Omni chat and speech)**：[Qwen3-Omni](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_omni.html)、[MiniCPM-o 4.5](https://sgl-project.github.io/sglang-omni/cookbook/minicpm_o.html)、[Ming-Omni](https://sgl-project.github.io/sglang-omni/cookbook/ming_omni.html)、[Ming-3.0-audio-flash](https://sgl-project.github.io/sglang-omni/cookbook/ming3_audio.html) —— 多模态理解与文本/音频协同回复。
- **原生全双工对话 (Native full-duplex conversation)**：[MiniCPM-o 4.5](https://sgl-project.github.io/sglang-omni/cookbook/minicpm_o.html)、[PersonaPlex](https://sgl-project.github.io/sglang-omni/cookbook/personaplex.html)、[NemotronLabs VoiceChat](https://sgl-project.github.io/sglang-omni/cookbook/nemotron_voicechat.html) —— 通过 `/v1/realtime` 进行实时双向听说交互，包括使用 MiniCPM-o 进行摄像头实时视觉输入。参阅[全双工指南](https://sgl-project.github.io/sglang-omni/developer_reference/full_duplex_guide.html)。
- **级联式语音助手 (Cascaded voice assistants)**：将 ASR → LLM → TTS 各服务组合为支持流式、按轮次进行的交互对话，原生配备语音活动检测（VAD）、对话历史上下文管理、轮次接管（Turn taking）与随时打断机制。参阅[流式级联指南](https://sgl-project.github.io/sglang-omni/developer_reference/full_duplex_guide.html#streaming-cascade)。
- **图像与视频生成 (Image and video generation)**：[LLaDA2.0-Uni](https://sgl-project.github.io/sglang-omni/cookbook/llada2_uni.html) 支持文本/图像理解、图像生成与编辑；[Cosmos3 Nano](https://sgl-project.github.io/sglang-omni/cookbook/cosmos3_nano.html) 支持交错多模态理解与生成，具备原生图像、视频与动作路由。
- **音乐生成 (Music generation)**：[MiniMax Music 3](https://sgl-project.github.io/sglang-omni/cookbook/minimax_music3.html) —— 歌词 + 风格描述 → 32 kHz 立体声歌曲。
- **语音合成生成 (Speech generation)**：[Higgs Audio v3](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html)、[MOSS-TTS](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts.html)、[MOSS-TTS Local](https://sgl-project.github.io/sglang-omni/cookbook/moss_tts_local.html)、[Fish Audio Drama 3](https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_drama3.html)、[Fish Speech S2-Pro](https://sgl-project.github.io/sglang-omni/cookbook/fishaudio_s2_pro.html)、[Qwen3-TTS](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_tts.html)、[Fun-CosyVoice3](https://sgl-project.github.io/sglang-omni/cookbook/fun_cosyvoice3.html)、[Voxtral TTS](https://sgl-project.github.io/sglang-omni/cookbook/voxtral_tts.html)、[Ming-Omni-TTS](https://sgl-project.github.io/sglang-omni/cookbook/ming_tts.html)、[Ming-3.0-TTS](https://sgl-project.github.io/sglang-omni/cookbook/ming3_tts.html)、[dots.tts](https://sgl-project.github.io/sglang-omni/cookbook/dots_tts.html)、[ZONOS2](https://sgl-project.github.io/sglang-omni/cookbook/zonos2.html) —— 通过 `/v1/audio/speech` 提供服务，批量、流式与音色特性依具体模型而定。
- **指令驱动语音生成与编辑 (Instruction-driven speech generation and editing)**：[AuK 与 AuK-Flash](https://sgl-project.github.io/sglang-omni/cookbook/auk.html) —— 文本与语音指令 → 语音生成；音频与编辑指令 → 语音定向编辑。
- **音频转录 (Audio transcription)**：[Qwen3-ASR](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_asr.html)、[Fun-ASR](https://sgl-project.github.io/sglang-omni/cookbook/fun_asr.html)、[ARK-ASR](https://sgl-project.github.io/sglang-omni/cookbook/arkasr.html)、[Nemotron 3.5 ASR](https://sgl-project.github.io/sglang-omni/cookbook/nemotron3_5_asr.html)、[Ming-3.0-ASR](https://sgl-project.github.io/sglang-omni/cookbook/ming3_asr.html)、实验性 [Whisper](https://sgl-project.github.io/sglang-omni/cookbook/whisper_asr.html) 以及通过 `/v1/audio/transcriptions` 接口的 [MOSS-Transcribe-Diarize](https://sgl-project.github.io/sglang-omni/cookbook/moss_transcribe_diarize.html)（MOSS-TD 还支持返回说话人标签与时间戳，配置 `response_format=verbose_json`）。
- **说话人日志分离 (Speaker diarization)**：[Nemotron 3 Diarization](https://sgl-project.github.io/sglang-omni/cookbook/nemotron_diarization.html) —— 对上传音频文件与实时流式音频识别说话人间期，包括重叠语音场景。
- **语音翻译 (Speech translation)**：[Hy-SimulMT](https://sgl-project.github.io/sglang-omni/cookbook/hy_simulmt.html) —— 中文语音 → 英文文本与英文语音 → 中文文本，支持文件翻译与实时同传。端点与模型支持详见[语音翻译指南](https://sgl-project.github.io/sglang-omni/basic_usage/audio_translations.html)。

## 硬件支持 (Hardware Support)

| 后端 | 支持状态 | 说明 |
|---------|--------|-------|
| **NVIDIA CUDA** | 官方支持 | 默认主力后端，覆盖所有模型与特性。 |
| **Apple Silicon** | 实验性支持 | 在 macOS arm64 上通过 Torch MPS 运行 Qwen3-ASR。可通过 [`install.sh`](./install.sh) 安装并参考 [Qwen3-ASR 指南](./docs/cookbook/qwen3_asr.md#apple-silicon)。 |
| **Intel GPU (XPU)** | 实验性支持 | 通过 PyTorch XPU 支持 Intel Arc GPU。**端到端支持 Qwen3-ASR、Qwen3-TTS、ZONOS2 与 Qwen3-Omni**（ZONOS2 配备解码图优化；Omni thinker 支持跨多 XPU 张量并行）。请按照 [Intel XPU 指南](./docs/get_started/installation_xpu.md)进行安装；系统将自动检测后端。 |

更多模型指南（包括实验性与科研探索路径）可在[实战手册 (Cookbook)](https://sgl-project.github.io/sglang-omni/)中查阅。

## 快速上手 (Quick Start)

- [安装指南 (Installation)](https://sgl-project.github.io/sglang-omni/get_started/installation.html)
- [TTS 使用指南 (TTS usage)](https://sgl-project.github.io/sglang-omni/basic_usage/tts.html)
- [Qwen3-Omni 使用指南 (Qwen3-Omni usage)](https://sgl-project.github.io/sglang-omni/basic_usage/qwen3_omni.html)
- [Qwen3-ASR 实战手册 (Qwen3-ASR cookbook)](https://sgl-project.github.io/sglang-omni/cookbook/qwen3_asr.html)
- [Nemotron 3.5 ASR 实战手册 (Nemotron 3.5 ASR cookbook)](https://sgl-project.github.io/sglang-omni/cookbook/nemotron3_5_asr.html)
- [MOSS-Transcribe-Diarize 实战手册](https://sgl-project.github.io/sglang-omni/cookbook/moss_transcribe_diarize.html)
- [全模态路由分发器 (Omni router)](https://sgl-project.github.io/sglang-omni/basic_usage/omni_router.html)
- [开发者参考文档 (Developer reference)](https://sgl-project.github.io/sglang-omni/developer_reference/main.html)

## 社区交流与支持 (Community & Support)

SGLang-Omni 诚挚欢迎广大开发者的参与，涵盖推理系统、算子优化、调度算法、跨阶段通信、模型运行器与缓存效率、新模型接入适配、基准评测以及生产级部署等多个方向。欢迎加入 [SGLang Slack 社区](https://slack.sglang.io)或查阅[开发者参考文档](https://sgl-project.github.io/sglang-omni/developer_reference/main.html)。

若有企业或组织希望对 SGLang-Omni、TTS 或全模态模型推理服务提供赞助或合作支持，可联系赵晨阳（Chenyang Zhao）：[zhaochenyang@lmsys.org](mailto:zhaochenyang@lmsys.org)。

## 学术引用 (Citation)

若在学术研究或工程项目中使用了 SGLang-Omni，请引用我们关于[统一交互式运行时](https://www.sglang.io/blog/omni-interactive-runtime)与[阶段-进程-GPU 拓扑架构（SPG Topology）](https://www.sglang.io/blog/spg-topology)的技术博客：

```bibtex
@misc{sglangomniinteractive2026,
  title={Unified Runtime for Full-Duplex Interactive Intelligence with {SGLang Omni}},
  author={{SGLang-Omni Team}},
  howpublished={\url{https://www.sglang.io/blog/omni-interactive-runtime}},
  year={2026},
  month={oct}
}

@misc{sglangomni2026,
  title={Fish Audio Drama 3 Meets {SGLang-Omni}: Day-0 Support and the Stage--Process--GPU Topology},
  author={{SGLang-Omni Team}},
  howpublished={\url{https://www.sglang.io/blog/spg-topology}},
  year={2026},
  month={oct}
}
```

## 致谢 (Acknowledgments)

SGLang-Omni 扎根于 SGLang 繁荣的生态体系，并汲取了来自 TTS、语音与全模态开源社区的诸多杰出成果。我们衷心感谢所有模型研发团队、系统架构贡献者与合作伙伴，共同携手让开源多模态推理服务更加快速、可靠且易于扩展。

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月28日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
