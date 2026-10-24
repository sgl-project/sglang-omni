# Voxt 模型栈现代化方案

初始分析：2026-09-17；后续计划调整：2026-09-18
分支：`chore/model-stack-modernization`
分析基线：`9807381e4b76e4bf1bcb6f692683136208322718`
Voxt 清理提交：`95ade09`

本文记录实施前基线与目标。当前分支已开始代码实施；实际完成项、检查结果和阻断项见 [实施记录](ModelStackModernizationImplementation.zh-CN.md)。不能把本文的目标或预期性能视为已经验证的结果。目标是把依赖、本地 ASR / LLM 目录、说话人分离、推理链路和设置 UI 收敛到更短、更清楚的路径上，同时保持现有核心业务和用户流程不变。

## 0. 当前进度与接续入口（2026-09-18）

| 项目 | 当前状态 | 下一步 |
|---|---|---|
| FluidAudio / VBx 与隐藏模型 | Voxt 源码已清理并提交 | macOS 编译、迁移和替代能力验收 |
| Audio fork 同步与升级 | 已推送 fork `main`，包含上游 `3e97855` 的祖先历史，0 behind | 不再重复同步或重做 fork 候选 |
| Audio fork 构建 / 测试 | Xcode 26.5 / Swift 6.3.2；669 个 Swift Testing 和 2 个 XCTest 通过 | 不能代替 Voxt 测试或真实模型回放 |
| Voxt MLX 依赖 | 工作区已切入 Audio `2a6e75d` / MLX `0.31.6` / LM `c6446cf` 及 API 适配 | Mac 真实解析、编译、loader / 模型回归 |
| UI / 链路优化 | 安装校验后台缓存、live 分组增量缓存、投递取消和非阻塞回收已落地工作区 | Swift 编译、取消竞态和 Instruments 验证 |
| 发布验收 | 未完成 | 实际 lockfile、质量 / 延迟 / 内存 / 包体报告 |

目标 fork revision：`2a6e75d28ae6a399ba7c7aec842384ef7a3142b5`。它与通过 CI 的 `33e72855c0169377ae5ddae83b0ce5da0673e920` 文件树相同（`4f1c756bff39e3ad320e2b12ce3f1726dd2a268b`），仅补充上游合并历史。[成功运行](https://github.com/hehehai/mlx-audio-swift/actions/runs/35309904287)。

以下现状分析的旧数字以分析基线为准；当前实施状态以本节和 [实施记录](ModelStackModernizationImplementation.zh-CN.md) 为准。已删除的 fork 开发分支不再作为接续入口。A–D 的后续源码已实施在工作区；E 的集中本地验收脚本已提供，但本地 Linux 无法执行 Xcode，因此未触发新 Actions，也不能标为产品验收完成。

## 1. 目标与硬约束

### 1.1 要达成的结果

1. 升级模型相关依赖，尤其是 MLX 栈。
2. 清理本地 ASR / LLM 目录：删除无用、隐藏、过期模型，以及它们的加载、配置、能力、UI、测试链路。
3. 完全移除 FluidAudio：包括 Offline VBx、FluidAudio 流式说话人分离回退、包依赖、下载、配置和 UI；独立审计发布包，确保不再包含 sherpa-onnx / ONNX Runtime。
4. 用 MLX + Apple CPU / GPU 把 ASR 和 LLM 链路提速，降低心智负担。
5. 检查设置、目录刷新、会议详情等 UI 阻塞，只做让现有流程更顺的重构。
6. 降低 App 体积，但不牺牲默认可用的本地能力。

### 1.2 硬约束

- 核心业务不变：听写、会议、翻译、改写、笔记、远程模型。
- 用户主流程不变：不增加新的配置步骤，不把简单选择拆成更多向导。
- 已安装模型的磁盘文件不自动删除；只停止识别、展示和加载。
- 旧选择必须可迁移到仍保留的默认模型，不能让设置页或启动卡死。
- 不为优化而引入新的运行时开关、实验通道或“高级模式”。
- 在当前分支分阶段实施；源码完成与 macOS 编译 / 真机验收分开记录。

### 1.3 明确不做

- 不重做听写 / 会议 / 翻译 / 改写的产品流程。
- 不删除 OmniVAD。它是独立 C 运行时，不是 ONNX，体积约 7 MB，承担会议 / 听写 VAD。
- 不删除 Hy-MT2 GGUF 翻译。`llama.swift` 只服务这条专用翻译路径，不是 sherpa-onnx。
- 不把远程 ASR / 远程 LLM 供应商清单纳入本轮删除。
- 不把设置页改成全新信息架构。目录可以更短，交互不能更绕。

## 2. 现状结论

### 2.1 依赖真实状态

分析基线（`9807381`）的 Xcode 直接依赖；不是当前分支的完成状态：

| 包 | 当前钉死方式 | 当前值 | 上游最新 | 结论 |
|---|---|---|---|---|
| `mlx-audio-swift`（fork `hehehai/mlx-audio-swift`） | exactVersion | `0.1.3-voxt.12` | 上游 `Blaizzy/mlx-audio-swift` 发布 `v0.1.3`，`main` 已到 2026-09-13 | 继续走 fork tag，但要评估是否同步更新 `main` |
| `mlx-swift-lm` | revision | `d242429` | 发布 `3.31.4`；`main` 已到 `c6446cf`（2026-09-15，首 token 清 MLX cache） | 这是提速关键，但受工具链卡住 |
| `mlx-swift` | 传递依赖 | 实际是 `0.31.4` | 发布 `0.31.6`，`Package.swift` 要求 Swift 6.3 | 不能只在 Voxt 里改 version，fork 把 `mlx-swift` 钉死在 `0.31.4` |
| `FluidAudio` | exactVersion | `0.15.6` | `0.15.7` | **本轮删除，不升级** |
| `llama.swift` | exactVersion | `2.10549.0` | 同 tag | 保留，服务 GGUF 翻译 |
| Sparkle | upToNextMajor `2.6.4` | 解析时会漂 | `2.10.0` | 可随 resolved 文件锁到新版本 |
| GRDB | upToNextMajor `7.10.0` | 解析时会漂 | `7.11.1` | 同上 |
| swift-log | upToNextMajor `1.6.0` | 解析时会漂 | `1.15.1` | 同上 |
| PermissionFlow | exact `2.11.2` | 已是最新 | `2.11.2` | 不动 |
| FaviconFinder | upToNextMajor `5.1.5` | 已是最新 | `5.1.5` | 不动 |

最初文档记录的是 Xcode 26.3 / Swift 6.2 上限。现在已从 fork 成功 CI 确认 Xcode 26.5 配套 Swift 6.3.2，并验证新依赖可构建。Voxt 仍需验证自身 API 与产品目标，不再把工具链版本是否满足作为未知阻断项。

### 2.2 当前源码未发现 sherpa-onnx 运行时依赖

当前源码和 Xcode 直接包引用中未发现 sherpa-onnx。现有引用是旧配置迁移；是否存在传递依赖或发布包残留，仍需在 macOS 上检查实际解析图与构建产物：

- `Voxt/Settings/Features/FeatureSettings.swift`：`sherpa:` 选择迁到默认 MLX ASR
- `Voxt/Settings/TranscriptionTypes.swift`：`sherpaOnnx` 引擎名迁到 `.mlxAudio`
- `VoxtTests/FeatureSettingsStoreTests.swift`：对应回归

本轮要做的是：

1. 删除历史运行时入口，只保留隔离的一次性配置迁移及其回归测试。
2. 完全删除 FluidAudio，确认其独占依赖（包括 NemoTextProcessing）不再进入解析与链接链路。
3. 独立审计 sherpa / ONNX Runtime：不能用“删除 FluidAudio”代替对发布包的验证。

纠正：`NemoTextProcessing.xcframework` 是文本规范化组件，不能仅凭它是预编译产物就判定它使用 ONNX。此前“它是 ONNX 大头”的说法没有依据，本文不再采用。

### 2.3 VBx 不是 sherpa，是 FluidAudio

`MeetingDiarizationMode.offlineVBx` 默认开启。实现是：

- `FluidAudioMeetingSpeakerDiarizationEngine`
- `OfflineDiarizerManager` / `DiarizerManager`
- `MeetingOfflineVBxModelStorage`
- 设置、引导、功能页里的 “Offline VBx”

FluidAudio `0.15.6` 的 `Package.swift` 带预编译 `NemoTextProcessing.xcframework`，并包含 FastClusterWrapper、MachTaskSelfWrapper 和包资源。删除 FluidAudio 后，这些独占依赖也应退出 Voxt 的构建图；实际包体收益需测量，不能把下载的 XCFramework 大小当作 App 可节省体积。

**已确认决策：完全移除 FluidAudio，而不仅删除 VBx 选项。** 当前直接使用 FluidAudio API 的只有上述说话人分离引擎和模型管理器。ASR、LLM、OmniVAD、MLX Silero、GGUF 翻译不依赖它。

Sortformer v2 已经在产品里，走 `MLXAudioVAD`，不需要 FluidAudio。保留独立说话人分离的 Sortformer 路径以及 MOSS 原生说话人输出；不保留 FluidAudio streaming fallback、隐藏开关或可选编译后门。

### 2.4 模型目录比 UI 看起来更重

ASR 目录在 `Voxt/Transcription/MLXModelSupport.swift`：

- 可见 11 个
- 隐藏兼容 25 个
- 加载器、能力表、体积表、提示调参、会议 live mode 都还为隐藏模型服务

LLM 目录在 `Voxt/Core/Models/CustomLLMModelSupport.swift`：

- 可见 12 个
- 隐藏兼容 16 个
- `displayModels(includingInstalled:)` 会让已下载的隐藏模型重新出现在设置页

这和“降低心智负担”相反。隐藏模型不是无成本兼容层，它们会：

- 拉长 `primeDownloadedStateCacheIfNeeded()` 的磁盘扫描
- 保留整条 family 加载 / 参数 / UI 分支
- 让已安装用户的目录再次变长

本轮策略：**隐藏模型整链删除，不再保留 hiddenSupport。** 旧选择迁到仍可见的默认或同系列主推模型。磁盘上的旧文件保留，设置页不再展示，运行时不再加载。

### 2.5 运行时已经有可用的加速骨架，但被目录和双引擎拖住

已经存在、应保留并收紧的能力：

- Qwen3 ASR 的 KV cache 量化（`MLXASRKVCachePolicy.conservativeQwen`）
- `Qwen3ASRMemoryEfficientLoader` / `MemoryEfficientModelContainerLoader`
- `MeetingLocalInferenceCoordinator` 单车道优先级队列
- 会议 native streaming：Qwen / Cohere / MOSS / Nemotron
- LLM warmup、idle unload、deep idle `Memory.clearCache()` + `Stream.synchronize()`
- Silero / OmniVAD 两套本地 VAD，能量值兜底

主要拖累：

- `MLXTranscriber.swift` 4117 行，为 10+ family 做特化
- `MLXSTTModelLoader` 用 repo 字符串匹配所有历史模型
- VAD 热路径里 `MainActorSync.run` 读设置
- 设置页用 `AnyView` 包一层生命周期观察，目录快照随下载进度整表刷新
- VBx 专用人数约束无法直接移植到 Sortformer；`numSpeakers` 是否受 checkpoint 输出维度限制必须先验证，不能当作任意可调参数

## 3. 目标架构

保留三条本地推理主链，删除第四条。

```text
听写 / 会议 ASR
  用户音频 -> OmniVAD / Silero / energy
           -> MLXAudioSTT（可见模型）
           -> 文本 / 时间戳 / 说话人（MOSS）

会议说话人分离
  会议音频 -> MLXAudioVAD Sortformer v2
           -> 说话人片段 -> 现有 smoothing / assembly

本地文本
  增强 / 翻译 / 改写 -> MLXLLM / MLXVLM（可见模型）
  专用翻译可选       -> llama.swift Hy-MT2 GGUF

删除
  FluidAudio 整包、Offline VBx、FluidAudio streaming fallback
  全部 hiddenSupport ASR / LLM
  sherpa-onnx 历史名称以外的任何运行时假设
```

默认推荐保持不变，避免用户重新学习：

| 角色 | 保持 |
|---|---|
| 默认本地 ASR | `mlx-community/Qwen3-ASR-0.6B-4bit` |
| 默认本地 LLM | `mlx-community/Qwen3.5-4B-OptiQ-4bit` |
| 默认会议说话人分离 | Sortformer v2（从 Offline VBx 迁过来） |
| 默认专用翻译 GGUF | Hy-MT2 1.8B Q4_K_M |

## 4. 依赖升级方案

### 4.1 工具链闸门已通过，下一步是 Voxt 集成

固定接续环境为 Xcode 26.5 / Swift 6.3.2（fork CI 已验证）。Mac 本地仍应打印版本确认；不得为兼容旧 Swift 6.2 编译器单独降级新依赖中的一项。

目标依赖必须作为一组切换：

| 依赖 | 升级前引用 | 已切入工作区的目标 |
|---|---|---|
| Audio fork | exact `0.1.3-voxt.12` | revision `2a6e75d28ae6a399ba7c7aec842384ef7a3142b5` |
| mlx-swift | `.12` 的 exact `0.31.4` | 新 fork 的 exact `0.31.6` |
| mlx-swift-lm | revision `d2424294a6c3bbd0de37a0761d80efc05e6813dd` | revision `c6446cf7bfb7cea76408013b614d4b2c530eaa03` |
| swift-transformers | 旧图解析值待 lockfile 确认 | 新 fork 的 exact `1.3.4` |
| swift-huggingface | 旧图解析值待 lockfile 确认 | 新 fork 的 exact `0.10.2` |

没有必须先发新 tag 的技术阻断；目标 revision 已在远程可获取。工程不得引用浮动 `main`、已删除的开发分支或本机绝对路径。

### 4.2 Voxt 集成步骤

1. 一次性修改 Audio / LM 工程引用、API 适配和 `tools/audit_model_stack.py` 预期，不能只改版本号。
2. 新模型 `prepare()` 生命周期、有效 EOS、chat conventions、typed prefill 已切入正式源码。保留预量化 loader 与初始 `.remainder` chunking，不同时改变生成策略；一次性 patch 已删除，后续直接审查正式 loader。
3. 补齐 loader 回归：普通量化 / OptiQ 混合量化、Qwen3.5、Qwen3 VL / Gemma VLM、thinking 关闭、EOS 与尾块、取消 / 切模型。patch 是适配起点，不是这些行为已验证的证明。
4. 处理未知 checkpoint 必须明确失败；移除目录不等于忽略模型配置错误。模型声明与 tokenizer 模板的工具 / reasoning 协议需要一起核对，不能只假定名称匹配就兼容。
5. 在统一工具链的 Mac 上真实解析整个 Voxt 图，提交工作区 `Package.resolved`。fork CI artifact 只能帮助核对共同依赖，不能直接复制成 Voxt lockfile。
6. 一次性 patch、隔离快照生成器及其专用测试已删除，正式源码成为唯一 loader；使用 `tools/run_model_stack_validation.sh` 集中验证，不维护两套应用实现。
7. 可在发布时为已验证 revision 补不可变 tag，但不阻塞本分支集成，也不复用旧 tag。

LM `#620` 是首个生成 token 处的 cache 管理变更，不直接等于 ASR 首包或 LLM TTFT 提速，必须测量完整链路。

### 4.3 其他依赖

- **删除** `FluidAudio` 包引用、product、`#if canImport(FluidAudio)` 全部分支。
- **保留** `llama.swift`，不升级除非 Hy-MT2 加载失败。
- Sparkle `2.10.0` / GRDB `7.11.1` / swift-log `1.15.1` 已在 `95ade09` 更新并固定；FaviconFinder `5.1.5`、PermissionFlow `2.11.2`、llama.swift `2.10549.0` 保留。Voxt API 兼容仍需完整构建确认。
- 工作区 `Package.resolved` 已解除 gitignore；重新盘点确认它原本已受版本控制，内容仍是旧依赖图。必须真实重新解析、审查并更新，而不只是修改工程 pin；CI 和发布严格使用同一锁文件。PR 首轮已因旧锁与新 revision 不匹配失败，不能再写成“尚无锁文件”。

### 4.4 体积相关的依赖事实

本轮确定删除 FluidAudio 及其独占依赖。当前环境没有 macOS Release 构建产物，不能宣称它是包体最大来源或给出确定减量；需用相同架构、配置、工具链做删除前后对照。

不该误删：

| 产物 | 大约体积 | 去留 |
|---|---:|---|
| `Voxt/Frameworks/libomnivad.dylib` | 3.6 MB | 留 |
| `Voxt/Resources/OmniVAD/*.omnivad` | 3.5 MB | 留 |
| `Voxt/VoxtIcon.icon/Assets/Voxt.png` | 4.0 MB | 本轮不动 |
| FluidAudio + 独占依赖 / 资源 | 待 Release 链接与资源审计 | 全部移除 |
| `llama.swift` binary | GGUF 运行时 | 留 |

隐藏模型权重是用户下载缓存，不随 App 分发。删除目录项主要降低维护成本和扫描范围，不等于从 DMG 中减去权重体积。FluidAudio 移除带来的二进制 / 资源减量，以 Release App、zip、DMG 分别实测为准。

## 5. 本地模型清理

### 5.1 清理规则

1. 可见列表保持“少而能解释”。
2. 隐藏模型不再兼容展示，整链删除。
3. 旧选择映射到同系列仍保留的主推模型；没有同系列则映射到默认模型。
4. 不扫描、不加载、不校验已删除 repo 的目录，避免设置页被历史缓存拖慢。
5. 不在 UI 上增加“卸载旧隐藏模型”向导。需要的话只在存储管理的现有清理入口里顺带忽略未知目录。

### 5.2 ASR：保留

继续作为默认可见目录，不改用户已经能看到的主选项：

| 阶段 | 模型 | repo | 角色 |
|---|---|---|---|
| 默认 / 低配 | Qwen3 0.6B 4bit | `mlx-community/Qwen3-ASR-0.6B-4bit` | 默认 |
| 平衡 | Qwen3 1.7B 6bit | `mlx-community/Qwen3-ASR-1.7B-6bit` | 主推质量 |
| 高精度 | Qwen3 1.7B 8bit | `mlx-community/Qwen3-ASR-1.7B-8bit` | 高质量 |
| Whisper 快 | Whisper Large v3 Turbo | `mlx-community/whisper-large-v3-turbo` | Whisper 主入口 |
| Whisper 准 | Whisper Large v3 | `mlx-community/whisper-large-v3-mlx` | Whisper 高精度 |
| Whisper 轻 | Whisper Small | `mlx-community/whisper-small-mlx` | 低资源 Whisper |
| 欧洲 25 语 | Parakeet v3 | `mlx-community/parakeet-tdt-0.6b-v3` | 英文 / 欧洲语 |
| 流式 | Nemotron | `mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit` | 流式 |
| 多语言事件 | SenseVoice | `mlx-community/SenseVoiceSmall` | 中英日韩 + 事件 |
| 会议说话人 | MOSS | `OpenMOSS-Team/MOSS-Transcribe-Diarize` | 会中带说话人 |
| 实时多语言 | Cohere 03-2026 | `beshkenadze/cohere-transcribe-03-2026-mlx-fp16` | 实时高质量 |

本轮不把可见 ASR 再砍到更少。再砍会改变现有设置页和已有用户选择，超出“流程不变”。

### 5.3 ASR：整链删除

从 catalog、capability、loader、live mode、hint、测试、文案全部删除：

| repo | 原因 |
|---|---|
| `mlx-community/whisper-tiny-mlx` | 被 Qwen3 0.6B / Whisper Small 覆盖 |
| `mlx-community/whisper-base-mlx` | 历史迁移残留 |
| `mlx-community/Qwen3-ASR-0.6B-6bit` | 和 0.6B 4bit、1.7B 6bit 重叠 |
| `mlx-community/Qwen3-ASR-0.6B-8bit` | 同上 |
| `mlx-community/Qwen3-ASR-0.6B-bf16` | 体积接近 1.7B，阶段不清晰 |
| `mlx-community/Qwen3-ASR-1.7B-4bit` | 夹在 0.6B 4bit 与 1.7B 6bit 中间 |
| `mlx-community/Qwen3-ASR-1.7B-bf16` | 过高资源，默认无区分度 |
| `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` | 全系列隐藏，会议还显式排除 |
| `mlx-community/Voxtral-Mini-4B-Realtime-6bit` | 同上 |
| `mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16` | 8.89 GB，不适合产品目录 |
| `Mediform/canary-1b-v2-mlx-q8` | 欧洲语已被 Parakeet v3 覆盖 |
| `UsefulSensors/moonshine-tiny` | 英文轻量被 Qwen / Parakeet 覆盖 |
| `facebook/wav2vec2-base-960h` | 英文 CTC 旧路径 |
| `facebook/mms-1b-fl102` | 9.6 GB 适配器模型，UI 解释成本高 |
| `mlx-community/parakeet-tdt_ctc-110m` | Parakeet 只留 v3 |
| `mlx-community/parakeet-tdt-0.6b-v2` | 被 v3 替代 |
| `mlx-community/parakeet-ctc-0.6b` | 解码变体 |
| `mlx-community/parakeet-rnnt-0.6b` | 解码变体 |
| `mlx-community/parakeet-tdt-1.1b` | 英文高配价值有限 |
| `mlx-community/parakeet-tdt_ctc-1.1b` | 同上 |
| `mlx-community/parakeet-ctc-1.1b` | 同上 |
| `mlx-community/parakeet-rnnt-1.1b` | 同上 |
| `mlx-community/GLM-ASR-Nano-2512-4bit` | 被 Qwen3 0.6B 覆盖 |
| `mlx-community/granite-4.0-1b-speech-5bit` | 语种少，和默认入口重叠 |
| `mlx-community/FireRedASR2-AED-mlx` | sherpa 已移除后的隐藏残留 |

删除后必须一起丢掉的代码，而不是留空分支：

- `MLXLiveMode.nativeVoxtralLive`
- `MMSLanguageAdapterOption` 全类型
- `MLXASRConfigurationCapability` 中的 `granitePrompt` / `canaryTask` / `moonshineDecoding` / `mmsAdapter` / `voxtralDelay`
- `MLXModelFamily` 中的 `graniteSpeech` / `voxtralRealtime` / `canary` / `moonshine` / `wav2vec2CTC` / `mmsCTC` / `lasrCTC`
- `MLXSTTModelLoader` 对应 `fromDirectory` 分支
- `MLXTranscriber` 里 Voxtral / Granite / Canary / Moonshine / Wav2Vec2 / MMS / FireRed / GLM-ASR 推理分支
- `ASRHintLocalTuning` 里只为这些 family 服务的字段和 UI
- `LocalModelSeriesClassifier.fireRedSeriesID` 以及 FireRed 分组排序
- 会议里 “Hidden support models are excluded from meeting optimization.” 这种 Voxtral 特判

旧 repo 迁移表只保留**仍存在的目标**：

```text
mlx-community/Parakeet-0.6B            -> mlx-community/parakeet-tdt-0.6b-v3
mlx-community/Voxtral-Mini-4B-Realtime-2602
mlx-community/Voxtral-Mini-4B-Realtime-2602-6bit
mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16
mlx-community/Voxtral-Mini-4B-Realtime-6bit
mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit
                                       -> mlx-community/Qwen3-ASR-0.6B-4bit
mlx-community/FireRedASR2
mlx-community/FireRedASR2-AED-mlx      -> mlx-community/Qwen3-ASR-0.6B-4bit
mlx-community/GLM-ASR-Nano-4bit
mlx-community/GLM-ASR-Nano-2512-4bit   -> mlx-community/Qwen3-ASR-0.6B-4bit
```

Whisper 短 ID 迁移保留，因为可见 Whisper 还在：

```text
tiny / base                            -> mlx-community/whisper-small-mlx
small                                  -> mlx-community/whisper-small-mlx
medium / large-v3-turbo                -> mlx-community/whisper-large-v3-turbo
large-v3                               -> mlx-community/whisper-large-v3-mlx
```

未知或已删 ASR repo，一律落到 `MLXModelCatalog.defaultModelRepo`。

### 5.4 LLM：保留

继续作为默认可见目录：

| 阶段 | 模型 | repo |
|---|---|---|
| 视觉 | Qwen3 VL 4B Instruct 4bit | `lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit` |
| 低配 | Qwen3.5 2B 4bit | `mlx-community/Qwen3.5-2B-4bit` |
| 默认 | Qwen3.5 4B OptiQ 4bit | `mlx-community/Qwen3.5-4B-OptiQ-4bit` |
| 高配 | Qwen3.5 9B OptiQ 4bit | `mlx-community/Qwen3.5-9B-OptiQ-4bit` |
| 中文备用 | GLM 4 9B | `mlx-community/GLM-4-9B-0414-4bit` |
| 非 Qwen 轻量 | Ministral 3 3B | `mlx-community/Ministral-3-3B-Instruct-2512-4bit` |
| 极轻 | LFM2 1.2B 4bit | `mlx-community/LFM2-1.2B-4bit` |
| 轻量 MoE | LFM2 8B A1B 3bit | `mlx-community/LFM2-8B-A1B-3bit-MLX` |
| 高端 | Qwen3.6 27B 4bit | `mlx-community/Qwen3.6-27B-4bit` |
| Gemma 轻 | Gemma 4 E2B IT 4bit | `mlx-community/gemma-4-e2b-it-4bit` |
| Gemma 平衡 | Gemma 4 E4B IT 4bit | `mlx-community/gemma-4-e4b-it-4bit` |
| Gemma 高配 | Gemma 4 12B IT OptiQ 4bit | `mlx-community/gemma-4-12B-it-OptiQ-4bit` |

可见 LLM 本轮也不再压缩。27B 已经在目录里，删它会改变现有高配用户选择。

### 5.5 LLM：整链删除

| repo | 迁到 |
|---|---|
| `Qwen/Qwen2-1.5B-Instruct` | Qwen3.5 2B |
| `Qwen/Qwen2.5-3B-Instruct` | Qwen3.5 4B OptiQ |
| `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | Qwen3 VL 4B |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | Qwen3.5 9B OptiQ |
| `mlx-community/Qwen3-0.6B-4bit` | Qwen3.5 2B |
| `mlx-community/Qwen3-1.7B-4bit` | Qwen3.5 2B |
| `mlx-community/Qwen3-4B-4bit` | Qwen3.5 4B OptiQ |
| `mlx-community/Qwen3-8B-4bit` | Qwen3.5 9B OptiQ |
| `mlx-community/Qwen3.5-4B-4bit` | Qwen3.5 4B OptiQ |
| `mlx-community/Qwen3.5-0.8B-OptiQ-4bit` | Qwen3.5 2B |
| `mlx-community/gemma-2-2b-it-4bit` | Gemma 4 E2B |
| `mlx-community/gemma-2-9b-it-4bit` | Gemma 4 E4B |
| `mlx-community/gemma-3-1b-it-qat-4bit` | Gemma 4 E2B |
| `mlx-community/gemma-3n-E2B-it-lm-4bit` | Gemma 4 E2B |
| `mlx-community/gemma-3n-E4B-it-lm-4bit` | Gemma 4 E4B |
| `mlx-community/Qwen3-30B-A3B-4bit` | Qwen3.6 27B |

保留仍有用的 alias：

```text
Qwen/Qwen3-8B-4bit                         -> mlx-community/Qwen3.5-9B-OptiQ-4bit
Qwen/Qwen2.5-7B-Instruct                   -> mlx-community/Qwen3.5-9B-OptiQ-4bit
mlx-community/Qwen3.5-2B-MLX-4bit          -> mlx-community/Qwen3.5-2B-4bit
mlx-community/Qwen3.5-0.8B-4bit-OptiQ      -> mlx-community/Qwen3.5-2B-4bit
mlx-community/Qwen3.5-0.8B-OptiQ-4bit      -> mlx-community/Qwen3.5-2B-4bit
```

`canonicalModelRepo` 负责旧 ID 映射，不把任意未知 repo 伪装成默认模型。不支持的选择由 FeatureSettings / manager 的选择校验回落到默认值；加载边界拒绝非目录模型，不需要新 UI。

### 5.6 参数与能力面收口

ASR 删除隐藏 family 后，能力枚举只保留实际会走到的项：

- live：`batchPreview` / `nativeQwenLive` / `nativeStreamingLive` / `nativeNemotronLive`
- 配置：`recognitionPreset` / `languageRouting` / `whisperTemperature` / `qwenContext` / `senseVoiceITN` / `cohereLongForm` / `mossPromptAndOutput` / `nemotronLatency`
- VAD 策略：`standard` / `preserveTimeline`（MOSS） / `modelManaged`（Cohere / Nemotron）

LLM 生成参数保持现有默认，不增加新旋钮：

- thinking 默认 off
- Qwen3 family `repetitionPenalty = 1.05`
- prefillStepSize 按 prompt 长度 256 / 512 / 768
- token budget 继续按任务类型封顶

如果 Swift 6.3 升级成功，再评估 mlx-swift-lm 新 API 是否能替换 `MemoryEfficientModelContainerLoader` 里的部分手工加载。**没有实测收益前，不删这个 loader。** 它是当前预量化模型省内存的关键。

## 6. 完全移除 FluidAudio，并独立审计 ONNX 残留

### 6.1 产品行为

独立会议说话人分离引擎只保留 Sortformer v2；MOSS 自带的说话人输出不受影响。FluidAudio 相关配置区域、选项、下载入口、引导文案全部移除，不只是隐藏。

用户侧变化应尽量无感：

- 功能设置里不再出现 “Offline VBx / Sortformer v2” 二选一。
- 保留 Sortformer 的下载和状态；参数仅保留经验证有效的项，不展示空转的人数约束。
- 旧值 `offlineVBx` 读取时直接当成 Sortformer。
- 引导页和功能摘要只显示 Sortformer。
- 不增加“你的说话人分离引擎已更换”的额外步骤。

这是本轮唯一允许的默认值变化，因为目标就是删 VBx。用静默迁移，而不是新的确认流程。

### 6.2 代码删除面

主要文件：

- `Voxt/Meeting/SpeakerAnalysis/MeetingSpeakerDiarizationEngines.swift`
  删除 `FluidAudioMeetingSpeakerDiarizationEngine` 整个 actor，包括 offline、streaming、fallback、共享实例和两套运行时配置。删除 `import FluidAudio` / `#if canImport(FluidAudio)`；工厂只返回 Sortformer。
- `Voxt/Meeting/SpeakerAnalysis/MeetingDiarizationModelManager.swift`
  删除 `downloadOfflineVBx`、`MeetingOfflineVBxModelStorage`。
- `Voxt/Meeting/SpeakerAnalysis/SpeakerDiarizationSettings.swift`
  `MeetingDiarizationMode` 变成单值或直接删除 enum，设置存储固定为 Sortformer。
- `Voxt/Settings/Features/FeatureMeetingSections.swift`
  删除 VBx 配置区域及引擎选择器，保留 Sortformer 下载状态与有效参数，不新增页面。
- `Voxt/Settings/Features/FeatureSettings.swift` / `FeatureSettingsStore.swift`、`Voxt/App/VoxtApp.swift`
  移除 VBx 默认值、双写和运行时读取。旧 `offlineVBx` 仅在一次性迁移中识别；迁移完成后不再写回。
- `FeatureAvailabilitySections.swift`、`OnboardingSettingsSteps.swift`、`OnboardingGuideView.swift` 及本地化资源
  清理 FluidAudio / VBx 名称、帮助文案、状态和入口；不增加引导步骤。
- `Voxt.xcodeproj/project.pbxproj` 和实际 `Package.resolved`
  删除 FluidAudio package、product、Frameworks build file 等引用；重新解析，确认 NemoTextProcessing 等独占产物不再参与构建。
- 测试、`tools/run_vad_damaged_cache_smoke.sh` 及使用文档
  去掉 VBx 默认假设和运行时测试，仅保留旧配置迁移用例，更新为 Sortformer 的有效行为。

旧 FluidAudio 下载缓存不再扫描、下载或加载，不自动删除用户磁盘文件。已有会议音频、转写、说话人标签和用户改名照常读取，不因引擎移除而重跑或清空。

灵敏度字段里 `fluidAudio*` 阈值全部删除。Sortformer 继续用现有 `threshold` / `minDuration` / `mergeGap`，必要时把现有三档灵敏度映射过去，不新增第四档。

### 6.3 清理 VBx 专用参数，不制造无效 Sortformer 配置

原方案把 Max 2–6 直接映射到 `SortformerConfig.numSpeakers`，这个结论撤回。上游默认值为 4 不等于模型支持任意改变输出人数；修改结构参数可能与已下载权重不匹配。

实施时：

1. 核对实际 Sortformer checkpoint、模型配置、权重输出维度和上游支持的运行时约束。
2. 删除 VBx 的 `offlineSpeakerBounds`、`fluidAudio*` 阈值和仅服务 FluidAudio 的配置传递。
3. 若没有受支持的人数约束 API，则删除相应人数提示控件和旧配置，不把它们假接到 `numSpeakers`，也不通过改名或随意合并标签伪造支持。
4. 灵敏度只映射到实际有效的阈值 / 后处理参数，并验证实时与终稿一致；无法生效的字段删除。
5. 验收覆盖 2 人、4 人、超过模型容量、重叠讲话、跨音频块身份稳定性。超过容量不得静默宣称已正确区分；若关键会议能力不满足，作为发布阻断项处理，不恢复 FluidAudio 后门。

复用现有设置区域，不增加用户操作步骤；简化参数不代表可以默认接受说话人分离质量退化。

### 6.4 完全移除的验收边界

同时检查源码、依赖解析和 Release 产物，不能只用一次文本搜索判断二进制已清理：

- 源码 / 工程：无 `import FluidAudio`、`canImport(FluidAudio)`、FluidAudio 引擎、下载和回退入口。
- 配置 / UI：不再提供 VBx 选项或保存 VBx 专用参数；旧值只允许出现在隔离迁移代码与测试中。
- SPM 图：FluidAudio 及其独占 binary / wrapper / 资源不再被 Voxt 引用；缓存目录里已有文件不等于仍被链接。
- Release App：用 `find` 检查 framework、bundle、模型文件名；用 `otool -L` 检查 Mach-O 动态链接，用链接映射 / 符号检查静态链接来源。普通 `rg` 默认跳过二进制，不能作为唯一门禁。
- 独立确认不存在 sherpa、ONNX Runtime 的运行时库及 `.onnx` 模型。若发现残留，追踪真实依赖来源，而不是预设它来自 FluidAudio。
- 对移除包资源、独占文本规范化组件、App / zip / DMG 分别记录前后大小。

历史分析文档、迁移测试、用户历史文本中出现名称不算运行时残留。`llama.cpp` / GGUF 保留。

## 7. ASR / LLM 链路优化

原则：先减分支，再调参数，最后才考虑换 API。用户路径保持“选一个模型，下载，使用”。

### 7.1 ASR 热路径

当前听写 / 会议共用 `MLXTranscriber`。隐藏 family 删除后，按 live mode 收成 4 条，而不是 12 条：

1. **Qwen native live**：听写预览 + 会议 native session。保留 KV cache 8bit / group 64 / quantizedStart 256。
2. **Nemotron native live**：cache-aware 流式。VAD 交给模型，`vadPolicy = modelManaged`。
3. **Streaming live（MOSS / Cohere）**：MOSS 保时间线；Cohere 可走模型内 long-form VAD。
4. **Batch preview（Whisper / Parakeet / SenseVoice）**：短窗口预览 + Final 全量。SenseVoice 继续用 Silero 切长音频。

会议 `MeetingMLXNativeLiveSession` 在删掉 Voxtral 后不再需要 “hidden support excluded” 异常。可见模型要么走 native live，要么走现有 batch 会议路径，选择逻辑对用户仍然不可见。

VAD 热路径去掉 `MainActorSync.run`：

- 核实后的现状：`activity` 已读取 actor 内缓存，不是每帧同步读主线程。同步桥接实际位于初始化 / 偏好刷新和 offline speechRanges。
- `LocalVADMode.stored()` 本来就是 `nonisolated` 的 UserDefaults 读取，可直接调用，删除多余 `MainActorSync`。
- 保留现有 refresh 和热路径缓存，不新增设置项。

### 7.2 会议 GPU 调度

`MeetingLocalInferenceCoordinator` 单车道是正确的，Apple GPU 上多模型并行更容易抖动。本轮不改成多车道。

只做两件小事：

1. live ASR feed 继续最高优先级。
2. 说话人分离、总结、详情翻译继续 `waitsWhileRecording`。
3. 删除 VBx 后，终稿说话人分离只跑 Sortformer，少一次 FluidAudio 模型加载和一次失败回退。

### 7.3 LLM 热路径

保持现有 `CustomLLMModelManager.runLocalPromptRequest`。可做的加速都是静默的：

1. 集成新 LM 的 cache 管理修复后，对比 TTFT、总耗时、峰值内存；不预设 `#620` 会缩短首 token 延迟。
2. 核实后的现状：`customLLMWarmupReposForIdle()` 没有调用者，本地模型实际按会话预热，idle 只处理远程连接。删除这个死方法，保留现有按需本地预热，不额外增加空闲模型加载。
   - 本地流式预览增加 50 ms 更新间隔，第一块立即送出，结束时补发尾块，减少主线程全文清洗与 UI 更新。
   - 不改变最终输出或添加用户设置。
3. `primeDownloadedStateCacheIfNeeded()` 只扫可见 catalog，不再扫 16 个隐藏 LLM + 25 个隐藏 ASR。
4. 生成参数不把 temperature / topP 暴露成新的必填项。高级 sheet 已经存在，保持原样。

### 7.4 Apple CPU / GPU 利用

现有策略已经对：

- 推理在 `Task.detached` 里跑，UI 在 MainActor。
- MLX 用 Metal；idle 时 `Stream.gpu.synchronize()` + `Memory.clearCache()`。
- 预量化权重避免 throwaway graph。

本轮不要引入 Core ML 双后端，也不要为 ASR 再包一层 GCD 线程池。真正的速度来自：

- 更少的模型 family 分支
- 移除 FluidAudio 的 offline → streaming 错误回退（现有代码并非运行时再回退到 Sortformer）
- 更新的 MLX cache / KV 实现
- 热路径不再同步等主线程

## 8. UI 交互与阻塞

目标不是重做界面，而是让现有页面在打开、下载、滚动、会议更新时不再卡。

### 8.1 必须修：设置模型页

`ModelSettingsObservation.contentWithLifecycle` 用多层 `AnyView` 包 `onChange` / `onReceive`。类型擦除增加理解成本，但仅凭存在 `AnyView` 不能断定它导致整页重绘。移除时按多个具名 `some View` 属性保留编译复杂度边界，实际刷新收益需 Instruments 验证。

改法：

- 去掉 `AnyView` 包装，直接链式 modifier。
- `refreshCatalogSnapshot()` 继续 debounce，但下载进度只更新对应行的 progress，不重建整个 catalog snapshot。
- `handleOnAppear` 里的 repo 规范化、provider 回退保持；磁盘扫描放到已有 manager 的后台路径，主线程只读缓存。
- 目录生成不再为决定“隐藏模型是否展示”遍历安装快照；缩小扫描集合。首次打开速度是否改善需要实测，不宣称已明显变快。

用户看到的仍是同一个模型目录，只是更顺。

### 8.2 必须修：会议详情列表

`MeetingDetailViewModel.refreshTranscriptListCaches()` 在每次分段 mutation 时重算 speaker ordinals、displayedSegments、speakerGroups。直播会议时这是主线程抖动来源。

改法：

- 直播更新走增量合并，已有 `updateLiveSegments` / `MeetingDetailVirtualList` 方向继续。
- 搜索、重命名说话人、删除分段才做全量重算。
- 不改用户可见的搜索 / 翻译 / 摘要交互。

### 8.3 建议修：模型目录信息

当前推荐徽章同时打在 Qwen3、Whisper、Gemma、Hy-MT2、部分远程供应商上。删除隐藏模型后，徽章规则应收成：

- 本地 ASR 系列：Qwen3
- 本地 LLM 系列：Qwen3.5（现有代码却给了 Gemma，这是不一致，应改掉）
- 专用翻译：Hy-MT2
- 远程保持现有，不在本轮扩大

这是降低选择成本，不是新功能。

### 8.4 明确不改的 UI

- 听写 overlay、快捷键、权限、历史列表的产品结构
- 远程模型配置 sheet
- 词典、提示词、增强策略
- 初版模型栈任务不重做引导；后续经明确要求纳入六步引导精简，见 `docs/OnboardingGuide.zh-CN.md`。它作为单独审查范围，不改变听写 / 翻译 / 会议的核心业务链路。
- 不为“高级用户恢复隐藏模型”增加入口

### 8.5 代码简洁性从删除中来

本轮重构的主收益是删除，而不是搬文件。

预计自然变短的文件：

| 文件 | 现在 | 删除后预期 |
|---|---:|---|
| `MLXModelSupport.swift` | 1102 | 能力表和 MMS / Voxtral / FireRed 大段消失 |
| `MLXTranscriber.swift` | 4117 | 去掉多个 family 特化 |
| `MLXModelManager.swift` | 1975 | loader 分支和扫描集合变小 |
| `CustomLLMModelSupport.swift` | 975 | 去掉 16 个 hidden 及展示表 |
| `MeetingSpeakerDiarizationEngines.swift` | FluidAudio 约占后半 | 只留 Sortformer |
| `LocalModelSeriesGrouping.swift` | FireRed 特例 | 删除 |

不要在本轮把 `MLXTranscriber` 强行拆成 10 个文件。先删分支，再看是否还需要按 live mode 拆。拆文件本身不给用户带来速度。

## 9. 接续实施批次（取代初版“最后才升 MLX”的顺序）

每批次保持业务入口、快捷键、输出语义和配置流程不变。阶段提交不等于每次 push；优先本地验证整个批次后，再统一推送一次。

### A. Voxt 接入已通过库级 CI 的 MLX 组合（P0）

范围：`Voxt.xcodeproj/project.pbxproj`、`MemoryEfficientModelContainerLoader.swift`、`CustomLLMModelManager.swift`、依赖审计、测试 / release workflow、实际 lockfile。

1. 按第 4 节同步固定 Audio / LM 引用并适配所有使用到的 API，不再等待发布 fork tag。
2. 保留现有预量化层加载，增加 loader 生命周期 / EOS / thinking / VLM / 输出尾块回归，不把新参数开放成额外 UI。
3. 将 fork 已验证的工具链修复应用到 Voxt 的本地验收入口和 CI：`xcode-select`、`SDKROOT`、compiler PATH 必须来自同一 Xcode；默认不移除编译检查或失败测试。
4. 实际解析完整 Voxt 依赖图、审计并入库 `Package.resolved`。保持 `-onlyUsePackageVersionsFromResolvedFile` 的后续构建语义。
5. 回归通过后删除一次性候选 patch / generator；正式源文件成为唯一实现。保留独立构建旧基线的能力，但不留用户可见的旧引擎开关。

出口：本地 Mac 的 Debug / Release build、全量 XCTest、默认 ASR / LLM / VLM 冒烟通过；没有 FluidAudio 重新进入依赖图。仅 Linux 静态通过时，本批次仍是“待 Mac 验证”。

### B. 模型安装状态后台化与设置页局部刷新（P1）

范围：ASR / LLM / GGUF / Sortformer manager 及 `ModelInstallSnapshots`、`ModelSettingsObservation`。

1. 先列出磁盘校验、目录枚举、大小统计、删除 / 移动及模型加载的实际执行线程；复用现有 manager / snapshot，不新增平行状态源。
2. 将耗时文件检查放到明确的后台隔离边界，MainActor 只发布不可变快照；`Task {}` 不等于离开 MainActor。
3. 扫描结果携带 storage-root / repo / revision，切目录、卸载、取消下载后的旧结果不能覆盖新状态。
4. 下载进度只更新行状态；目录成员只在安装 / 卸载 / 选择等生命周期事件变化时重建。状态未知沿用现有 loading 表达，不能临时误报“未安装”并触发重复下载。
5. 后台校验仍必须覆盖缺失分片、损坏 config、未完成下载和外置盘失联，不能靠减少检查来换取快感。

出口：慢盘 / 大缓存 / 切目录 / 同时下载删除的确定性测试通过；模型页无主线程目录遍历，打开和下载期间 UI 响应有 Instruments 证据。

### C. 会议 live 列表增量处理（P1）

范围：`MeetingDetailViewModel.updateLiveSegments`、`MeetingTranscriptListSupport` 和现有虚拟列表。

1. 现有翻译状态局部更新保留；新增针对 live snapshot 的稳定 ID / revision 差分，避免每次字幕变更都重新排序和分组全部历史。
2. partial 文本可被模型修订：不能假设只有末尾追加，也不能丢弃旧段修正。只重建受影响分组；搜索、全量替换、说话人改名等明确走全量重建。
3. 异步计算使用值快照及 revision，较早结果不可覆盖较新字幕。保留滚动锚点、编辑状态、搜索结果和翻译缓存。
4. 节流只作用于可替换的显示快照，不丢终稿、存储或错误事件。

出口：长会议、乱序 / 重复更新、partial 改写、翻译搜索、重命名、删除撤销和会话切换测试通过；实际更新成本与受影响数据规模相符。

### D. 推理所有权、背压、取消与内存回收（P1，先可靠再调速）

范围：`MeetingLocalInferenceCoordinator`、`MeetingMLXNativeFeedScheduler`、模型生命周期和 `IdleMemoryReclamation`。

1. 保留现有优先级有界车道，不未经测量就增加 GPU 并发。核对 permit 是否覆盖真实推理时段，而不只是 `feedAudio` 入队动作。
2. 审查当前 `try? withPermit` 后仍推进 `pendingOffset` 的路径：取消 / 拒绝必须显式处理，不得把未投递音频标成已消费。10 秒队列上限触发时也必须保留可恢复音频和清晰状态。
3. finish 必须排空已接收音频、保留最后结构化结果；cancel 必须停止投递、释放任务和 permit，不能无限等待 event stream。
4. 本轮采用更保守的回收设计：删除全局 `Stream.gpu.synchronize()` / `Stream.cpu.synchronize()`，后台仅清理 allocator 未使用缓存，不触碰在用模型或等待新会话的 GPU 工作。增加 pending load 阻断。若未来恢复显式全局同步，必须先建立覆盖全部使用者的独占权，不能只搬到 detached task。
5. 不用 `@unchecked Sendable`、无界 detached task 或每 token 全局清 cache 代替模型所有权控制。

出口：停止 / 取消 / 队列过载 / 模型切换 / 内存压力 / 回收时再启动等测试通过；不能丢音频或卡住 UI。尚未得到 GPU profile 的参数调整保持现状。

### E. 一次性本地验收包与 CI 复核（P0 发布闸门）

1. Mac 本地完成：语法 / 工程 / 依赖审计、Debug / Release build、全量 XCTest、选定模型回放和损坏缓存 smoke，生成结果摘要 / xcresult。
2. 确认迁移幂等且不动历史数据；旧 ID、缺失替代权重和下载失败沿用现有状态 / 重试入口。
3. 按第 6.4 节审计 Release 动态 / 静态链接及资源，记录 App / zip / DMG 体积。相同机型与配置测量 ASR 首包 / Final、LLM TTFT / 总耗时、峰值内存和 UI 主线程时间；无基线则不写提速百分比。
4. 本地集中修完同类问题后再统一提交 / push，一轮 CI 验证最终 SHA。CI 失败先阅读完整日志、批量排查，不逐条错误提交试跑。
5. 没有 Mac 环境时，明确停在待验证边界；不能用 tree-sitter、Python 测试或 fork 测试代替 app 类型检查。若只能使用 CI 完成 macOS 验证，先说明限制并集中准备后再运行。
6. 纯历史 merge 且 tree hash 与已通过版本相同才可跳过重复 CI；代码、依赖或 workflow 变更不能援引旧绿色结果。
7. 对所有退役 runtime 设防回归门禁；保留必要旧数据迁移测试，不为追求“零旧字符串”删除迁移安全性。

**Plan 完成定义**：A–D 的实施及对应测试完成，E 的 app 构建、质量、性能与包体证据齐备。允许 tag 晚于集成，但不允许把缺失的验证标成完成。后续要求已将 onboarding / 录音引导也纳入本分支：单独记录六步导航、配置草稿及真实体验状态的实现与验收，不把引导源码合入等同于 AppKit 交互已验证。

## 10. 迁移矩阵

| 旧状态 | 新行为 | 用户是否要操作 |
|---|---|---|
| 默认 ASR Qwen3 0.6B 4bit | 不变 | 否 |
| 默认 LLM Qwen3.5 4B OptiQ | 不变 | 否 |
| 可见目录里的其它 ASR / LLM | 不变 | 否 |
| 隐藏 ASR 仍被选中 | 静默迁到第 5.3 节目标 | 否 |
| 隐藏 LLM 仍被选中 | 静默迁到第 5.5 节目标 | 否 |
| 已下载的隐藏模型文件 | 留在磁盘，设置页不显示，运行时不加载 | 否 |
| 说话人分离 = Offline VBx | 一次性迁到 Sortformer，按现有安装策略准备模型；失败显示现有下载 / 重试状态，保存会议音频 | 无需重配；下载失败可能需要重试 |
| VBx 专用人数 / 阈值参数 | 迁移后清理；只保留 Sortformer 经验证有效的参数 | 否 |
| FluidAudio 缓存 / 历史会议 | 缓存停止使用但不自动删除；历史内容、说话人标签和改名保持 | 否 |
| 说话人分离 = Sortformer | 不变 | 否 |
| 翻译 = Hy-MT2 GGUF | 不变 | 否 |
| 远程 ASR / LLM | 不变 | 否 |
| 历史 `sherpa:` / `sherpaOnnx` / `whisperKit` | 继续迁到 MLX 默认 / Whisper 映射 | 否 |

## 11. 测试与验收

### 11.1 必须自动覆盖

- catalog：可见集合、canonical 映射、已删 repo fallback
- FeatureSettings：`sherpa:`、`offlineVBx`、隐藏 ASR / LLM 选择的迁移
- FluidAudio 移除：迁移幂等、旧参数清理、无运行时回退、替代模型缺失 / 下载失败
- Sortformer 实际支持的参数与容量边界；历史会议与说话人改名不受迁移影响
- 设置页生命周期：无 `AnyView` 回归可用现有 snapshot 测试收紧
- idle memory reclamation 现有测试保持
- MeetingLocalInferenceCoordinator 现有测试保持

### 11.2 必须真机覆盖，但不进默认 CI

用 `VOXT_RUN_MODEL_TESTS=1`：

1. Qwen3 0.6B 听写：开始说话到首包文字。
2. Qwen3 1.7B 6bit Final 质量抽检。
3. Qwen3.5 4B：增强、翻译、改写。
4. 会议：Sortformer 终稿说话人；MOSS 若已安装则 live 说话人。
5. 切模型、idle 卸载、再次录音，确认不会卡死设置页。

### 11.3 发布验收

- App 包内无 FluidAudio、无 onnxruntime、无 sherpa。
- 链接的 MLX 产品仍是 `MLXAudioCore` / `MLXAudioSTT` / `MLXAudioVAD` / `MLXLLM` / `MLXLMCommon` / `MLXVLM`。
- 不链接 `MLXAudioTTS` / `MLXAudioSTS` / `MLXAudioUI`。
- 对比同配置 Release App / zip / DMG 体积，记录 FluidAudio 和独占资源移除的实际收益，不预设降幅。
- 默认用户：装一个 ASR、一个 LLM，就能听写和增强，不需要先理解 VBx、hidden 模型或 ONNX。

### 11.4 性能对照（同机前后对比，不设拍脑袋指标）

在同一台 Apple Silicon Mac 上记录，作为阶段 4 是否合并的依据：

| 场景 | 记什么 |
|---|---|
| 听写首包 | 开始讲话到 overlay 第一段非空文字 |
| 听写 Final | 松键到插入完成 |
| LLM 增强 | 转写完成后到增强文本稳定 |
| 会议 10 分钟 | 平均 CPU、峰值内存、说话人分离耗时 |
| 打开模型设置 | 主线程到首屏目录可交互 |

没有前后对比，不接受“感觉更快”作为阶段 4 的合并理由。阶段 1–3 的合并理由是删除和去阻塞，不依赖 MLX bump。

## 12. 风险

| 风险 | 处理 |
|---|---|
| Xcode / SDK / JIT 子进程混用工具链 | fork 已验证 Xcode 26.5 / Swift 6.3.2；Voxt 同步统一系统 xcode-select、SDKROOT 和 compiler PATH，不只设置 DEVELOPER_DIR |
| Sortformer 在多人会议上的容量或质量不满足现有核心业务 | 以真实 checkpoint 和会议样本验证；不随意改 `numSpeakers`。关键退化阻断发布，修复替代链路，不恢复 FluidAudio |
| 有用户长期使用隐藏模型 | 静默映射到同系列主推。不在 UI 上挽留 |
| `MemoryEfficientModelContainerLoader` 与新 mlx-swift-lm API 不兼容 | 阶段 4 先适配 loader，失败则暂缓 lm bump |
| 缺失或不匹配的 `Package.resolved` | 集成批次在 Mac 生成并审查整个 app 依赖图后入库；不手写 / 拼接，不拿 fork lockfile 代替 |
| FluidAudio 移除不完整或混淆 ONNX 来源 | 源码、工程、resolved 图、链接映射、动态库与资源联合审计；旧名称只允许存在于迁移 / 测试 / 历史文档 |

## 13. 建议执行顺序（一句话）

已完成的源码清理不重复做；现在先接入通过库级 CI 的固定 MLX 组合，再完成安装状态后台化、会议增量更新与推理可靠性检查，最后集中进行 Voxt 本地验收及一轮 CI 复核。

用户仍然是“选模型、下载、说话”，不增加流程。只在有测试 / 实测证据时将对应项标为完成，不把 fork 通过、源码提交或静态检查当作产品交付完成。
