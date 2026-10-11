# 上下文增强完整删除与权限精简方案

> 状态：**实施前方案基线，不是完成记录**。基于 `04962c1` 源码审计。
> 代码实施进度、与方案的差异及待验收项见 [实施记录](ContextEnhancementRemovalImplementation.zh-CN.md)。
> 当前环境为 Linux，未运行 Xcode、TCC 或音频设备真机验证。
> 权限依据与公开实现对照见 [权限最小化诊断](PermissionMinimizationAssessment.zh-CN.md)。

## 1. 推荐决策与最终目标

采用 **“完整删除上下文增强 + 基础流程权限最小化”**，不采用隐藏开关、默认关闭、保留实验入口的折中方式。

交付目标：

1. 全局 Fn/组合键使用已有辅助功能授权，**删除输入监控的申请、检查门槛、设置项及默认 HID 监听路径**。
2. 普通录音静音改为输出设备静音，**删除它对系统音频录制权限及 Process Tap 的依赖**。
3. 删除上下文增强的窗口内容/结构采集、截图、模型能力判断、图片请求、调试预览、持久化配置及专属提示词。
4. 因截图唯一业务入口被删除，**一并删除屏幕录制权限的请求、检测、引导与用途说明**。
5. 系统音频录制仅服务于真正采集其他应用声音的会议模式，按需授权；不再属于基础必需项，也不再为了静音而申请。
6. 基础本地/远程语音输入、Fn 快捷键、跨应用文本输出及普通设备静音，目标只需 **麦克风 + 辅助功能**。

这里的“删除那两个权限”指删除两个基础功能对它们的需求，不是破坏会议采集能力，也不是应用主动撤销用户的系统 TCC 授权。

**静音的明确语义**：静音当前输出设备，包括 Voxt 自身；不保留“只静音其他应用”的高级 Tap 后门。若以后再次需要该能力，按独立功能重新评估，不纳入这次默认方案。

## 2. 删除边界：避免同名能力误删

### 2.1 产品名称与源码映射

设置界面 `FeatureLanguageToolSections.swift:154–181` 的“上下文增强”位于 `rewriteContent`，实际控制：

```text
featureSettings.rewrite.appContext.textEnabled
featureSettings.rewrite.appContext.screenshotEnabled
```

源码中的 `.rewrite` 对应本次用户所指的转写/改写功能，不应只凭“转写”二字删 `.transcription`，而留下实际运行入口。

此外，`.transcription.appContext` 也保留着配置、执行计划参数和 `captureTranscriptionAppContextIfNeeded`。当前仓库符号搜索只找到该采集入口的定义，未找到调用点；应作为同一能力的遗留链路一起删除，实施时再做调用与资源引用复核。

### 2.2 删除与保留清单

| 能力/结构 | 处理 | 原因 |
|---|---|---|
| 上下文增强 Content Text / Screenshot 开关 | 删除 | 本次明确退出的产品能力 |
| 窗口 AX 树遍历、可见文本、窗口标题、控件摘要、自动附加选中文字 | 删除该采集服务及用途 | 不再把前台页面内容作为额外 LLM 上下文 |
| 窗口截图、临时 PNG、JPEG 压缩、图片预算、Base64 图片输入 | 删除 | 当前应用内图片输入的生产链路来自此能力 |
| 调试窗口自动抓取前台 App、上下文 payload、截图预览 | 删除 | 不能让调试入口绕过产品删除继续采集 |
| 普通识别结果润色、标点整理、翻译、口述改写 | 保留 | 不是上下文采集能力 |
| 用户选中文本后明确触发翻译/改写 | 保留 | 是用户操作指定的任务输入，不是额外读取整个页面 |
| 改写会话历史、继续对话、previousResponseID | 保留 | 已有会话内容不是屏幕上下文 |
| 改写答案的 JSON/结构化输出、Markdown 展示 | 保留 | “结构删除”指上下文数据/窗口结构，不是移除答案结构 |
| App Enhancement / App Branch 应用分组及提示词路由 | 保留 | 独立功能，依赖 App/URL 身份而非窗口截图 |
| 应用身份快照、输出目标 PID/bundleID | 保留必要部分 | 防止切换窗口导致输出或分组错位 |
| 词典、词典作用域、自动学习、ASR hint/context bias | 保留 | 与屏幕增强不同，部分仍需焦点文本或 App 身份 |
| 会议/文件转录、摘要、音频附件、模型下载 | 保留 | 非本次删除对象 |

**隐私说明边界**：不能宣传“Voxt 不再读取任何其他应用文本”。选中文本处理、输入交付和已启用的词典学习仍可能读取焦点内容；准确表述是“不再为上下文增强自动读取窗口内容或截图并附加给模型”。

## 3. 已确认的完整链路

### 3.1 正常执行

```text
FeatureLanguageToolSections：开启上下文增强
  → FeatureSettings / FeatureSettingsStore：保存两套 appContext 配置
  → FeatureSettingsView：隐私提示、截图授权请求

RecordingSessionFlow：记录目标应用身份快照（共享，不能整条删掉）
  → TranslationSupport：captureRewriteAppContextIfNeeded
  → LLMExecutionPlanning：按 provider/settings 决定采集
  → TranscriptionAppContextCaptureService
       ├─ AX：窗口/控件/选中文本/可见文本，遍历窗口树
       ├─ BrowserContextResolver：附加浏览器 URL
       └─ screencapture -l windowID：PNG → 压缩 JPEG
  → app context block + RewriteAppContextGuidance + image attachments
  → LLMExecutionPlan / LLMCompiledRequest
       ├─ Remote Responses：input_image + data URL
       └─ CustomLLM：CIImage → UserInput.Image → session.streamDetails(images:)
```

采集服务位于 `Voxt/Core/Transcription/TranscriptionAppContextSupport.swift`。它不仅截屏，还组合 App/Bundle ID、窗口标题、URL、焦点控件、选中文本、可见文本；这些内容结构均属于删除范围。

### 3.2 调试执行

```text
ModelDebugCore.valuesWithRuntimeContext
  → 独立采集前台 App 上下文
  → DebugRewriteAppContextPayload（含 Base64 图片）
  → ModelDebugSupport 再构造上下文、图片请求
  → ModelDebugWindowComponents：字符数、图片数、图片预览卡片
```

所以只删设置和正常录音链路不算完整删除。调试路径、专用序列化类型和请求 metadata 必须同批退出。

## 4. 文件级删除/修改清单

下列共享文件只删相关代码，不能整文件清空。

### A. 配置、界面与迁移

| 文件 | 操作 |
|---|---|
| `Voxt/Settings/Features/FeatureSettings.swift` | 删除 `TranscriptionAppContextSettings`；删除 `TranscriptionFeatureSettings`、`RewriteFeatureSettings` 的 `appContext` 字段、init 参数、CodingKeys 和解码 |
| `Voxt/Settings/Features/FeatureSettingsStore.swift` | 删除默认构造、sanitize、storageRepresentation 中的 appContext 传递；通过既有迁移流程重写规范 JSON |
| `Voxt/Settings/Features/FeatureLanguageToolSections.swift` | 删除整个上下文增强折叠区、两个 Toggle 和相关 Badge |
| `Voxt/Settings/Features/FeatureSettingsView.swift` | 删除启用敏感内容提示、截图授权请求、截图 Badge，以及只为它存在的状态；保留提醒事项刷新、其他 Toast/滚动行为 |
| `Voxt/Settings/Onboarding/OnboardingSettingsData.swift` | 删除截图需求参数与上下文增强关联；同时更新录音静音/输入监控提示 |
| `Voxt/Settings/Models/ModelSettingsReusableSections.swift` | 删除“自动注入 app context”的承诺，保留转录原文及词典说明 |

### B. 采集与运行时

| 文件 | 操作 |
|---|---|
| `Voxt/Core/Transcription/TranscriptionAppContextSupport.swift` | 整文件删除：采集结果、能力判断、AX 树遍历、窗口选择、截图子进程、图片压缩 |
| `Voxt/App/TranslationSupport.swift` | 删除采集 await、文本/图片额外预算及 buildRewriteExecutionPlan 的 appContext 参数 |
| `Voxt/App/LLMExecutionPlanning.swift` | 删除三处 capture helper、两种 plan builder 的 appContext 参数、`.app` block、上下文使用规则、附件传递和附件专用日志 |
| `Voxt/Core/TextPromptBuilders.swift` | 删除 `RewriteAppContextGuidance`；修正 direct-answer 中仍允许从 app context 推断的规则 |
| `Voxt/App/EnhancementPromptFlow.swift` / `Voxt/App/VoxtApp.swift` / Recording 生命周期 | 保留 App Branch、词典作用域及输出目标用到的轻量 App 身份；必要时重命名以区分身份路由和已删除的内容采集，不机械删除 `EnhancementContextSnapshot` |

删除采集 await 后，保留既有会话 generation、取消检查、模型使用锁和迟到结果拒绝规则；不能把本次删除变成生命周期重构。

### C. 统一 LLM 图片输入链路

当前源码扫描中，生产端图片来自上下文截图与其调试 payload；无独立“用户上传图片”功能入口。因此推荐连同 **应用层图片请求抽象** 完整删除，而不是永远传空数组。

| 文件 | 操作 |
|---|---|
| `Voxt/Core/LLM/LLMExecutionPlan.swift` | 删除 `LLMImageAttachmentDetail`、`LLMImageAttachment`、`LLMInputAttachment`、图片预算扩展、`.app` block kind、plan/compiled request 的 attachments |
| `Voxt/Core/LLM/LLMExecutionPlanCompiler.swift` | 删除附件编译透传；保留 input/glossary/conversation/metadata |
| `Voxt/Core/LLM/RemoteLLMMessages.swift` | 删除图片 data URL、`input_image` 构造、附件参数及附件独占 helper；保留 Responses 的纯文本、多轮消息格式 |
| `Voxt/Core/LLM/RemoteLLMRuntimeClient.swift` | 删除 request 附件分支，简化为纯文本消息；保留 previousResponseID、结构化答案和错误重试 |
| `Voxt/Core/Models/CustomLLMModelSupport.swift` | 删除 `CustomLLMRequestPlan` 图片字段及各 builder 传递 |
| `Voxt/Core/Models/CustomLLMRequestRuntime.swift` | 删除 `userInputImages`、CIImage 解码及只为此存在的 CoreImage import |
| `Voxt/Core/Models/CustomLLMModelManager.swift` | 推理仅传文字，删除从应用附件生成 images 的代码；模型加载分支另见依赖章节 |
| `Voxt/App/LLMExecutionPlanning.swift` 等 plan 构造点 | 删除附件参数，包括翻译、摘要、分段复制及测试构造中的空数组占位 |

不能一概删除远程 Responses API、图片以外的 content block、通用模型/音频附件或全项目中的 `context` 字样。

### D. 调试链路

- `Voxt/Windows/ModelDebugCore.swift`：删除 `valuesWithRuntimeContext` 及其异步采集调用、`captureRewriteAppContext`、`serializedRewriteAppContextCapture`。
- `Voxt/Core/Models/ModelDebugSupport.swift`：删除 `ModelDebugRuntimeValueKey` 的专用键、payload/attachment Codable 类型、app-context block、图片预算、附件预览拼接、截图字符/数量 metadata、`LLMDebugImagePreview`。
- `Voxt/Windows/ModelDebugWindowComponents.swift`：删除 app context/图片指标行和 `LLMDebugImagePreviewCard`。
- 保留模型调试本身、编译后文本预览、ASR 音频片段、VAD 图和用户手工输入变量。

### E. 提示词和本地化

- 修改 `Voxt/Resources/Prompts/{en,zh-Hans,ja}/*-rewrite.txt`：不再指示模型读取当前 App、界面文本或截图。
- 保留“有选中文本则处理源文本；否则依据口述及已有会话生成；缺少必要对象时请用户选中/提供内容，不编造”。
- “回复屏幕上最后一条消息”在没有选中文本、没有会话内容时将不再工作，应给出简短补充指引，不恢复自动采集。
- 更新 `AppPromptDefaults.swift` 的旧默认提示词 digest 迁移，不能把原默认模板误认成用户自定义而永久保留旧承诺。
- 清理英/简中/日 `Localizable.strings` 的功能专属键、隐私提示、截图和输入监控权限说明；共享的“内容”“结构”“上下文”“图片”等词只在确认无引用后删除。
- 更新 Rewrite/Prompt/Onboarding 使用文档、Architecture 和相关目录 README；历史审计/历史方案保留原证据并标记被本方案取代，不全库抹除历史。

## 5. 权限清理：三条明确结果

### 5.1 输入监控：从基础应用链路删除

修改：

- `HotkeyManager.swift`：删除双权限 guard、输入监控请求和 prompt 标记；辅助功能与实际 Tap 安装结果决定可用性。
- `HotkeyEventTapInstallation.swift`：保留 `.defaultTap`、独立 run loop、超时恢复和 generation 管理；优先级/位置调整须以 Fn 实测为依据。
- `HotkeyRecorderView.swift`：删除 `HotkeyRecorderHIDMonitor` 及 IOKit HID 输入监听；移除 `requestInputMonitoring` 和 `.listenOnly` 专用采集路径，改为本地事件优先，必要时辅助功能授权下的短期 Tap。
- `AccessibilityPermissionManager.swift`：删除输入监控 helper/status 字段；仍被使用的 AX 授权逻辑保留。
- 权限枚举、设置页、PermissionGuidance、两套 Onboarding、侧栏告警、日志与测试同步删除 inputMonitoring 分支。

验收不仅是“不弹窗”：输入监控从未授权/关闭时，Fn、组合键、录制快捷键、长按松开、双击、鼠标快捷键均应正常；Secure Input 与系统保留快捷键限制需正确降级，不引导多授一项权限解决所有问题。

### 5.2 普通静音：移除系统音频采集依赖

- 替换 `SystemAudioMuteController.swift` 的 Process Tap/聚合设备实现，使用可写的输出设备 `kAudioDevicePropertyMute`。
- 移除该控制器对 `SystemAudioCapturePermission` 的任何依赖；保留 begin/stop/cancel/error/quit 的既有调用点。
- 删除 `GeneralSettingsView.swift` 和配置式 Onboarding 中打开静音即请求系统音频权限的逻辑。
- 文案改为“录音时静音当前输出设备”，明确 Voxt 自身提示音也受影响。
- 状态保存使用设备 UID + 原始状态 + 会话所有权；原本静音不取消，用户更改不盲目覆盖，设备切换不恢复错设备，旧恢复任务不干扰新录音。
- 录音结束立即恢复，不等待 ASR/LLM；正常退出、取消、失败应幂等清理。强杀无法保证 cleanup，需保守恢复与提示策略。
- 本期主 mute 不可写时报告该设备不支持、继续录音；不偷偷触发 Process Tap 授权。音量降零后备可另加，但必须同样通过状态与设备验收。
- 开始提示音在静音前播放，结束提示音在恢复后播放；录音中的提示音改用视觉提示或接受无声，不临时解静音造成背景声音漏出。

**保留的会议例外**：`MeetingSystemAudioCapture.swift` 使用 `.unmuted` Tap 真正采集会议声音，与静音控制器是两条职责。保留它、`NSAudioCaptureUsageDescription` 及会议系统音频权限入口；用途说明改为仅描述会议。

`SettingsPermissionRequirementResolver` 不再无条件列出 systemAudioCapture。权限页可在“会议/可选功能”展示和说明；基础侧栏告警不因用户未使用系统会议采集而标记缺权限。麦克风-only 与导入文件流程不得被该权限阻断。

### 5.3 屏幕录制：随截图能力彻底删除

本次扫描中，屏幕录制请求和检查均服务于上下文截图，没有发现会议使用屏幕画面捕获。

删除：

- `SettingsPermissionKind.screenCapture`、`OnboardingContextualPermission.screenCapture`。
- `ScreenCapturePermission` helper，所有 `CGRequestScreenCaptureAccess` / `CGPreflightScreenCaptureAccess` 调用。
- `PermissionGuidance` 的屏幕录制导航分支、设置页按钮、Onboarding 分支及截图需求参数。
- `Voxt/Voxt/Info.plist` 的 `NSScreenCaptureUsageDescription`。
- 截图子进程调用、相关说明、默认提示词和测试中的截图需求断言。

**不要用 Screen Recording 的 API 代替 AudioCapture 查询**。即使 macOS 系统设置将屏幕与音频权限放在相邻/同组界面，也不能混淆两个功能的授权契约。

### 5.4 私有 TCC 治理

前次审计已发现 `SystemAudioCapturePermission.swift` 使用私有 `TCCAccessPreflight/Request`。本次联合方案包含治理项：

- 普通静音先完全脱离该模块。
- 会议经用户主动开始系统音源采集后，通过公开 Core Audio Tap 流程触发授权，按运行结果处理拒绝、重试及资源释放。
- 不能再让 SPI 不可用的 unknown 状态在启动前永久挡住公开捕获流程；也不能以本地缓存或全零 PCM 伪装成准确授权状态。
- 移除私有 SPI 的变更单独提交、单独真机验收，不与“截图删除已完成”混称为已验证。

### 5.5 最终权限矩阵

| 场景 | 权限 |
|---|---|
| 本地/远程基础语音输入 + 全局快捷键 + 自动输出 | 麦克风、辅助功能 |
| 普通输出设备静音 | 无新增采集权限 |
| 上下文增强/截图 | 功能和权限入口均不存在 |
| Apple Direct Dictation | 额外 Speech Recognition |
| 会议仅麦克风 | 麦克风 |
| 会议系统音源/混合音源 | 按需系统音频录制；混合另需麦克风 |
| 文件转录 | 文件选择授权，不需要麦克风/系统音频采集 |
| App Branch 浏览器 URL | 需要时按浏览器申请自动化 |
| Reminders 同步、通知 | 保留各自按需授权 |

辅助功能仍是快捷键、输入交付、选中文字等共用需求，不因上下文删除而移除。沙盒 audio-input、用户选择文件、浏览器 Apple Events 等 entitlement 同理，不能凭功能名称整批删除。

## 6. 依赖精简：独占链路删除，共享运行时不误删

### 6.1 本期推荐：保留模型目录兼容性

**删除上下文增强不等于必须删除所有多模态架构模型。** 当前 `supportsImageInput(repo:)` 被复用于决定模型加载工厂：

```text
CustomLLMModelManager
  → supportsImageInput(repo:)
  → MemoryEfficientModelContainerLoader.load(supportsVision:)
  → VLMModelFactory / processorRegistry
```

其 true 分支覆盖 Qwen3-VL、Ministral-3 和多个 Gemma-4 模型。即使新的请求只含文字，直接删除 MLXVLM 或把该判断一律改 false，也会把这些仍可用于改写/翻译的已支持模型错误地交给文字工厂加载。

推荐本期：

1. 删除面向截图的 `TranscriptionAppContextCapabilityResolver`、远程 vision 型号白名单、图片预算及输入 API。
2. 将目录中的 `supportsImageInput` 拆掉产品含义，改为内部的加载后端选择，例如 `loadingBackend = .llm / .vlm`；它只负责模型架构，不再对外表示应用支持图片功能。
3. 保留必要的 MLXVLM 工厂与模型 processor；验证已有模型的纯文字请求可用。
4. 删除模型目录中的“适合截图/图文上下文”推荐和 Vision 功能标签，改为实际支持的文字任务描述；不要强迫用户重新下载或切换已选模型。

这属于保留其他文字任务共享的架构依赖，不是保留上下文增强链路。其体积/加载开销仍存在，不能声称本期已完全去除视觉模型运行库。

### 6.2 如果后续还要求去掉 MLXVLM 二进制依赖

需新增一个**模型目录收缩**变更，而不是顺手删 import：

- 为上述五个 VLM 路径模型逐一验证是否有兼容的纯文本加载方式；没有则退出支持。
- 为所有引用这些 repo 的转录、翻译、改写、会议、笔记标题等选择设置设计替代/用户选择流程。
- 调整 repo alias、生成参数、目录展示、下载、安装状态和模型验收脚本；不能把一个模型的调参直接复制到另一个模型。
- 保留磁盘已有权重供用户手动清理，不在升级时强删；不自动下载数 GB 替代模型。
- 无剩余加载消费者后，删除 VLM loader、trampoline、processor 配置读取及 Xcode 的 MLXVLM product/build references。
- `mlx-swift-lm`、MLXLMCommon/MLXLLM、MLXAudio 和 mlx-swift 仍有其他消费者，不能删除整个包或清空 lockfile。

该变更风险和用户影响明显大于上下文删除，**不建议与本期基础能力删除强绑定**。

### 6.3 其他依赖去留

| 依赖 | 决策 |
|---|---|
| ImageIO、CoreImage | 删除功能独占 import/转换代码；它们是系统框架，不宣称因此删除一个 SwiftPM 包 |
| UniformTypeIdentifiers | 文件导入、拖放、词典/日志导出仍使用，保留 |
| FaviconFinder | App Branch URL 图标仍使用，保留 |
| PermissionFlow / SystemSettingsKit | 麦克风、辅助功能、自动化、会议等仍使用，保留 |
| ApplicationServices / AppKit / CoreGraphics | AX 输出、快捷键、UI 等仍使用，保留 |
| MLXAudio / MLXLLM / MLXLMCommon / llama.swift | ASR/文本 LLM/GGUF 仍使用，保留 |
| Core Audio Process Tap | 从静音控制器移除；会议采集保留 |

包引用删除必须以剩余 import、加载路径、target product 和真实构建证据为准，不以全仓库字符串出现次数为准。

## 7. 配置、提示词与数据迁移

### 7.1 配置迁移

- 删除 Codable 的 appContext key 后，旧 JSON 中的未知字段应被忽略；用原始旧 JSON fixture 测试，避免删类型后测试无法构造旧数据。
- 覆盖旧 `enabled` 单开关格式及 `textEnabled/screenshotEnabled` 子开关格式，分别测试两套 feature 配置。
- 使用 `FeatureSettingsStore.migrateIfNeeded` 的既有显式写入点重存，移除旧字段；保持 `load()` 无写入副作用，防止 SwiftUI 读写循环。
- 不新建“已移除能力”永久兼容类型或迁移专用运行时后门；测试 fixture 中保留旧键足够。
- 确保模型、用户提示词、快捷键、笔记同步和其他功能开关不被整体恢复默认。

### 7.2 默认提示词迁移

审计基线三份 rewrite 默认模板的 trim 后 SHA-256：

| 语言 | Digest |
|---|---|
| en | `896482a35261df59cd0e37f50265e7ca9b28b942796f250381bf26d5bbbf0b74` |
| zh-Hans | `d662ecb5d34cada98d06f112cd51b7bd299b5114f5531a5b8d21a0a18d08c4d1` |
| ja | `5ed3a58145fa218cac67277b95042d84a4d153eb30de7e411668555d4378972f` |

修改模板前重新核对；将原默认加入既有 legacy digest 集合，确保升级读取的是新默认。用户修改过的提示词必须逐字保留，不能凭含“截图/上下文”就删除。发布说明提示这类用户自行调整旧自定义规则。

### 7.3 残留文件、日志和授权

- 采集服务目前用 `defer` 删除 `voxt-app-context-UUID.png` 临时文件；异常强杀可能遗留。可增加一次性、受限于本应用临时目录和严格文件名模式的清理，不递归扫用户目录。
- 调试结果当前主要保存在内存；本次审计未发现专门的持久化截图图库，不应凭空设计数据库破坏性迁移。
- 原有调试日志可能已包含窗口文本、URL 或提示词，功能删除不会追溯擦除日志、导出文件或远程服务已收到的数据。沿用日志清理能力并说明边界，不自动清空用户历史。
- 不删除转录历史、改写会话、词典、用户模型权重或会议录音。
- 停止请求后，macOS 里旧授权记录可能仍存在；提供可选的手动撤销输入监控/屏幕录制指引。系统音频授权是否撤销取决于用户是否需要会议系统音源。

## 8. 分阶段实施与提交边界

| 阶段 | 工作 | 完成标准 |
|---|---|---|
| P0 基线与用例 | 固定旧 JSON、默认 prompt digest、请求 fixture、权限矩阵；确认用户已有未提交修改 | 有可重放的升级/行为测试，不混入会议文件任务工作 |
| P1 上下文能力整链删除 | 同一可构建变更覆盖 UI/config/采集/runtime/debug/prompt/图片输入 | 没有隐藏采集入口、空附件壳或无法编译的中间提交 |
| P2 屏幕/输入监控权限删除 | 删截图权限整链；改快捷键管理器和录制器 | 两权限均不再索取，已有快捷键语义通过测试与真机验收 |
| P3 无采集静音 | 改控制器、开关请求、状态恢复、设备不支持提示 | 系统音频拒绝时普通静音和转录仍可用，不影响会议代码 |
| P4 会议授权与策略统一 | 消除全局 systemAudio 必需假设；移除 TCC SPI；按模式和用户意图授权 | 普通/麦克风会议/文件导入不受影响；系统会议首次授权/拒绝/撤权正确 |
| P5 依赖与文档收尾 | 加载后端命名、目录文案、独占 import 清理、测试替换、使用文档更新 | 清单逐项有删除/保留依据；完整构建和目标模型纯文字冒烟通过 |

若 P4 的公开授权行为尚未通过真机验证，应明确标记该阶段未完成；不能用“静音已不索权”冒充私有 TCC 已移除。

回退按提交/版本执行，不通过暗藏上下文开关实现。迁移只移除退出功能的配置，旧版本解码缺失 appContext 默认关闭；自定义 prompt/其余偏好保留，确保回退不会丢失核心数据。

## 9. 测试与验收

### 9.1 自动化测试调整

| 测试 | 处理 |
|---|---|
| `TranscriptionAppContextSupportTests` | 随服务删除，不保留仅为已移除能力而存在的图片压缩测试 |
| `FeatureSettingsStoreTests` | 删除开关持久化断言，新增旧 JSON 可解码、重存无 appContext、其他偏好完整且迁移幂等 |
| `SettingsPermissionSupportTests` / `OnboardingSupportTests` / `SettingsTypesTests` | 不再期望输入监控/屏幕录制；会议需求按模式；普通静音不要求 systemAudio |
| `LLMExecutionPlanCompilerTests` | 去掉图片与 app block 测试，保留词典、会话和纯文字编译断言 |
| `RemoteLLMRuntimeClientMessagesTests` | 删除图片/模型 vision 能力测试，补纯文字首轮、previousResponseID、多轮回放 payload 精确断言 |
| `CustomLLMModelConfigurationTests` / `CustomLLMRequestRuntimeTests` | 删除图片字段 fixture；保留文字请求/参数/历史完整性 |
| `CustomLLMModelSupportTests` | 将图像支持测试改为模型加载后端分类，不把 VLM 模型误切到 LLM 工厂 |
| `ModelDebugSupportTests` | 删除截图 payload 测试，补 rewrite 调试请求不执行捕获且仅包含显式输入/历史 |
| `AppPromptDefaultsTests` / `PromptBuildersTests` | 新默认无屏幕依赖；三语言旧默认自动升级；自定义 prompt 不改；缺失目标正确指引 |
| Hotkey 系列与新增设备静音测试 | 不申请 ListenEvent；Fn 手势不退化；mock 设备验证只恢复自身修改、失败/设备切换/连续会话 |

不把纯字符串搜索当成行为测试；不能通过测试时真实更改宿主机静音状态。权限与设备副作用使用可替换接口/测试桩，保持 UserDefaults suites 和临时目录隔离。

### 9.2 残留检查

在生产 Swift/Info.plist 中检查以下专属符号应归零（迁移 fixture、负向测试和历史方案允许出现）：

```text
TranscriptionAppContextSettings / TranscriptionAppContextCaptureService
TranscriptionAppContextCapabilityResolver / RewriteAppContextGuidance
DebugRewriteAppContextPayload / __VOXT_DEBUG_REWRITE_APP_CONTEXT_CAPTURE__
LLMImageAttachment / LLMInputAttachment / LLMDebugImagePreview
CGRequestScreenCaptureAccess / CGPreflightScreenCaptureAccess
NSScreenCaptureUsageDescription / /usr/sbin/screencapture
CGRequestListenEventAccess / CGPreflightListenEventAccess
HotkeyRecorderHIDMonitor
```

同时审查：

- 生产配置无 `.transcription.appContext` / `.rewrite.appContext`。
- 应用请求构造无图片 `input_image` / data URL 路径；纯文本 Responses content blocks 可以保留。
- `SystemAudioMuteController` 无 CapturePermission、Process Tap、Aggregate Device 依赖。
- `AudioHardwareCreateProcessTap` 在会议中允许保留，不能要求全仓库归零。
- `.app` 是多个枚举共享名字，不能要求全局归零；只删 `LLMContextBlockKind.app`。
- VLM 工厂按共享模型边界保留；如果执行额外模型目录收缩，再单独验收 MLXVLM 引用归零。

### 9.3 构建与真机门禁

```bash
xcodebuild build -project Voxt.xcodeproj -scheme Voxt -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO
xcodebuild test -project Voxt.xcodeproj -scheme Voxt -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO
```

自动化先运行上述重点测试，再运行全套；未签名测试通过不等于发布授权行为通过。

发布等效签名/沙盒下，在 macOS 15、26 和干净账户验证：

1. 基础配置首次使用只需麦克风/辅助功能；拒绝输入监控、屏幕和系统音频仍可完成普通转录/改写与支持设备的静音。
2. 快捷键运行、设置录制、后台、Fn/Globe 系统冲突、长按松开、休眠恢复、Secure Input 的行为。
3. 上下文旧开关为 true 的升级用户，正常与调试入口均不读取整个窗口、不启动 screencapture、不发送图片。
4. 有选区改写、无选区生成、继续对话、字典作用域、App Branch、文本注入和自动学习均不回归；无法定位屏幕内容时不编造。
5. 至少一个远程 Responses、一个 Chat Completions、本地 LLM，以及每个保留 VLM 加载家族的纯文字请求成功；关注历史与结构化输出未丢失。
6. 内建、蓝牙、USB、HDMI/DP/虚拟输出；原本静音、用户中途调整、设备切换、重复录音、启动失败、正常退出/强杀后的恢复边界。
7. 会议麦克风、系统、混合模式以及文件导入分别验证；只有实际使用系统音源时需要系统音频授权。
8. 设置/Onboarding/侧栏三处显示一致；关闭可选功能不会一直出现基础权限缺失告警。

## 10. 收益与不承诺项

可以确认的结构收益：删除整套上下文采集服务、截图子进程、AX 页面遍历、图片压缩/序列化、provider vision 白名单、调试截图链路及屏幕授权分支；降低敏感窗口信息进入提示词/日志的机会。

运行时收益主要发生在原先开启上下文增强的用户：不再等待截图和 AX 遍历，不再增加这部分 prompt/图片预算。默认未启用用户的提速可能有限。

不承诺未经测量的包体、内存、延迟下降数值；共享 MLXVLM 若保留，不能宣传视觉运行库体积已消失。也不承诺所有音频设备都具备可写静音属性。

**最终验收口径**：上下文增强在 UI、配置、正常执行、调试、请求和截图权限上全部退出；基础流程不依赖输入监控或系统音频录制，屏幕录制不再申请；其他功能仅保留其确实需要的按需权限。
