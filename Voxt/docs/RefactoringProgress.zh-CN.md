# 分阶段重构实施记录

关联：[全项目评估](RefactoringAssessment.zh-CN.md)、[源码地图](Architecture.md)、[回归矩阵](LocalRegressionMatrix.md)。

状态定义：**已实施**表示代码已修改并完成可用的静态检查；**已验收**必须包含 macOS 编译、XCTest 和相关人工检查。实施环境为 Linux，没有 Swift/Xcode；下面各批均不能仅凭静态检查标记为已验收。提交及 PR 状态以 Git 历史和 GitHub 为准；合入前仍需完成下述验收。

## 阶段总览

| 阶段 | 范围 | 实施状态 | 验收状态 |
| --- | --- | --- | --- |
| 0 | 远程 LLM 拆分、旧收尾流水线删除、首轮文档修正 | 已实施，见评估报告 | CI 测试工作流通过；Release / 人工待验收 |
| 1 | MLX 纯逻辑/独立推理边界、Onboarding、Settings 组件 | 已实施 | CI 通过；有状态驱动仍待阶段 5 |
| 2 | 核实的孤儿 UI、下载动作、会议空包装 | 已实施 | CI 通过；人工待验收 |
| 3 | 大测试套件、目录归位、回归入口维护 | 已实施 | CI 通过；人工待验收 |
| 4 | Remote ASR / 会议 provider 会话契约与去重 | 已实施，新增 36 个定向测试 | `fd38d43` macOS CI 通过；真实 provider 人工待验收 |
| 5A | 请求/启动任务、会议 session-token 与清理屏障、MLX 校正任务所有权 | 已实施，新增 20 个测试 | `b771be1` macOS CI 通过；设备/模型验收待执行 |
| 5B | 热键监听安装对象与业务状态、MLX native-live 任务/use 释放 | 已实施，新增 15 个测试 | `5936e62` macOS CI 通过；设备/模型验收待执行 |
| 5C | 录音身份/提交、模型加载退出、会议导入资源与最终化快照 | 已实施，新增 22 个测试 | `25776d2` macOS CI 通过；整体集成验收仍待执行 |
| 6A | Dictionary/History/MeetingDetail 拆分、孤儿调用链与 UI 清理 | 已实施 | `c55101a` macOS CI 通过；UI 待验收 |
| 6B | 模型/续传/GGUF 与远程配置职责、退役本地 LLM API 清理 | 已实施 | `1b393a3` macOS CI 通过；模型/下载待人工验收 |
| 6C | 文本交付代次/剪贴板所有权、历史/权限/连通性职责与孤儿链清理 | 已实施，净新增 20 个测试 | `2d62d99` macOS CI 通过；编辑器/UI 待验收 |
| 6D | 词典建议退役链、持久化兼容与远程设置校验/快照 | 已实施，新增 12 个测试 | `f1a1a0f` macOS CI 通过；UI/扫描待验收 |
| 6E 集中收尾 | 剩余热点复审、异步/取消边界、流式故障注入与 Release 门禁 | 已实施，新增 49 个测试 | `d817b78` 完整 XCTest / 无签名 Release CI 通过；真机/模型验收仍待执行 |
| 最终验收 | 真实设备/权限/provider/编辑器/模型与性能对比 | 外部验收未执行 | 见集中收尾清单；不标记全部已验收 |

阶段 0–3 的 `fa087ed` 已通过 [macOS CI Tests 工作流](https://github.com/hehehai/voxt/actions/runs/35433366619)。这是前一批的证据，不能替代阶段 4 新增行为的编译和测试，也不代表 Release 构建、模型回放及真实设备验收已完成。

阶段 4 的 `fd38d43` 也已通过 [macOS CI Tests 工作流](https://github.com/hehehai/voxt/actions/runs/35436703019)，日志包含 `TEST SUCCEEDED`。阶段 5A 的 `b771be1` 已通过 [macOS CI](https://github.com/hehehai/voxt/actions/runs/35438916477)，阶段 5B 的 `5936e62` 已通过 [macOS CI](https://github.com/hehehai/voxt/actions/runs/35440624827)。阶段 5C 的 `25776d2` 已通过 [macOS CI](https://github.com/hehehai/voxt/actions/runs/35443091401)。阶段 6A 的 `c55101a` 已通过 [macOS CI](https://github.com/hehehai/voxt/actions/runs/35445401328)。阶段 6B 的 `1b393a3` 已通过 [macOS CI](https://github.com/hehehai/voxt/actions/runs/35448285265)。阶段 6C 的 `2d62d99` 已通过 [macOS CI](https://github.com/hehehai/voxt/actions/runs/35452065137)。阶段 6D 的 `f1a1a0f` 已通过 [macOS CI](https://github.com/hehehai/voxt/actions/runs/35454152580)。集中收尾代码提交 `d817b78` 已通过 [完整 XCTest / Release CI](https://github.com/hehehai/voxt/actions/runs/35482326300)，发现 1,763 项：1,740 通过、23 项模型门禁跳过、0 失败。后续纯文档提交单独标明，不冒称其 SHA 与此 run 相同。

阶段 3 中不涉及行为的文件归位提前实施；这不表示存储及同步生命周期重构已完成。不得因为暂时没有 Mac 就把阶段 4–6 的风险或验收项删掉。

## 阶段 1：从大文件中分离职责

### MLX

`Voxt/Transcription/MLXTranscriber.swift` 从 3,855 行降至 2,274 行，新增职责文件：

| 文件 | 职责 | 行数 |
| --- | --- | ---: |
| `MLXTranscriptionModels.swift` | 规划、结果、回放使用的值类型 | 112 |
| `MLXTranscriptionPlanning.swift` | 采样选择、VAD 分段、校正节奏、Final 预算 | 318 |
| `MLXTranscriptMerging.swift` | 顺序合并、稳定前缀和隐藏预览合并 | 218 |
| `MLXLiveTextPreview.swift` | 原生语言选择、Qwen 协议头和可见文本 | 231 |
| `MLXCaptureBuffers.swift` | 原有加锁采样缓冲、VAD pre-roll、合并电平投递 | 250 |
| `MLXDetachedInference.swift` | 已解析配置及独立推理；保留 nonisolated 边界 | 377 |
| `MLXStructuredTranscript.swift` | live-ended / batch 共用的片段可靠性规则 | 77 |

- 两份完全相同的 `mergeStablePrefix` / `longestCommonPrefix` 算法合并复用；迁移前先比对原函数体。
- 任务创建、取消传播、会话 revision、模型 pin/unpin 和采样实例所有权仍由原 transcriber 持有；没有为了行数把运行时状态全面开放给扩展。
- **剩余 2,274 行仍是热点**。分离 native-live、capture 和 finalization 的状态所有权属于阶段 5，不能把此次提取描述成已完成生命周期重构。

### Onboarding

`OnboardingGuideView.swift` 从 2,868 行降至 440 行。

- 主视图保留 SwiftUI 状态及生命周期装配。
- `OnboardingGuidePermissions`、`Practice`、`ModelSelection`、`Configuration`、`Modals`、`Shortcuts` 按步骤/职责整理。
- `OnboardingGuideModelRows`、`Components`、`Styles` 与 `SelectableGuideTextView` 分离展示组件和 AppKit 桥接。
- 跨文件实际使用的成员改为 internal，文件内 helper 继续 private；没有更改练习的 session ID、焦点、确认保存、权限轮询或麦克风 watchdog 行为。
- 步骤扩展仍共享主视图状态，这是结构整理，**不是宣称已完成状态解耦**。后续提取子视图时要保持 SwiftUI state identity。

### Settings 外壳

`SettingsView.swift` 从 2,047 行降至 844 行。

- 分离 `SettingsSidebar`、`SettingsSidebarHeader`、`SettingsSidebarFooter`。
- 分离反馈弹窗和通知弹窗；反馈地址通过参数传递，不扩大外壳私有常量的访问范围。
- 导航及观察状态保留 private；通知列表自己的状态仍归通知视图所有。
- 外壳仍略高于 800 行复审线，后续继续按导航/页面装配边界整理，而不是再机械切开所有状态。

## 阶段 2：删除依据

| 删除 / 简化项 | 核实依据 | 保留覆盖 |
| --- | --- | --- |
| `GuideBullet`、旧 onboarding `hotkeyBinding(for:)` | 无调用；当前快捷键配置使用保留 trigger behavior 的 `shortcutBinding` | `OnboardingGuideTests` 及人工引导验收 |
| `GeneralModelStorageCard` | 全仓仅声明，当前设置/引导有实际使用的路径选择组件 | 设置与模型存储回归 |
| 4 个 `ModelDownloadPresentationSupport` 动作工厂及其 localized wrapper | 无调用；保留实际使用的 statusText 和 DownloadState | `ModelDownloadStatusSnapshotTests` |
| `RemoteASRMeetingConfiguration` | `hasValidMeetingModel` 与已有 `isConfigured` guard 完全相同；resolved 配置原样返回；其余配置/status helper 没有调用 | `MeetingStartPlannerTests`、`MeetingASRSupportTests` |
| 不可达 `.remoteASRMeetingUnavailable` 分支和仅供该分支使用的 provider 参数 | 前一个 guard 已排除相同条件；provider 身份仍在 RemoteProviderConfiguration 中 | 原 8 个 planner 测试保留，仅移除 9 处无用实参 |
| `transcribeMeetingChunk`、2 个未使用 typealias | 无调用；实际结构化 chunk 接口保留 | 会议转录测试 |

仍在用的 meeting aliases 保留；adapter 移至 `MeetingTranscriberAdapters.swift`。不改持久化枚举值、数据库迁移、本地化 key 或协议/selector 回调。

## 阶段 3：测试和目录

### 测试

四个大套件共 **230 个测试方法**按行为分组；迁移前后逐个核对方法体和断言，未删除测试：

| 原套件 | 原行数 | 归类 |
| --- | ---: | --- |
| HotkeyManagerTests | 2,588 | 通用派发、Note、修饰键、恢复、双击、粘贴、长按、鼠标、会话停止 |
| RemoteModelConfigurationTests | 1,752 | 通用配置、ASR、凭据读取/写入/迁移、Codex、端点迁移 |
| MLXModelManagerTests | 1,214 | 目录策略、生命周期、安装/存储；Custom LLM 配置单独成组 |
| MeetingDetailViewModelTests | 1,108 | 摘要、live 更新、文本/说话人编辑、翻译 |

共享 fixture 放在 `VoxtTests/TestSupport/*TestCase.swift`，不在基类声明测试方法；保留 MainActor、默认配置恢复、临时目录清理、受控 continuation 和原有 manager 保留策略。迁移不是修改测试的等待/调度策略。

阶段 3 结束时应用 425 个 Swift 文件、152,512 行，26 个文件仍 >1,000 行；测试 186 个 Swift 文件、39,560 行，最大文件 874 行，静态 `func test…` 数仍为 1,585。行数包含注释和空行；测试文本保留不等于 XCTest 已成功发现/运行。

### 目录

原样移动，不修改路径解析、数据格式或存储位置：

- `HistoryRepository`、`HistoryValueResolver` → `Core/History/`。
- `DictionaryRepository` → `Core/Dictionary/`。
- Note store、Obsidian/Reminders export store 和 sync coordinator → `Core/Notes/`。

Xcode 使用同步目录组，无需手工添加 Swift build phase 条目；仍需 Mac 验证 target membership。`Info.plist` 和资源目录未移动，音频夹具未修改。

### 回归脚本修复

`tools/run_local_regression_matrix.sh`：

- 删除开发者机器的绝对路径，改为从脚本位置解析仓库根目录。
- 增加 `refactor` 分组，包含 core 和拆分后的测试家族；旧 suite 名不再代表原整套覆盖。
- 删除指向不存在测试类的 `whisper` / `diagnostic` 分组；未知/已删除组返回非零，不产生“零测试通过”的假象。**不是删除 MLX Whisper 支持或迁移测试。**
- `all` / `full` 不再重复运行已包含在 core 的三个 VAD suite。
- 普通和 build-for-testing 路径均显式关闭签名并严格使用锁文件。
- 新增 5 项 Python CLI 测试，使用假的 xcodebuild 检查 suite 存在、覆盖、路径、参数、去重及失败传播；不冒充 Swift 测试。

## 阶段 4：远程 ASR 协议与完成契约

### 职责拆分与共享

- `RemoteASRTranscriber.swift`：3,547 → 1,259 行。文件请求、Aliyun 流、Doubao 流和响应投影分离；录音/generation 状态仍归 transcriber，剩余有状态驱动属于阶段 5。
- `MeetingRemoteProviderLiveSession.swift`：1,869 → 51 行，仅保留工厂；已有 Base / Doubao / Aliyun Fun / Qwen 类型整体分离。基类 561 行，各 provider 文件均不超过 211 行。
- 原 `RemoteASRSupport.swift` 的 11 个支持声明按文本解析、端点、Aliyun、StepFun、Gemini 原样归位，新文件最大 359 行。
- 会议与短句复用同一份 `DoubaoPacketCodec`，移除重复常量、帧构造、gzip、整数序号/文本提取处理和未调用的旧文件上传实现。保留 dictation 全文与 meeting utterance 时间片段的不同投影。
- Aliyun 端点解析复用已有 `RemoteASREndpointSupport`；PCM 转换复用原 `RemoteASRTranscriber` 的静态实现，不改变采样算法。会议的模型路由和认证头保持原策略。

### 有意改变的行为（不是仅移动代码）

1. **有界解压、拒绝损坏报文**：豆包 WebSocket 短句路径现在与会议一样限制压缩输入 2 MiB、解压输出 8 MiB、扩张比 64（允许 1 MiB 基础窗口）。损坏 gzip 不再当成普通文本回退；未知压缩类型明确失败。整个帧在复制/解码前也受尺寸限制。自建兼容服务的非规范响应需要人工复核。
2. **终包和序号**：区分 flag 2 无序号终包与 flag 3 负序号终包；不再把时间戳等任意 JSON 数字识别成 sequence，不让越界整数转换崩溃。非 JSON 文本回退及 JSON metadata 不会抹掉线上的终包标记。
3. **会议完成仅一次**：提前完成先记录终态；重复 finish 共用完成结果和截止时间，重复回调不会重复 `.finished`。超时从停止请求开始计时，保留原 1.8 秒预算，并覆盖握手/发送期间的等待。
4. **有序 drain 与取消**：握手完成时先发送已缓冲音频，再发送 finish；等待期间的 append 不会越过队列。取消清理队列，不把取消的 partial 提升为 final。失败时在 `.failed` 移除会话 token 前保存可用 partial，然后关闭 socket / receiver / keepalive。
5. **响应 actor 冻结终态**：5 类 provider 的最终等待传播取消，终态、超时返回或取消后不再接受迟到文本/错误。保留各 provider 的拼接规则及 StepFun / Gemini grace window。
6. **握手与错误隔离**：文件转录的握手 gate 记录成功/失败，增加 20 秒上限并响应取消；所有退出路径清理接收任务。错误回调绑定创建时的 generation，不能污染新录音；取消不触发 partial fallback 或交付。

### 新增回归与边界

| 套件 | 方法数 | 重点 |
| --- | ---: | --- |
| `DoubaoPacketCodecTests` | 13 | 帧布局、负序号、无序号终包、截断、gzip 限制、非零 Data 索引、两种文本投影 |
| `RemoteASRResponseStateTests` | 10 | provider 终态、partial drain、取消、迟到事件、握手成功/失败/超时 |
| `RemoteASRCompletionTests` | 5 | 有/无 partial 的失败、取消不交付、旧 generation 的结果/错误隔离 |
| `MeetingRemoteSessionLifecycleTests` | 8 | 提前确认、并发 finish、可控超时、握手/发送失败、取消 drain、重复完成 |

Fake 会话复用真实基类，通过既有 override 边界注入故障；截止时间可控，不依赖真实服务或麦克风。测试不等于真实 URLSession WebSocket、provider 账户/服务行为、设备或模型质量验收。远程 LLM 的 transport 故障注入不包含在本批。

`refactor` 回归组已包含这些测试和原有 ASR/会议协议覆盖。阶段 4 结束时应用 440 个 Swift 文件、152,041 行，24 个文件仍 >1,000 行；测试 190 个 Swift 文件，静态 XCTest 方法增加至 1,621。

## 阶段 5A：在途任务与会话所有权

本批优先减少平行状态和竞态，不为达成文件行数目标继续扩大 private 成员访问范围。

### 实施内容

- `TrackedTaskStore`：统一 MainActor 任务登记、取消和退出等待；每次 invocation 使用独立 ID。取消不删除在途任务，任务退出才注销，避免同一业务请求 ID 的旧任务清掉新任务。
- `LLMRequestLifecycle`：收敛 current request ID 与任务集合；AppDelegate 不再直接持有 `activeLLMRequestID` / `llmTasksByRequestID`。旧请求不能入队或执行，已取消但仍在清理的任务继续阻止深度空闲回收。
- 录音启动通过 `TrackedTaskStore` 管理；新启动等待所有旧启动退出后才操作同一 transcriber/音频引擎。取消只是请求，不能假设 CoreAudio 或权限请求立即终止。
- `MeetingLiveSessionRegistry`：session 与 token 成对存储，区分 active 与 draining。drain 期间的最终文本仍有效；被替换/取消后，旧 token 的 partial、final、failed、finished 都被拒绝。旧 finish 返回不能清掉新 session。
- 会议音频提交取消后仍被跟踪；移除 completed-ID 辅助集合和按 `isCancelled` 提前丢任务的逻辑。
- 会议资源清理现在捕获旧 transcriber/session/model use，按单一 cleanup 屏障等待在途提交与 scheduler drain 后再重置 VAD、archive 并释放 model use。新会议/文件导入等待旧清理完成，避免异步 cleanup 误取消新会话。
- 会话 revision 在清理时立即失效；capture epoch 在清理时递增，旧麦克风回调在修改电平/启动 watchdog 状态前被拒绝。chunk 推理 await 前后校验会话有效性。
- `MLXCorrectionPassCoordinator`：统一校正 pass ID、kind、task 所有权；被取消的 pass 保留串行槽位到真正退出，新 pass 不与旧推理重叠。父任务取消传播到实际推理，过期 revision 不启动/返回输出。
- MLX 最终化在 archive await 后再次核验 revision / cancellation，过期归档清理而不是接管新会话输出。未改变模型参数、校正策略和原生 stream 的模型 pin 规则。

### 测试与范围限制

新增 20 个确定性测试：`TrackedTaskStoreTests`（5）、`LLMRequestLifecycleTests`（4）、`MeetingLiveSessionRegistryTests`（5）、`MLXCorrectionPassCoordinatorTests`（5），以及 `MeetingCaptureTimelineTests` 的 cleanup epoch 用例。共享 `ManualTaskBarrier` 模拟忽略取消、仍需释放的在途原生操作，不靠 sleep 推测时序。均纳入 `refactor` 回归组。

这些测试覆盖提取出的所有者契约；尚不能替代真实 CoreAudio、完整 MeetingSessionCoordinator 启停、模型推理与内存回收的集成验收。本批主动改变了启动串行化、清理等待及旧事件拒绝语义，需要重点验证快速连按、取消后重启、暂停恢复、切换双音源、应用退出。

5A 结束时，热键监听和 native-live 的 task/use 所有权留到 5B；应用完整会话、模型管理器和会议导入/最终化继续作为剩余工作，不因通过单元测试而视为完成。

## 阶段 5B：热键监听与 native-live owner

### 热键

- `HotkeyEventTapInstallation` 拥有一个 tap、source、callback context 和独立 run loop。停止时从 manager 的路由锁内摘出旧 owner，在锁外等待旧线程退出；新安装的线程不会被旧 stop 操作关闭。
- CGEvent 回调不再直接携带未保留的 HotkeyManager 指针，而是使用安装对象持有的 context / weak manager。source 移除块保留 context 到回调线程执行清理，避免释放过程中遗留裸 manager 指针。
- `HotkeyEventTapRunLoop` 将启动中的 thread 也纳入条件变量管理。启动超时后 stop 先记录停止请求，即使线程稍后才开始，也不会再进入长期运行状态；停止后的 owner 不复用。
- deferred event / 恢复请求校验安装代次；主队列业务回调校验状态代次，stop/reset 后不再投递旧 action。实际 App callback 在路由锁外执行，避免重新引入 event tap 超时。
- 权限重试不在 sleep 期间强持有 manager；stop 使重试 ID 失效。取消的长按/双击 fallback task 在拿锁后再次检查取消，不能命中后来复用的 binding ID。
- 六种业务的 36 个平行字段归并为 `HotkeyBusinessState` 记录，统一状态读写和 reset；不改快捷键优先级、鼠标/修饰键/长按/双击算法。对 14 个核心路由方法逆向还原存储替换后，正文与上批一致。
- 删除 4 个未调用的旧 tap-cancel helper 和 `clearNoteTransientState`。`HotkeyManager.swift` 从 2,293 行降至 1,858 行，仍需继续按边界整理，不机械拆文件。

### Native MLX

- `MLXNativeLiveRuntime` 将 installed session、event/feed task 和转交的 model-use release 配对。替换先摘除旧 owner；旧 event 不能更新新状态，旧 retirement 不能清除新 stream。
- retirement task 等待 Voxt 的两个任务退出后恰好释放一次 model use；关机等待全部 retirement。它不是可跳过的普通取消任务，避免取消 cleanup 自身漏掉 use 释放。未显式关闭而被销毁的 owner 通过 isolated deinit 取消 stream，并仅捕获旧 entry 完成退出/use 释放，不在析构后捕获 self。
- native setup 复用 `TrackedTaskStore`，被取消的 setup 保留至退出；Qwen / streaming / Nemotron 三份加载和 pin-transfer 流程合并，具体模型的 StreamingConfig 保持原值。
- 空闲回收同时检查 setup 和 retirement，不能仅因 UI 已停止就销毁仍在退出的 runtime。`MLXTranscriber.swift` 从本批前 2,223 行降至 2,065 行。
- **依赖边界限制**：检查了固定 Audio revision 的源码，库的同步 `cancel()` 仅发出取消/关闭事件流，不提供等待内部 decode / Metal 工作完全退出的 API。因此这里保证的是 Voxt task/use owner 的退出顺序，不能声称已经证明底层推理完全静止；真正的 native quiescence 仍需模型回放及必要的依赖 API 支持。

新增 `HotkeyManagerLifetimeTests`（4）、`HotkeyEventTapRunLoopTests`（4）、`MLXNativeLiveRuntimeTests`（7），共 15 项；纳入 `refactor`，静态 XCTest 方法数为 1,656。run-loop 测试创建专用线程和普通 CF source，不安装系统 event tap、不请求权限；runtime 测试使用 fake stream，不加载模型。真实事件监听、权限恢复、睡眠唤醒和模型内存行为仍须人工/模型验收。

上述内容是 5B 的完成边界。5C 对录音身份、模型加载和会议导入/最终化的后续处理见下一节；阶段 6 的剩余大文件与最终回归尚未实施。

## 阶段 5C：录音身份、模型加载退出和会议导入/最终化

### 录音与结束流程

- `RecordingSessionLifecycle` 统一 session ID、取消、单次输出认领、正在结束及已结束标记；录音/选中翻译/失败复位/应用退出使用明确的 begin/cancel/invalidate 转移，不再分散写五个字段。
- 取消立即使旧输出无效，但保留取消前 ID 的清理资格；新会话开始后旧结束请求不能再清理新会话。旧 complete-end 回调也不能清掉新 ending ID。
- `SessionEndFlow` 删除固定顺序上的 protocol + 5 个 stage 包装，按原顺序直接执行隐藏界面、恢复音量、结束音、复位和残余捕获清理；171 → 98 行。
- 原静态 end-decision helper 在迁移后只剩测试引用，已删除；3 个已有 end-flow 测试改测真正的生命周期转移，保留原断言意图，未为删行而删除测试。
- 输出认领不再让已取消会话进入交付流程。实际文本注入、历史/词典快照和 UI 状态仍在原组件内；这不是宣称全部 AppDelegate 状态或外部编辑器事务已经解耦。

### 模型加载与关机

- `SharedModelLoadCoordinator<Value>` 归入 `Core/Models/`，去掉 `Any` 模型值和 `as! Value`，只对退出等待句柄做类型擦除。
- 分开“仍可共享给 waiter 的当前 load”和“取消后尚未退出的 load”。`cancelAll` 保留后者，后续应用关机仍能等待，不再依赖只覆盖特定调用路径的额外 termination 数组。
- 过期 generation 即使晚到的是错误而非结果，也转为 CancellationError，避免写坏替代 load 的模型状态。
- ASR / Custom LLM 的深度空闲回收检查改用 outstanding load，旧 cancelled native loader 退出前不再当作空闲。原 `hasPendingModelLoad` 保留当前 waiter 语义。
- 两个 manager 的重复 shutdown 调用共用完整退出 task，不只等待 active count 后提前返回。加载/下载/active use 全部退出后才释放缓存。`MLXModelManager.swift` 1,967 → 1,843 行。
- 不强制串行所有模型加载、不修改模型参数/目录，也不把 Swift load task 完成当作底层 Metal quiescence 证明。

### 文件导入与会议最终化

- `MeetingImportedFileAnalyzer` 在等待旧会议 cleanup **之前**就登记任务，关闭无法取消的空窗。cancel 捕获当次 pipeline/task；迟到取消不会命中新导入，调用方取消会传递到实际任务。
- `MeetingImportedFilePipeline` 独立拥有导入 transcriber、标准化音频路径和 model use，不再复用/修改 live coordinator 的 transcriber / active engine 字段。只有成功结果保留音频；失败或清理中取消会删除临时结果，清理可重复执行。
- 导入在清理结束前保持 busy；owner 析构也会取消该次任务及 pipeline。文件队列的延迟取消同样复核 task ID 和 cancelling 状态。
- `MeetingFinalizationContext` 固定停止时的 session ID、capture mode、引擎/模型、时长和 visible snapshot；三次 recovery checkpoint 与最终结果复用同一元数据，移除三份重复构造。
- finalization task 在 checkpoint 收尾完成前持续占用会议生命周期；重复 stop 返回同一 task，不能在旧任务尚未退出时开始新会议再被旧 task 清引用。
- `MeetingSessionCoordinator.swift` 1,996 → 1,821 行；文件导入代码移到独立资源所有者，不是仅把 coordinator 的 private 状态改成 internal 后拆 extension。

### 覆盖和未完成项

新增：`SharedModelLoadCoordinatorTests`（6）、`MeetingImportedFileAnalyzerTests`（7）、`RecordingSessionLifecycleTests`（6）、`MeetingFinalizationContextTests`（3），共 22 项；静态 XCTest 方法数 1,678。聚焦组同时补入已有文件队列和 recovery checkpoint 测试。

测试用受控 task barrier / fake pipeline 覆盖取消窗口、并发拒绝、清理中取消、旧任务隔离、checkpoint 元数据和单次结束。真实文件解码、模型/设备生命周期、并发 shutdown 的完整硬件路径仍依赖 Mac 集成验收。

阶段 5 的这组核心边界已实施，不能推导出所有异步路径已逐行审计或全部大类已拆完。剩余 UI/编辑器事务快照、模型下载状态、大文件、孤儿代码/测试去重和性能基准继续列入阶段 6，库内部 native 退出限制保留为明确待验收项。

## 阶段 6A：Dictionary / History / MeetingDetail

### 清理依据

- 词典学习里 `semanticChangeSummary` 的两个重载、候选聚类/评分/词缀扩展及专用类型/常量构成封闭的 private 子图，没有来自业务入口的调用。当前请求实际走 `typefluxChangeSummary`。共删除 **28 个 private 声明（约 760 行，含声明间空行）**，不是按单次符号出现机械删除。
- 保留活跃的 token/LCS 删除检测：`containsSemanticDeletionOnlyChangeGroup` 仍用于“等待用户完成替换”的观察策略，不能连同旧评分管线一起删掉。保留的 76 个学习声明逐个核对，除跨文件访问级别外正文不变。
- `DictionaryStore` 删除无调用的 `entriesByCategory`、`setCategoryExpanded`、`importProjectTerms`、`activeEntriesAcrossAllScopesForRemoteSync` 及旧项目导入结果类型；现行分类、热词、项目扫描/一键摄入和 JSON 导入入口保留。
- History 删除两个无调用的批量建议/修正快照更新入口、仅供旧入口使用的 suggestion merge helper 和两个孤儿 entry-copy wrapper；实际修正结果更新、历史追加和序列化字段保留。
- 删除 8 个孤儿 UI 类型：旧 `DictionarySuggestionRow` 及其独占容器/badge；旧 `DictionaryHeaderIcon` 及其独占图形；`MeetingPrimaryIconButtonStyle`。保留当前 toolbar 的 `DictionaryHeaderIconButton`，也不删除 AppKit/SwiftUI 协议回调。

### 职责拆分

| 原文件 | 本批前 | 本批后 | 新职责文件 |
| --- | ---: | ---: | --- |
| DictionaryLearningMonitor | 2,161 | 377 | Prompt / TextScope / TextDiff，最大 455 行 |
| DictionaryStore | 1,777 | 996 | DictionaryModels（410）、DictionaryStoreQueries（264） |
| TranscriptionHistoryStore | 1,344 | 917 | TranscriptionHistoryModels（371） |
| MeetingDetailWindow | 1,409 | 232 | WindowView（762）、PlaybackController（89）、PlaybackPane（315） |
| MeetingDetailViewModel | 1,206 | 981 | MeetingDetailPresentation（214）、MeetingTranscriptExporter（22） |

- Store 仍唯一持有 Published 状态、缓存、reload generation 和持久化写入。Query 扩展只读取不可变依赖；未把 Store 的 private setter 全面开放。
- Dictionary / History 的模型和 Codable 声明原样移动（除已无人使用的项目导入结果类型），保留 CodingKeys、旧字段回退、枚举 raw values 和数据路径；没有 schema migration，也没有删除历史兼容字段。
- MeetingDetail 的窗口/controller 所有权不变；播放器仍由主视图持有。播放控件子视图只接管自身开关/缩放/popover，scrubbing 通过 binding 与主视图同步；虚拟列表的 equatable 比较和回调原样归入现有 transcript components 文件。
- ViewModel 拆出只读展示策略，Published setter、异步任务、编辑/翻译/摘要变更仍在原所有者。并未把这次文件拆分描述成新的独立业务模块。

### 测试与规模

- 仅删除 1 个可证实冗余测试：`SessionEndFlowTests.testSessionEndExecutionDecisionAllowsFreshSession`。同文件 `testSessionEndExecutionDecisionRejectsDuplicateInFlightSession` 已在相同初始状态先断言首次 `.execute`，再断言重复结束；首次结束契约仍被覆盖。其余测试方法不变。
- 聚焦回归新增现有词典学习/匹配/异步 Store、历史 Codable/会话/修正、会议详情格式/虚拟列表/cache suite；没有新增只检验搬文件的 Swift 测试。
- 应用 Swift 行数 **151,965 → 150,761，净减少 1,204 行**（含注释和空行）；>1,000 行文件 **23 → 18**。测试方法 **1,678 → 1,677**，测试文件仍全部 <1,000 行。
- 剩余热点：MLXTranscriber、CustomLLMModelManager、HotkeyManager、MLXModelManager、MeetingSessionCoordinator，以及下载、远程配置、文本输入和历史设置等。Store / ViewModel 虽低于 1,000 行，仍高于 800 行复审线，后续只按真实职责边界继续整理。

本批 Linux 已完成声明/模型正文对比、孤儿引用/跨文件 private 检查和工具回归。**这些不是 macOS 编译或 UI 验收结果。** 重点人工检查词典分类/导入/自动学习、历史兼容数据，以及会议详情搜索/编辑/说话人、播放速率/缩放/高亮/滚动联动。

## 阶段 6B：模型、续传和配置边界

### 清理与保留

- Custom LLM 删除 7 个没有生产调用的旧重载（raw/system-prompt enhance、无 repo enhance、两个 translate、两个 rewrite）、3 个独占私有 helper，以及 5 个退役 request builder。现行 `executeCompiledRequest`、`enhance(userPrompt:repo:)`、词典扫描及真实模型生成测试路径保留。
- 删除只由旧测试引用的 `CustomLLMRemoteSizeCache` 与通用 `CustomLLMRepoSelection` helper；manager 原有“支持检查 → canonical repo / fallback”策略改为直接返回 String，不再装箱未使用的 requested/effective 字段。输出 JSON/resultText 解析仍用于现行编译请求，**没有一起删除**。
- ASR / Custom LLM 的 `sizeTask`、`prefetchTask` 从未创建实际任务；删除始终为 nil 的占位字段、取消/等待代码及无调用的 no-op API。真正执行工作的 VAD / diarization prefetch/size task 不动。
- 续传删除无调用的 HubClient 工厂、快照 getter、递归 partial purge、未使用 formatter，以及 delegate 的只写字段/参数。实际 URLSession 协议回调、ETag/Range/416、sidecar、重试与取消实现保留。
- 删除无人调用的 HTTP HEAD probe helper；`makeDownloadContext` 去掉未使用的 token/cache 参数。实际文件下载的 bearer token 路径保留，元数据请求的原有认证行为未改变。

### 职责整理

| 原文件 | 本批前 | 本批后 | 提取边界 |
| --- | ---: | ---: | --- |
| CustomLLMModelManager | 2,000 | 1,429 | Stateless request/runtime policy、tokenizer adapter、catalog values |
| MLXModelManager | 1,843 | 1,610 | STT factory、catalog values、通用 FileManager helper |
| MLXModelDownloadSupport | 1,387 | 491 | 续传 types（175）、delegate（210）、传输/sidecar（443）归 Core/Models |
| GGUFTranslationModelManager | 1,100 | 668 | GGUFTranslationRuntime（434），actor/底层资源代码原样移动 |
| RemoteProviderConfiguration | 1,294 | 965 | ModelOptions（229）和模型/端点 resolution（106） |

- `CustomLLMRequestRuntime`（255 行）只接收计划、设置和 tuning；container/model use、异步任务及 Published diagnostics 仍由 manager 管理，没有全面开放 private setter。
- `LocalModelManagerCatalog`（228 行）保留原有嵌套类型路径/值和 catalog 转发，加载和下载的可变状态仍在 manager。
- 下载源重试候选收敛到 `ModelDownloadSourceSelection.attemptCandidates`；保留 resume 固定已选源、fresh 按探测耗时尝试可达源的原策略。
- GGUF runtime 和远程配置的受保护 model/Codable 声明逐段比对原样保留。`RemoteStoredCredentialPresence`、presence 字段和 runtime configuration 构造器仍为同文件 `fileprivate`；没有为了拆文件弱化凭据边界。仅纯兼容性规范化 helper 跨文件共享。
- 没有重写下载算法、调整推理参数、修改依赖 pin、模型清单、数据格式或音频夹具。manager 尚超过 1,000 行的部分不以任意切片冒充职责解耦。

### 测试与规模

- 删除 4 个退役代码专属测试（两个旧 request builder、两个旧 size-cache helper）；两个旧 repo helper 测试改测实际 manager 选择/fallback。
- 新增 `CustomLLMRequestRuntimeTests` 6 项：编译请求任务映射/字段、结构化输出与文本回退、预算优先级、默认参数及 prefill 边界；下载源测试新增 3 项重试顺序/固定源契约。
- XCTest 静态方法数 **1,677 → 1,682**；不是为了维持数量而保留旧 API 测试。聚焦组补入现有续传/校验/模型目录/安装缓存及 GGUF 的三个非模型用例；installed GGUF 仍走显式模型门禁，避免重复执行。
- CLI 测试现在还校验 `-only-testing` 的方法名存在，而非只检查 suite，避免改名后静默选择零测试；source-selection 测试补充隔离 defaults 的 teardown。
- 应用 Swift 行数 **150,761 → 150,285，净减少 476 行**；>1,000 行文件 **18 → 15**。剩余重点仍是录音、模型管理器、热键、会议协调、远程连通性和设置大文件，以及 UI/编辑器交付事务与最终验收。

本批 Linux 工具检查、受保护声明/运行时正文对比通过；随后 `1b393a3` 已通过本批 macOS CI。真实暂停/续传、网络切换、下载源回退、模型安装/取消/卸载和 GGUF/native 内存行为保留人工/模型验收要求。

## 阶段 6C：文本交付与设置/连通性边界

### 删除依据与职责

- 核查提交/转译/重写生产入口、测试及 selector/协议路径后，删除无入口的预览注入/二次替换调用链：旧 finalize/preview/replacement helper、专属事务类型/字段、AX range setter 和 async callback bridge。现行一次提交、AX 读取、词典学习、选中文本读取保持。
- 删除无调用的结构化流式预览入口及 parser `preview` 包装、`extractRewriteAnswerPayload` 转发；四个旧预览测试改测实际 extractor/纯文本流式 normalizer，保留截断 JSON 与畸形 chunk 夹具覆盖。当前 conversation streaming、最终 payload 解析及仍被最终交付调用的空内容标题 helper 保留。
- 连通性删除 5 个不可达 helper：HTTP GET、通用 WebSocket、旧 Aliyun HTTP 探测、重复默认端点和日志目标 getter。保留实际 provider 分发、端点安全校验、请求/返回解析及凭据边界。
- 连通性拆为 ASR 请求、流式 ASR、LLM 和 WebSocket 传输/诊断文件。31 个保留函数正文比对不变；**未改超时、取消或网络资源策略**，也不宣称已经完成这些边界的运行时审计。
- 历史列表/删除值类型归入 `HistorySettingsData`，Note 控件与带编辑状态的 row 分开；row 的 footer 仍为同文件 private。权限探测输入/结果与 nonisolated native helper 独立，视图保留私有状态、取消任务和持久化，不改原探测/授权策略。
- 37 个权限函数与 24 个 History 类型按正文核对（仅必要可见性及等价本地化调用调整）；保留 AX 输入实现逐段比对不变。没有修改存储格式/迁移键、语言资源、模型参数、依赖 pin、签名或音频夹具。

| 原文件 | 本批前 | 本批后 | 边界 |
| --- | ---: | ---: | --- |
| TextInputIO | 1,203 | 839 | 保留输入/AX 读取；输出转入 TextOutputDelivery，删除旧替换链 |
| SessionTextIO | 776 | 630 | 清理旧链、合并答案注入、检查交付代次 |
| VoxtApp | 1,003 | 994 | 删除旧替换事务状态，装配剪贴板 writer |
| RemoteConnectivityTester | 1,284 | 57 | 验证/分发入口；协议 helper 分组 |
| HistorySettingsComponents | 1,119 | 389 | 普通历史行/工具栏；Note 控件和 row 各自归位 |
| HistorySettingsView | 1,072 | 966 | 保留列表状态与任务，提取值类型/工具栏 |
| PermissionsSettingsView | 1,007 | 813 | 保留 UI/任务；浏览器探测值与 native checks 分离 |

### 有意修正的交付行为

- `TextInjectionTransaction` 在排队工作真正执行前检查有效性，并对重复执行、重复/重入 completion 做单次保护。自动交付完成回调也检查原 session，再更新 UI、历史、词典和结束流程，不再让旧回调修改新会话。
- `RecordingSessionLifecycle.outputGeneration` 与回调 ID 分离：正常结束可继续已发出粘贴的后续按键；开始新会话、取消或关闭答案使旧代次失效。Auto Key 执行时还检查前台 PID。取消之后用户主动发起的新手动请求不被取消标记永久阻断。
- 手动答案注入合并两份重复流程，固定文本、目标应用和历史 ID；入口 Task、隐藏后的延迟粘贴及完成回调均检查代次，避免跨会话恢复旧窗口或更新新历史。
- `PasteboardTextWriter` 到实际写入时才读取原文本，按 token/changeCount 恢复；后续观察到的复制（即使字符串相同）不会被旧恢复覆盖。连续临时粘贴继承用户原始基线，不把前一次临时结果当成原剪贴板；保留结果模式废止旧恢复。
- **边界限制**：按键发出不等于编辑器确认接收，前台 PID 也不是编辑器/窗口级 ACK；仍沿用纯文本剪贴板恢复，不恢复富文本/其他格式，changeCount 检查不是跨进程原子 CAS。真实焦点、权限、快速连按及剪贴板并发仍需 Mac 验收。

### 测试与规模

- 新增 `TextInjectionTransactionTests` 8 项、`PasteboardTextWriterTests` 8 项及生命周期代次 4 项；静态 XCTest 方法 **1,682 → 1,702**。四个旧 parser 用例转向当前生产入口，不为保留测试而保留退役 API。
- 剪贴板测试使用独立命名 NSPasteboard 并释放，不读写 general clipboard；注入用受控回调，不发送真实键盘事件，也不是完整 AppDelegate/UI 端到端测试。
- 聚焦组增加两组新 suite 和现有 `RemoteProviderConnectivityTesterTests`；原安全、权限、历史、词典与会话测试保留，CLI 继续校验 suite/方法 selector。
- 应用 **483 个 Swift 文件、149,996 行**，本批净减少 **289 行**；>1,000 行文件 **15 → 9**。测试 **205 个文件、41,493 行**，均低于千行。
- 剩余热点：MLXTranscriber（2,065）、HotkeyManager（1,858）、MeetingSessionCoordinator（1,821）、MLXModelManager（1,610）、HotkeySupport（1,438）、CustomLLMModelManager（1,429）、RemoteASRTranscriber（1,259）、RemoteProviderSheetState（1,149）、DictionarySuggestionStore（1,128）。不以文件缩小宣称性能改善或全项目重构完成。

## 阶段 6D：词典建议兼容与远程设置

### 退役链与保留边界

- 当前录音后的 `previewDictionarySuggestions` 和 store 的旧 `discoverSuggestions` 始终返回空数组。删除这条空发现链、专属 draft / apply / evidence helper；`persistDictionaryEvidence` 收敛为原位置的 `dictionaryStore.recordMatches`。新历史仍写入空建议数组，**保留历史字段及旧快照解码**。
- 删除自动历史扫描 no-op 与三个调用点；显式一键扫描仍通过 `applyHistoryScanCandidates` 直接写入词典，编辑后自动学习是另一条活跃路径，均未删除。
- 核对 store 类型全部生产调用、测试与 UI 后，删除无入口的 pending/status/dismiss/add/bulk-add/reset/count API，以及始终为空且无人读取的非 Codable `snapshotsByHistoryID`。去掉 History 视图未使用的观察依赖、孤儿 scope label 和旧测试 factory。
- **没有删除旧 `dictionary-suggestions.json` 文件、Codable 字段、枚举 raw value 或迁移键**。旧文件的 reload、去重、状态优先级、证据合并和必要写回原样保留；这里只删除退役写入入口，不把存量兼容代码当死代码。Store 增加 defaults/文件路径注入，默认路径与行为不变，测试不访问用户数据。
- 建议/历史值归 `DictionarySuggestionModels`（99 行），过滤设置、旧 prompt 兼容及候选策略归 `DictionaryHistoryScanPolicy`（366 行）。可变进度、私有 Published setter、reload generation 与写回仍在 store；模型和策略正文定向比对不变。

### 远程设置

- `RemoteProviderSheetSnapshot`（217 行）负责配置、generation/ASR 值装配；`RemoteProviderSheetValidation`（233 行）负责端点与字段校验。父 SwiftUI view 继续持有状态；没有为了拆文件扩大 private setter 或宣称独立状态所有者。
- 删除 15 个无调用/退役菜单属性及 helper，其中包括 test-only OpenAI token 校验包装和重复整数解析转发；删除两项只用于退役 UI 的 `@State` 及初始化赋值。**Core 持久化 OpenAI/provider 兼容字段仍保留**。
- 一个旧 OpenAI 测试改测保存流程实际调用的 `validationMessageForGenerationSettings`，不再通过测试引用维持旧 API。
- 93 个保留声明正文核对通过；唯一表达式简化是用相同实现的 `parsedOptionalInt` 替代 Ollama 专属转发。credential edit intent、端点安全策略、参数/JSON 校验顺序、默认值及快照装配语义不变。
- 本批没有修改 Codex 模型列表加载或连接测试 task 的取消/迟到回调处理，也未重新设计 legacy async reload；这些运行时边界继续保留后续审查项。

### 测试与规模

- 新增 `DictionarySuggestionStoreTests` 8 项：旧枚举/证据 Codable round-trip、历史快照、去重写回、scope/状态优先级、直接扫描写入、成功 checkpoint、失败/取消进度及隔离设置持久化。复用 `TemporaryDirectory` / `TestDoubles` 并清理 defaults suite。
- 远程设置新增 4 项：空白/合法 token、负数/小数/溢出、ASR 跳过非活动 generation 字段、OMLX schema 格式校验；保留原测试，聚焦组加入上述两个 suite。
- `DictionarySuggestionStore` **1,128 → 389**；`RemoteProviderSheetState` **1,149 → 623**。应用 **487 个 Swift 文件、149,596 行**，本批净减少 **400 行**；千行文件 **9 → 7**。测试 **206 个文件、41,725 行**，静态方法 **1,702 → 1,714**，所有测试文件低于千行。
- 剩余千行热点为 MLXTranscriber、HotkeyManager、MeetingSessionCoordinator、MLXModelManager、HotkeySupport、CustomLLMModelManager、RemoteASRTranscriber。继续按实际所有权推进，不以文件行数证明性能收益。

## 阶段 6E：集中代码收尾

本轮不再把七个热点分别推迟到下一次“继续”。完成能够在当前环境实施的代码事项，建立 Release / 证据归档门禁，并明确真实 Mac 外部验收。完整处置理由及验收操作见 [集中收尾清单](RefactoringCloseout.zh-CN.md)。

### 异步边界与修复

- `RemoteProviderSheetOperations` 独立持有 Codex 模型列表/连接测试结果与任务代次。替换、关闭和本地校验失败使旧任务失效；取消后任务仍追踪至退出，延后 SwiftUI 回调复核结果身份。
- `ConnectivityWebSocketSession` 成对拥有专属 socket/session，关闭幂等；超时先关闭接收传输再等待 task-group 退出，取消覆盖 send/receive。原 provider payload/解析不改，解决仅 cancel task group 可能等待不返回的接收任务。
- 词典同步 reload 也更新代次；异步读取在后台进行并等待真实退出。取消/旧结果不发布，损坏/读取失败保留当前快照与原文件。旧文件模型显式声明值类型的 nonisolated/Sendable 边界，不改 Codable 字段。
- `RemoteASRPreviewController` 为每次循环独立保留去重状态；请求返回后复核录音/代次/取消，旧预览不能清除新会话状态或发布结果。完成任务及退休预览均可由关机等待；临时快照清理也覆盖 copy 失败。
- MLX 的 session/prewarm task store 保留取消后的旧循环、最终化、预加载与 watchdog；关机/空闲回收不只检查最新 task 槽位。采集启动取消请求 stop，但仍等待 native start 返回，不宣称硬超时退出。
- 会议 VAD 准备在旧 cleanup 后执行，纳入取消和退出等待；停止的 recording-active 更新纳入 finalization 顺序，移除独立游离任务。
- 下载大小校验绑定所下载 repo，避免 UI 切换模型后错用另一模型大小；Custom LLM 在 metadata await 前固定目录并检查取消。统一显示进度估计，整数转换前限幅；catalog 数值和校验容差不变。
- 热键偏好转换使用有界整数转换；损坏的负数/超范围值不再导致迁移崩溃。有效编码、迁移键、预设和手势路由规则保持。
- 为 `RemoteLLMRuntimeClient` 注入可选 URLSession，默认仍动态使用现有代理 session。隔离 URLProtocol 故障覆盖真实 Chat/Responses 执行入口；取消不再进入零 chunk 回退，partial 后失败不重试。

### 职责与保留决定

- 热键拆为值/匹配、左右修饰键、持久化/迁移及展示；仅跨文件必要的纯 canonical helper 调整访问级别，存量 Codable 正文核对不变。
- MLX 提取明确输入的 `MLXInferenceConfiguration`，推理预算/语言/预设/MOSS/Cohere 规则正文保持；会议提取 `MeetingLiveTranscriptPresentation`，不迁走 session token/翻译任务。
- Remote ASR 分离捕获格式/转换、request hint context、WAV 快照与错误展示；保留的捕获及上下文正文核对不变。
- 应用目前 **500 个 Swift 文件、约 150k 行**，仍有 **5 个**千行有状态协调文件。它们经过评审后保留私有状态、锁和资源生命周期，不以全面开放状态或机械切片冒充完成。准确行数随修复变化，以源码和收尾记录为准。
- 未变更依赖 pin、音频夹具、模型默认参数、持久化格式或签名配置。

### 验证

新增 49 项（设置 7、WebSocket 7、reload 6、ASR preview 4、热键 3、进度 3、推理规划 5、会议展示 5、repo 校验 2、LLM 故障 7），静态方法总数 **1,763**。聚焦脚本纳入新 suite；原回归覆盖不删。

CI 增加 unsigned Release build、xcresult/discovery/summary 及 `/usr/bin/time -l` 原始资源记录。首个收尾 run `35478088469` 因 `@concurrent` 随新增 helper 错位导致编译失败；已恢复至原异步校验方法。第二轮 run `35478584544` 又发现说话人模型下载仍引用旧进度 helper，已改接通用实现并复查全部调用点。第三轮 run `35479399460` 的 `68701e5` 已完成 1,763 项 XCTest：1,740 通过、23 个模型门禁跳过、0 失败；随后 Release 在 Swift 6.3.2 的 `EarlyPerfInliner` / 泛型 coordinator 析构处崩溃。将 Entry 改为独立 nonisolated 泛型值，保持任务类型安全及 Release 优化，等待重新验证。该 run 还显示冷实例 Remote ASR 完成测试耗时约 600 秒；清理改为只移除实际采集过的 input node，避免清理时懒初始化硬件。第四轮 `35481423733`（`39ed131`）XCTest 再次 1,740 通过/23 跳过/0 失败，冷清理测试降至 0.0013 秒；Release 仍在同一析构处崩溃，单独提取 Entry 不足以解决。最终改为成员级 MainActor 隔离，保留类型安全/私有状态及 Release 优化。第五轮 `35482326300`（`d817b78`）**XCTest 和 Release 全部通过**：1,740 通过、23 跳过、0 失败，新增 suite 与原 6 项加载协调器测试均已核对执行。冷清理回归 0.0025 秒。前面失败记录保留，不冒称失败 run 通过；详细命令统计及外部限制见收尾清单。

## 后续专项：启动模型检测与全仓静态复审

用户反馈首次打开误报模型未安装，本轮 `4e07cbb` 修复了 unknown→missing 错判和 Settings 外壳漏订阅 installationRevision；选择器检测中显示 Loading，保留不可选。共享 100ms 合并通知，不加轮询/主线程 scan，也不把检测中当作已安装。

同时进行了全仓只读收集及重点链路审查：716 个应用/测试 Swift 文件，区分词法孤儿候选与框架入口；确认删除 3 个孤儿 UI 类型、4 个普通/私有词典 helper、3 个无调用测试工厂，不删测试方法/数据格式。修复 structured metadata 密钥字段名未脱敏的问题，新增 8 个 Swift / 4 个工具测试。详细未解决风险与实测方案见 [全项目性能、安全与精简审查](PerformanceSecurityReview.zh-CN.md)。

`4e07cbb` 已通过 [完整 XCTest / Release CI 35490900072](https://github.com/hehehai/voxt/actions/runs/35490900072)：1,771 发现、1,748 通过、23 模型门禁跳过、0 失败；新增 suite 逐项核对通过。这不意味着 SEC-02 CDP 边界、长录音内存、同步导入/上传、子进程退出及真机 profile 已完成。本次新发现也说明前批“代码收尾”不能被解释为未来无缺陷或全部性能安全验收通过。

## 验证与下一门禁

Linux 已执行：

- Python 工具测试：10 项通过。
- Shell 语法检查、模型源码/锁文件审计、`git diff --check`：通过。
- 保留函数/测试正文、目录移动内容、删除引用与 Markdown 链接的静态核对。

集中收尾代码 `d817b78` 已完成完整 XCTest/Debug 测试构建和 unsigned Release CI。单独 `refactor` 聚焦组在 Linux 仅核对命令与 selector，尚未独立跑 Mac（其 suite 已进入完整 XCTest）。真实 Mac 外部验收仍未完成；以下命令供复现，不将未运行项写成通过：

```bash
xcodebuild build -project Voxt.xcodeproj -scheme Voxt -configuration Debug -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO
xcodebuild build -project Voxt.xcodeproj -scheme Voxt -configuration Release -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO
bash tools/run_local_regression_matrix.sh refactor
xcodebuild test -project Voxt.xcodeproj -scheme Voxt -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO
```

人工检查：六步引导的前进/后退/关闭、三种练习、权限和麦克风切换；设置导航、通知、反馈、历史/Note 编辑、词典一键扫描及远程 provider 保存/测试；会议远程启动配置；本地 ASR live/final/取消；延迟粘贴时取消/重启、关闭答案、正常结束后的 Auto Key、切换应用/窗口和连续用户复制。核对新 suite 的测试发现数量，不能只看 xcodebuild 退出码。

阶段 4–6E 的已识别代码事项已集中处理，LLM 流式故障注入已补齐；五个大型 owner 按评审决定保留。最终设备/模型/编辑器验收和运行时性能前后数据仍缺失。构建命令资源统计不代表应用/Metal 峰值内存，不宣称性能已提升或所有验收已经结束。
