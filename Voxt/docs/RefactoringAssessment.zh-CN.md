# 全项目重构评估与执行清单

实施进展和最新文件规模见 [分阶段重构记录](RefactoringProgress.zh-CN.md)。下文基线及首批记录保留，不能用来推断后续阶段已经验收。

## 范围与结论

盘点基线：`81c04a3`。统计 `Voxt/`、`VoxtTests/` 的 Swift 源文件（含注释、空行），不含依赖、`build/`、`tmp/` 和音频夹具。评估覆盖目录结构、文件规模、声明/引用、测试分布、构建配置与文档；重点阅读入口、收尾、远程 LLM、转录和会议协调代码。**不是对全部代码逐行完成语义审查，也不是编译器级死代码分析或性能基准。**

结论：保留现有模块划分，分批收敛职责，不重写整个应用。优先删除已失去入口的旧实现，再拆分纯逻辑和协议实现，最后处理录音、会议、热键的状态所有权。仅把大类切成多个 extension，不能解决共享状态耦合。

## 基线

| 模块 | Swift 文件 | 行数 |
| --- | ---: | ---: |
| App | 46 | 17,719 |
| Core | 130 | 44,354 |
| Hotkey | 8 | 4,752 |
| Meeting | 38 | 13,039 |
| Settings | 114 | 41,648 |
| Transcription | 18 | 15,827 |
| Windows | 39 | 15,743 |
| 应用合计 | 393 | 153,082 |
| 测试 | 156 | 39,396 |

- 应用：98 个文件 >500 行，47 个 >800 行，30 个 >1,000 行，7 个 >2,000 行。
- 测试：23 个文件 >500 行，5 个 >1,000 行；静态匹配到 1,585 个 `func test…` 方法，不等于实际运行/发现的测试数量。
- 38 个文件含 `AppDelegate` 扩展。它仍持有录音、LLM 任务、文本注入、窗口、词典、会议和历史相关状态。
- 应用和测试使用 Xcode 文件系统同步组；移动 Swift 文件通常无需添加手工 build phase 条目，但仍需验证 target membership 和跨文件访问级别。

## 重点问题与拆分方向

以下行数均为基线，不是本批修改后的行数。

| 优先级 | 文件 / 范围 | 行数 | 方向与风险 |
| --- | --- | ---: | --- |
| P1 | `Transcription/MLXTranscriber.swift` | 3,855 | 先提取规划、文本合并、采样缓冲值类型，再分离捕获与推理；保留 actor、取消和模型 lease 语义 |
| P1 | `Transcription/RemoteASR/RemoteASRTranscriber.swift` | 3,547 | 按 provider 提取协议会话，统一录音入口；不要把所有 private 状态改为 internal 来凑行数 |
| P1 | `Settings/Onboarding/OnboardingGuideView.swift` | 2,868 | 分离步骤视图、AppKit 练习输入和展示组件；实际练习仍走业务链路 |
| P0 | `Core/LLM/RemoteLLMRuntimeClient.swift` | 2,369 | 删除废弃入口，拆分请求构造、Responses/Chat 执行、provider 参数和运行策略；本批已处理 |
| P2 | `Hotkey/HotkeyManager.swift` | 2,293 | 拆事件适配、触发状态、动作派发；重点保持单击/双击/长按/修饰键侧别规则 |
| P2 | `Core/Dictionary/DictionaryLearningMonitor.swift` | 2,161 | 分离观察、候选判定、持久化；明确异步取消及文本隐私边界 |
| P1 | `Settings/Shell/SettingsView.swift` | 2,047 | 外壳只保留导航和页面装配，首页卡片/弹窗/徽标观察独立 |
| P2 | `Core/Models/CustomLLMModelManager.swift` | 1,995 | 下载/安装状态与加载/推理生命周期分开，复用已有缓存和磁盘操作支持 |
| P2 | `Meeting/MeetingSessionCoordinator.swift` | 1,994 | 分离文件导入和 live 会话生命周期；保留 stop/drain、双音源时序和迟到回调隔离 |
| P2 | `Transcription/MLXModelManager.swift` | 1,967 | 同上，避免再造通用模型框架；分别验证安装、切 root、引用计数和回收 |
| P2 | `Meeting/MeetingRemoteProviderLiveSession.swift` | 1,869 | 与 Remote ASR 核对协议重复，先比较契约再复用，不把会议和短句工作流强行合并 |
| P2 | Dictionary/History store、MeetingDetail | 1,200–1,800 | 拆模型、查询、变更、展示；保留数据库迁移、分页及虚拟列表缓存约束 |

### 目录与依赖边界

- `Settings` 是设置界面，`Windows` 是独立窗口和悬浮层，`App` 是装配/流程；不存在 `Voxt/UI/`。已修正根目录开发说明。
- 基线中 Repository 仍在 Core 根部；阶段 3 已归位到 History/Dictionary 子目录，持久化行为不变。
- 阶段 3 已将笔记及外部同步收敛到 `Core/Notes/`；不要因文件数量创建无业务边界的 `Helpers/`、`Common/`。
- 共享偏好类型目前部分位于 `Settings/Shell`，会被运行时引用。后续应把纯偏好/领域值与 SwiftUI 展示属性分离，而不是宣称当前已有严格单向分层。
- `AppDelegate` 的任务、会话 ID、输出事务要逐项确定唯一所有者，再提取协调器；本批未改变这些状态的所有权。

## 删除规则与已核实结果

单次文本出现只是候选，不是删除结论。必须核对生产调用、测试、协议实现、`@objc`/selector、SwiftUI/AppKit 回调、条件编译、脚本、资源和持久化格式。迁移代码即使只运行一次，也不等于无用代码。

| 已处理项 | 依据 |
| --- | --- |
| `SessionFinalizeStage`、`SessionFinalizePipelineRunner`、7 个旧 stage | runner 无实例化；stage 只在定义处出现。实际提交走 `preparedDeliveryContext → deliverCommittedOutput → finalizeCommittedOutputPostDelivery` |
| `SessionFinalizeContext` 的 suggestions/history ID 字段 | 仅废弃 stage 读写；实际交付后逻辑自行生成词典建议和历史 ID。保留的上下文改为不可变快照 |
| 5 个 Remote LLM 旧重载 | `enhance(text:systemPrompt:…)`、两个 `translate`、两个 `rewrite` 无调用。业务入口已使用编译请求；仍在用的 `enhance(userPrompt:…)` 和词典扫描保留 |
| `LLMExecutionLatencyProfile` | 仅声明，无运行时、测试或配置引用 |
| SessionTextIO 的两个私有 helper | `resolvedLeadMs` 无调用；实例 `normalizedOutputText` wrapper 无调用，静态实现继续保留 |

阶段 2 已核验并清理会议空操作封装、`GeneralModelStorageCard` 和模型下载动作 helper，依据见实施记录；其余旧快捷键入口仍待逐项审查。协议方法如 `dropEntered`、`windowShouldClose`、`dismantleNSView` 不能因为没有显式调用而删除。

## 测试整理

- 没有足够证据认定某整个测试套件无用；初步测试方法正文比对也未发现完全相同的重复块。**本批没有为减少行数而删除测试。**
- 将远程 LLM 原 2,049 行、78 个测试按端点、流解析、消息、Responses 请求、Codex、本地 provider payload、生成设置拆成 7 个 suite；原方法名、方法体和断言保留，suite 名有调整。
- 阶段 3 已按行为契约拆分 `HotkeyManagerTests`（原 2,588 行）、`RemoteModelConfigurationTests`（原 1,752 行）、`MLXModelManagerTests`（原 1,214 行）和 `MeetingDetailViewModelTests`（原 1,108 行）；230 个测试方法正文和断言保留。
- 只合并“相同前置条件 + 相同行为 + 相同断言”的重复场景；不同 provider、边界、失败路径、持久化迁移不能当作重复。参数化时保留能定位输入的失败信息。
- 默认单测、已安装模型回放、人工设备/权限验收保持分层。模型测试通过 `ModelTestGate` 显式启用，跳过不等于通过。
- 当前远程 LLM 套件偏重 payload/解析；未找到对其网络执行入口的直接故障注入测试。修改重试/流式策略之前，需补上零 chunk 回退、已有 partial 禁止重放、取消、Codex 强制流式等契约测试。

## 首批完成的结构整理

- `RemoteLLMRuntimeClient.swift`：2,369 → 303 行。原实现收敛到 7 个职责文件，单文件最大 422 行；没有新增框架或 provider 类层级。
- `SessionTextIO.swift`：1,194 → 779 行。准备逻辑在 `SessionOutputPreparation.swift`，时序日志在 `SessionTimingLogging.swift`。
- 原 `Core/SessionFinalizePipeline.swift` 删除；实际使用的答案/会话模型归入 `Core/LLM/RewriteAnswerModels.swift`，交付快照在 `Core/SessionFinalizeContext.swift`。
- 更新中英文 Prompt 文档的真实收尾流程、源码目录说明、测试入口和依赖锁文件状态。
- 这是结构和死代码清理，不宣称降低了实际延迟、内存或编译时间；这些收益需另行测量。

## 后续执行顺序与验收

1. **P0：合入门禁**：在 Mac 上验证本批 Debug/Release 编译和完整 XCTest；检查新 suite 的测试发现数量。未通过前不要叠加录音状态重构。
2. **P1：纯逻辑与 UI**：MLX 规划/合并、Onboarding 步骤、Settings 壳层。每次只处理一个职责域，保持现有测试和 UI 行为。
3. **P1/P2：协议会话**：Remote ASR、会议远程 provider；先补失败/取消契约，再收敛重复握手、解析和 drain。
4. **P2：状态所有权**：录音/会议/热键/模型生命周期；在 Mac 和真实设备上验证启动、停止、取消、切设备、睡眠唤醒、并发回调与内存回收。
5. **P3：存储与测试归档**：Repository 归位、笔记同步边界、测试按域整理；更新回归脚本中的 suite 名称，逐条判定测试去重，保留升级迁移覆盖。

建议新文件以 200–500 行为常见范围，>800 行必须复审，>1,000 行原则上拆分或说明原因。这是维护目标，不是为了达标机械拆文件的硬门禁。注释优先说明生命周期、线程/actor、重试约束、兼容原因，不为每个显然的函数重复名称。

每批需记录：删除依据、前后文件规模、行为不变点、验证命令、实际结果及未验证项。以“减少重复状态和入口、降低理解成本、保住行为覆盖”为目标，不设盲目的净删行比例。

## 首批验证状态

首批环境：Linux，无可用 `swift` / `xcodebuild`。后续各阶段的新增检查和未通过的 Mac 门禁见实施记录。

已执行：

- `python3 -B -m unittest discover -s tools -p 'test_*.py' -v`：5 项通过。
- `python3 tools/audit_model_stack.py --resolved Voxt.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved`：通过；当前锁文件已匹配审计要求，旧依赖不一致说明已修正。
- 远程 LLM 拆分前后声明正文静态对比、78 个测试方法正文核对、跨文件 private 引用检查、`git diff --check`。

必须补跑（macOS，不能用静态检查替代）：

```bash
xcodebuild build -project Voxt.xcodeproj -scheme Voxt -configuration Debug -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO
xcodebuild build -project Voxt.xcodeproj -scheme Voxt -configuration Release -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO
xcodebuild test -project Voxt.xcodeproj -scheme Voxt -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO
```

聚焦测试入口见 [VoxtTests/README.md](../VoxtTests/README.md)。后续音频/模型修改另跑 [本地回归矩阵](LocalRegressionMatrix.md)、真实设备验收及必要模型回放；保留首 partial、stop-to-delivery、峰值内存和取消回收的前后数据，不把拆文件当作性能证明。
