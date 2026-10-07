# 模型栈现代化：实施与验证记录

分支：`chore/model-stack-modernization`
初始清理提交：`95ade09`；后续 A–D 源码与测试作为本分支的阶段性提交一起审查。提交 / PR 不代表产品验收完成。
本地环境：Linux x86_64，无 Xcode / macOS SDK / Apple GPU；另临时安装官方 Swift 6.3.2 Linux 工具链用于纯 Foundation 代码验证（不加入项目依赖）。

**状态：新 MLX 接入和后续核心代码路径已实施，并提供集中本地验收入口；产品级验收仍未完成。** 本地实施期间没有用多次 Actions 逐条试错；提交本轮 PR 后由现有 workflow 统一验证，暂不直接合并或发版。 不能把源码改完、Python 测试或 fork 的绿色 CI 当作 Voxt 发布验收通过。

## 1. 初始清理（已提交）

- FluidAudio / Offline VBx / 流式回退、下载 / 存储和相关 UI 全链路移除。
- 删除 25 个隐藏 ASR、16 个隐藏 LLM 及专用参数 / 加载分支；保留可见 ASR 11 个、LLM 12 个和 Hy-MT2 GGUF。
- 保留退役 ID 的迁移、历史会议及原缓存文件，不修改音频 fixtures。
- 删除目录成员的安装扫描、类型擦除包装和死代码；增加本地预览节流及会议翻译缓存局部更新。
- Sparkle `2.10.0`、GRDB `7.11.1`、swift-log `1.15.1` 固定引用。
- 增加源码 / resolved 图 / 发布产物审计。

## 2. Audio fork 的已验证证据

- fork `main` 已包含上游 `3e978558404df4ad1bbb0a5634a03df2b0f9dfa5` 的完整祖先历史。
- `e87d5c5` 修复 Parakeet 对新 LM compile API 的使用；`CompiledTrace` 的权重更新、模型释放回归通过。
- `33e7285` 统一 Xcode / SDK / CPU JIT 子进程工具链。
- [Actions 35309904287](https://github.com/hehehai/mlx-audio-swift/actions/runs/35309904287)：Xcode 26.5 / Swift 6.3.2，resolve、build、669 个 Swift Testing 和 2 个 XCTest 通过。
- `2a6e75d28ae6a399ba7c7aec842384ef7a3142b5` 为后续历史合并；与 `33e7285` 的 tree 均为 `4f1c756bff39e3ad320e2b12ce3f1726dd2a268b`，因此未重复跑 CI。
- 临时 fork 分支已经删除，正式消费不可变 revision；没有创建新 release tag。

边界：fork workflow 排除 `SmokeTests`，部分网络 / 模型测试仍受环境变量控制；这不是所有真实模型、多语言精度和长会议性能的验证报告。

## 3. A：新 MLX 已切入 Voxt 工作区

| 组件 | 现行工作区引用 |
|---|---|
| Audio fork | revision `2a6e75d28ae6a399ba7c7aec842384ef7a3142b5` |
| mlx-swift | fork exact `0.31.6` |
| mlx-swift-lm | revision `c6446cf7bfb7cea76408013b614d4b2c530eaa03` |
| swift-transformers | fork exact `1.3.4` |
| swift-huggingface | fork exact `0.10.2` |

- 工程和审计脚本已同步，不引用浮动 main、本地绝对路径或不存在的 tag。
- 新模型 `prepare()` 生命周期、有效 EOS、model-declared chat conventions、typed prefill 已切入正式 loader / manager。
- 保留预量化 / OptiQ 层加载；prefill 暂保留 `.remainder`，避免依赖升级同时改变分块策略。
- loader 标记 `@concurrent`，防止 approachable concurrency 下将同步权重处理留在调用者 MainActor。
- 删除一次性 `mlx-next.patch`、隔离候选生成器及其专用测试，避免维护两套实现。
- 新增 `tools/configure_xcode.sh`，测试与发布 workflow 共用一致 Xcode / SDK / compiler PATH，并采用 fork 已验证的 MetalToolchain 安装步骤；共享开发机不自动 sudo 切工具链。

**待验证**：Swift 类型检查、真实 OptiQ/VLM 权重、tokenizer / model reasoning 和 EOS 行为。纠正早先盘点：仓库已有受版本控制的旧 `Package.resolved`（Audio `.12` / MLX `0.31.4` / LM `d242429`），并非没有锁文件。必须针对新工程引用在 Mac 重新解析、审查并更新，不能手写或复制 fork 的锁文件。

## 4. B：安装校验与文件操作

主要新增：`ModelInstallationCache.swift`、`ModelDiskOperations.swift`、`ModelWeightFileValidation.swift`。

- ASR / LLM / GGUF / Sortformer 的安装目录检查使用后台扫描和不可变结果；MainActor 只接收状态。阻塞扫描使用并发上限为 2 的文件操作队列，不占用 Swift cooperative executor。
- Silero VAD 辅助仓库单独列入 artifact 管理白名单，不进入 ASR 目录或 STT loader；VAD 的目录与模型校验也移出主线程，避免清理隐藏 ASR 时误伤有效 VAD。
- 存储目录变化会取消旧任务并切换 revision；下载完成、失败、进度和加载结果均不能回写到新 root。模型切换和 root 切换不清零仍在使用的 lease。
- 合并同 repo 的并发扫描；cache invalidation 身份防止旧 storage root / 卸载前结果回写。
- 等待扫描可取消，不需等待慢盘返回；取消单个等待者不取消其他调用者共用的扫描。
- 模型行新增内部 `checking` 状态，未知时不显示可点的安装按钮。它不是新配置步骤。
- 推理前置检查不把“正在扫描”当作“未安装”；真实异步执行入口等待安装校验。
- 安装完成信号单独触发目录刷新，不依赖当前选中模型的下载进度。
- ASR / LLM / GGUF 卸载改为 async；使用中的模型不能被删除，等待下载 / 加载结束后再后台移除磁盘文件。取消下载的清理同样固定目标路径后在后台执行，不继承已取消任务的清理中止标志。
- ASR shadow 目录准备、完整性验证 / 文件移动和 Sortformer 验证加载移出主线程。
- 增加 safetensors index 完整分片检查及路径限制；允许 Hugging Face snapshot 的正常符号链接。空权重、坏 config、缺失分片不算安装成功；GGUF 至少检查文件 magic，实际结构由 runtime 加载校验。
- 现有模型选择、暂停 / 继续、重试、下载源和卸载确认流程保留。

新增 `ModelInstallationCacheTests` 覆盖异步检查、合并扫描、失效结果、取消、分片缺失和 checking 操作状态。原 manager 生命周期测试改为显式等待扫描完成，补充 Silero artifact、跨 root 加载和切模型时保留 lease 的回归。

模型回放、内存与 GGUF 集成测试也已等待初始安装扫描，避免把 unknown 当作未安装而全部 XCTSkip。六组 ASR 集成测试不再硬编码开发者 `/Users/...` 模型路径；复用当前配置或 `VOXT_MODEL_STORAGE_ROOT`，覆盖时在 teardown 恢复原偏好。

对于配置好的本地模型，前置扫描尚未完成时不提前判定“未安装”并切换到其他 provider；请求进入异步校验，会议摘要选项保留用户配置的待检查本地模型。

## 5. C：会议 live 快照增量缓存

新增 `MeetingTranscriptListCache.swift` 并接入 `MeetingDetailViewModel`。

- 全快照仍会做线性比较以检测任意旧段修订，不假设永远只在尾部追加。
- 文本修改复用说话人序号；严格时间顺序的追加只为新增 speaker 分配序号；删除、重排、时间修订或同时间戳排序才完整重建。
- 复用未变的说话人分组，仅对变化组重新排序 / 计数；缓存只保留当前快照，删除后旧组不会残留。
- 保留完整搜索过滤语义、翻译局部更新、说话人改名和历史编辑；不异步发布可能过期的 UI 快照。
- 新增测试覆盖旧段改写、删除 / 撤销、搜索子集、乱序快照和名称变更。

不是 O(1) 列表承诺：读取 / 比较传入的全量 snapshot 仍为 O(n)，减少的是重复的全量排序、分组构建和词数计算。收益需 Instruments 测量。

## 6. D：背压、取消和回收

- 拆出可注入投递操作的 `MeetingMLXNativeFeedScheduler`。
- 成功投递才推进音频 offset；permit 失败不会吞音频，也不会在 finish 假报成功。
- cancel 后悬挂的投递恢复时不再重新写回已清空队列的 offset。
- 队列过载明确终止当前 live 路径并发送失败事件，不再静默压缩时间线；捕获音频由现有会议 archive 保留供终稿处理。
- native event 消费遇到 ended / failed 即结束；finish 增加 120 秒 watchdog，缺失结束事件时取消队列和消费者，避免永久卡住会议终稿。
- 修正推理协调器延迟取消留下无限 tombstone 的问题；保留已有优先级和队列上限。
- 深度回收不再同步等待全局 CPU / GPU stream；后台仅调用 allocator 的空闲缓存回收和 malloc pressure relief，不修改在用模型状态。
- 模型加载中的状态也阻止 idle reclamation，避免只有尚未生成容器的加载任务被漏判。

**设计边界**：原生 `feedAudio` 是同步入队 API，permit 覆盖的是投递而不是整个后端异步 GPU 推理周期。本轮没有声称已将所有原生推理全局串行化，也没有盲目提高 GPU 并发。取消后的 native 后端资源完成时机、超时和长会议质量仍需真机验证。

新增测试覆盖正常排空、拒绝投递不消费、悬挂投递取消、显式过载、pending load 回收阻断。

## 7. E：集中本地验收入口

新增 `tools/run_model_stack_validation.sh`：

1. 检查同一套 Xcode / SDK / JIT 环境。
2. Python 工具测试与源码审计、diff 检查。
3. 真实 SPM 解析和兼容组合审计；`--update-lock` 显式允许当前升级重算旧 lockfile。
4. Debug build、完整 XCTest（xcresult）、Release build。
5. Release App、静态 link maps 和动态库 / 资源审计，输出 App 大小。
6. 保存工具链、基线提交、工作区差异、解析图、模型测试开关、日志和结果状态。

脚本不提交、不 push、不触发 Actions、不创建 tag。模型测试仍需已有权重和显式 opt-in；其结果不能用工具测试代替。

```bash
# 在 Xcode 26.5 / Swift 6.3.2 的 Mac 上，确认系统 xcode-select 与 DEVELOPER_DIR 一致。
bash tools/run_model_stack_validation.sh --update-lock
# 有测试权重时，再在本地显式执行模型回放：
VOXT_RUN_MODEL_TESTS=1 VOXT_MODEL_STORAGE_ROOT="/absolute/path/to/existing/models" \
  bash tools/run_model_stack_validation.sh
```

## 8. 当前检查结果与未完成门禁

| 检查 | 结果 / 状态 |
|---|---|
| Python 审计工具测试 | 5 个通过 |
| 源码禁用 runtime / 固定依赖审计 | 通过 |
| shell、workflow YAML、pbxproj 包引用静态检查 | 通过 |
| Swift 6.3.2 实际 parser | 修改 / 新增 Swift 文件解析通过；不是全 app 类型检查 |
| Swift 6.3.2 隔离 Foundation 编译与运行 | 直接编译真实安装缓存、分片校验、文件删除、音频投递调度器、协调器及会议列表缓存；失效结果、删除、投递失败 / 取消、字幕修订 / 追加 / 删除 / 撤销断言通过 |
| 隔离运行边界 | 会话接口和 macOS thermal state 在临时 harness 中使用测试替身，MLX 后端与 UI 未参与；不能据此宣称整个应用编译通过 |
| `git diff --check` | 通过 |
| 本地完整验收脚本 | 在入口明确拒绝：`Xcode validation requires macOS.` |
| Voxt 新依赖 lockfile、Debug / Release / XCTest | **待更新旧锁并在 Mac 执行；PR 首轮 CI 已在依赖解析阶段失败，未进入 app 编译 / 测试** |
| ASR / LLM / VLM 质量、长会议、取消、内存压力回放 | **待 Mac 执行** |
| Instruments、峰值内存和 App / zip / DMG 对比 | **未测量** |

因此当前不能标记“整个 plan 验收完成”。先在 Mac 收集完整批次的构建 / 测试日志，集中修复同类问题后再统一提交并跑一轮 CI；不以多次 Actions 单点试错。纯 Foundation 隔离编译和静态审计不能代替完整 macOS / MLX 的类型检查和回放。

## 9. 后续纳入：六步交互引导

按后续要求，原本独立保留的 onboarding / 录音练习改动现在也纳入同一分支与 PR #150：

- 引导收敛为权限、模型、语音输入、语音翻译、选中翻译、探索更多六步；三个练习可跳过，老用户不被强制重新引导。
- `OnboardingModelDraft` 在明确确认时合并到最新配置，浏览或关闭未确认页不重写用户的混合模型路由。
- `OnboardingPracticeState` 使用真实业务事件，并校验步骤、来源窗口和 session ID；失败、取消、重试和迟到回调隔离。
- 复用实际快捷键、音频反馈、文本注入和翻译结果窗口；新增中英日文案及 `OnboardingGuideTests`。
- 具体流程、测试与人工验收见 [六步交互引导](OnboardingGuide.zh-CN.md)。AppKit 交互、权限、焦点和真实输入验证仍待 Mac 执行。

### PR 依赖解析阻断（历史记录）

后续状态（`81c04a3`）：当前已提交的 `Package.resolved` 通过升级兼容集静态审计。下面保留的是当时的失败记录，不表示当前锁文件仍不匹配；macOS 构建及模型行为仍需独立验证，当前策略见 [依赖文档](MLXAudioDependency.md)。

[PR 首轮 CI 35322684144](https://github.com/hehehai/voxt/actions/runs/35322684144) 因旧 lockfile 与新 revision 不匹配失败：`an out-of-date resolved file was detected`，退出码 74。这不是 XCTest 断言失败；app 编译和测试尚未开始。

本次引导提交不修改这个已知依赖阻断，不伪造锁文件。为了避免重复消耗同一个必然失败的 Actions 运行，本次提交使用 `[skip ci]`，PR 保持 Draft / 待验证，不沿用旧绿色结果。更新并审查实际 lockfile 后，再统一运行完整 CI。
