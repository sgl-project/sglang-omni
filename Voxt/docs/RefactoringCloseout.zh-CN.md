# 重构集中收尾与验收清单

关联：[分阶段记录](RefactoringProgress.zh-CN.md)、[源码地图](Architecture.md)、[回归矩阵](LocalRegressionMatrix.md)。

## 结论的边界

阶段 0–6D 后列出的代码事项在本轮集中处理，不再按文件逐批等待下一次“继续”。**代码实施、自动门禁、真实环境验收是三种不同状态**：

- 已实施：下面的异步修复、职责整理及测试。
- 自动门禁：代码提交 **`d817b78`** 已通过 [macOS Tests 工作流 35482326300](https://github.com/hehehai/voxt/actions/runs/35482326300)：XCTest/Debug 测试构建及无签名 Release 构建均成功。后续纯文档提交不冒称使用同一 HEAD；任何新代码必须重跑门禁。
- 未完成的外部验收：真实设备/TCC 权限、provider 账户、编辑器交付、模型/native 退出和运行时性能对比。Linux 实施环境不能代做这些项目，不能因此宣称“所有优化及验收全部完成”。

没有修改依赖 pin、模型清单/调参默认值、音频夹具或个人签名配置。PR 仍为草稿，未合入、未发布。

## 本轮已处理的代码事项

| 范围 | 处理与保证 | 不作的承诺 |
| --- | --- | --- |
| 远程设置 | `RemoteProviderSheetOperations` 独立持有请求代次、结果和在途任务；替换/关闭使旧结果失效，SwiftUI 延后回调复核结果身份 | 不是完整窗口/UI 自动化 |
| WebSocket 探测 | 专属 socket/session 成对关闭；超时先关闭传输，再等待 losing receive；取消同样覆盖 send，关闭幂等 | 不代替真实代理/TLS/provider 测试 |
| 词典旧文件 | 同步 reload 也废止旧异步读取；失败保留当前快照和原文件，丢弃取消/迟到结果；等待真实文件读取退出 | 不提供跨进程文件写入 CAS |
| Remote ASR | 预览循环各自持有去重状态；取消/新会话后不发布；完成任务和退休预览均保留到退出，临时快照收尾覆盖复制失败；清理只触及真正采集过的 input node，不为清理懒初始化硬件；排队的 tap 回调也复核录音代次 | 不改变识别模型或分包/采样参数 |
| MLX 会话 | 跟踪取消后尚未退出的循环、finalization、preload、watchdog/prewarm；关机等待、空闲回收检查在途任务；启动取消请求停止引擎 | AVAudioEngine/native 调用没有可证明的硬退出期限 |
| 会议 | VAD 准备在旧 cleanup 后执行，并被取消/退出屏障追踪；停止的 recording-active 更新纳入 finalization 顺序 | 真实双音源、睡眠唤醒、设备切换仍需验收 |
| 模型下载 | 校验大小绑定下载 repo，而非 UI 当前选中模型；Custom LLM 在 metadata await 前固定目录并检查取消；复用显示进度估计并防整数转换溢出 | 进度估计不是下载字节校验，也不是性能收益 |
| 热键偏好 | 值/持久化/展示分离；非法整数偏好不再在 UInt 转换处崩溃；有效编码键、raw value、预设及路由语义保留 | 不改变组合键优先级/触发时间参数 |
| 远程 LLM | 可注入隔离 URLSession；真实执行入口的故障注入覆盖首段前回退、partial 后不重试、取消不触发新回退 | 不替代 provider 账户或真实链路压测 |

`MLXInferenceConfiguration` 接收明确输入快照；MOSS/Whisper/Cohere/Qwen 的参数规则保持。`MeetingLiveTranscriptPresentation` 只处理展示值，token、翻译和资源仍归 coordinator。

## 为什么没有把所有文件强拆到千行以内

千行是复审信号，不是完成条件。本轮已复审剩余七个热点，`HotkeySupport` 与 `RemoteASRTranscriber` 已降至千行以内；以下五个保留为有状态所有者：

| 文件 | 本轮处置与保留理由 |
| --- | --- |
| `MLXTranscriber.swift` | 提取推理规划，补齐退休任务；采集、revision、model pin 与最终化仍需同一 owner 协调 |
| `HotkeyManager.swift` | tap/run-loop owner 已在前批分离；复核安装/业务代次、锁外回调及锁内原子路由，保留同一把锁保护的手势状态机 |
| `MeetingSessionCoordinator.swift` | 提取纯展示策略，修正 VAD 准备顺序；音源 epoch、session token、stop/cleanup 屏障继续保持一处 |
| `MLXModelManager.swift` | 修正按 repo 的大小校验、共享进度估计；加载、use、删除、下载及 storage revision 的协调保持私有 |
| `CustomLLMModelManager.swift` | 固定异步下载的目标目录、共享进度估计；container/use、加载、下载/删除和退出仍保持同一 owner |

这是评审后的保留决定，不是将状态全部改成 internal 再按行数切片。它们仍是维护热点；本轮结束也不意味着未来不存在改进空间或潜在缺陷。

## 自动化证据

Tests 工作流现在保存 `validation-evidence`（7 天保留）：

- `VoxtTests.xcresult`、`test-summary.json`、`test-discovery.json`；
- `debug-test.log`、`release-build.log`，包含 `/usr/bin/time -l` 的原始命令资源统计与 Xcode build timing summary。

`68701e5` 的 run `35479399460` XCTest 通过但 Release 触发 Swift 6.3.2 泛型析构优化器崩溃；单独提取 Entry 后 `39ed131` / run `35481423733` 仍复现。最终采用成员级 MainActor 隔离，**`d817b78` / run `35482326300` 已同时通过 XCTest 和 Release**，没有关闭优化或使用 unsafe 存储。

核对 xcresult：总计 **1,763**，**1,740 通过、23 跳过、0 失败**。新增各 suite 全部执行通过，包括 7 项真实执行入口的 LLM 流式故障注入；6 项 SharedModelLoadCoordinator 契约也全部通过。23 项跳过均为需安装模型/显式开启的回放、内存与 GGUF 等集成门禁，不视为模型验收。

冷实例清理测试在最终 run 为 **0.0025 秒**（此前约 600 秒），这是该测试场景的观测，不外推为整体应用提速。原始命令统计：Debug test 595.34 秒、RSS 1,057,259,520 字节；Release build 708.42 秒、RSS 247,201,792 字节。这是本次虚拟 runner 的命令观测，**不是应用/Metal 总峰值或前后性能对比**。CI 现在也限定单测试默认 120 秒、最多 300 秒，避免硬件意外初始化或死等待无界阻塞。

必须核对 HEAD SHA、所有新增 suite 的发现/执行及 skip 数。**模型门禁 skip 不是模型通过；构建命令的 RSS 不是整个应用/Metal 的峰值内存。**

聚焦命令仍为：

```bash
bash tools/run_local_regression_matrix.sh refactor
xcodebuild build -project Voxt.xcodeproj -scheme Voxt -configuration Release -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO
```

新增回归覆盖设置请求、探测 socket、旧文件读取、预览代次、热键坏偏好、进度溢出、推理规划、会议展示及 LLM 流式故障。没有为了降低测试数量删除失败/取消/迁移覆盖。

## 需要真实 Mac 的验收（尚未执行）

每项记录设备/系统、提交 SHA、操作、期望、实际结果和日志位置；不能把空白勾选为通过。

- [ ] 普通录音、翻译、重写、选中文本翻译；停止、取消及快速重启至少 10 轮。
- [ ] 隐藏答案后立刻取消/重开；切换应用及同应用窗口；Auto Key；粘贴期间用户复制及连续粘贴。按键发出不等于编辑器 ACK。
- [ ] 撤销/恢复麦克风、辅助功能、输入监听、屏幕/系统音频与浏览器自动化权限；睡眠/唤醒及输入设备拔插。
- [ ] 会议双音源、暂停/恢复、取消启动、停止最终化、导入文件取消/重试；核对说话人、翻译与归档。
- [ ] 设置切换 Codex auth 文件、连续加载模型列表、测试连接时关闭窗口；真实 provider 的成功、拒绝、超时、断网及代理切换。
- [ ] 两个本地模型同时下载、切换当前模型/存储位置、暂停/续传、取消/删除；校验模型不误判，旧操作不写新位置。
- [ ] 老版本历史/词典文件与迁移偏好；一键词典扫描及失败/取消后的 checkpoint。
- [ ] 已安装模型回放、取消/退出、空闲回收及 GGUF/native 内存行为。固定 MLX Audio 的同步 `cancel()` 不是内部 decode/Metal 完成证明。

```bash
VOXT_RUN_MODEL_TESTS=1 bash tools/run_local_regression_matrix.sh full
```

需预先安装所测 checkpoint，逐项解释 skip。更完整的设备/打包流程见 [VAD 人工验收](VADManualAcceptance.zh-CN.md)。

## 性能对比（尚未取得运行时前后数据）

- 全 PR 可用实际分支基线 `bdf5182` 对比最终 SHA；只比较本轮则用 `f1a1a0f`。建议用两个 git worktree，不改当前工作区。
- 相同 Mac、系统、电源状态、音频/模型/provider/网络条件；分别记录冷启动与热启动，避免把下载或缓存命中差异当优化。
- 每个场景重复采样，记录启动/首个 partial/final 延迟、录音停止至交付、模型加载/退出、CPU/RSS/Metal 内存、识别准确性及失败率。
- Debug/Release 构建分别记录清缓存与增量构建；CI 的单次时间/RSS 只能作为观测，不足以得出前后改善结论。
- 使用无敏感内容的固定音频，不把真实凭据或用户原文写入验收报告。

只有自动门禁、相关人工场景和性能对比均有结果，才能升级为“已验收”；不得用行数减少或 CI 绿色替代这些证据。
