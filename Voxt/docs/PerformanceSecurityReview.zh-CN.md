# 全项目性能、安全与精简审查

关联：[源码地图](Architecture.md)、[集中收尾验收](RefactoringCloseout.zh-CN.md)、[回归矩阵](LocalRegressionMatrix.md)。

## 范围与结论

本轮从代码基线 `8b4d293` 开始，首批修复提交 `4e07cbb`。这是**全仓静态收集 + 重点链路人工审查**，不是“每个函数均人工证明安全”，也不是已完成真机 UI/CPU/RSS/Metal/磁盘性能验收。发现未解决项不能因为上一轮重构收尾或 CI 绿色而隐去。

- 收集对象：Git 跟踪的全部 `Voxt/`、`VoxtTests/` Swift；不扫描 build/tmp 生成物，不读取用户模型、凭据、数据库或录音。
- 补充检查：Xcode product dependency、Package.resolved、Tests 工作流、Info.plist、entitlements、Keychain、endpoint/TLS、日志、存储/迁移、脚本调用和资源路径。
- 规模（`4e07cbb`）：应用 **500 文件 / 149,766 行**，测试 **216 文件 / 42,734 行**；应用目录 App 51、Core 163、Hotkey 14、Meeting 49、Settings 135、Transcription 44、Windows 44。
- 已修复：启动模型误报及提示漏刷新、模型选择器把检测中显示成未安装、结构化日志 metadata 裸密钥漏脱敏。
- 已精简：确认无入口的 UI/词典包装、私有展示 helper 和测试工厂；**未删除测试方法、用户数据、Codable 字段、迁移或依赖 pin**。
- 尚有优先处理项：CDP 原始 socket 的资源/整数边界、主线程导入/音频文件读取、长录音内存、阻塞子进程取消和数据库失败恢复。下表逐项给出证据与建议，未实测的项不标作已验收。

## 可复现的只读清单

```bash
python3 -B tools/audit_source_inventory.py --output /tmp/voxt-review-$(date +%s).json
python3 -B -m unittest discover -s tools -p 'test_*.py' -v
python3 tools/audit_model_stack.py --resolved Voxt.xcodeproj/project.xcworkspace/xcshareddata/swiftpm/Package.resolved
```

工具记录 HEAD、工作区是否有修改、位置/符号名和 imports，不输出命中日志正文，不覆盖已有报告，不删除文件。CI 的 `validation-evidence` 增加 `source-review-inventory.json`。这是词法清单，不是 Swift 调用图、漏洞扫描结论或性能 profile；多行表达式、动态调用和宏需要人工补查。

`4e07cbb` 快照的检查点数量：

| 规则 | 应用位置 | 应用+测试 | 解释 |
| --- | ---: | ---: | --- |
| 同步文件读取 | 45 | 68 | 是否阻塞 UI 要检查调用上下文；nonisolated 不等于后台执行 |
| 目录遍历 | 21 | 26 | 检查触发频率、去重与文件数；不据此断言慢 |
| Timer 建立 | 5 | 5 | 检查可见性/播放门禁、暂停与销毁 |
| Task / detached 位置 | 289 | 347 | 检查取消/代次/资源退出；不等于 289 个泄漏 |
| unchecked Sendable / unsafe | 37 | 52 | 核对锁、actor、线程回调边界，不机械删标注 |
| fatalError / preconditionFailure / try! | 19 | 25 | 区分 bundle 不变量、coder init、抽象协议钩子与真实用户输入 |
| Process / AppleScript / open URL | 13 | 13 | 检查参数构造、权限、超时与退出 |
| 文件移动/删除 | 88 | 129 | 不是孤儿文件数量；核对根目录和业务引用 |
| 日志敏感字段复审 | 187 | 188 | 含安全的字符数/布尔值；只是候选位置 |

单次词法引用共 2,003 项，其中应用 154 项（84 函数、55 属性、8 类型、7 let）；测试 `test...` 的动态发现解释了大部分其余命中。**154 不是可删数量**，剩余清单见本次 CI artifact，尚未逐个完成业务可达性证明。

## 已修复：首次启动模型误报

### 原因

1. manager 的 `isModelDownloaded()` 启动后台检查后立即返回缓存；首次缓存为空时 `?? false`。
2. `ModelConfigurationIssueResolver` 将这个 false 当成已确认未安装。
3. Settings 外壳将缺失提示存到 State，但只监听下载数量/配置/窗口激活等，没有监听安装扫描的 `installationRevision`。扫描完成时下载数仍是 0，旧警告不更新。
4. 模型列表自身已有安装事件订阅，导致列表和顶部警告可能不一致；窗口重新显示/配置变化才重新计算旧警告。不是切换菜单把模型重新安装了。

### 修复与性能约束

- resolver 先检查现有 `isCheckingInstallation`，只有检查结束且确实不存在才显示安装警告；未知不等于已安装，也不跳过后续真正加载验证。
- `ModelInstallationObservation` 合并两个 manager 的 installationRevision，100ms debounce；外壳和模型列表共用。只监听安装变化，**不监听高频下载进度、音量或全部 objectWillChange**。
- debounce 也确保在 Published 的 willSet 之后读取已提交状态；后台完成无需菜单/鼠标交互。
- 选择器检测中显示现有本地化 `Loading…`，保持不可选，不误导用户重新安装。
- 不增加同步磁盘读取、Timer 或启动全模型强制扫描；原缓存请求合并与 storage revision 保留。
- 未拆分安装验证与磁盘占用统计：它可能延长扫描，但本轮缺乏真实目录耗时证据，贸然加第二次遍历会增加 I/O。先修状态和通知，再按 profile 决定。

新增 5 项测试：冷启动已安装、确认未安装各业务 scope、无交互自动刷新、remote 配置缺失不被隐藏、选择器检测中契约。使用隔离 root/defaults 和微小文件夹，不加载真实模型。完整安装缓存的去重/旧代次/取消测试保留。

## 已修复：结构化日志脱敏

`VoxtLogRedactor.redactedMetadata` 原实现只检查 value 字符串：`["apiKey": "opaque-secret"]` 不含 `apiKey=` 前缀，因此会漏掉。嵌套 dictionary/array 也存在同样问题。这是可构造的脱敏机制缺口，**不是已确认真实用户密钥被写入日志的证据**。

修复为递归检查规范化字段名，secret 容器整体隐藏；tokenCount、model 等安全 metadata 保留。敏感字段集合静态缓存，不反复创建正则或遍历日志目录。新增 3 项回归覆盖裸值、嵌套/数组、普通字段中的内联秘密。已有 HTTP header、正文、省略 LLM 原文和 home 路径脱敏测试不删。

`.visible` 明确绕过脱敏、未知字段名、任意用户正文不应指望正则全部识别；调用处仍须最小化日志。已检查 `VoxtLog.llm` 默认不求值/输出用户 prompt 与 response，其他日志类别及第三方 SDK 输出仍需真实运行检查。

## 精简处置与不删清单

| 项目 | 本轮处置 / 依据 |
| --- | --- |
| `FlowTagBadgeStrip` / 独占 `FlexibleTagLayout` | 无 view 装配、预览、测试、资源入口，删除整文件；当前词典的 `DictionaryFlexibleTagLayout` 是不同活跃实现，保留 |
| `FeatureHintBanner` | 无生产/测试/资源引用，删除；其他 Settings row 不动 |
| AppDelegate `resolveDictionaryCorrection` / `resolveDictionaryMatches` | 无入口的旧 wrapper；现行 SessionOutputPreparation 使用 matcher/correction 快照，词典持久化和学习路径保留 |
| DictionarySettingsView `scopeLabel` / `historyScanSummaryText` | 私有无调用展示函数，非 selector/协议钩子，删除 |
| 测试 `makeSelectorEntry` / `makeCatalogEntry` / `makeURLItem` | 三个无调用 factory 删除；测试方法、fake/barrier 与模型门禁不动 |
| NSApplication/NSWindow delegate、SwiftUI Layout、URLSession callbacks | 单次词法引用也可能由框架调用，保留；不能凭 rg 删除 |
| `StoredBranchURLItem` / `StoredAppBranchGroup`、旧 JSON / raw value | 候选已收集，但含存储语义需进一步证明兼容历史后再删；本轮不动 |
| SQLite migrations、Keychain legacy migration、security-scoped bookmarks | 属于用户升级/权限边界，不是无用数据；保留 |
| 旧模型目录、partial/sidecar、历史 WAV、checkpoint | 未运行删除；清理须有 DB/任务/租约/引用证据，避免误删可恢复文件 |
| TestModelManagers / Fake URLProtocol / ManualTaskBarrier | 测试依赖仍活跃；静态全局 manager 可能污染标准 defaults/文件扫描，登记隔离改进，不能为了精简删覆盖 |

直接链接的 13 个 package product 均有应用 import 使用：MLXAudioCore/STT/VAD、Logging、Sparkle、LlamaSwift、MLXLMCommon/VLM/LLM、PermissionFlow、SystemSettingsKit、FaviconFinder、GRDB。未发现可证明“仅测试占位”的直接 package；XCTest/CoreFoundation 的 test-only import 是系统框架，不是可删第三方包。传递依赖无直接 import 并不等于无用；没有改变 lockfile 或升级依赖。当前项目没有 WhisperKit import/product，历史文档的旧依赖描述不能作为删包依据。

## 性能/安全风险分级与处置

优先级表示需要处理/验证的顺序，不是 CVSS 或已经复现的数据泄漏等级。

| ID / 优先级 | 证据、影响 | 状态 / 下一步 |
| --- | --- | --- |
| UI-01 / P1 | 冷缓存 false + 外壳漏订阅，误报未安装 | `4e07cbb` 完整 XCTest / Release CI 通过；真实启动/多模型目录验收待执行 |
| SEC-01 / P1 | metadata 字段名未参与脱敏 | 本轮已修复、补回归；仍须审查实际日志 export |
| SEC-02 / P1 | `App/TextInputCDP.swift` WebSocket 127 长度直接 UInt64→Int、按对端长度分配；readUntilClose 无总量限制；debugger URL host 来自返回值 | **未修复**。本地 Electron 调试端口也不是可信边界；需长度/host/总预算/取消与 socket 故障注入。不能直接截断正文或限制合法大编辑器内容而声称行为不变 |
| IO-01 / P1 | `RemoteASRFileRequests` 在 MainActor transcriber extension 中 `Data(contentsOf:)`、base64/multipart 拼装；`DictionarySettingsView.importDictionary` 同步读/解析/写入 | **确认同步路径，未测 UI 阻塞量**。需要后台流式准备/事务进度，保持上传契约、导入失败原子性及取消清理；不通过静默限制文件大小“优化” |
| MEM-01 / P1 | MLX/Remote sample store 保留完整录音；WAV export/合并、快照/重采样/JSON base64 可能同时持有多份 | **资源增长风险，未证实泄漏**。32-bit mono 16kHz 单份约 230MB/小时，48kHz 约 691MB/小时（理论值）；测长录音，设计磁盘 spool 时必须保留 Final/重写音频语义 |
| CPU-01 / P2 | 安装验证内部遍历后仍遍历 allocated size；unknown repo 由查询按需检查，队列并发 2 | 保留有界后台队列/缓存；测 1/10/30 模型、小文件多/外置盘。不要恢复主线程 scan；可再设计单遍扫描/二阶段元数据 |
| CPU-02 / P2 | waveform/播放器各有 timer，model pane 轮询和多个 Published | 已复核显式 stop/deinit/可见性门禁；不是看到 Timer 就删。Time Profiler 对比隐藏窗口/暂停后 wakeup，未实测 idle CPU |
| LIFE-01 / P2 | `ps`、`screencapture` Process.waitUntilExit，截图在 detached 内；Task 取消不会自动终止子进程 | **未修复**。补 process deadline/退出回收，测试管道满/退出失败；单纯 detached 不是取消屏障 |
| DISK-01 / P2 | 日志滚动默认 2MiB×6、内存 2000 行，但单行/排队量无字节上限；latestLines 用 queue.sync 读文件 | 有轮转、0600 创建、redactor；仍需限制过大响应日志和导出时阻塞，保留诊断可用性；不是完全有界内存证明 |
| DATA-01 / P2 | `VoxtDatabase.init` 初始化/迁移失败 fatalError | 保留现有 fail-fast，禁止通过重建/清空数据库吞掉错误。需无损恢复 UI、磁盘满/只读/坏库注入与备份方案 |
| SEC-03 / P2 | 下载 destination 使用标准化路径前缀；weight index 特意允许 HuggingFace blob symlink | 有 `..` 防护与分片检查，**不等于无符号链接/TOCTOU 风险**。检查写入目标既存 symlink/root 授权，不能把合法只读缓存链接一概禁用 |
| TEST-01 / P2 | 部分 catalog tests 用标准 defaults + 全局 manager；有些测试使用 yield/sleep 等时序 | 活跃依赖不是占位。逐步注入目录/defaults 与受控 barrier；本轮新增模型提示测试已隔离 |
| NATIVE-01 / 外部门禁 | native decode/Metal 的同步 cancel 无内部 worker await API；TCC、音源和 AX 无法在 Linux 实测 | 延续集中收尾限制：task 退出/CI 绿色不等于 Metal 完全静止或编辑器确认接收 |

### 已检查的安全保护（不是安全证明）

- `RemoteEndpointSecurityPolicy`：拒绝 URL 内 user/password；带凭据的非 loopback 明文 HTTP/WS 被拒绝，诊断端点去 query/fragment。Loopback 明文保留本地 Ollama/兼容服务业务。
- `VoxtNetworkSession`：未发现自定义信任所有证书逻辑；代理认证以 proxy protection space 分支处理，其余 default handling。仍需测试重定向/代理 challenge 及 provider 特殊 header，不保证第三方全链路无泄漏。
- `VoxtSecureStorage`：Data Protection Keychain + WhenUnlockedThisDeviceOnly，保留 legacy migration 和更新失败处理；实际签名 identity/TCC 需签名 Mac 验证。
- Info.plist / entitlements：sandbox、选中文件 bookmark、网络 client、microphone、EventKit、浏览器 Apple Events、Codex 目录和 Sparkle helper 例外；未发现 ATS 全局放开。例外有业务用途，未粗暴删除。
- SwiftPM：lockfile/依赖栈审计通过不等于 CVE/供应链安全证明；本轮未接入外部漏洞数据库，依赖源码和发布签名仍需独立门禁。
- SQLite：核查 repository 查询中的动态 WHERE/placeholder 为程序组装、用户值走 arguments；未据单次 grep 宣称所有 SQL 均无注入。
- 会议音频队列有 10 秒上限，Doubao gzip 有尺寸/扩张限制；其他字节流不因此自动获得同样保护。屏幕/输入上下文是敏感数据，需真实权限和功能开关验收。

## 本轮自动验证证据

代码修复 `4e07cbb` 已通过 [macOS CI 35490900072](https://github.com/hehehai/voxt/actions/runs/35490900072)：

- XCTest 发现 **1,771** 项：**1,748 通过、23 个模型门禁跳过、0 失败**；Debug 测试构建及 unsigned Release 均成功。
- 下载 xcresult summary/discovery 核对：新增模型提示 suite 5/5、安装缓存 5/5、日志脱敏 9/9、功能选择器 21/21 通过。
- 无交互自动刷新回归测试耗时约 0.24 秒（包含两次 100ms 合并窗口），冷安装样例测试约 0.046 秒；是小型临时文件测试，不作为真实大模型扫描速度结论。
- Linux 工具测试 **14 项通过**，shell 语法、模型依赖审计、diff whitespace 和文档链接检查通过。
- 本报告及新增 CI inventory 归档步骤后续提交，不能将其 SHA 与上述代码验证 SHA 混称相同；当前 PR HEAD 的 CI 单独记录。

## 实测方案与完成标准

当前 Linux 没有 Xcode/Instruments、真实 TCC、GPU/麦克风/provider 账户；**本轮没有实际 UI 帧时间、idle CPU、应用 RSS/Metal 或磁盘 I/O 前后对比数据**。CI 的构建 RSS 不能代替它们。

| 场景 | 方法 / 记录 |
| --- | --- |
| 首次打开、1/10/30 已安装模型 | Release、相同目录/缓存条件；扫描起止、警告状态、首次可用时间；不交互等待后自动刷新；用 File Activity 检查扫描次数 |
| 空闲 5 分钟、窗口显示/隐藏、模型页/历史滚动 | Time Profiler / Hangs / SwiftUI Instruments；主线程栈、刷新频次、CPU/wakeup；不得只看肉眼流畅 |
| 短句、30/60 分钟录音、双音源会议 | Allocations / VM Tracker / Metal；首 partial/final 延迟、峰值及退出后内存、音频丢帧、准确率；模型冷/热分开 |
| 大字典/长历史导入、导出、日志导出 | 同一数据量，主线程最大阻塞、数据库事务、峰值内存、写入量；坏文件/磁盘满确保旧数据不丢失 |
| 下载、续传、取消、换盘、断网/代理 | 验证 ETag/Range、磁盘增长、partial 清理和旧代次；本地 mock server 注入重定向/坏包，不能用真密钥 |
| UI 文本交付/权限撤销/睡眠设备拔插 | 真实编辑器、多个窗口/TCC；输入目标、剪贴板、取消与资源退出都需验收 |

基线 `8b4d293` 与最终提交同一 Mac/系统/电源/模型/数据，至少区分冷/热启动并重复采样，报告中位数/p95、最大值、失败/skip 数。不要将“没有提示”当安装验证成功；明确未安装仍应提示，remote 未配置仍应提示。使用假凭据与无隐私音频，保存可公开的 Instruments trace/摘要和提交 SHA。

**本轮交付是已验证修复 + 可复现清单 + 风险账本，不是“全项目性能、安全全部通过”。** SEC-02、IO-01、MEM-01 等仍需要专门实现/验收，不能为了业务不变而掩盖问题，也不能为追求精简删除兼容或测试防线。
