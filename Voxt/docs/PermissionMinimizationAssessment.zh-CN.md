# 权限最小化诊断与方案评估

> 状态：实施前的源码审计、Apple 文档及开源实现对照。后续代码修改见 [实施记录](ContextEnhancementRemovalImplementation.zh-CN.md)；**macOS 真机验证仍未完成**。
> Voxt 源码基线：`04962c1`。审计环境为 Linux，无法验证 TCC、签名沙盒和实际音频设备行为。
> 本文区分已确认的代码事实、可行的替代方案和待验证项，不将其他闭源产品的内部实现视为已知事实。
> 后续联合方案：[上下文增强完整删除与权限精简](ContextEnhancementRemovalPlan.zh-CN.md)。该方案进一步删除截图能力及屏幕录制授权入口；本文的截图按需授权、高级 Tap 静音等内容是此前备选，不代表联合方案最终保留。

## 1. 结论

基础语音输入可以朝 **麦克风 + 辅助功能** 两项授权收敛，条件是使用本地/远程 ASR，并将静音改为不采集系统音频的实现。

目前权限偏多有四个直接原因：

1. **快捷键运行时权限门槛过高**：已经使用 `.defaultTap`，却在创建之前强制要求辅助功能与输入监控同时获准。
2. **快捷键录制器引入另一套权限路径**：默认启动键盘 HID 监听，同时显式请求输入监控并建立 `.listenOnly` Tap。只修改运行时监听器不能消除这项授权。
3. **通过音频采集机制实现静音**：创建 Core Audio Process Tap、Aggregate Device 并启动 IOProc，即使丢弃音频，也进入系统音频采集的权限模型。
4. **权限清单把可选功能当成基础需求**：设置页无条件把系统音频录制列为必需；引导页把输入监控列为必需。已有测试也固定了这些产品策略。

另有一个独立风险：系统音频授权检查/请求使用私有 `TCC.framework` SPI。应与权限精简一起治理。

**不能直接删除所有系统音频权限代码**：Voxt 的会议功能实际采集其他应用声音，该场景确实需要授权；系统听写、截图上下文、浏览器 URL 和提醒事项也有各自独立的按需授权。

## 2. 输入监控：原因与替代方案

### 2.1 当前调用链

| 位置 | 已确认事实 | 影响 |
|---|---|---|
| `Voxt/App/HotkeyLifecycle.swift:137` | `setupHotkey()` 调用 `hotkeyManager.start()` | 启动快捷键管理器即可进入请求流程 |
| `Voxt/Hotkey/HotkeyManager.swift:288–306` | `guard accessibilityGranted, inputMonitoringGranted`；两者缺失时分别请求 | 辅助功能已授权但输入监控未获准时，不尝试创建实际 Tap |
| `Voxt/Hotkey/HotkeyEventTapInstallation.swift:29–34` | `.defaultTap`，依次尝试 HID/session 位置 | 与上述“双权限同时必需”的门槛不匹配 |
| `Voxt/Core/Security/AccessibilityPermissionManager.swift:73–88` | `CGPreflightListenEventAccess` / `CGRequestListenEventAccess` | 主动引入输入监控授权流程，而非只诊断 Tap 是否可用 |
| `Voxt/Hotkey/HotkeyRecorderView.swift:135–136, 170–175, 252–270` | 录制快捷键先开 HID，再请求输入监控，创建 `.listenOnly` Tap | 即使运行时修复，进入快捷键录制仍可能索权 |
| 同文件 `488–554` | `IOHIDManager` 匹配 Keyboard/Keypad 并监听输入值，用于 Fn/Space | 属于原始设备输入监听，不是普通本地按键处理 |
| `Voxt/Settings/Onboarding/OnboardingGuidePermissions.swift:8–16` | 输入监控属于全部必需项，全部获准才通过 | 引导阻断基础使用 |
| `Voxt/Settings/Onboarding/OnboardingSupport.swift:114–125` | 转录权限包含输入监控 | 旧/配置式引导也有同样门槛 |
| `Voxt/Settings/SettingsPermissionSupport.swift:108–113` | 输入监控属于基础权限列表 | 设置告警持续引导用户增加授权 |

### 2.2 macOS 的实际区分

Apple 在 WWDC 2019 Session 701 明确区分：

- `.listenOnly` 是被动监听，对应输入监控授权路径。
- `.defaultTap` 可修改/拦截事件，对应辅助功能授权路径。
- `NSEvent.addGlobalMonitorForEvents` 的键盘事件文档也明确提到辅助功能授权，但它只能观察，不能拦截；并且不接收本应用自身的事件。
- `RegisterEventHotKey` 注册具体组合键，不等于监听整个键盘，可实现无需这些隐私弹窗的普通全局快捷键。

因此，“全局快捷键/Fn 一定需要单独授予输入监控”不是准确的产品约束。Voxt 已需要辅助功能来向其他应用交付文本，使用该授权下的事件 Tap 是合理路线，不是绕过 TCC。

也不能反过来说“辅助功能与输入监控是同一个权限”或“辅助功能自动打开输入监控设置”。两者是不同授权路径，`CGPreflightListenEventAccess()` 的结果不能作为所有快捷键后端的统一可用性判据；其返回值及兼容行为需按系统版本实测。

### 2.3 推荐路线

**第一阶段：保留现有 `.defaultTap` 状态机，不重写整个快捷键系统。**

1. 运行时不再将输入监控作为必需条件，也不在启动/重试时主动请求。
2. 根据辅助功能状态与真实 Tap 安装结果报告可用性；安装失败不能直接归因为缺少输入监控。还可能是签名、沙盒、Tap 位置、事件掩码或资源问题。
3. 快捷键录制以本地 `NSEvent` 为默认路径；确需处理 Fn/系统冲突时，评估复用现有辅助功能授权下的 Tap，或建立短生命周期 `.defaultTap`。
4. 不再默认启动原始 HID 监听。若真机证据证明极少数设备需要它，应明确做成用户主动选择的兼容功能，并解释额外权限，而不是全体用户的前置条件。
5. 同步移除两套引导、权限列表、录制器提示和测试中“输入监控必需”的假设。

**第二阶段可选：普通组合键采用 `RegisterEventHotKey`。**

优点是只注册所需组合，不接收全部键盘事件，权限及事件暴露面都更小。可参考 KeyboardShortcuts 的封装，但不必为了这次精简立即新增依赖。

不能把它当作当前功能的完全替代：纯 Fn、纯修饰键、左右区分、双击、长按组合、鼠标侧键和事件吞掉策略需要分别设计；系统保留键和重复注册也可能失败。混合后端应避免重复触发。

### 2.4 Fn 的边界

Fn/Globe 的核心信息通常来自 `.flagsChanged`、Fn 标志及键码，不必然需要直接监听 HID。需区分：

- 能观察 Fn；
- 能识别 Fn + 组合键；
- 能阻止 macOS 同时触发输入法切换、Emoji 或听写。

这三者不是一回事。额外输入监控权限也不保证能抢占全部系统保留快捷键。应测试并对冲突提供修改系统 Fn 行为/快捷键的指引，而不是反复索权。

Tap 的位置与模式也不是一回事：`cghidEventTap` 中的 “hid” 不等于 `IOHIDManager`。可评估优先 session Tap，但不能仅通过改位置就宣称权限问题已解决。

## 3. 系统音频录制：静音为何触发授权

### 3.1 当前实现不是单纯调音量

`Voxt/Core/SystemAudioMuteController.swift` 的流程为：

1. `24–32`：要求 `SystemAudioCapturePermission` 返回 authorized。
2. `80–104`：创建排除本应用的全局 `CATapDescription`，设置 `muteBehavior = .muted`。
3. `107–174`：创建 Process Tap、私有 Aggregate Device、IOProc，调用 `AudioDeviceStart`。
4. IOProc 不处理样本，但维持 Tap 工作。
5. `35–43`：停止并销毁这些资源来恢复播放。

Apple 官方 Core Audio Tap 示例说明：从包含 Tap 的 Aggregate Device 首次开始录制时，系统会请求系统音频录制权限；同时必须提供 `NSAudioCaptureUsageDescription`。不保存、不上传、不读取回调样本，并不让这条采集路径免于授权。`isPrivate` 表示音频对象的可见性，也不是权限豁免。

此外，`Voxt/Settings/GeneralSettingsView.swift:255–271` 在用户打开静音开关时就主动请求权限；配置式引导 `OnboardingSettingsData.swift` 也有类似流程。并非只有真正录音时才请求。

当前选择有实际好处：尝试只静音其他进程、保留 Voxt 自身提示音，而且不直接改写用户的设备音量。问题在于，它为普通静音引入了音频采集权限及额外资源生命周期。

### 3.2 方案比较

| 方案 | 是否需要系统音频采集权限 | 能力/代价 | 建议 |
|---|---|---|---|
| Core Audio 输出设备 `kAudioDevicePropertyMute` | 不依赖采集授权；仍须检查设备属性与沙盒实际行为 | 静音该设备的全部声音，包括 Voxt；部分设备不支持可写 mute | 基础静音首选 |
| 设备音量属性降到 0/降低后恢复 | 不依赖采集授权 | 可作为明确的后备；主音量/分声道支持不一致，恢复状态更复杂 | 在属性检查与状态管理完善后增加 |
| Voice Processing 的 other-audio ducking | 不是 Process Tap 采集路线 | macOS 14+ 有公开配置；针对非语音音频压低，不等于任意其他进程完全静音；需启用语音处理、验证 ASR 音质及设备路由 | 独立实验，不作为即插即用替换 |
| 暂停/恢复媒体播放器 | 不必采集系统音频 | 不是静音：播放进度暂停；不能覆盖所有声音。Apple Events 可能增加自动化授权，MediaRemote 等路径还可能涉及私有接口 | 不作为默认替代 |
| 现有 Process Tap 静音 | 需要 | 可实现进程筛选，保留自身输出；多出权限和聚合设备资源 | 如确有产品需求，保留为明确选择的高级模式 |

**关键产品取舍：少一项权限并不保证完全保留“只静音其他应用，自己仍响”的语义。**

建议普通模式改名为“录音时静音当前输出设备”，明确包括本应用声音。开始提示音可先播放再静音，结束时先恢复再播放；录音期间的中间提示音仍会受影响。不要临时解除静音播放提示音，否则其他应用也可能同时漏音。

如果需求只是降低媒体背景声而非完全静音，应先对公开 voice-processing ducking 做原型验证。当前几条录音引擎未看到启用语音处理的代码，加入它可能改变 AGC/回声消除、输入格式、延迟和设备兼容性，不应只为省授权偷偷改变采集链路。不要直接套用 iOS `AVAudioSession.duckOthers` 为原生 macOS 通用答案，也不建议调用私有 `AudioDeviceDuck`。

### 3.3 输出设备静音的工程要求

不能只做“开始设 true、结束设 false”：

- 先判断属性存在、可写，保存设备 UID、原始 mute/音量和实际改写值。
- 用户原本已静音时，结束后必须保持静音；读取失败不能当成未静音。
- 只恢复本会话成功修改的状态；处理用户录音期间自行调音量/静音的行为，避免无条件覆盖。状态相同不总能证明所有权，无法判断时采用保守策略。
- 设备切换时不能对“当前默认设备”直接写回旧设备状态；分别跟踪旧设备恢复和新设备是否接管。
- 有些 USB/HDMI/DisplayPort/虚拟/多输出设备没有可写的主 mute 或音量；属性探测失败时提示不支持，或走明确可控的后备，不偷偷切回会索权的采集方案。
- 默认输出设备不覆盖所有显式路由到其他设备的应用输出，功能描述不能承诺“所有系统声音”。
- 保留当前代码在停止麦克风后立即恢复声音的时机，不等 ASR/LLM 完成。
- 正常结束、取消、启动失败、引擎报错、退出、睡眠、连续录音均需幂等释放与恢复；防止旧会话的延迟恢复影响新会话。
- 进程被强杀时无法保证执行清理，设备可能保留静音。需设计保守的异常恢复/用户提示，不能简单在下次启动时总是取消静音。

## 4. 权限策略和私有 API 的附加问题

### 4.1 系统音频被无条件列为必需

`SettingsPermissionRequirementResolver.requiredPermissions` 固定返回的基础集合包含：

```swift
[.microphone, .systemAudioCapture, .accessibility, .inputMonitoring]
```

`muteSystemAudioWhileRecording` 虽在 context 中传入，却没有用于此处的条件判断；context 也没有会议采集模式。即使静音关闭、只用麦克风、不用会议，也会显示缺少系统音频权限。`SettingsView.swift:676–682` 将它用于侧栏告警。

但实际普通录音的 `RecordingCaptureFlow.swift:239–265` 只硬性检查麦克风和所选引擎需要的 Speech；辅助功能未授权时甚至允许继续，只提示文本注入可能不可用。会议则在 `MeetingSessionFlow.swift:369–385` 按 `usesMicrophone` / `usesSystemAudio` 检查。

这说明 **UI 的“必需权限”、引导门槛和运行时真实能力不一致**。应统一成按功能/入口/采集模式计算的策略，而非“应用具备某功能，所以所有用户预先授权”。

`SettingsPermissionSupportTests` 明确测试了功能关闭时仍包含这些基础权限；`OnboardingSupportTests` 也固定输入监控为必需。它们不是系统要求的证据，而是需要跟随产品策略一起修改的测试。

### 4.2 私有 TCC SPI

`Voxt/Core/Security/SystemAudioCapturePermission.swift:44–63` 动态加载：

```text
/System/Library/PrivateFrameworks/TCC.framework/Versions/A/TCC
TCCAccessPreflight
TCCAccessRequest
kTCCServiceAudioCapture
```

风险包括系统升级兼容性、非公开返回值/调用约定、审核合规性，以及符号加载失败后把合法功能阻断。当前 symbol 不可用会得到 unknown，但会议前置检查把所有非 authorized 状态都拦住，无法进入正常捕获启动流程。

建议：

- 普通静音完全脱离此模块。
- 真正需要系统音频的会议，由用户明确点击启用/开始后，通过公开 Process Tap 捕获流程触发系统授权，按 Apple 文档处理失败、拒绝、重试和资源清理。
- 不假设存在跨版本通用的公开 AudioCapture preflight 等价接口。应允许“尚未确认”，不能用本地 Bool 假装实时授权状态，也不能把静音 PCM 当成拒绝权限。
- 不要用 `CGPreflightScreenCaptureAccess` 替代系统音频采集授权检测；它们不是同一个权限契约。也不要仅为获得查询接口就改成屏幕捕获路线。
- `NSAudioCaptureUsageDescription` 为会议保留；删除普通静音用途的说明，而不是删除整个 key。

### 4.3 建议的最小权限矩阵

| 功能 | 建议授权 |
|---|---|
| 本地/远程语音识别 + 跨应用 Fn 快捷键和文本输出 | 麦克风 + 辅助功能 |
| 普通输出设备静音 | 不额外请求系统音频采集 |
| 仅应用内录音、手动复制结果 | 麦克风；不应因辅助功能/输入监控缺失而完全不可用 |
| Apple Direct Dictation | 额外 Speech Recognition |
| 会议：仅麦克风 | 麦克风；不需系统音频采集 |
| 会议：系统声音 / 混合音源 | 系统音频采集；混合音源另需麦克风 |
| 改写截图上下文 | 开启并使用时申请屏幕录制 |
| 浏览器当前 URL 的自动分组 | 需要时对目标浏览器申请 Apple Events 自动化 |
| 同步 Apple Reminders | 用户启用同步时申请 Reminders |
| 下载完成等通知 | 可选通知授权，不影响录音能力 |
| 用户选择文件/模型目录 | 文件选择器与 security-scoped bookmark，不升级为全盘访问 |

沙盒 entitlement、Info.plist 用途说明和 TCC 授权是不同层次。移除一个用途说明不会自动解决索权；删除 audio-input entitlement 反而会破坏麦克风。当前项目启用 App Sandbox，应保留安全边界，并使用真实发布签名验证行为，不以移除沙盒作为默认修复。

## 5. 同类应用：可以证实什么

### 5.1 VoiceInk：可核查的直接参考

审阅版本：`Beingpax/VoiceInk@173cbb2b3aa0a18ab4035aa1bc9dc9fc215e88b6`。

- `ShortcutMonitor.swift`：使用 `.cgSessionEventTap + .defaultTap`，接收 keyDown/keyUp/flagsChanged 和鼠标事件；修饰键逻辑处理按下、松开和独立手势。
- `OnboardingPermissionModels.swift`：必需项只有麦克风与辅助功能；截图为额外可选权限。
- `MediaController.swift`：读取默认输出设备，用 `kAudioDevicePropertyMute` 和 `AudioObjectSetPropertyData` 静音，先检查属性可写性；跟踪用户原来是否已静音以及恢复代次。
- `PlaybackController.swift`：媒体暂停是另一项功能，使用 `MediaRemoteAdapter` 和模拟播放键，不应将它与设备静音混为一谈。

这证明两条替代路线已有实际开源产品实现，**不证明其所有硬件边界、沙盒配置或异常恢复都满足 Voxt**。例如 VoiceInk 的设备恢复仍应结合上文要求独立设计，不能逐行照搬。

该项目为 GPL-3.0；可参考 API 和设计思路，直接复制实现前必须核对许可证兼容性。

### 5.2 KeyboardShortcuts：普通组合键的参考

审阅版本：`sindresorhus/KeyboardShortcuts@772133d9dbe800fdac0473226822994c5c162c58`。

`HotKey.swift` 使用 Carbon `RegisterEventHotKey`；README 明确说明不产生权限弹窗、支持沙盒和 Mac App Store。适合普通组合键，但不是 Voxt 全部 Fn/纯修饰键/鼠标手势功能的一对一替代。

### 5.3 Typeless、豆包、微信

本次没有这些闭源应用的源码或受控运行时证据，不能宣称它们具体使用了哪个 API。用户观察到“只要两项权限”与上述公开路线相容，但也可能包含播放暂停、ducking、输入法组件或不同辅助进程架构。

需要进一步对比时，在干净 macOS 账户上记录版本和授权状态，用同一组场景观察：

1. 音量/静音状态是否改变，媒体进度是否继续。
2. 本应用提示音、通知声、第二个播放器是否同样消失。
3. 默认设备切换、多输出、蓝牙耳机下是否仍生效。
4. 麦克风启动但没有声音输入时，与持续说话时，压低行为是否不同。
5. Fn 系统行为冲突时能否同时抑制系统动作。

这些实验可帮助区分设备静音、暂停、ducking 与进程级静音；界面观察只能提供线索，不能单独证明内部实现。若涉及输入法组件，应与 Voxt 菜单栏应用的权限模型分开评估。

## 6. 建议实施顺序

### P0：先减少确定的过度索权

- 修复 `.defaultTap` 的双权限前置要求。
- 修改快捷键录制器：本地事件优先，移除默认 HID 路径及默认输入监控请求。
- 更新两套引导、设置页、告警和本地化文案，输入监控不再作为基础条件。
- 系统音频权限按实际会议采集需求展示/申请，不再无条件列为缺失。
- 保留菜单/界面发起录音和手动复制的降级体验。

### P1：替换普通静音

- 在现有 `SystemAudioMuteController` 职责范围内实现设备属性方案，保留会话结束恢复的既有调用点。
- 移除普通静音开关的系统音频授权请求，修改其功能语义和用途说明。
- 完成原始状态、设备 UID、外部用户修改、连续会话和异常恢复设计。
- 有需要再增加明确的高级 Process Tap 模式，默认不能因硬件不支持而静默触发新授权。

### P2：权限模型和实现治理

- 移除私有 TCC 请求/查询依赖，会议使用公开捕获授权流程。
- 统一引导、设置页、侧栏告警、运行时的需求计算。
- 评估普通组合键使用 Carbon；评估 ducking 的音质/设备收益与成本。
- 旧用户保留偏好，不自动修改系统 TCC；应用停止请求后，旧授权不会自行从系统设置消失。如确认不再使用，应提供可选的手动撤销指引。

## 7. 验证门禁（均待执行）

### 自动化

- 修改 `SettingsPermissionSupportTests`、`OnboardingSupportTests` 及 `SettingsTypesTests` 中相关断言。
- 对权限后端注入测试状态：辅助功能获准、输入监控未获准时仍尝试安装 runtime Tap；不调用输入监控请求。
- 验证“本地/远程 ASR + 不用会议/截图/提醒事项”不会产生额外必需权限。
- 静音控制用可替换的设备读写接口测试原本静音、属性不可写、读写失败、连续会话、旧恢复任务、设备切换和用户干预；不以单元测试修改宿主机真实输出状态。
- 保留 Fn/纯修饰键/双击/长按/左右区分/鼠标热键既有行为测试。

### 真机

在 macOS 15 与 26 的受支持版本、发布等效签名和沙盒构建下验证；不要把 Xcode/Terminal 代为获准后的表现当作正式应用的证据。

| 场景 | 验收目标 |
|---|---|
| 干净账户，首次启动基础配置 | 仅在用户操作相应功能时申请麦克风/辅助功能，不额外申请输入监控/系统音频 |
| 输入监控从未授予、明确关闭；仅辅助功能获准 | Fn、Fn+Shift、Fn+Space、普通组合键、鼠标侧键分别验证 |
| 打开快捷键录制界面 | 不因 HID/被动 Tap 再次索权；前后台行为正确 |
| 系统保留键、Secure Input、休眠唤醒、Tap 超时 | 不把所有失效都误诊为缺权限；不绕过安全输入，不出现卡住的长按状态 |
| 内建、蓝牙、USB、HDMI/DP、虚拟/多输出设备 | 不支持时明确降级；不偷偷索取采集授权 |
| 原本已静音、录音中调音量/切设备、连续录音 | 不覆盖用户状态、不恢复错设备、不提前解除新会话静音 |
| 停止、取消、启动失败、正常退出、强杀后重启 | 恢复行为符合明确策略，强杀限制可见 |
| 会议麦克风模式 / 系统模式 / 混合模式 | 按模式申请与阻断；拒绝系统音频不妨碍普通转录 |
| 功能中途撤权、更新、重新签名 | 能重新诊断并解释；不依赖过期 Bool 缓存 |

本环境未运行 `xcodebuild`，也未测出任何实际 TCC/硬件结果。上述建议必须通过这些门禁后，才能宣称 Voxt 已实现“两权限基础体验”。

## 8. 公开参考资料

1. [Apple WWDC 2019：Advances in macOS Security，Session 701](https://developer.apple.com/videos/play/wwdc2019/701/)：输入监听与 `.defaultTap` / `.listenOnly` 的授权区别。
2. [Apple：CGEvent.tapCreate](https://developer.apple.com/documentation/coregraphics/cgevent/tapcreate(tap:place:options:eventsofinterest:callback:userinfo:))：Tap 创建、事件掩码和可用性。文中部分历史权限描述较旧，不能代替现代系统实测。
3. [Apple：NSEvent.addGlobalMonitorForEvents](https://developer.apple.com/documentation/appkit/nsevent/addglobalmonitorforevents(matching:handler:))：辅助功能、只能观察及不含自身事件的限制。
4. [Apple：Capturing system audio with Core Audio taps](https://developer.apple.com/documentation/coreaudio/capturing-system-audio-with-core-audio-taps)：用途说明与开始采集时的系统授权。
5. [Apple：voiceProcessingOtherAudioDuckingConfiguration](https://developer.apple.com/documentation/avfaudio/avaudioinputnode/voiceprocessingotheraudioduckingconfiguration)：公开的非语音音频 ducking 配置。
6. [VoiceInk：ShortcutMonitor](https://github.com/Beingpax/VoiceInk/blob/173cbb2b3aa0a18ab4035aa1bc9dc9fc215e88b6/VoiceInk/Features/Shortcuts/Coordination/ShortcutMonitor.swift)。
7. [VoiceInk：MediaController](https://github.com/Beingpax/VoiceInk/blob/173cbb2b3aa0a18ab4035aa1bc9dc9fc215e88b6/VoiceInk/Infrastructure/SystemIntegration/Media/MediaController.swift)。
8. [VoiceInk：OnboardingPermissionModels](https://github.com/Beingpax/VoiceInk/blob/173cbb2b3aa0a18ab4035aa1bc9dc9fc215e88b6/VoiceInk/Features/Onboarding/State/OnboardingPermissionModels.swift)。
9. [VoiceInk：PlaybackController](https://github.com/Beingpax/VoiceInk/blob/173cbb2b3aa0a18ab4035aa1bc9dc9fc215e88b6/VoiceInk/Infrastructure/SystemIntegration/Media/PlaybackController.swift)。
10. [KeyboardShortcuts：README](https://github.com/sindresorhus/KeyboardShortcuts/blob/772133d9dbe800fdac0473226822994c5c162c58/readme.md) / [HotKey.swift](https://github.com/sindresorhus/KeyboardShortcuts/blob/772133d9dbe800fdac0473226822994c5c162c58/Sources/KeyboardShortcuts/HotKey.swift)。
