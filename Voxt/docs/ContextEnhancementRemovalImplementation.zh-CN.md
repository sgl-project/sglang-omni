# 上下文增强删除与权限精简：实施记录

> 状态：业务代码与测试已修改；Linux 静态门禁通过。**Xcode 构建、XCTest 和发布签名 macOS 真机验收尚未执行。**
> 方案基线：[完整删除方案](ContextEnhancementRemovalPlan.zh-CN.md)；授权机制依据：[权限诊断](PermissionMinimizationAssessment.zh-CN.md)。

## 已实施

### 上下文能力整链删除

- 删除 `TranscriptionAppContextSupport.swift` 及专属采集/压缩测试，退出窗口 AX 树扫描、窗口截图子进程、图片压缩、远程型号图像能力白名单。
- 从 transcription/rewrite 配置、store、设置 UI、执行计划、模型调试入口中删除 appContext。
- 删除图片附件类型、预算、Base64 图片 payload、Responses 图片构造、本地 CIImage 输入、调试图片 metadata 和预览卡片。应用 LLM 请求现在只传文字。
- 更新英/简中/日默认 rewrite 提示词，不再假定模型能看到当前屏幕。原三个默认模板通过 digest 迁移；自定义提示词不改。
- 删除专属本地化键；更新输入范围与模型加载职责文档。

保留 App Branch、目标应用身份快照、选中文本处理、词典学习、会话历史、结构化答案及会议/文件功能。它们不是上下文采集的替身，也不能因为名称包含 context 就一起删掉。

### 权限与快捷键

- 删除输入监控请求/检查、引导必需项和权限页面分支。
- 运行时继续使用 `.defaultTap`，前置条件只有辅助功能；真实 Tap 安装失败仍走现有重试。
- 快捷键录制器删除 HID 监听和 `.listenOnly`，使用本地事件或辅助功能已获准时的短期 `.defaultTap`；加入 Tap 超时重新启用、本地鼠标按钮处理和 Escape 取消处理。
- 删除屏幕录制的枚举、检测/请求 API、导航分支和 Info.plist 用途说明。
- 静音和截图不再参与必需权限计算；未启用的翻译/改写功能不因所存 ASR 选择引入 Speech 授权要求。

### 普通设备静音

`SystemAudioMuteController` 改为当前输出设备的 `kAudioDevicePropertyMute`，无 Process Tap、Aggregate Device、IOProc 或授权调用。

- 检查属性可写性和原始状态；用户本来已静音时不接管恢复责任。
- 记录设备 ID + UID，避免设备断开后 HAL ID 被复用导致恢复错设备。
- 监听默认输出与 mute 属性；换输出先恢复旧设备，再尝试新设备；观察到用户取消静音后不再次强制静音，也不接管用户后续自行静音。
- 会话 ID 拒绝迟到的属性通知；重复结束幂等；正常释放、录音停止、取消、退出和系统睡眠时恢复。
- 开始提示音播放后再应用静音；待执行任务检查录音会话、取消、停止与应用退出状态，避免录音结束后才静音。
- 不支持静音的设备继续录音并显示提示。设置明确说明全部设备声音（包括 Voxt）都会受影响。

### 会议系统音频

- 删除 `SystemAudioCapturePermission.swift` 中私有 TCC framework 的动态加载与请求/查询。
- 会议不再用未知 preflight 状态阻止启动，由用户主动开始实际 Core Audio Tap 捕获触发系统授权。
- 权限页只展示会议用途说明和系统设置入口，不用缓存伪造“已授权/未授权”。
- 保留会议 `.unmuted` Tap 和 `NSAudioCaptureUsageDescription`，用途说明只描述会议。
- Tap/启动错误增加拒绝授权后的操作提示；修复读取 Tap format 抛错时未销毁已建音频资源的路径。

### 模型依赖

保留现有文字生成模型目录及所需 MLXVLM 工厂。原 `supportsImageInput` 改为 `requiresVLMFactory`，明确它仅决定加载架构。移除截图推荐及 Vision 功能标签。没有删除用户模型权重、SwiftPM 模型包或共享 PermissionFlow/FaviconFinder。

## 测试变更与已执行检查

新增/调整的原生用例：

- `SystemAudioMuteControllerTests`：模拟设备测试原始静音、幂等恢复、不可读/不可写、默认设备切换、用户干预、HAL ID 复用、迟到通知、恢复失败和设备重新可用。
- `FeatureSettingsStoreTests`：旧单开关/子开关 JSON 被忽略并在显式迁移时剔除；读取不回写；自定义 prompt 和其他偏好保留；重复迁移语义幂等。
- `RetiredRewritePromptTests`：三语言原默认模板升级，用户改过的模板原样保留。
- 权限、Onboarding、prompt、LLM 编译、远程文字消息、本地模型加载后端和调试 payload 的测试同步更新。

已执行：

```bash
python3 -m unittest discover -s tools -p 'test_*.py' -v
# 20 tests passed，其中 6 项为本次新增的静态删除门禁

git diff --check
# passed
```

新增门禁 `tools/test_permission_minimization.py` 检查运行时代码不存在退休的采集/附件/授权 API，快捷键仍使用 modifying Tap，静音无采集依赖、会议有真实 Tap，Info.plist 正确，旧提示词 fixture 的 digest 与迁移表一致，新文案覆盖三语言。

还对改动的 Swift 做了 tree-sitter 相对基线语法扫描；该解析器不支持新的 `isolated deinit` 等部分 Swift 语法，**这不是 Swift 编译或类型检查结果**。

## 尚未执行的发布门禁

当前 Linux 环境没有 Swift/Xcode/macOS SDK，以下必须在 Mac 上补齐：

```bash
xcodebuild build -project Voxt.xcodeproj -scheme Voxt -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO
xcodebuild test -project Voxt.xcodeproj -scheme Voxt -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO
```

之后使用发布等效签名/沙盒在 macOS 15、26 验证：

1. 干净账户中仅麦克风+辅助功能，且输入监控、屏幕、系统音频均未获准时，基础流程和支持设备的静音正常。
2. Fn/组合键/长按/双击/鼠标及录制快捷键；Secure Input、系统 Fn 冲突、睡眠恢复。
3. 旧上下文配置为 true 的升级用户和模型调试均不采集窗口或发送图片；App Branch、选中文本、会话/词典功能不回归。
4. 保留的各本地模型加载家族可运行纯文字请求；远程 Responses/Chat 多轮和结构化答案正确。
5. 内建/蓝牙/USB/HDMI/虚拟输出的 mute 能力、用户中途调节、设备切换和会话结束恢复。
6. 会议系统音源首次授权、拒绝后重试、途中撤权、启动失败；麦克风-only 与文件导入不被系统音频权限阻断。

## 已知边界，不作为已解决事项

- 输出设备静音不是“只静音其他应用”。部分设备无可写的主 mute；没有自动切回采集 Tap，也没有音量降零后备。
- 强杀进程、设备拔出、HAL 恢复失败时可能保留设备静音，需要用户手动检查。代码不会下次启动无条件解静音，避免覆盖用户自己的状态。
- 系统音频属性是跨进程共享状态，没有原子所有权比较。快速外部 mute/unmute 可能被系统合并通知；实现尊重观察到的外部变化，不宣称彻底解决跨进程竞争。
- 保留原模型意味着 MLXVLM 加载库体积仍在。彻底删除该库需另做模型支持迁移，不是仅删 import。
- 旧截图曾用临时文件并通过 defer 清理；本次没有扫描删除历史临时目录，也没有清空用户日志、导出或远程服务数据。删除能力不等于追溯擦除历史内容。
- macOS 旧授权记录不会自动消失；用户可手动撤销输入监控/屏幕录制。是否撤销系统音频取决于是否需要会议系统音源。

本轮没有运行真实 TCC 或音频硬件，不能把静态门禁通过描述为“两权限体验已完成真机验证”。
