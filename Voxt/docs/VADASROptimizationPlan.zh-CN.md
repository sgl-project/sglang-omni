# VAD 与 ASR 链路优化执行方案

## 当前结论

当前分支不接入 `speech-swift` 依赖，也不把 `SpeechVAD` / `VoicePipeline` 作为运行时代码路径。`speech-swift` 仅作为链路设计参考：统一 VAD 决策层、按业务场景选择策略、让 ASR 只消费稳定的音频段或 gate 结果。

OmniVAD-Kit 的接入规划见 [OmniVAD-Kit 接入规划](./OmniVADIntegrationPlan.zh-CN.md)。OmniVAD 应作为新的本地 VAD 后端接入现有 VAD 设置和策略层，而不是在业务代码里新增旁路兜底方案。

App 当前使用的本地 VAD 运行路径是：

- `Automatic`：由 Voxt 按业务场景选择本地 VAD 策略，当前优先使用本地 MLX Silero。
- `Silero`：明确使用 `mlx-community/silero-vad` 本地模型。
- `Energy`：只使用本地能量阈值检测，用于轻量对照和问题定位。
- `Off`：关闭本地 VAD gate，保留录音、停止、取消和最终 ASR。

VAD 是本地 ASR 链路优化，不增加远端 VAD 设置项。远端 provider 自己的 realtime/server VAD 逻辑仍由对应 provider 管理，不能和本地 gate 互相截断。

## 阶段 1：依赖收敛

- 已移除 `speech-swift` SwiftPM package、`SpeechVAD` product、实验后端代码和真实后端测试。
- Xcode 包解析后，`Package.resolved` 与 `project.pbxproj` 不再包含 `speech-swift` / `SpeechVAD`。
- 保留现有 MLXAudio / MLX / WhisperKit 等项目既有依赖，不新增 speech-swift 传递依赖。

## 阶段 2：统一 VAD 模式

- 新增全局 `localVADMode` 偏好，只暴露一个用户设置：`Automatic`、`Silero`、`Energy`、`Off`。
- 设置入口放在 `Settings > General > Advanced > VAD`。
- 不再在转录、翻译、改写、会议各处散落独立 VAD 开关；各业务场景的阈值和策略写在代码常量里。

## 阶段 3：共享策略层

- 新增 `ASRVoiceActivityUseCase`：`transcription`、`translation`、`rewrite`、`meeting`。
- 新增 `ASRVoiceActivityRuntimePolicy`，把全局模式和业务场景解析成实际后端。
- 新增按业务场景区分的 `ASRVoiceActivityConfiguration.profile(for:)`，用于控制 chunk、静音、前后 padding、触发阈值等策略常量。

## 阶段 4：业务链路接入

- 转录、翻译、转写复用同一个本地 VAD gate 策略，按当前 workflow 选择对应 profile。
- 会议实时 VAD 改为读取全局 `localVADMode`，默认 `Automatic/Silero` 走本地 MLX Silero，`Energy` 走本地能量检测，`Off` 放行音频。
- 模型调试页展示当前 VAD 模式、实际帧级 VAD 后端和本地 gate 状态，便于排查业务异常。

## 阶段 5：安全与异常保护

- VAD 输入会归一化 NaN/Inf 采样、采样率、时间戳和音量，避免坏音频值进入模型或 segment 状态。
- 损坏或缺失 Silero 模型缓存不应导致启动崩溃、会议采集死锁或无法停止。
- 日志只记录 VAD 模式、后端、概率、耗时和 segment 事件，不记录原始音频、完整转写或提示词。
- `Off` 模式是明确的用户选择，用于问题定位和快速止损；它不能影响最终 ASR、ESC 取消或语音结束指令。

## 阶段 6：验证与上线门禁

- `tools/run_local_regression_matrix.sh vad` 覆盖 VAD 规划、Feature Settings 存储和模型调试快照。
- `tools/run_vad_phase6_smoke.sh` 串联 Release build、启动 smoke、损坏 Silero 缓存 smoke、四种 `localVADMode` 偏好 smoke 和验收报告生成。
- `tools/validate_vad_acceptance_report.sh` 要求 smoke 日志启用 VAD/Silero fatal/error denylist 和隐私 denylist。
- 手工验收见 `docs/VADManualAcceptance.zh-CN.md`，重点覆盖会议、转录、翻译、转写、长录音、停止/取消、设备切换、睡眠唤醒、签名包启动。

## 当前剩余风险

- 普通录音的实时 gate 仍主要依赖已有音量输入，会议路径才有帧级 Silero 模型判断；后续如要让转录/翻译/转写也完全使用帧级 Silero，需要调整 capture buffer 向下游传递方式。
- VAD 参数目前采用代码常量，需要用真实音频验收结果继续收敛阈值。
- 长时间静音、嘈杂环境和低音量远场讲话仍必须通过真实 app 验收后再作为稳定默认策略发布。
