# OmniVAD-Kit 接入规划

## 目标

在不破坏现有 `Automatic` / `Silero` / `Energy` / `Off` 体验的前提下，把 OmniVAD-Kit 接入为 Voxt 的一等本地 VAD 后端。用户仍然只在 `Settings > General > Advanced > VAD` 里切换 VAD 模式；当选择 OmniVAD 后，转录、翻译、改写、会议等使用本地 VAD gate 的功能都切到 OmniVAD。

接入要解决两件事：

- 抽离：业务代码不直接依赖 Silero、OmniVAD 或 Energy 的实现细节，只消费统一的 voice activity decision / segment event。
- 兼容：已有用户设置、模型调试页、server VAD、远端 ASR、本地 MLX ASR、停止/取消/历史写入边界保持现有行为。

## 非目标

- 不在 backend 未完成前把 OmniVAD 选项暴露给用户。
- 不把 OmniVAD 当作“出错时偷偷兜底”的备用方案。它应作为用户显式可选的核心 VAD 后端。
- 不把远端 provider 的 server VAD 和本地 VAD 互相串接。server VAD active 时，本地 gate 仍按现有策略禁用。
- 不把原始音频、完整转写、完整提示词写入日志。

## 当前状态

Voxt 当前 VAD 体系由以下层组成：

- `LocalVADMode`：用户可见模式，当前为 `automatic`、`silero`、`energy`、`off`。
- `ASRVoiceActivityBackendKind`：运行时实际后端，当前为 `off`、`energy`、`mlxSilero`。
- `ASRVoiceActivityRuntimePolicy`：按用户模式和业务场景解析实际后端。
- `ASRVoiceActivityBackend`：帧级 VAD backend 协议。
- `ASRVoiceActivitySegmenter`：统一把帧级判断转换成 speech start/end/forced/rejected 事件。
- `MeetingVoiceActivityDetector`：会议链路的 VAD 接入点。
- `RecordingVoiceActivityFrameDecider`：转录、翻译、改写链路的 VAD 接入点。
- `ModelDebugCatalog.vadSnapshot`：模型调试页展示 mode、frame VAD、local gate。

现有 `Automatic` 默认解析到 MLX Silero。这个默认值在 OmniVAD 完成真实音频验收前不改变。

## OmniVAD 能力分层

OmniVAD-Kit 提供三类能力，应分阶段使用：

| 能力 | 入口 | Voxt 用途 | 接入阶段 |
| --- | --- | --- | --- |
| Stream VAD | `stream-vad.omnivad` | 实时录音、会议、翻译、改写的本地 gate | 第一阶段 |
| Batch VAD | `vad.omnivad` | 长音频和导入文件的 speech segment 规划 | 第二阶段 |
| AED | `aed.omnivad` | 区分 speech / singing / music，减少背景音乐或视频误送 ASR | 第三阶段 |

第一阶段只让 Stream VAD 进入主路径。Batch VAD 和 AED 先作为独立能力验证，不直接影响默认业务行为。

## 用户设置兼容

扩展 `LocalVADMode`：

```swift
enum LocalVADMode {
    case automatic
    case silero
    case omni
    case energy
    case off
}
```

兼容规则：

- `automatic`：初期仍解析为 `.mlxSilero`。
- `silero` / 历史值 `mlxSilero`：继续解析为 `.silero`。
- `omni` / 可选历史别名 `omnivad`：解析为 `.omni`。
- `energy`：继续走轻量能量检测。
- `off` / 历史值 `disabled`：关闭本地 gate。
- 未知值：继续回落到 `.automatic`。

扩展 `ASRVoiceActivityBackendKind`：

```swift
enum ASRVoiceActivityBackendKind {
    case off
    case energy
    case mlxSilero
    case omniStream
}
```

UI 只新增一个用户可见选项 `OmniVAD`。模型调试页必须能显示实际 backend 是 `OmniVAD`，而不是只显示 mode。

## 抽象边界

新增 OmniVAD 代码时，保持以下边界：

- `OmniVADBridge`：最薄 C bridge，只负责 C API 映射、错误码和内存释放。
- `OmniVADResourceLocator`：只负责从 app bundle 找 `libomnivad.dylib` 和 `.omnivad` 文件。
- `OmniVADRuntime`：负责加载、签名包兼容、资源缺失错误、版本诊断。
- `OmniStreamVoiceActivityBackend`：实现 `ASRVoiceActivityBackend`，输出统一的 `ASRVoiceActivityFrameDecision`。
- `OmniAEDClassifier`：第三阶段独立使用，不混入第一阶段 gate。

业务层只依赖 `ASRVoiceActivityBackend` 或既有 detector/decider，不直接调用 OmniVAD C API。

## 运行时设计

### 录音 / 转录 / 翻译 / 改写

`RecordingVoiceActivityFrameDecider` 改为按 backend kind 持有具体 backend：

- `.energy` -> `ASREnergyVoiceActivityBackend`
- `.mlxSilero` -> `ASRSileroStreamingVoiceActivityDetector`
- `.omniStream` -> `OmniStreamVoiceActivityBackend`
- `.off` -> 不产生 frame decision

保留现有 `ASRVoiceActivitySegmenter`，让 OmniVAD 先输出帧级 speech 概率或 speech 布尔值，再由统一 segmenter 处理 speech start/end。这样不会让每个 backend 自带的状态机直接支配业务停止、取消和 final suppression。

### 会议

`MeetingVoiceActivityDetector` 需要按 speaker/source 维护独立 OmniVAD stream state：

- microphone / system audio 互不共享 state。
- `MeetingSpeaker.me` / `MeetingSpeaker.them` 互不共享 state。
- 重置会议、切换 VAD mode、设备热插拔时销毁或 reset 对应 state。

server VAD active 时继续按现有逻辑返回 `.server`，本地 OmniVAD 不参与 gate。

### 长音频 / 导入文件

第一阶段不替换 SenseVoice long-form Silero segmentation。第二阶段再增加 Batch VAD 规划：

- 统一返回 `[ASRVoiceActivitySegment]` 或 sample range。
- 由 ASR engine 自己决定是否接受 batch VAD segment。
- 空 segment 不直接报 fatal，必须走可恢复错误路径。

### AED

第三阶段接入 AED 时，不直接替代 VAD：

- AED 输出用于判断音频事件类别。
- speech 和可选 singing 才进入 ASR。
- music 可用于抑制背景音触发、改善会议系统音频。
- AED 结果进入 debug telemetry 和验收报告，不写入用户内容日志。

## 打包与构建

推荐把 OmniVAD 作为 app 内置资源，而不是运行时下载：

```text
Voxt/Resources/OmniVAD/
  vad.omnivad
  stream-vad.omnivad
  aed.omnivad
Voxt/Frameworks/
  libomnivad.dylib
```

构建要求：

- `libomnivad.dylib` 和 app 一起签名。
- macOS deployment target 不高于 Voxt 支持的最低版本。
- dylib 使用稳定的 `@rpath` / `@loader_path`。
- Release build、notarization、Gatekeeper 启动 smoke 都要覆盖。

如果后续选择源码集成，应优先把 ncnn 静态链接进一个可控 dylib，避免把 ncnn 的构建细节扩散到 Xcode 主工程。

## 错误与故障保护

OmniVAD 模式下出现资源缺失、dylib 加载失败、模型损坏、推理错误时：

- 不崩溃。
- 不阻塞停止、取消、final ASR。
- 当前 session 内降级到 Energy，并记录一次 warning。
- 模型调试页显示 backend unavailable 或 degraded 状态。

这类降级只用于故障保护，不是产品策略。`Off` 仍然是唯一明确关闭本地 gate 的用户选择。

## 日志与隐私

允许记录：

- mode
- effective backend
- source
- probability
- segment event
- elapsedMs
- sampleCount
- error class / sanitized error message

禁止记录：

- 原始音频样本
- 完整转写文本
- 完整提示词
- 用户词典明文
- 未脱敏路径中的敏感内容

## 测试计划

### 单元测试

- `LocalVADMode.resolved(rawValue: "omni") == .omni`
- `LocalVADMode.resolved(rawValue: "omnivad") == .omni`
- `ASRVoiceActivityRuntimePolicy.effectiveBackend(mode: .omni, *) == .omniStream`
- `automatic` 仍解析为 `.mlxSilero`
- `off` 禁用 local gate
- server VAD active 时本地 gate disabled
- OmniVAD unavailable 时 session 内降级到 energy
- meeting 不同 speaker 的 stream state 独立

### 启动与打包 smoke

- 缺失 `libomnivad.dylib` 不崩溃。
- 缺失或损坏 `stream-vad.omnivad` 不崩溃。
- Release app idle 300 秒 CPU / RSS 稳定。
- 签名包启动不出现 dyld load error。

### 真实音频验收

沿用 `docs/VADManualAcceptance.zh-CN.md`，新增 OmniVAD 维度：

- 静音会议
- 键盘/鼠标噪声
- 背景音乐/视频
- 远场低音量
- 双人重叠讲话
- 30 分钟长录音
- 20 次快速开始/停止
- ESC 取消
- 语音结束指令
- 设备热插拔
- 睡眠唤醒

## 分阶段执行清单

### 阶段 0：规划与保护

- 新增本规划文档。
- 不改默认 VAD 行为。
- 不暴露不可用 UI 选项。

### 阶段 1：抽象与设置兼容

- 扩展 `LocalVADMode` 和 `ASRVoiceActivityBackendKind`。
- 更新 `ASRVoiceActivityRuntimePolicy`。
- 更新模型调试页 snapshot。
- 增加 enum / policy 单元测试。
- UI 先可由 feature flag 或内部 build gate 控制是否展示 `OmniVAD`。

### 阶段 2：原生 Bridge 与资源定位

- 引入 `OmniVADBridge`。
- 引入 `OmniVADResourceLocator`。
- 增加资源缺失和损坏模型测试。
- 只验证加载，不进入业务路径。

### 阶段 3：Stream VAD 后端

- 实现 `OmniStreamVoiceActivityBackend`。
- 接入 `RecordingVoiceActivityFrameDecider`。
- 接入 `MeetingVoiceActivityDetector`。
- 保留 Energy 故障保护，但不作为 OmniVAD 接入的主路径。
- 扩展 VAD smoke 脚本覆盖 OmniVAD。

### 阶段 4：真实音频验收

- 用 OmniVAD 跑完整手工验收矩阵。
- 和 Silero / Energy 结果对比。
- 根据远场低音量、背景音乐、短语音触发结果调整阈值。

### 阶段 5：Batch VAD

- 为长音频导入和 SenseVoice long-form 增加 batch segmentation 实验路径。
- 不直接删除 Silero long-form 路径。
- 验证空音频、纯静音、长会议录音、重叠讲话。

### 阶段 6：AED

- 接入 AED debug 分类。
- 对系统音频和背景音乐场景做 gating 实验。
- 验证是否把 singing 当作可转写内容。

### 阶段 7：默认策略评估

只有当 OmniVAD 在真实音频验收中稳定优于或不弱于 Silero，并且打包/签名/启动 smoke 全部稳定后，才考虑把 `Automatic` 默认从 `.mlxSilero` 切到 `.omniStream`。

## 发布阻断条件

- 任意崩溃、dyld 加载失败、签名失败。
- 任意开始/停止/取消死锁。
- OmniVAD 模式下明显丢失远场低音量讲话。
- 背景音乐/键盘噪声导致大量空 ASR。
- 日志泄露原始音频、完整转写或完整提示词。
- `Off` 模式行为被改坏。
- server VAD 与本地 OmniVAD 同时 gate 同一路音频。
