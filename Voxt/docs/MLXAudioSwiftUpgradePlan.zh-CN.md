# MLX Audio Swift 全面升级、优化与重构方案

更新时间：2026-07-20

## 1. 文档目的

本文评估 `mlx-audio-swift` 从 Voxt 之前使用的能力基线升级到当前 fork 后，对 Voxt 可以产生的全部直接和间接影响，并给出可执行的升级、替换、优化、重构和验证方案。

本文覆盖：

- 本地 ASR/STT 模型加载、离线转录、实时预览和最终转录。
- 普通转录与会议转录的差异化处理。
- VAD、长音频切段、静音抑制、端点检测和说话人分离。
- 模型配置表单、语言支持标记、结果结构和调试指标。
- SwiftPM 依赖范围、编译成本、fork 维护方式和发布回滚。
- 当前未使用但可能对 Voxt 有价值的语音增强能力。

本文不把上游新增的所有 TTS 模型自动视为 Voxt 应接入的功能。Voxt 当前没有链接 `MLXAudioTTS`，因此 TTS 只评估依赖和未来产品影响，不进入本轮 ASR 实施范围。

本文替代 `MLXModelSupport2025.zh-CN.md` 中已经过时的 MLX Audio 运行时结论；`VADASROptimizationPlan.zh-CN.md` 仍可作为现有 VAD 链路的历史设计记录。

## 2. 审查基线

### 2.1 当前依赖状态

| 项目 | 当前值 |
|---|---|
| Voxt package URL | `https://github.com/hehehai/mlx-audio-swift.git` |
| Voxt pin | exact version `0.1.3-voxt.7`（commit `7b440f768f5fc2a9c4b4c837084a9faeb4e62ba8`） |
| Voxt `mlx-swift-lm` pin | `343cae3799054b2e138ebfb1ae8d7d0f6c6a4a5b` |
| fork 分支 | `codex/moss-asr-configuration` |
| fork 发布标签 | `v0.1.3-voxt.7` |
| upstream 最新 release/main | `v0.1.3`, `542fffacb3be8de47024b3b54888f71d72d46d30` |
| fork 额外提交 | MOSS streaming prompt/失败传播、Nemotron 增量 event session、强类型 segments、语言来源、Qwen streaming segments 与协议元数据过滤；已同步 upstream Qwen attention cache 与 Voxtral realtime 性能修复 |
| 审查区间 | `v0.1.2..v0.1.3-voxt.7` |

截至本文日期，Voxt 锁定 fork 中已发布的最新 Voxt tag，并已迁移强类型 segments 与语言来源语义，同时纳入 upstream 的 Qwen attention cache 和 Voxtral realtime 性能修复。`mlx-swift 0.31.6` 要求 Swift 6.3，因此 CI 固定使用 Xcode 26.5；本地开发也必须使用包含 Swift 6.3 的 Xcode。

### 2.2 Voxt 实际链接范围

Voxt 直接链接：

- `MLXAudioCore`
- `MLXAudioSTT`
- `MLXAudioVAD`

Voxt 当前没有链接：

- `MLXAudioTTS`
- `MLXAudioSTS`
- `MLXAudioLID`
- `MLXAudioUI`
- `MLXAudioG2P`

`MLXAudioSTT` target 当前声明依赖 `MLXAudioCodecs`，因此 Codecs 的大规模扩张可能影响 Voxt 的干净编译，即使 App 没有直接 import codec 模块。

Voxt 还直接链接 `MLX`、`MLXLLM`、`MLXLMCommon` 和 `MLXVLM`，用于自定义本地 LLM/VLM。`mlx-audio-swift` 对 `mlx-swift-lm` 的最低版本升级因此不是纯 ASR 传递依赖变化：重新解析该依赖可能同时影响本地摘要、LLM tokenizer/model factory 和语音模型，必须作为跨功能升级验证。

## 3. 执行结论

### 3.1 必须优先处理

1. MOSS 普通/会议最终推理改成支持 prompt 的可抛错、可取消 stream，移除 `fatalError` 风险。
2. MOSS streaming 不再静默吞掉错误，并为全部实时 session 建立可传播的失败合同。
3. Nemotron 切换到新的真正增量 `NemotronASRStreamSession`，停止使用旧的重复 mel 处理 session。
4. 建立统一模型能力描述，不再用 repo substring 和 `Multilingual` 标签推导语言、表单和实时能力。
5. 修复实时 Qwen 的 Automatic language 被重新强制成用户主语言的问题。
6. 为 Cohere/Voxtral 的 VAD 增加无语音和失败策略，禁止静音时自动转回整段识别。

### 3.2 高收益改进

1. 保存和消费 Qwen、Whisper、Parakeet、Nemotron 等模型已经返回的时间戳 segments。
2. 将包中的 `[[String: Any]]` segment 改为强类型、可发送的数据结构。
3. 为 Voxtral 和 Nemotron 暴露官方支持的延迟档位。
4. 将 SenseVoice 长音频检测改用 `SileroVAD.getSpeechTimestamps`，保留 Voxt 的 overlap 和无语音行为。
5. 固定使用 Silero VAD v6，并在真实中文、会议和噪声语料上校准阈值与端点延迟。
6. 将 VAD 的 gate、trim、segment 三种用途分离，避免时间戳模型接收被压缩的音频时间轴。

### 3.3 当前不应直接替换

1. 不用 `STT.loadModel(modelRepo:)` 替换 `MLXModelManager`。包 loader 目前只解决远端下载和 `fromPretrained`，不覆盖 Voxt 的模型目录、多下载源、断点状态、修复和 writable shadow。
2. 不用 `segmentSpeech` 整体替换 SenseVoice 自定义长音频链路。它没有 overlap，且无语音时返回整个音频。
3. 不为追求统一而把 MOSS、Voxtral、Nemotron 全部塞入现有 `StreamingInferenceSession`。它们的流式状态和结果能力不同，应先统一协议而不是抹平能力。
4. 不因为包新增 TTS/codec 就把这些产品加入 Voxt。没有业务入口时只增加构建、模型管理和测试成本。

## 4. 上游更新与 Voxt 影响总览

| 更新领域 | 上游/当前 fork 能力 | Voxt 当前状态 | 建议 |
|---|---|---|---|
| Qwen3-ASR | 自动语言、context、chunked prefill、取消、重复保护、streaming | 已使用 context 和 streaming；Automatic live 有偏差；高级重复配置未接 | 修复 live language；保留安全默认；可增加高级重复设置 |
| Voxtral Realtime | 真正增量 stream、可配置 transcription delay | 已使用增量 session，但自建 event wrapper；未暴露 delay | 保留增量实现；包内统一 event adapter；增加 delay 表单 |
| Nemotron ASR | cache-aware streaming、新增真正增量 session、时间戳 | 已接旧 event session；未消费 segments；语言 locale 可能不匹配 | 优先切新 session；暴露 chunk latency；按 prompt dictionary 路由语言 |
| Cohere Transcribe | 模型、streaming、Silero VAD、标点 task token | 已接标点、温度、max token、VAD/fixed 策略 | 修复无语音/失败策略；限制官方 14 种语言；可增加 live delay |
| MOSS Transcribe Diarize | 长音频、时间戳、说话人、prompt/hotwords、窗口式实时输出 | 已区分普通/会议并支持 prompt；最终推理仍用同步 fatal path | 增加 prompt-aware throwing stream；错误传播；max token/声学事件配置 |
| Whisper | 新的 MLX Whisper family、语言检测、segments | 已接模型；表单中的 preset 对固定 30 秒窗口无效；segments 被丢弃 | 隐藏无效 preset；消费 segments；翻译/原生 timestamp 等待包接口 |
| Canary | 25 种语言 ASR、X-English 翻译、标点控制 | fork 和 App 已接 task controls；包硬编码 no timestamp | 保持现有表单；包增加 timestamp task token 后再开放开关 |
| Parakeet | batch、混合 TDT、bf16、性能优化、aligned segments | v3 可见但仍描述成英文；语言提示不参与解码；segments 被丢弃 | 更新为 25 种欧洲语言自动识别；隐藏语言提示；消费时间戳 |
| SenseVoice | ASR、LID、情绪、事件、ITN | 已接语言/情绪/事件/ITN；长音频切段为 App 自实现 | 使用包时间戳 API 简化检测；保留 metadata 合并和 overlap |
| FireRed/GLM/Granite | 加载、兼容和模型修复 | 隐藏支持或已有专用 prompt | 维持兼容；不扩大默认展示；统一 capability metadata |
| Moonshine/Wav2Vec2/MMS/LASR | 新增模型；MMS adapter runtime switching | 已隐藏支持；MMS 使用自由文本 code | MMS 改 adapter picker；其他保持隐藏测试 |
| Silero VAD | v6、stream state、批量概率、speech timestamps | 已统一使用 v6；长音频已使用 timestamp API | 保持单一 v6 checkpoint；继续校准真实语料阈值 |
| SpeechSegmenter | 通用 Silero 切段 | Cohere/Voxtral 已使用 | 增加 no-speech/failure policy 和 overlap 能力 |
| Sortformer | streaming 结果转换性能优化 | 已用于会议实时和最终说话人分析 | 已自动继承；增加性能回归指标即可 |
| SmartTurn | 最近 8 秒语义端点判断 | 未接 | 作为 VAD 后置端点判断实验，不替代 VAD |
| FSMN VAD | 帧级/离线 VAD | 未接 | 仅作为中文长音频实验后端，不进入首批实现 |
| DeepFilterNet | 离线和流式语音增强 | `MLXAudioSTS` 未链接 | 作为后续噪声增强实验，不进入本轮默认链路 |
| ModelUtils | 下载 progress handler | Voxt 有自定义模型下载管理 | 不替换；仅复用不冲突的模型解析能力 |
| Core 音频工具 | `StreamingWAVWriter` 复用 buffer，并支持 `MLXArray`/`Data` 直接写入 | Voxt 当前没有使用该 writer | 不为升级而迁移；以后若增加 MLX 音频归档，再用它替换重复 buffer 分配 |
| 基础依赖 | `mlx-swift-lm` 最低版本从 2.x 升至 3.x；STT/Codecs 引入 `MLXFast` | Voxt 直接固定并使用 `mlx-swift-lm`；已解析 `mlx-swift 0.31.4` | 同时验证 ASR 与本地 LLM/VLM；做 clean build、取消/并发和内存验证 |
| TTS/Codecs | 大量新 TTS 与 codec | TTS 未链接；Codecs 被 STT 传递依赖 | 不接 TTS；评估移除 STT 对 Codecs 的无用 target dependency |

## 5. 当前 Voxt MLX ASR 链路

### 5.1 模型管理

`MLXModelManager` 负责：

- 模型 catalog 和 canonical repo 迁移。
- 自定义模型存储根目录。
- 下载状态、镜像源、缓存检查和损坏目录修复。
- 本地目录加载和 Qwen writable shadow。
- 模型实例缓存和 active-use 生命周期。

它根据 repo 名称手工分派到各模型的 `fromDirectory` 或 `fromModelDirectory`。这段分派有重复，但承担了包统一 loader 当前没有覆盖的产品职责。

### 5.2 普通转录

普通 MLX 转录当前包含：

1. 音频采集和重采样。
2. 可选的帧级本地 VAD。
3. 实时模型 session 或周期性 batch preview。
4. 停止后 quick/final correction。
5. 文本清理、context 泄漏抑制和历史写入。

Qwen、Granite、SenseVoice、MOSS、Cohere 和 Voxtral 有专用分支，其余模型使用 `STTGenerationModel.generateStream`。

### 5.3 会议转录

会议链路与普通转录不同：

- 麦克风与系统音频有独立 source label。
- VAD 负责生成或完成 meeting chunks。
- Sortformer 可做实时或最终说话人分析。
- MOSS 可对完整资产或最终 chunk 输出时间戳和 speaker ID。
- 会议 UI 依赖稳定的 start/end、speaker 和 final/provisional 状态。

因此普通转录的“去掉静音再识别”不能直接应用到要求原始时间轴的会议结构化转录。

## 6. 目标架构

### 6.1 统一模型能力描述

新增单一数据源，例如 `MLXASRModelCapability`：

```swift
struct MLXASRModelCapability: Sendable {
    let family: MLXModelFamily
    let supportedLanguageCodes: Set<String>?
    let languageRouting: LanguageRouting
    let liveMode: LiveMode
    let outputCapabilities: Set<OutputCapability>
    let configurationCapabilities: Set<ConfigurationCapability>
    let vadPolicy: MLXVADAudioPolicy
    let supportedPurposes: Set<MLXTranscriptionPurpose>
}
```

它应统一驱动：

- 模型 catalog 的“支持主语言”标签。
- 配置表单显示哪些控件。
- `ASRHintResolver` 的语言编码。
- 最终推理参数。
- live session 工厂。
- 是否消费 timestamps/speaker/language metadata。
- VAD 是 gate-only、trim 还是 segment。

这将替换以下脆弱逻辑：

- 多处 `repo.lowercased().contains(...)`。
- 用 `Multilingual` 展示标签推导任意主语言支持。
- `generic` family 同时代表 Parakeet、Voxtral、Nemotron 等完全不同模型。
- Model Settings 和 Feature Settings 中重复的主语言判断。

### 6.2 统一强类型结果

包当前使用 `[[String: Any]]` 表示 segment，并通过 `@unchecked Sendable` 规避并发检查。建议在 fork 先新增兼容 API：

```swift
public struct STTTranscriptSegment: Sendable, Codable {
    public let text: String
    public let startSeconds: Double?
    public let endSeconds: Double?
    public let speakerID: String?
    public let language: String?
    public let emotion: String?
    public let event: String?
}

public enum STTLanguageSource: Sendable, Codable {
    case detected
    case requested
    case outputTarget
    case modelDefault
}
```

`STTOutput` 应区分：

- 检测到的输入语言。
- 调用方请求语言。
- 翻译后的输出语言。
- 模型只回显请求语言但没有执行检测的情况。

迁移期间保留旧 `segments`，增加 `typedSegments`。待 Voxt 和示例完成迁移后，再考虑删除 `Any` 字段。

### 6.3 统一实时 session 协议

目标不是让所有模型共用同一算法，而是共用生命周期和事件合同：

```swift
public protocol STTLiveSession: AnyObject, Sendable {
    var events: AsyncThrowingStream<STTLiveEvent, Error> { get }
    func feedAudio(samples: [Float])
    func finish()
    func cancel()
}
```

统一事件至少应包含：

- `displayUpdate(confirmed:provisional:)`
- `segmentFinalized(STTTranscriptSegment)`
- `stats(StreamingStats)`
- `ended(STTOutput)`
- error 通过 throwing stream 传播

模型内部仍保留不同实现：

- Qwen：encoder window + token agreement。
- Cohere：窗口式 streaming。
- MOSS：短窗口重复转录，不是模型原生无限 streaming。
- Voxtral：原生 causal incremental session。
- Nemotron：cache-aware FastConformer/RNNT session。

完成后可以删除 Voxt 私有的 Voxtral event wrapper，并合并四套几乎相同的 session setup/lifecycle 代码。

### 6.4 统一请求配置

当前 `STTGenerateParameters` 能表达语言、目标语言、标点、chunk 和重复惩罚，但不能表达 Qwen context、Granite prompt 或 MOSS prompt。

建议采用以下一种方式：

1. 小范围兼容方案：增加 `prompt`、`context` 和明确的 `task` 字段。
2. 长期方案：增加模型特定 typed options，避免一个字段在不同模型上含义不一致。

长期方案示例：

```swift
enum STTModelOptions: Sendable {
    case qwen(context: String, language: String?)
    case moss(prompt: String, repetition: RepetitionOptions)
    case canary(source: String, target: String, punctuation: Bool, timestamps: Bool)
    case cohere(language: String, punctuation: Bool)
}
```

短期优先保证 MOSS prompt 可以走 throwing stream，长期再统一全部专用 downcast。

## 7. 关键链路改造

### 7.1 MOSS 最终推理

#### 当前问题

Voxt 为传递 MOSS prompt，直接调用同步 `MossTranscribeDiarizeModel.generate(...)`。包内部将任何错误，包括 `CancellationError`，转换为 `fatalError`。

影响：

- 用户停止、取消或快速切换模型时存在 App 进程崩溃风险。
- 错误无法显示在模型调试页。
- 与其他 `AsyncThrowingStream` 模型的取消语义不一致。

#### 方案

包侧：

1. 让 prompt 成为 generation parameters 或新增公开的 prompt-aware stream overload。
2. stream 内继续执行 chunk cancellation check。
3. 所有模型错误通过 throwing continuation 结束。
4. 同步 `generate` 保留兼容，但不再作为 Voxt 生产路径。

App 侧：

1. MOSS 最终推理改用 stream。
2. 从 `.result` 读取 typed segments。
3. 普通模式按配置渲染 plain/speaker/timestamp text。
4. 会议模式保留完整结构化 segment。

#### 影响

- 可靠性：高收益。
- 行为变化：正常成功结果不变。
- 风险：中，主要是 stream token 拼接和最终结果一致性。
- 测试重点：取消、长音频、空 prompt、自定义 prompt、hotwords、无 EOS、模型错误。

### 7.2 MOSS 实时输出

MOSS 当前实时能力是窗口式增量转录：

- partial window 最多约 2.5 秒。
- final window 通常约 4 秒。
- 窗口结果追加为 confirmed text。

它不是类似 Voxtral 的原生连续 decoder cache。因此：

- 不应向用户宣称“原生 streaming”。
- `DelayPreset` 的 token promotion 配置对 MOSS 不生效。
- MOSS 适合普通功能的实时预览，但最终结果仍应在停止后完整校正。
- 会议模式应优先保存最终结构化 segments，而不是把 preview 当最终数据。

需要修复的包行为：

- decode error 不能静默 `return`。
- stop flush 的错误必须结束为失败状态。
- stats 应标明 window duration 和 final/partial pass。
- `maxDecodeWindows` 不应间接承担 MOSS window size；应增加明确的 `windowDurationSeconds`。

### 7.3 Nemotron

#### 当前问题

包同时保留了旧的 `NemotronASRStreamingSession` 和更新后的 `NemotronASRStreamSession`。Voxt 使用旧 session，而新 session：

- 只处理新音频并复用 encoder/RNNT cache。
- 约束 lazy graph，避免 session 越长计算图越大。
- 最终结果与整段 streaming 路径对齐。
- 暴露时间戳 segments。
- 支持 80、160、320、560、1120ms latency ladder。

#### 方案

优先在包中让统一 event session 封装新 raw session，而不是在 Voxt 再写第二个 wrapper。

语言应在模型加载后根据 `promptDictionary` 解析。不能把 App 的 `zh` 直接假设为模型接受的 key；模型可能要求 `zh-CN` 等 locale。

#### 影响

- 长时间实时转录性能和内存预计明显改善。
- 较小 chunk 会降低延迟但可能增加 WER。
- 切换实现后必须对比最终文本、partial 顺序、时间戳和峰值内存。

### 7.4 Voxtral

Voxt 已使用 `makeStreamSession`，这是正确方向。剩余工作：

- 把私有 event wrapper 下沉到包统一 session 协议。
- 暴露 transcription delay。
- 温度固定为 0，不向普通用户提供采样设置。
- 不展示无效的“跟随主语言”配置；Voxtral 本身处理多语言，不使用 Voxt 当前传入的 language 字段。

建议表单预设：

| 档位 | delay | 定位 |
|---|---:|---|
| 最快 | 240ms | 最低延迟，允许更多识别误差 |
| 平衡 | 480ms | 官方推荐默认 |
| 准确 | 960ms | 更稳定文本 |
| 字幕 | 2400ms | 高延迟、高稳定性 |

保留高级值：80ms 的倍数，80-1200ms，以及 2400ms。

### 7.5 Qwen3-ASR

已经获得的收益：

- chunked prefill 降低长音频 prefill 图规模。
- async evaluation 改善 token loop。
- cancellation 已传播。
- greedy loop guard 限制重复 token 卡死。
- context bias 已用于字典和术语。
- 自动语言检测可用于混合语言。

待修复：

- 普通 batch 在 Automatic 时传 `nil`，但 native live 在没有 hint 时重新强制用户主语言，两个路径行为不一致。
- 专用 `generateStream` 调用没有传 repetition fields，因此未来增加高级配置时不会生效。
- streaming stats 被 App 丢弃。

建议配置：

- 语言：Automatic / 跟随主语言。
- Recognition Context：保持现有字典模板。
- Live Latency：Realtime / Agent / Subtitle。
- Advanced：repetition penalty/context，仅高级测试可见，默认保持包安全值。

### 7.6 Cohere

现有表单已经覆盖主要有效配置：

- 明确语言。
- 标点和大小写。
- max output tokens。
- temperature。
- 固定切段或 VAD 长音频策略。

需要补充：

- 语言限制为官方 14 种：`en/de/fr/it/es/pt/el/nl/pl/vi/zh/ar/ja/ko`。
- 不提供 Automatic；官方模型没有显式自动语言检测。
- 不提供 timestamps/diarization，因为模型不支持。
- 可增加 Qwen/Cohere streaming 共用的 live delay preset。
- VAD 失败和无语音不能自动退回整段音频。

### 7.7 Canary

当前 fork 和 App 已支持：

- 25 种欧洲语言 ASR。
- X 到 English 翻译。
- English 到 X 翻译。
- 标点和大小写。
- max tokens 和 temperature。

剩余差距是 timestamp。官方 Canary 支持 ASR 的词级/段级时间戳和翻译的段级时间戳，但当前 Swift 实现始终加入 `<|notimestamp|>`。

方案：

1. 包增加 `timestamps` task option。
2. tokenizer 选择 timestamp token。
3. decoder 输出解析成 typed segments。
4. App 增加“时间戳”开关，仅在包完成后显示。

### 7.8 Whisper

当前包实现：

- 自动或指定语言。
- 固定 30 秒窗口。
- 每个窗口输出 segment。
- tokenizer 已认识 transcribe/translate 和 timestamp token。

当前 App 的 Recognition Preset 只改变 `chunkDuration/minChunkDuration`，但 Whisper 不使用这些值，因此该控件是无效配置。

方案：

- 立即隐藏 Whisper Recognition Preset。
- 保留 temperature 和语言。
- 消费当前已有的 30 秒 segments。
- 后续包增加 task 和 timestamp parsing 后，再提供 Translate to English 与真实 token timestamp。

### 7.9 Parakeet

Parakeet v3 是 25 种欧洲语言自动识别模型，不是英文专用模型，也不需要调用方语言提示。

方案：

- 更新模型描述和语言能力表。
- 不显示“跟随主语言”；该字段当前只被写入 `STTOutput.language`，不参与模型推理。
- 消费 aligned word/sentence segments。
- 文件批量转录或会议积压处理可评估 `generateBatch`，实时会议不要为了 batch 等待而增加延迟。

### 7.10 SenseVoice

现有实现保留：

- `zh/en/yue/ja/ko/auto` 路由。
- ITN。
- language/emotion/event metadata。
- 长音频 metadata 合并。

可直接简化：

1. 使用 `SileroVAD.getSpeechTimestamps` 替换 `predictProba` 后的自定义概率状态机。
2. 继续用 Voxt 的 `splitSenseVoiceRange` 增加 overlap。
3. 无语音仍返回结构化错误，不转整段识别。

## 8. 其他模型配置

| 模型 | 当前配置 | 建议 |
|---|---|---|
| Moonshine | max tokens、temperature、English | 已足够，继续隐藏支持 |
| Wav2Vec2 | 无采样配置、English | 正确，继续隐藏支持 |
| MMS | ISO-639-3 自由文本 | 包公开 `availableAdapterLanguages`，App 改为 picker；保留手工高级输入 |
| LASR CTC | checkpoint 决定语言/词表 | 不增加无效配置，继续隐藏 |
| GLM-ASR | 通用 preset/language | 建立明确 zh/en capability，避免 generic 推导 |
| Granite Speech | prompt bias | 保留专用 prompt；明确支持语言列表 |
| FireRed ASR 2 | 通用语言/离线识别 | 继续隐藏；不标 realtime |

## 9. VAD、端点检测和说话人链路

### 9.1 三种 VAD 音频策略

新增 `MLXVADAudioPolicy`：

```swift
enum MLXVADAudioPolicy {
    case gateOnly
    case trimSilence
    case segment
}
```

含义：

- `gateOnly`：完整音频继续送给模型，只用 VAD 抑制无语音最终结果和触发 endpoint。适合 MOSS 结构化输出、Voxtral、Nemotron 和要求原始时间轴的会议。
- `trimSilence`：删除长静音但保留前后 context。适合普通 Qwen/Cohere dictation，减少幻觉和计算。
- `segment`：按 speech ranges 独立推理。适合 SenseVoice 和长文件处理。

当前 Voxt 对普通 MLX 录音统一使用过滤后音频，且 native live feed 也读取过滤 buffer。目标架构应按 capability 决定策略，避免时间戳和流式模型的时间轴被压缩。

### 9.2 Silero v6

包已支持 `mlx-community/silero-vad-v6`。Voxt 固定使用 v6，不再提供 v5 运行时选项：

- VAD 模型管理、实时 gate、会议和长音频分段共享单一 v6 repo。
- 已有 v5 模型目录不主动删除，但不再作为失败回退。
- 使用中文、远场、电话和噪声语料校准 v6 的 threshold、speech recall、false positive 和 endpoint delay。

### 9.3 SpeechSegmenter 策略缺口

包当前：

- 把概率按约 256ms block 聚合。
- 合并短间隔 speech runs。
- 按 max chunk 切分。
- 无语音时返回整个音频。

建议增加：

```swift
enum NoSpeechPolicy { case returnEmpty, useWholeAudio }
enum VADFailurePolicy { case throwError, fixedChunks, useWholeAudio }
```

同时增加 overlap 或公开 speech ranges，让 App 决定如何切 audio。Voxt 生产默认应使用 `returnEmpty + throwError`，显式选择 fixed chunks 时才允许 fixed fallback。

### 9.4 SmartTurn

SmartTurn 判断“用户是否完成一个表达”，不是判断“当前帧有没有人声”。正确组合：

```text
Silero/OmniVAD 检测 speech end
        -> 收集最近最多 8 秒语音
        -> SmartTurn 判断 endpoint
        -> 触发 intermediate/final action
```

用途：

- 缩短固定 2 秒 trailing silence。
- 避免用户短暂停顿时过早提交。
- 改善连续 dictation 的交互节奏。

上线前必须测试中文和中英混合语音。SmartTurn 不应成为默认依赖，先作为实验模型和 feature flag。

### 9.5 FSMN VAD

FSMN VAD 当前更适合离线或长音频检测，没有与 Silero 相同的轻量 streaming state API。它可以作为中文会议/文件测试后端，但不应直接替换实时 Silero。

若后续需要接入，应先在包中抽象通用 speech-range protocol，再避免 App 针对 FSMN 增加旁路。

### 9.6 Sortformer

Voxt 已使用 Sortformer 做：

- 会议实时 speaker activity。
- 最终会议 speaker diarization。

上游本轮优化减少了 prediction 到 segments 的读取成本，Voxt 通过依赖更新已经自动获得收益。后续只需要增加 benchmark，不需要重写引擎。

MOSS 已带 speaker labels 时，继续跳过二次 Sortformer 是正确策略；否则会增加耗时并可能覆盖模型原生 speaker consistency。

## 10. 模型配置表单目标

### 10.1 表单生成原则

- 表单由 capability 驱动，不由 `MLXModelFamily.generic` 推导。
- 不显示模型不消费的参数。
- 不把所有 `StreamingConfig` 内部参数暴露给普通用户。
- 普通模式显示稳定预设；模型调试模式可显示高级值。
- dictation 和 meeting 可保存不同配置，MOSS 当前做法推广到确有业务差异的模型。

### 10.2 建议配置矩阵

| 模型 | 普通设置 | 高级/测试设置 |
|---|---|---|
| Qwen | 语言模式、context、live latency | repetition penalty/context、decode interval |
| Voxtral | transcription delay | max tokens/session、原始 delay value |
| Nemotron | language/auto、stream latency | prompt locale、chunk ms |
| Cohere | 明确语言、标点、长音频策略、max tokens | temperature、live delay、VAD profile |
| MOSS Dictation | output mode、hotwords、custom prompt、max tokens | repetition、window duration |
| MOSS Meeting | timestamp/speaker/event output、hotwords、max tokens | repetition、window duration |
| Canary | task、source/target、标点、max tokens | temperature、timestamps（包完成后） |
| Whisper | language、temperature | task/timestamps（包完成后） |
| Parakeet | 长音频 preset | batch size（离线测试） |
| SenseVoice | language、ITN | VAD profile |
| MMS | adapter picker | ISO-639-3 手工输入 |

### 10.3 VAD 表单

VAD 是 workflow 配置，不应复制到每个模型表单。建议提供：

- Backend：Automatic / Silero / OmniVAD / Energy / Off。
- Silero Model：固定 v6，只展示下载状态。
- Sensitivity：Low / Standard / High。
- Endpoint：Fast / Balanced / Stable。
- SmartTurn：Off / Assist。
- Advanced：threshold、min speech、min silence、speech pad、max segment。
- Dictation 和 Meeting 使用独立保存值。

当前存在 `meetingSileroVADSensitivity` preference key，但没有完整运行时消费和表单，可以在能力重构时统一处理，避免继续保留无效偏好。

## 11. App 侧重构清单

### 11.1 `MLXModelSupport`

- 增加 capability registry。
- 模型可见性与能力分开管理。
- 修正 Parakeet v3 描述和语言列表。
- 为 Cohere、MOSS、Voxtral、Nemotron、SenseVoice 建立精确语言集合。
- 保留 legacy repo canonical mapping。

### 11.2 `ASRHintResolver`

- 根据 capability 决定 automatic/required/unsupported。
- Qwen live Automatic 传 `nil`。
- Nemotron 在模型加载后将 base language 映射到真实 prompt dictionary key。
- Voxtral/Parakeet 不发送无效 language hint。
- Cohere 遇到不支持语言时不静默退回 English，应阻止选择或显示明确配置错误。

### 11.3 `MLXTranscriber`

- MOSS 改 throwing stream。
- 统一 live session factory 和 setup lifecycle。
- 保留完整 `STTOutput`，不只保存 text。
- stats 只在结束时汇总记录，避免实时日志过量。
- VAD audio policy 按模型和 purpose 选择。
- typed segments 转成普通 history metadata 或 meeting segments。
- 取消、stop、model pin/unpin 使用同一生命周期实现。

### 11.4 Meeting

- MOSS speaker segment 继续作为最高优先级 speaker 数据。
- 非 MOSS 模型使用原生 timestamp segment 改善会议边界，但 speaker 仍来自音频 source 或 Sortformer。
- Nemotron/Parakeet timestamps 不应被误当 diarization。
- 保持原始时间轴时禁止 trim silence。
- 完整资产处理与实时 chunk 处理分开选择参数。

### 11.5 Model Settings

- 删除无效控件。
- 增加 Voxtral/Nemotron family。
- MMS 改 adapter picker。
- MOSS 增加 max tokens 和 event preset。
- VAD 配置移到 workflow/global section。

## 12. fork 侧改造清单

### P0：可靠性合同

1. MOSS stream 支持 prompt。
2. MOSS 同步 fatal path 不再作为推荐 API。
3. live session 错误可传播。
4. `StreamingInferenceSession` unsupported model 从 `preconditionFailure` 改为 throwing/failable factory。

### P1：统一结果和 session

1. typed transcript segment。
2. language provenance。
3. common live session protocol。
4. Voxtral/Nemotron event adapters。
5. ended event 返回完整 output，而不是只有 text。

### P1：VAD 语义

1. no-speech policy。
2. failure policy。
3. 公开 speech ranges 或 overlap。
4. Cohere/Voxtral 不自行吞掉 VAD 错误。

### P2：模型任务

1. Canary timestamps。
2. Whisper task/translate/timestamps。
3. MMS available adapter languages。
4. MOSS 明确 window duration。

### P2：本地加载

增加统一 local loader：

```swift
STT.loadModel(modelDirectory:modelType:)
```

完成后 Voxt 可以减少模型类型分派，但模型存储、下载、repair 和 shadow 仍由 App 管理。

### P3：SwiftPM 构建结构

审查显示 `MLXAudioSTT` 声明依赖 `MLXAudioCodecs`，但 STT source 当前没有直接 import `MLXAudioCodecs`。应验证删除该 target dependency 后是否完整构建。

如果可删除，收益包括：

- Voxt 干净构建不再编译约 45 个 codec Swift 文件。
- 新 TTS codec 不会继续扩大 ASR App 的编译成本。
- 依赖图更符合模块职责。

如果部分 STT 模型未来确实需要 codec，应拆成可选 model target，而不是让全部 STT 用户承担整个 codec target。

包的基础依赖也发生了需要单独验证的变化：

- `mlx-swift-lm` 的最低要求从 `2.30.3` 提升到 `3.31.3`。
- `MLXAudioSTT` 和 `MLXAudioCodecs` 新增 `MLXFast`，STT 还新增了 `MLXAudioVAD` target dependency。
- Voxt 自己也直接 pin 并消费 `mlx-swift-lm`，所以重新解析版本会同时影响 ASR 和本地 LLM/VLM；这些变化会扩大 SwiftPM 解析、链接和 clean build 的影响面，但不会自动改善 Voxt 的业务链路。

建议在 fork CI 中增加最小产品构建矩阵：分别只依赖 `MLXAudioCore + MLXAudioSTT`、`MLXAudioVAD` 和完整 package，防止某个新增模型把全部 ASR 消费者的依赖图继续扩大。Voxt 侧需要比较升级前后的 clean build 时间、最终 App 二进制依赖、首次模型加载、并发取消和峰值内存，并回归本地 LLM/VLM 的 tokenizer 加载、文本生成和取消。

`AudioUtils.StreamingWAVWriter` 已加入可复用 `AVAudioPCMBuffer`，并支持从 `MLXArray` 或 `Data` 写入。Voxt 当前没有调用它，不建议仅为了采用新 API 重写现有录音保存；只有后续链路本来就持有 MLX tensor 并需要持续写 WAV 时，才会有明确的拷贝和分配收益。

## 13. 未使用模块的处理

### 13.1 TTS

本轮新增 OmniVoice、IndexTTS、MOSS-TTS、MOSS-TTS-Nano、Irodori、Kokoro、KittenTTS、Fish Speech、Echo、Chatterbox 等大量实现。

Voxt 当前没有语音合成产品入口，也没有链接 `MLXAudioTTS`。结论：

- 不新增 TTS product dependency。
- 不增加 TTS 模型目录和下载 UI。
- 不为 TTS 新增模型存储抽象。
- 如果以后加入“朗读结果”功能，单独立项评估系统 TTS 与本地 MLX TTS，而不是借本轮 ASR 重构顺带接入。

### 13.2 Codecs

Codecs 对 Voxt 当前没有直接产品能力。重点是隔离编译依赖，不是展示 codec 模型。

### 13.3 DeepFilterNet

DeepFilterNet 可能改善噪声会议和远场识别，但会新增 `MLXAudioSTS`、模型下载、额外延迟和音质失真风险。

建议后续实验顺序：

1. 仅文件转录或会议最终处理。
2. 比较 raw 与 enhanced 的 WER/CER、VAD recall 和 speaker diarization。
3. 确认收益后再考虑 live，且必须可以关闭。

不建议将语音增强作为 ASR 失败时的隐式 fallback。

### 13.4 Language Identification

Voxt 当前可利用 Qwen、Whisper、Nemotron等模型自己的 auto language。单独引入 `MLXAudioLID` 会增加模型和延迟。

只有在需要跨模型统一语言检测、并且现有模型无法提供可靠 language metadata 时，再单独评估 LID 前置阶段。

## 14. 分阶段实施计划

### Phase 0：基线与测试资产

- 固定当前 fork commit 和 App branch。
- 建立可重复音频集。
- 记录每个可见模型的文本、首字延迟、RTF、峰值内存、取消行为。
- 增加静音、短音频、长音频和错误模型目录测试。

影响：无产品行为变化，是后续比较基线。

### Phase 1：包可靠性

- MOSS prompt-aware throwing stream。
- streaming error propagation。
- unsupported live model 改 throwing factory。
- VAD no-speech/failure policy。

影响：错误从崩溃/静默变成可观测失败；需要更新 App 错误展示和测试。

### Phase 2：直接能力升级

- Nemotron 新增量 session。
- Voxtral/Nemotron latency 配置。
- Qwen live Automatic 修复。
- Silero v6 固定运行时和下载入口。
- SenseVoice 使用 `getSpeechTimestamps`。

影响：实时行为和延迟可能变化；通过 feature flag 分模型上线。

### Phase 3：能力和表单重构

- capability registry。
- 精确语言矩阵。
- model-specific settings。
- 删除无效表单。
- MMS adapter picker。

影响：设置存储结构变化；必须保留 Codable 默认值和旧配置迁移。

### Phase 4：结构化结果

- 包 typed segments/language provenance。
- App 保留完整 output。
- 会议使用非 MOSS timestamps。
- history/debug 展示检测语言和性能摘要。

影响：跨 Transcriber、Meeting、History；属于高 blast-radius 改动，应独立分支和完整测试。

### Phase 5：架构收敛

- common live session protocol。
- 删除 Voxt 私有 Voxtral wrapper。
- 合并 live setup/lifecycle。
- local directory loader。
- 验证移除 STT -> Codecs dependency。

影响：主要是维护性和构建速度，用户行为原则上不变。

### Phase 6：实验能力

- SmartTurn endpoint assist。
- FSMN VAD A/B。
- DeepFilterNet 文件/会议增强。
- Parakeet batch backlog。

影响：全部先隐藏或 feature flag，不进入默认稳定路径。

### 当前实施状态

截至 2026-07-10，推荐的首批实施范围已经完成：

- MOSS 最终推理改为 prompt-aware throwing stream，并保留最终结构化 segments。
- streaming failure 通过事件传播，App 会记录失败并释放 live session。
- Cohere/Voxtral 使用严格 VAD 路径；无语音返回空结果，不再回退整段识别。
- Qwen live Automatic 保持自动语言检测，不再强制映射为用户主语言。
- SenseVoice 长音频检测复用 `SileroVAD.getSpeechTimestamps`，保留 Voxt overlap 分段。
- fork 已发布 `v0.1.3-voxt.1`，Voxt 使用 exact version 固定。

第二批直接能力升级也已完成：

- Nemotron event session 改为封装缓存式 `NemotronASRStreamSession`，不再使用旧的重复 mel/encoder 状态实现。
- Nemotron 增加 80/160/320/560/1120ms streaming latency 表单，并按 checkpoint prompt dictionary 解析 locale。
- Voxtral 增加 240/480/960/2400ms transcription delay 表单并传入 native session。
- Silero 固定使用 v6 并提供下载入口；实时 gate 与长音频分段共享同一 checkpoint。
- 该阶段发布为 `v0.1.3-voxt.2`；当前基线已继续推进到 `v0.1.3-voxt.7`。

验证结果：fork 首批 7 项测试及第二批 Nemotron 定向测试、Voxt 125 项 ASR/VAD/设置定向测试和 Voxt 完整构建均通过。真实模型取消、静音语料及 30 分钟 session 仍需按上线门禁进行人工回归和性能采样。

## 15. 影响和复杂度矩阵

| 项目 | 收益 | 风险 | 改动范围 | 建议优先级 |
|---|---|---|---|---:|
| MOSS throwing stream | 防崩溃、正确取消 | 中 | fork + transcriber | P0 |
| streaming error propagation | 可诊断、无静默失败 | 中 | fork + live UI | P0 |
| VAD no-speech policy | 减少静音幻觉 | 中 | fork + Cohere/Voxtral | P0 |
| Nemotron 新 session | 延迟/内存/准确性 | 中高 | fork adapter + transcriber | P1 |
| Qwen Automatic 修复 | 行为正确 | 低 | transcriber + tests | P1 |
| capability registry | 消除重复和错误表单 | 中 | catalog/settings/resolver | P1 |
| Voxtral/Nemotron latency | 用户可测试质量/延迟 | 中 | settings + session | P1 |
| Silero v6 | 噪声和边缘场景潜在改善 | 中 | model manager + settings | P1 |
| SenseVoice timestamp API | 减少重复实现 | 低中 | transcriber + tests | P1 |
| typed output | 时间戳和 metadata 统一 | 高 | fork + transcriber + meeting/history | P2 |
| Canary timestamps | 新能力 | 中高 | tokenizer/decoder/App | P2 |
| Whisper task/timestamps | 翻译和时间戳 | 中高 | tokenizer/decoder/App | P2 |
| local loader | 减少加载分派 | 中 | fork + model manager | P2 |
| 移除 Codecs dependency | 减少编译成本 | 中 | Package.swift + CI | P2 |
| SmartTurn | 更自然 endpoint | 高不确定性 | 新模型 + capture | P3 |
| DeepFilterNet | 噪声增强潜力 | 高不确定性 | 新 product + audio pipeline | P3 |

## 16. 验证方案

### 16.1 音频语料矩阵

至少覆盖：

- 中文普通话、英文、中英混合。
- 粤语、日语、韩语。
- 单人近场、双人会议、多人会议。
- 系统音频、麦克风、混合 source。
- 安静、键盘、风扇、音乐、人群、电话压缩。
- 短于 1 秒、5-30 秒、5 分钟、30 分钟以上。
- 完全静音、只有环境声、只有呼吸/咳嗽。
- 快速 stop、ESC cancel、录音中切模型、睡眠唤醒。

### 16.2 指标

文本质量：

- 中文 CER。
- 英文/欧洲语言 WER。
- 热词命中率。
- prompt/context 泄漏率。
- 重复输出和异常尾字符率。

实时体验：

- 首次可见文本延迟。
- 最新文本刷新间隔。
- provisional 回退次数。
- stop 到 final 完成时间。
- partial/final 一致性。

性能：

- Real-time factor。
- token/s。
- 峰值 MLX memory。
- 30 分钟 session 的内存增长。
- clean/incremental Xcode build 时间。

结构化能力：

- timestamp 偏差。
- segment 覆盖率和重叠。
- speaker DER 或至少 speaker consistency。
- MOSS segment parse 成功率。

VAD：

- speech recall。
- false activation rate。
- endpoint delay。
- 静音 hallucination rate。
- v6 在中文、远场、电话和噪声语料上的阈值基线。

### 16.3 自动测试

fork tests：

- prompt 传播。
- cancellation 不触发 fatal。
- stream failure。
- no-speech policies。
- typed segment decode。
- language provenance。
- Nemotron chunk ladder。
- Canary timestamp task token。

Voxt tests：

- capability -> UI control matrix。
- primary language tags。
- hint encoding。
- settings Codable migration。
- VAD audio policy。
- MOSS dictation/meeting 配置隔离。
- structured segment -> meeting/history mapping。
- live lifecycle、cancel、model pin balance。

### 16.4 上线门禁

P0/P1 合入前必须满足：

- 所有现有单元测试通过。
- `xcodebuild build` 和 `xcodebuild test` 通过。
- MOSS 取消不崩溃。
- 静音不产生 Cohere/Voxtral 最终文本。
- Qwen Automatic 在 batch/live 行为一致。
- 30 分钟 Nemotron/Voxtral/MOSS session 无持续非预期内存增长。
- 模型旧设置和已下载目录仍可识别。

## 17. 兼容、迁移与回滚

### 17.1 设置迁移

- `MLXLocalTuningSettings` 新字段全部提供默认值。
- 保留 family key，必要时增加 repo-specific override。
- 旧 generic preset 可读取，但对不支持的模型不再显示或应用。
- MOSS dictation/meeting 现有设置不合并。
- MMS 旧文本 language code 自动映射到 picker selection。

### 17.2 模型目录

- 不移动现有 ASR 模型目录。
- Silero v6 使用独立 repo 目录；已有 v5 目录不删除但不再读取。
- canonical repo migration 保留。
- package local loader 必须接受现有目录，不重新下载。

### 17.3 fork 发布

每一批包修改：

1. 从当前 fork branch 派生小范围分支。
2. 补包测试。
3. 推送并创建新的 `v0.1.3-voxt.N` annotated tag。
4. Voxt 从 revision 临时 pin 切到 exact Voxt tag。
5. 上游接受同等实现并发布后，再迁回 upstream release tag。

不要把全部重构堆进一个长期不可回收的 fork commit。

### 17.4 回滚

- package 每个 phase 独立 tag。
- App 以 feature flag 控制 Nemotron 新 session 和 SmartTurn；Silero 固定使用 v6。
- typed output 迁移期同时保留 text-only 路径。
- 回滚不得删除新版本写入的用户配置；旧版本应忽略未知字段。

## 18. 明确不采用的方案

- 不把静默 fallback 当作正常错误处理。
- 不在 App 中复制新的模型 decoder 或 VAD 核心算法。
- 不用字符串正则作为所有模型结构化结果的长期合同。
- 不把 model prompt、语言和 task 全塞入无类型字典。
- 不让用户看到模型不消费的配置。
- 不让会议时间戳依赖被 trim 后的压缩音频时间轴。
- 不为未使用 TTS 功能扩大 Voxt 当前 package product 集合。
- 不在没有基线 benchmark 时默认启用 SmartTurn、FSMN 或 DeepFilterNet。

## 19. 推荐的首批实施范围

建议第一个实施分支只包含以下内容：

1. fork：MOSS prompt-aware throwing stream。
2. fork：stream error propagation。
3. fork：VAD no-speech/failure policy。
4. App：MOSS 最终推理切 stream。
5. App：Qwen live Automatic 修复。
6. App：SenseVoice 使用 `getSpeechTimestamps`。
7. 对应 package/App 单元测试与取消、静音回归测试。

这批改动优先解决崩溃、静默失败和静音幻觉，不同时引入 typed output 和 UI 大重构，便于验证和回滚。

第二批再处理 Nemotron 新 session、Voxtral/Nemotron latency 和 Silero v6；第三批处理 capability registry 与表单；第四批处理 typed output 和会议时间戳。

## 20. 参考资料

项目内：

- `docs/MLXAudioDependency.md`
- `docs/VADASROptimizationPlan.zh-CN.md`
- `Voxt/Transcription/MLXTranscriber.swift`
- `Voxt/Transcription/MLXModelManager.swift`
- `Voxt/Transcription/MLXModelSupport.swift`
- `Voxt/Core/Transcription/ASRHintLocalTuning.swift`
- `Voxt/Core/Transcription/ASRHintResolver.swift`
- `Voxt/Core/Transcription/ASRVoiceActivityPlanning.swift`
- `Voxt/Meeting/Capture/MeetingVoiceActivity.swift`
- `Voxt/Meeting/MeetingRealtimeDiarization.swift`

上游与模型资料：

- [mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift)
- [MOSS Transcribe Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B)
- [Voxtral Mini 4B Realtime](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602)
- [Cohere Transcribe](https://docs.cohere.com/docs/transcribe)
- [NVIDIA Nemotron 3.5 ASR](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
- [NVIDIA Canary 1B v2](https://huggingface.co/nvidia/canary-1b-v2)
- [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
- [SenseVoice Small](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)
- [Silero VAD v6 MLX](https://huggingface.co/mlx-community/silero-vad-v6)
- [Silero VAD releases](https://github.com/snakers4/silero-vad/releases)


本地 mxl-audio-swift repo path： /Users/guanwei/x/doit/mlx-audio-swift
