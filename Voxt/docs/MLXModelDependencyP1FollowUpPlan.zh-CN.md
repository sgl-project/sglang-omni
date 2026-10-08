# MLX 模型依赖 P1 后续实现计划

> 状态基准：`codex/model-dependency-p1-optimizations` 工作区  
> 依赖基线：`mlx-audio-swift` `0.1.3-voxt.10`  
> 总计划参考：[MLXAudioSwiftUpgradePlan.zh-CN.md](./MLXAudioSwiftUpgradePlan.zh-CN.md)

## 1. 目标

把「能力元数据 → 推理参数 → 会议/听写结果」这条链路收齐，避免：

- 模型返回不可靠时间戳却被当成会议分段
- 表单展示模型不消费的控件
- Qwen 等长会话路径缺少明确的 KV / latency 策略

本文件只覆盖**Batch A 之后的剩余 P1**；P2/P3（Canary timestamps、SmartTurn、DeepFilterNet 等）仍按总计划排队。

## 2. Batch A（已完成）

| 项 | 落地位置 | 验收点 |
| --- | --- | --- |
| 会议 Silero 灵敏度表单 + 运行时 | `MeetingSileroVADSensitivity`、`FeatureMeetingSections`、`MeetingVoiceActivity` | 设置改 onset 阈值；responsive < balanced < stable |
| ASR timing 能力元数据 | `MLXASRTimingGranularity`、`MLXModelCatalog` | Whisper=`chunk`，Parakeet/Nemotron/MOSS=`sentence` |
| 可靠 structured segments | `MLXTranscriber.reliableStructuredSegments` | chunk/none 返回空；sentence/word 过滤非法区间 |
| Qwen KV cache 策略 | `MLXASRKVCachePolicy.conservativeQwen` | batch + native stream 传入量化参数 |
| MOSS speakerID 可选化 | `MLXStructuredTranscriptSegment.speakerID: String?` | 无 speaker 时不强制拼 `moss:` |
| 依赖与文档 | `0.1.3-voxt.10`、README 精简、本地化 | 包与文案对齐 |

## 3. Batch B（已完成）— Capability 驱动表单

| 项 | 落地 |
| --- | --- |
| Whisper 摘要去掉无效 preset | `MLXConfigurationSummarySupport` |
| recognitionPreset 仅作用于有该 capability 的模型 | `resolvedInferenceConfiguration` |
| SenseVoice / Parakeet / CTC 说明改 capability 驱动 | `MLXASRConfigurationSheetView` |
| 主语言不支持时 inline 警告 | `unsupportedPrimaryLanguageWarning` |
| MMS 未知 adapter 保留原值并在推理前明确失败 | `MMSLanguageAdapterOption.validatedAdapterCode` + 设置页橙色提示 |

## 4. Batch C（已完成）— 非 MOSS 会议时间戳

| 项 | 落地 |
| --- | --- |
| `usesStructuredOutput` 改为 `isFinal && timingGranularity.providesReliableSegments` | `MeetingMLXSegmentMapping` |
| Nemotron/Parakeet 时间戳不得映射为 speaker ID | 仅 `mossTranscribeDiarize` 写入 `moss:` speaker |
| `preserveTimeline` 不再按 silence evidence 丢段 | `MeetingFinalSpeechValidator` |
| 定向测试 | `MeetingMLXSegmentMappingTests`、更新 preserveTimeline 测试 |

## 5. Batch D（已完成，含 fork Phase 5 关键）

| 项 | 落地 |
| --- | --- |
| `TranscriptionEvent.ended(STTOutput)` | fork `0.1.3-voxt.12`：Qwen / MOSS / Cohere / Nemotron stop 携带 text + segments + language |
| dictation / meeting 消费 ended output | `structuredSegmentsForLiveEnded`；MOSS 走 `mossStructuredSegments`（含 speaker） |
| 删除 Nemotron 私有 adapter | 直接用包 `NemotronASRStreamingSession` |
| 可靠时间戳与 batch 对齐 | sentence/word 可用；Qwen chunk / Cohere none 仍按 capability 过滤 |
| Voxtral 仍走 Voxt wrapper | `MLXVoxtralNativeStreamingSession`（ended 已改为 `STTOutput(text:)`） |

## 6. Batch E — 回归门禁

```bash
xcodebuild test -project Voxt.xcodeproj -scheme Voxt -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO \
  -only-testing:VoxtTests/MeetingMLXSegmentMappingTests \
  -only-testing:VoxtTests/MeetingFinalSpeechValidationTests \
  -only-testing:VoxtTests/ASRHintSettingsTests \
  -only-testing:VoxtTests/ASRVoiceActivityPlanningTests \
  -only-testing:VoxtTests/MLXModelSupportTests \
  -only-testing:VoxtTests/FeatureSettingsStoreTests
```

人工抽检：Qwen live 长会话；会议 Silero responsive/stable；Parakeet 会议边界；Whisper 无假分段；MMS 选无效 adapter 应报错。

## 7. 明确不在本轮

| 项 | 原因 |
| --- | --- |
| Canary / Whisper 官方 timestamps 扩展 | P2，依赖包侧 task token |
| SmartTurn / FSMN / DeepFilterNet | P3 实验 |
| Dictation 独立 Silero 灵敏度 | 另开小任务 |
| 删除全部私有 live wrapper（含 Voxtral） | Phase 5 后续架构收敛 |

## 8. 实施状态

| Batch | 状态 | 备注 |
| --- | --- | --- |
| A | 完成 | 待与 B/C/D 一并提交 |
| B | 完成 | capability 表单 + MMS 明确失败 |
| C | 完成 | 会议 timing 门控 + preserveTimeline |
| D | App 侧完成 | 包 ended payload 留给 Phase 5 / fork |
| E | 完成 | 定向测试通过；审查 P1（Silero 0.5、Nemotron live 分段接线）与 P2（Sensitive 文案、endedSegments 同步、mlx-swift 降级说明）已修复 |

更新记录：2026-07-23 落地 B/C/D App 侧实现。
