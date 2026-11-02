# 会议性能与安全修复总结

日期：2026-06-12

维护说明：下文是当时的修复和验证记录。测试拆分后，Doubao 压缩包边界回归位于 `RemoteModelConfigurationASRTests`；当前完整聚焦入口为 `bash tools/run_local_regression_matrix.sh refactor`，不要仅运行旧配置 suite 就认为已覆盖这些协议测试。当前文件结构见 [源码地图](Architecture.md)。

## 修复范围

本次修复覆盖前次审核中除日志外的项目：

- 长会议音频归档和最终处理的内存峰值问题。
- Doubao 实时会议协议包 gzip 解压缺少上限的问题。
- 会议详情页和说话人转写装配在长会议下的 CPU 放大问题。

日志相关问题按要求未处理。

## 已修复内容

### 1. 长会议音频内存峰值

涉及文件：

- `Voxt/Meeting/MeetingAudioArchive.swift`
- `Voxt/Meeting/MeetingSessionCoordinator.swift`
- `Voxt/Meeting/MeetingFinalTranscriptProcessor.swift`
- `Voxt/Meeting/MeetingSpeakerAnalysis.swift`

修复前，会议期间会把 microphone 和 system audio 的完整 Float 数组常驻内存；停止会议时还会再生成完整混音数组、完整最终转写资产和完整说话人分析资产。会议越长，内存峰值按时长线性增加。

修复后：

- 会议音频写入临时目录中的 Float32 分段文件，不再把整段会议音频常驻内存。
- 每个磁盘分段为 300 秒，即 5 分钟。
- 最终转写和说话人分析使用 `MeetingAudioAssetDescriptor` 描述符，逐段按需读取。
- 单个最终处理资产最大 300 秒，即 5 分钟。
- WAV 导出按 60 秒窗口混音和写入，不再一次性构造完整混音 PCM。
- `MeetingAudioArchive.reset()` 和 actor 释放时会清理临时分段目录。

用户侧语义：

- 正常长会议不会因为到达 5 分钟而停止或报错；5 分钟只是内部处理窗口。
- 用户仍然获得完整会议音频、最终转写和说话人分析。
- 只有写入临时文件失败、导出 WAV 超过 WAV 格式 4GB 数据块上限等真实资源错误才会进入异常路径。

### 2. Doubao gzip 解压防护

涉及文件：

- `Voxt/Meeting/MeetingRemoteProviderLiveSession.swift`
- `VoxtTests/RemoteModelConfigurationTests.swift`

修复前，Doubao server packet 使用 gzip 时会无上限解压；如果解压失败还会回退为原始 payload 继续解析。

修复后：

- gzip 压缩 payload 最大 2 MB。
- gzip 解压 payload 最大 8 MB。
- gzip 膨胀倍率最大 64 倍；小 payload 至少允许 1 MB 解压空间，避免误伤正常小包。
- gzip 解压失败、超过压缩体积、超过解压体积或超过膨胀倍率都会抛错，停止解析该包。
- 不再使用“解压失败后按原始 payload 继续解析”的 fail-open 行为。

用户侧语义：

- 正常 Doubao 实时转写包不受影响。
- 异常大包或 gzip bomb 会被拒绝；这是安全保护，属于异常输入场景。

### 3. 长会议 UI 和说话人装配 CPU

涉及文件：

- `Voxt/UI/MeetingDetailWindow.swift`
- `Voxt/Meeting/MeetingSpeakerTranscriptAssembler.swift`

修复前：

- 详情页每次播放时间变化都会线性扫描 transcript segments。
- 每个 transcript row 计算说话人标题时都会重新排序全部 segments 并重建序号表。
- 说话人装配每个 segment 都会扫描全部 speaker turns。

修复后：

- 详情页说话人序号表缓存为 view state，segments 变化时才刷新。
- 播放 active segment 查找改为二分查找。
- 说话人 turn 先按 audio source 建索引；segment 只扫描相关时间窗口附近的 turns。

用户侧语义：

- 显示结果不改变。
- 长会议详情页滚动、播放定位和说话人分析装配的 CPU 开销降低。

## 关键阈值

| 项目 | 当前阈值 | 达到阈值后的行为 |
| --- | ---: | --- |
| 音频磁盘分段 | 300 秒 / 5 分钟 | 内部分段，不影响录制和最终结果 |
| 最终处理资产窗口 | 300 秒 / 5 分钟 | 逐段读取和处理，不影响完整结果 |
| WAV 导出混音窗口 | 60 秒 | 分窗口写入 WAV，不影响导出内容 |
| Doubao gzip 压缩体 | 2 MB | 超过则拒绝该异常包 |
| Doubao gzip 解压结果 | 8 MB | 超过则拒绝该异常包 |
| Doubao gzip 膨胀倍率 | 64 倍 | 超过则拒绝该异常包 |

## 验证结果

已执行：

```bash
xcodebuild build -project Voxt.xcodeproj -scheme Voxt -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO
```

结果：通过。

已执行 focused tests：

```bash
xcodebuild test -project Voxt.xcodeproj -scheme Voxt -destination 'platform=macOS' CODE_SIGNING_ALLOWED=NO -only-testing:VoxtTests/MeetingAudioArchiveTests -only-testing:VoxtTests/RemoteModelConfigurationTests
```

结果：通过，执行 55 个测试，0 个失败。

## 新增测试覆盖

- `MeetingAudioArchiveTests.testAssetDescriptorsLoadAudioOnDemand`
  - 验证音频描述符可以按需加载混音资产。
- `RemoteModelConfigurationTests.testDoubaoParserRejectsOversizedCompressedPayload`
  - 验证 gzip 压缩体超过 2 MB 会被拒绝。
- `RemoteModelConfigurationTests.testDoubaoParserRejectsHighExpansionGzipPayload`
  - 验证高膨胀比 gzip payload 会被拒绝。
