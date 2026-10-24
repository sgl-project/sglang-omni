# 会议详情左侧转录列表优化

日期：2026-08-02  
对齐：[MeetingLocalPerformanceSafetyOptimizationPlan.zh-CN.md](MeetingLocalPerformanceSafetyOptimizationPlan.zh-CN.md) §8.3

## 目标

- 降低长会议详情页滚动 / 播放跟播时的主线程压力。
- 时间轴与发言标记列表保持**稳定、均匀**的卡片间距与正确多行高度。

## 最终实现

### 保留的性能手段

- 进度条继续订阅 `currentTime`；列表只在 `activeSegmentID` 变化时无动画滚动。
- `displayedSegments` / `speakerGroups` 缓存在 `MeetingDetailViewModel`。
- 列表包在 `Equatable` 子视图中，降低纯进度刷新带来的无效更新。
- 行模型扁平化为 `MeetingTranscriptVirtualRow`（Timeline / Speaker Marks 共用）。

### 列表承载

使用 SwiftUI `ScrollView` + `LazyVStack`（见 `MeetingTranscriptVirtualList`）。

曾尝试 `NSTableView` + `NSHostingView`（手动行高 / GeometryReader 回写 / `usesAutomaticRowHeights`）。在换行 `Text` 场景下，AppKit 测得的行高经常大于真实卡片高度，表现为：

- 多行卡片下方空白偏大
- 间距忽大忽小
- 卡片内部上下 padding 视觉不对称

因此放弃 AppKit 动态行高路径，改回 SwiftUI 懒加载列表以保证视觉正确。长列表的主要 CPU 放大器（播放 0.15s 整表跟滚、body 内反复 group/filter）已在上层消掉。

## 关键文件

| 路径 | 角色 |
|------|------|
| `Voxt/Windows/MeetingDetail/MeetingTranscriptVirtualList.swift` | LazyVStack 列表 + scrollRequest |
| `Voxt/Windows/MeetingDetail/MeetingTranscriptVirtualRows.swift` | 行模型 / flatten / diff 辅助 |
| `Voxt/Windows/MeetingDetail/MeetingTranscriptRowHeightCache.swift` | 文本高度估算（测试与后续备用） |
| `Voxt/Windows/MeetingDetail/MeetingDetailSegmentRow.swift` | 段行 / speaker header |
| `Voxt/Windows/MeetingDetail/MeetingDetailViewModel.swift` | 列表缓存 |
| `Voxt/Windows/MeetingDetail/MeetingDetailWindow.swift` | 接入与跟播 |

## 验收

- 单元：`VoxtTests/MeetingTranscriptVirtualListTests.swift`
- 手动：时间轴 / 发言标记下多行与单行卡片间距一致；播放跟播不带动画抖动；开关翻译、搜索、侧栏折叠正常。
