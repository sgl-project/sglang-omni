# Documentation / 文档导航

## 使用指南 / User guides

- [English introduction](../README.md) / [中文介绍](README.zh-CN.md)
- Remote models: [English](RemoteModel.md) / [中文](RemoteModel.zh-CN.md)
- Prompts: [English](Prompt.md) / [中文](Prompt.zh-CN.md)
- Rewrite: [English](Rewrite.md) / [中文](Rewrite.zh-CN.md)
- Meetings: [English](Meeting.md) / [中文](Meeting.zh-CN.md)
- [六步引导](OnboardingGuide.zh-CN.md) / [快捷键规则](HotkeyRules.zh-CN.md)

## 开发与维护 / Development

- [Contributing and build commands](../CONTRIBUTING.md)
- [Architecture and source map](Architecture.md)
- [全项目重构评估、清理依据与执行清单](RefactoringAssessment.zh-CN.md)
- [分阶段实施记录与待验收门禁](RefactoringProgress.zh-CN.md)
- [集中收尾、保留决策与最终验收清单](RefactoringCloseout.zh-CN.md)
- [全项目性能、安全与精简审查（含未解决风险）](PerformanceSecurityReview.zh-CN.md)
- [权限最小化诊断与方案评估](PermissionMinimizationAssessment.zh-CN.md)
- [文件任务分析资源调度、阶段释放与断点恢复方案](FileAnalysisResourceSchedulingPlan.zh-CN.md)
- [Sortformer 长文件退化修复（待 macOS 验收）](SortformerLongFileRepair.zh-CN.md)
- [上下文增强删除与权限精简实施记录（待 Mac 验收）](ContextEnhancementRemovalImplementation.zh-CN.md)
- [Test suites and focused checks](../VoxtTests/README.md)
- [MLX dependency policy and current pins](MLXAudioDependency.md)
- [Local regression matrix](LocalRegressionMatrix.md)
- [VAD 人工验收](VADManualAcceptance.zh-CN.md)
- [会议虚拟列表实现](MeetingDetailVirtualList.zh-CN.md)

## 方案与历史记录 / Plans and historical evidence

`*Plan*`、`*Evaluation*` 和修复/实施总结用于记录当时的约束、方案和验证证据，不自动代表当前行为或已完成状态。查当前依赖以 Xcode 项目、实际 `Package.resolved` 和依赖策略为准；查当前职责以源码地图为入口。

- [上下文增强完整删除与权限精简方案](ContextEnhancementRemovalPlan.zh-CN.md)
- [模型栈现代化方案](ModelStackModernizationPlan.zh-CN.md) / [实施记录](ModelStackModernizationImplementation.zh-CN.md)
- [ASR 跨模型优化](ASRCrossModelOptimization.zh-CN.md) / [首 partial 延迟评估](ASRFirstPartialLatencyEvaluation.zh-CN.md)
- [会议本地性能与安全优化方案](MeetingLocalPerformanceSafetyOptimizationPlan.zh-CN.md)
- [提示词分层方案](TextEnhancementPromptLayeringPlan.zh-CN.md)

历史记录中的旧路径、旧依赖和待办需结合记录上下文阅读；不应据此恢复已移除的运行时或绕过当前验收门禁。
