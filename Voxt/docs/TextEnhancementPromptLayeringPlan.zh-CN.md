# 文本增强提示词分层方案

本文档评估 Voxt 在 ASR 结果进入 LLM 处理时的提示词拆分方案，目标是把“基础 ASR 清理”和“结构/语义/场景任务”解耦，让用户可以勾选不同增强能力，并在实际执行时合并为一次 LLM 请求。

## 结论

推荐把提示词拆成三层：

1. 公共基础层：只处理 ASR 噪声，不改变原句语义和组织结构。普通转录、语音翻译、转写口述指令、会议实时翻译前置清理都可以复用。
2. 可选增强层：处理列表、分段、结构整理、轻度语义顺序修复等会改变文本形态的能力。默认只用于普通转录，不作为翻译/转写/会议摘要的公共前置。
3. 任务层：翻译、转写、应用增强、会议摘要各自保留独立目标规则。任务层可以引用公共基础层，但不应把结构增强硬编码进公共规则。

运行时不要串联多次 LLM。应由一个 Prompt Composer 根据当前功能、用户勾选能力、App Branch、词典、上下文、目标语言等输入，生成一个最终 `LLMExecutionPlan.promptContent`，再由现有 `LLMExecutionPlanCompiler` 合成一次请求。

## 当前实现观察

当前主要入口：

- 普通转录：`processStandardTranscription -> runStandardTranscriptionPipeline -> enhanceTextForCurrentMode -> resolvedEnhancementPrompt -> buildEnhancementExecutionPlan`
- 翻译：`runTranslationPipeline -> translateText -> resolvedTranslationPrompt -> buildTranslationExecutionPlan`
- 转写：`runRewritePipeline -> rewriteText -> resolvedRewritePrompt -> buildRewriteExecutionPlan`
- 会议实时翻译：调用 `translateMeetingRealtimeText -> translateText`
- 会议最终转写优化：当前是重跑 ASR 和确定性文本后处理，不是 LLM 文本增强链路

需要注意一个文档/实现差异：`docs/Prompt.zh-CN.md` 仍描述“语音翻译先增强再翻译”，但当前 `runTranslationPipeline` 只包含翻译和严格重试阶段，没有独立 EnhanceStage。后续如果要恢复“翻译前基础清理”，建议不要加一个额外 LLM stage，而是把公共基础层合并进翻译 prompt。

当前默认提示词存在重复和边界混合：

- `zh-Hans-enhancement` 同时包含基础清理、数字日期规范、标点词替换、结构列表整理、App 上下文规则。
- `zh-Hans-translation` 基本复制了清理规则，再追加翻译规则。
- `en-enhancement` 与 `en-translation` 同样重复。
- 列表/分段规则已经属于结构增强，不应作为翻译、转写口述指令的公共基础前置。

当前资源文件字符量：

| 文件 | 字符数 |
| --- | ---: |
| `zh-Hans-enhancement.txt` | 1,971 |
| `zh-Hans-translation.txt` | 1,899 |
| `en-enhancement.txt` | 5,810 |
| `en-translation.txt` | 5,736 |
| `zh-Hans-rewrite.txt` | 465 |
| `en-rewrite.txt` | 1,426 |

英文 prompt 明显更长。拆分后，翻译和增强不再各自复制整段清理规则，维护成本会下降；但单次请求的实际 prompt 长度不一定总是更短，因为 Composer 会按能力拼接。收益主要来自边界清晰、可配置、可测试。

## 能力拆分

### 公共基础层

这一层可以命名为 `basicASRCleanup`，建议默认开启，且作为翻译/转写/应用增强的前置公共能力。

范围：

- 去除无语义填充词和停顿词。
- 处理明确自我修正，只保留最终确认内容。
- 修正明显 ASR 识别错误、标点、空格、大小写。
- 把明确作为标点使用的口述标点词转换为标点符号。
- 规范数字、时间、日期、百分比、单位、电话号码等。
- 保护人名、产品名、术语、命令、代码、路径、URL、邮箱、数字。
- 保持原语言混合，不翻译、不总结、不扩写、不改变文风。
- 使用主语言只作为标点习惯和歧义消解参考。
- 如果无有效内容，返回空字符串。

边界：

- 不主动拆段。
- 不主动生成列表。
- 不重排句子。
- 不压缩、总结或改写语气。
- 不用 App 上下文补全未说出的信息。

### 可选基础子能力

如果用户希望更细粒度，可以把 `basicASRCleanup` 拆成多个 UI 勾选项，但底层仍可合并为一个模块组：

| 能力 | 推荐默认 | 公共可复用 | 说明 |
| --- | --- | --- | --- |
| `fillerCleanup` | 开 | 是 | 删除“嗯、呃、like、you know”等无语义停顿 |
| `selfCorrection` | 开 | 是 | 只处理明确改口，不处理历史叙述里的“这是不对的” |
| `punctuationAndSpacing` | 开 | 是 | 标点、大小写、空格、中英间空格 |
| `spokenPunctuation` | 开 | 是 | “感叹号/逗号/括号”在标点用途下转符号 |
| `numberDateNormalization` | 开 | 是 | 百分比、时间、单位、手机号等 |
| `entityProtection` | 开 | 是 | 保护专名、代码、路径、URL、邮箱 |
| `languagePreservation` | 开 | 是 | 保持原始语言混合，主语言仅作格式参考 |

第一期 UI 不建议暴露过细。建议只暴露“基础清理”一个总开关，内部默认包含以上子能力；高级配置可以后续再做。

### 结构增强层

这一层可以命名为 `structuralFormatting`，普通转录可选开启，翻译/转写默认不继承。

范围：

- 根据明确的顺序关系整理为编号列表。
- 根据明确并列关系整理为无序列表。
- 保留或生成必要换行，让内容更易读。
- 对明显的长句做轻度分段。
- 在不改变意思的前提下整理嵌套列表。

边界：

- 不能引入标题，除非用户口述了标题或当前任务明确要求。
- 不能为了“好看”重写句子逻辑。
- 不能把普通一句话强行拆成列表。
- 不能用于翻译前置公共层，否则会让翻译 prompt 同时承担结构改写和语言转换，增加不可控性。

### 语义整理层

这一层建议命名为 `semanticPolish`，第一期不建议默认开启。

范围：

- 去除重复表达。
- 对明显口语断裂做轻度顺句。
- 对短文本保持原样，对长文本只做局部连贯性修复。

风险：

- 容易越界为改写。
- 会改变用户口述风格。
- 对命令、代码、短消息的破坏概率高。

建议：先不做独立 UI 能力，等基础层和结构层稳定后再评估。

### App 上下文层

这一层不属于基础清理，应作为场景上下文规则附加：

- 普通转录 App Enhancement：可用于指代消解、识别目标输入场景、修正明显 ASR 错误。
- 转写 App Enhancement：可用于定位“这条消息”“这里”“最新一条”等目标。
- 翻译：默认不使用 App 上下文，除非将来明确有“根据当前 App 翻译选区/口述”的需求。

App 上下文规则必须保留当前约束：只消解指代，不机械复述界面文本，不补全不可见事实。

## 推荐默认组合

| 功能 | 推荐组合 | 说明 |
| --- | --- | --- |
| 普通转录 | `basicASRCleanup + structuralFormatting` | 为保持当前默认体验，可首版默认开启结构增强；设置中允许关闭 |
| 语音翻译 | `basicASRCleanup + translationTask` | 只做保守清理后翻译，不做列表/分段结构改造 |
| 选中文本翻译 | `translationTask` | 选中文本不是 ASR 结果，默认不加基础清理 |
| 转写口述指令 | `basicASRCleanup + rewriteTask` | 清理口述指令中的 ASR 噪声，但不把指令整理成列表 |
| 转写选中文本改写 | `basicASRCleanup(dictatedPrompt only) + rewriteTask` | 只清理口述指令，不清理用户选中的源文本 |
| 应用增强普通转录 | `basicASRCleanup + optional structuralFormatting + appContextRules + appGroupPrompt` | App 规则作为场景层合入一次请求 |
| 会议实时翻译 | `basicASRCleanup + translationTask` | 复用普通翻译 prompt composer |
| 会议摘要 | `transcriptSummaryTask` | 不复用结构增强；摘要本身是独立任务 |

## Prompt 组合模型

建议新增一个纯 Swift 组合器，例如：

```swift
struct PromptCapabilitySet: Codable, Hashable, Sendable {
    var basicASRCleanup: Bool
    var structuralFormatting: Bool
    var semanticPolish: Bool
}

enum PromptTaskKind: Hashable, Sendable {
    case transcription
    case translation(targetLanguage: TranslationTargetLanguage)
    case rewrite(structuredAnswerOutput: Bool, directAnswerMode: Bool)
    case transcriptSummary
}

struct PromptCompositionInput {
    let task: PromptTaskKind
    let capabilities: PromptCapabilitySet
    let userMainLanguage: String
    let userOtherLanguages: String
    let customTaskPrompt: String?
    let appBranchPrompt: String?
    let hasAppContext: Bool
    let glossaryPurpose: DictionaryGlossaryPurpose?
}
```

输出为：

```swift
struct PromptCompositionOutput: Equatable {
    let promptContent: String
    let delivery: LLMExecutionDelivery
    let profile: String
    let enabledCapabilities: PromptCapabilitySet
}
```

组合顺序建议固定：

1. Role：说明 Voxt 当前任务角色。
2. Input policy：说明输入是 ASR 文本、选中文本、口述指令还是会议片段。
3. Public cleanup rules：按能力插入基础层。
4. Optional formatting rules：只在启用结构增强时插入。
5. Task rules：翻译、转写、普通转录输出规则。
6. Context rules：App 上下文、截图、词典等。
7. Output contract：只返回最终文本/JSON，不附加解释。

组合器应该只生成规则，不嵌入长输入文本；输入文本继续放在 `LLMExecutionPlan.task` 和 `LLMContextBlock(kind: .input)`，交给 `LLMExecutionPlanCompiler` 生成动态 user prompt。

## 提示词模块草案

### 基础 ASR 清理模块

```text
基础 ASR 清理规则：
- 只清理语音识别造成的噪声，不翻译、不总结、不扩写、不改变写作风格。
- 删除无语义填充词、重复停顿和明显口头犹豫。
- 处理明确自我修正：若说话者否定、取消或改口，只保留最后确认的有效内容；历史叙述中的正误说明不要当作自我修正。
- 修正明显 ASR 识别错误、标点、空格、大小写，以及中英文连接处空格。
- 仅当口述标点词被用作标点时，将“逗号/句号/问号/感叹号/括号/引号”等转换为对应符号；若用户是在谈论这些词本身，则保留文字。
- 规范数字、百分比、时间、日期、单位和号码格式。
- 完整保留人名、产品名、术语、命令、代码、路径、URL、邮箱和数字。
- 保持原始语言混合。用户主要语言只用于标点习惯、格式习惯和歧义消解，不是翻译目标。
- 如果清理后没有有效内容，返回空字符串。
```

### 结构增强模块

```text
结构增强规则：
- 只有当原文明确包含顺序、步骤、并列或层级关系时，才整理为列表。
- 顺序关系使用编号列表；非顺序并列关系使用“-”列表；子项使用 Markdown 嵌套列表。
- 对长文本可增加必要换行，但不要改变原句事实、语气和论证顺序。
- 不要生成原文没有暗示的标题、小节或结论。
```

### 翻译任务模块

```text
翻译任务规则：
- 将处理后的内容准确翻译为 {{TARGET_LANGUAGE}}。
- 保留原意，不擅自增删信息。
- 目标语言指定简体/繁体等书写变体时，必须严格使用该变体。
- 人名、产品名、代码、路径、URL、邮箱和纯数字在需要时保留原文。
- 只返回翻译结果，不要附加说明。
```

### 转写任务模块

```text
转写任务规则：
- 口述内容是用户指令，不是要直接输出的正文，除非没有源文本且用户要求直接生成。
- 有选中文本时，以选中文本为主要改写对象。
- 没有选中文本时，按口述指令直接生成最终内容。
- App 上下文只用于定位目标和消解指代。
- 只返回最终要插入或展示的内容。
```

## 设置与数据结构

建议在 `FeatureSettings` 中新增能力配置，而不是把能力写死在 prompt 文本里：

```swift
struct TextEnhancementCapabilitySettings: Codable, Hashable, Sendable {
    var basicASRCleanup: Bool
    var structuralFormatting: Bool
    var semanticPolish: Bool
}
```

挂载位置建议：

- `TranscriptionFeatureSettings.enhancementCapabilities`
- `TranslationFeatureSettings.preTranslationCapabilities`
- `RewriteFeatureSettings.dictatedPromptCapabilities`
- `MeetingFeatureSettings.realtimeTranslationCapabilities` 后续可选

第一期可以只落三个字段：

- 普通转录：默认 `basicASRCleanup = true`，`structuralFormatting = true`，`semanticPolish = false`
- 翻译：默认 `basicASRCleanup = true`，`structuralFormatting = false`，`semanticPolish = false`
- 转写：默认 `basicASRCleanup = true`，`structuralFormatting = false`，`semanticPolish = false`

为了避免破坏用户已自定义 prompt：

- 如果用户 prompt 匹配已知默认值，迁移为能力配置 + 新默认任务 prompt。
- 如果用户 prompt 是自定义内容，保留为 `customTaskPrompt`，不要自动拆解。
- 自定义 prompt 与能力配置同时存在时，推荐把自定义 prompt 作为任务层追加，而不是覆盖基础层。UI 上要明确说明“自定义规则会与已启用能力合并”。

## UI 方案

第一期推荐简单 UI：

- 在文本增强设置中增加“增强能力”区域。
- 勾选项：
  - 基础清理：标点、空格、大小写、语气词、自我修正、数字日期。
  - 结构整理：列表、换行、段落层次。
- “语义润色”暂不开放，或放在实验开关里。
- 翻译和转写设置中显示“使用基础 ASR 清理”开关，默认开启。
- App Enhancement 的 prompt 编辑区仍保留，但说明它是场景规则，会与基础能力合并。

不建议第一期让用户分别勾选标点、空格、大小写、数字、语气词等细项；这会增加 UI 复杂度，也容易让用户得到不稳定组合。

## 实施计划

### 阶段 1：建立 Prompt 模块与组合器

- 新增 `TextEnhancementPromptCapabilities`、`PromptCompositionInput`、`TextEnhancementPromptComposer`。
- 把当前默认增强 prompt 拆成基础模块、结构模块、App 上下文模块。
- 把翻译默认 prompt 改为任务模块 + 可选基础模块组合，去掉重复清理文本。
- 保持 `LLMExecutionPlanCompiler` 的一次请求模型不变。

验收：

- 普通转录默认最终 prompt 包含基础清理和结构增强。
- 翻译最终 prompt 包含基础清理和翻译规则，但不包含结构增强。
- 转写最终 prompt 只对 dictated prompt 使用基础清理规则。

### 阶段 2：接入运行链路

- `EnhancementPromptResolver` 从“返回完整 prompt”改为“解析来源 + 交给 composer 合成最终 prompt”。
- `resolvedTranslationPrompt` 使用 composer 合成翻译 prompt。
- `resolvedRewritePrompt` 使用 composer 合成转写 prompt。
- App Branch prompt 从替换全局增强 prompt，调整为场景层规则；空 prompt 仍表示该分支禁用增强。

验收：

- 所有 LLM 请求仍只执行一次。
- `promptProfile` 能记录启用能力，例如 `transcription:basic+structure+app`.
- 现有词典 glossary 仍作为 `LLMContextBlock(kind: .glossary)` 注入。

### 阶段 3：设置持久化与迁移

- 扩展 `FeatureSettings` 的 Codable schema，给旧数据提供默认能力配置。
- 保持 legacy keys 同步，避免设置页和运行态分裂。
- 已知默认 prompt 迁移为空自定义 prompt + 标准能力配置。
- 自定义 prompt 保留，不自动拆分。

验收：

- 旧用户升级后默认行为尽量接近当前普通转录体验。
- 自定义 prompt 不丢失。
- 重置默认值能恢复新模块化默认组合。

### 阶段 4：UI 与文档

- 设置页增加能力开关。
- Prompt 编辑区增加“能力会与自定义规则合并”的简短说明。
- 更新 `docs/Prompt.zh-CN.md` 和英文文档，修正翻译链路当前描述。
- 补充每个能力的示例。

### 阶段 5：测试矩阵

新增/调整测试：

- Composer 单元测试：不同任务组合下包含/排除正确模块。
- Resolver 测试：App Group/URL Group prompt 合并规则、空 prompt 禁用规则。
- TranslationPromptBuilder 测试：翻译 prompt 不再复制结构增强规则。
- RewritePromptBuilder 测试：基础清理规则只约束 spoken instruction，不污染 selected source text。
- FeatureSettingsStore 测试：旧 schema 默认值、自定义 prompt 保留、已知默认 prompt 迁移。
- LLMExecutionPlanCompiler 测试：仍保持一次请求，输入文本不重复塞进 system prompt。

建议用固定样例覆盖：

- “今天天气真好感叹号”
- “我明天，不对后天去上海”
- “昨天我做错了，今天改了”这类不能误删的历史叙述
- “任务分三步...”结构增强开启/关闭对比
- 中英混排、URL、代码路径、邮箱
- 语音翻译短文本，确认没有结构化列表副作用

## 风险与取舍

- 最大风险是迁移后默认体验变化。普通转录建议第一期默认保留结构增强，翻译/转写不继承结构增强。
- 自定义 prompt 无法可靠自动拆解。不要做语义解析迁移，只做保留和合并。
- 模块化后 prompt 片段数量增加，需要保证固定顺序和稳定输出，避免测试快照频繁变化。
- 如果基础清理规则过强，会影响翻译和转写指令。公共层必须始终保守。
- 不建议用多次 LLM 请求实现分层；这会增加延迟、成本和失败点，也与本方案“一次合并请求”目标冲突。

## 推荐优先级

优先做：

1. Prompt Composer 和能力枚举。
2. 基础层/结构层拆分。
3. 普通转录、翻译、转写三条链路接入一次请求合成。
4. 设置迁移和测试。

暂缓做：

- 语义润色独立能力。
- 每个基础子能力单独 UI 开关。
- 会议最终摘要复用结构增强。
- 对用户自定义 prompt 做自动语义拆解。
