// EnhancementStrategyResolverTests.swift
// Provides Enhancement Strategy Resolver Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class EnhancementStrategyResolverTests: XCTestCase {
    func testShortTranscriptionUsesSinglePassWithNoHintWhenModelLimitUnknown() {
        let strategy = TaskLLMStrategyResolver.resolve(
            taskKind: .transcriptionEnhancement,
            rawText: String(repeating: "短", count: 120),
            promptCharacterCount: 80,
            baseGlossarySelectionPolicy: DictionaryGlossaryPurpose.enhancement.selectionPolicy,
            capabilities: .unknown
        )

        XCTAssertEqual(strategy.rawTextCharacterCount, 120)
        XCTAssertEqual(strategy.mode, .singlePass)
        XCTAssertEqual(strategy.contextBudgetPolicy, .standard)
        XCTAssertNil(strategy.outputTokenBudgetHint)
        XCTAssertFalse(strategy.truncationGuard.isEnabled)
    }

    func testLongTranslationWithTightModelLimitUsesSegmentedMode() {
        let strategy = TaskLLMStrategyResolver.resolve(
            taskKind: .translation,
            rawText: String(repeating: "长", count: 360),
            promptCharacterCount: 120,
            baseGlossarySelectionPolicy: DictionaryGlossaryPurpose.translation.selectionPolicy,
            capabilities: LLMProviderModelCapabilities(maxContextTokens: 8192, maxOutputTokens: 400)
        )

        XCTAssertEqual(strategy.rawTextCharacterCount, 360)
        XCTAssertEqual(strategy.mode, .segmented)
        XCTAssertEqual(strategy.contextBudgetPolicy, .reducedForLongInput)
        XCTAssertEqual(strategy.outputTokenBudgetHint, 400)
        XCTAssertTrue(strategy.truncationGuard.isEnabled)
        XCTAssertNotNil(strategy.segmentationCharacterLimit)
    }

    func testLongTranscriptionDisablesTruncationGuard() {
        let strategy = TaskLLMStrategyResolver.resolve(
            taskKind: .transcriptionEnhancement,
            rawText: String(repeating: "长", count: 360),
            promptCharacterCount: 100,
            baseGlossarySelectionPolicy: DictionaryGlossaryPurpose.enhancement.selectionPolicy,
            capabilities: .unknown
        )

        XCTAssertEqual(strategy.mode, .singlePass)
        XCTAssertFalse(strategy.truncationGuard.isEnabled)
    }

    func testTruncationGuardFallsBackForPrefixLikeOutput() {
        let original = String(repeating: "这是一段比较长的原始文本。", count: 30)
        let enhanced = String(original.prefix(60))
        let strategy = TaskLLMStrategyResolver.resolve(
            taskKind: .rewrite,
            rawText: original,
            promptCharacterCount: 100,
            baseGlossarySelectionPolicy: DictionaryGlossaryPurpose.rewrite.selectionPolicy,
            capabilities: .unknown
        )

        let guarded = TaskLLMStrategyResolver.applyTruncationGuard(
            outputText: enhanced,
            originalText: original,
            strategy: strategy
        )

        XCTAssertTrue(guarded.didFallback)
        XCTAssertEqual(guarded.text, original)
    }

    func testTranslationTruncationGuardKeepsCompactCrossLanguageOutput() {
        let original = String(repeating: "This roadmap item needs a careful product review before release. ", count: 12)
        let translated = "这个路线图事项发布前需要仔细做产品评审。"
        let strategy = TaskLLMStrategyResolver.resolve(
            taskKind: .translation,
            rawText: original,
            promptCharacterCount: 100,
            baseGlossarySelectionPolicy: DictionaryGlossaryPurpose.translation.selectionPolicy,
            capabilities: .unknown
        )

        let guarded = TaskLLMStrategyResolver.applyTruncationGuard(
            outputText: translated,
            originalText: original,
            strategy: strategy
        )

        XCTAssertFalse(guarded.didFallback)
        XCTAssertEqual(guarded.text, translated)
    }

    func testRewriteTruncationGuardKeepsDirectAnswerShorterThanPrompt() {
        let prompt = String(repeating: "帮我基于今天的任务安排写一个简短总结，语气自然一点，不要太长。", count: 12)
        let answer = "今天主要整理 Codex 体验评测、优化两个 Web coding 任务，并走查自媒体视频。"
        let strategy = TaskLLMStrategyResolver.resolve(
            taskKind: .rewrite,
            rawText: prompt,
            promptCharacterCount: 100,
            baseGlossarySelectionPolicy: DictionaryGlossaryPurpose.rewrite.selectionPolicy,
            capabilities: .unknown
        )

        let guarded = TaskLLMStrategyResolver.applyTruncationGuard(
            outputText: answer,
            originalText: prompt,
            strategy: strategy
        )

        XCTAssertFalse(guarded.didFallback)
        XCTAssertEqual(guarded.text, answer)
    }

    func testTruncationGuardKeepsHealthyLongOutput() {
        let original = String(repeating: "做 PPT 时，从资料库里挑素材，起稿就可以发起任务。", count: 12)
        let enhanced = String(repeating: "做 PPT 时，从资料库里挑素材，起稿就可以发起任务。", count: 11)
        let strategy = TaskLLMStrategyResolver.resolve(
            taskKind: .transcriptionEnhancement,
            rawText: original,
            promptCharacterCount: 100,
            baseGlossarySelectionPolicy: DictionaryGlossaryPurpose.enhancement.selectionPolicy,
            capabilities: .unknown
        )

        let guarded = TaskLLMStrategyResolver.applyTruncationGuard(
            outputText: enhanced,
            originalText: original,
            strategy: strategy
        )

        XCTAssertFalse(guarded.didFallback)
        XCTAssertEqual(guarded.text, enhanced)
    }

    func testLongTranscriptionEnhancementNeverFallsBackToRawText() {
        let original = """
        啊，可以开始画了。那我今天的周二代班的话，啊，第一个的话就是把我上周体验 Codex 一些感受记录下来，整理一个啊小白视角的 Codex 的一个体验评测分享。然后第二个的话就是有两个 Web coding 的小任务，主要是优化一下之前的一个产品体验。第一个是小红书图文编辑器的那个呃体验，有些 bug 要修一下。第二个是我做的一个 AI 小白学 AI 很方便的一个这边个产品，再把那个啊体验上优化一下。然后第三个任务就是啊之前用 Cloud Code 剪的那个 AI 做自媒体的视频的那个视频要走查一下，然后发布在小红书。然后第四个的话就是体验一下 Open Design 这个产品，进行一些各种 case 一些对比评测，然后整理一篇啊调研文档，准备下周发。嗯，然后我们关注这个。
        """
        let enhanced = """
        可以开始画了。今天周二代班。
        1. 第一个任务是把我上周体验 CodeX 的一些感受记录下来，整理一个小白视角的 CodeX 体验评测分享。
        2. 第二个任务有两个 Web coding 的小任务，主要是优化一下之前的一个产品体验：第一个是小红书图文编辑器的体验，有些 bug 要修。第二个是我做的一个 AI 小白学 AI 很方便的这个产品，再把体验优化一下。
        3. 第三个任务是之前用 Cloud Code 剪辑的 AI 做自媒体的视频要走查一下，然后发布在小红书。
        4. 第四个任务是体验 Open Design，进行一些 case 的对比评测，整理一篇调研文档，准备下周发。
        然后我们关注这个。
        """
        let strategy = TaskLLMStrategyResolver.resolve(
            taskKind: .transcriptionEnhancement,
            rawText: original,
            promptCharacterCount: 100,
            baseGlossarySelectionPolicy: DictionaryGlossaryPurpose.enhancement.selectionPolicy,
            capabilities: .unknown
        )

        let guarded = TaskLLMStrategyResolver.applyTruncationGuard(
            outputText: enhanced,
            originalText: original,
            strategy: strategy
        )

        XCTAssertFalse(guarded.didFallback)
        XCTAssertEqual(guarded.text, enhanced)
    }

    func testListFormattedLongTranscriptionEnhancementNeverFallsBackToRawText() {
        let original = """
        那我今天的周二代办的话，啊，第一个的话就是把我上周体验 Codex 一些感受记录下来，整理一个啊小白视角的 Codex 的一个体验评测分享。然后第二个的话就是有两个 Webcoding 的小任务，主要是优化一下之前的一个产品体验。第一个是小红书图文编辑器的那个呃体验有些 bug 要修一下。第二个是我做了一个 AI 小白学 AI 很方便的一个这边个产品，再把那个呃体验稍优化一下。然后第三个任务就是啊之前用 Cloud Code 做的那个 AI 做自媒体的视频的那个视频要走查一下，然后发布在小红书。然后第四个的话就是体验一下 Open Design 这个产品，进行一些各种 case 一些对比评测，然后整理一篇呃调研文档，准备下周发。然后我们摁住这个。
        """
        let enhanced = """
        周二代办：
        1. 把上周体验 CodeX 的一些感受记录下来，整理一个小白视角的 CodeX 体验评测分享。
        2. 有两个 Webcoding 小任务，主要是优化之前的一个产品体验：
           - 小红书图文编辑器有些体验 bug 要修一下；
           - 我做了一个 AI 小白学 AI 很方便的产品，再把体验稍微优化一下。
        3. 检查之前用 Cloud Code 做的 AI 做自媒体视频，然后发布在小红书。
        4. 体验一下 Open Design，进行一些 case 的对比评测，整理一篇调研文档，准备下周发布。
        """
        let strategy = TaskLLMStrategyResolver.resolve(
            taskKind: .transcriptionEnhancement,
            rawText: original,
            promptCharacterCount: 100,
            baseGlossarySelectionPolicy: DictionaryGlossaryPurpose.enhancement.selectionPolicy,
            capabilities: .unknown
        )

        let guarded = TaskLLMStrategyResolver.applyTruncationGuard(
            outputText: enhanced,
            originalText: original,
            strategy: strategy
        )

        XCTAssertFalse(guarded.didFallback)
        XCTAssertEqual(guarded.text, enhanced)
    }

    func testRealWorldLongTranscriptionEnhancementKeepsLLMOutput() {
        let original = """
        整理一些我周日的自媒体代办，我就给大家展示一下平常我用的这个 Tablis。首先我们按住 Fn 键，然后开始说话了。那我今天的周日代办的话，啊，第一个的话就是把我上周体验 Codex 一些感受记录下来，整理一个啊小白视角的 Codex 的一个体验评测分享。然后第二个的话就是有两个 Web coding 的小任务，主要是优化一下之前的一个产品体验。第一个是小红书图文编辑器的那个呃体验，有些 bug 要修一下。第二个是我做了一个 AI 小白学 AI 很方便的一个这么一个产品，再把那个呃体验稍微优化一下。然后第三个任务就是啊之前用 Cloud Code 剪的那个 AI 做自媒体的视频的那个视频要走查一下，然后发布在小红书。然后第四个的话就是体验一下 Open Design 这个产品，进行一些各种 case 一些对比评测，然后整理一篇呃调研文档，准备下周发。然后我们。
        """
        let enhanced = """
        我周日的自媒体代办，我就给大家展示一下平常我用的这个 Tablis。

        1. 把我上周体验 CodeX 的一些感受记录下来，整理一篇小白视角的 CodeX 体验评测分享。
        2. 做两个 Web coding 的小任务，优化之前的产品体验：
           - 小红书图文编辑器的体验，修复一些 bug。
           - 我做的 AI 小白学 AI 很方便这个产品的体验，进一步优化。
        3. 走查之前用 Cloud Code 剪的 AI 做自媒体视频，并发布在小红书。
        4. 体验 Open Design，进行各类 case 对比评测，整理一篇调研文档，准备下周发布。
        """
        let strategy = TaskLLMStrategyResolver.resolve(
            taskKind: .transcriptionEnhancement,
            rawText: original,
            promptCharacterCount: 100,
            baseGlossarySelectionPolicy: DictionaryGlossaryPurpose.enhancement.selectionPolicy,
            capabilities: .unknown
        )

        let guarded = TaskLLMStrategyResolver.applyTruncationGuard(
            outputText: enhanced,
            originalText: original,
            strategy: strategy
        )

        XCTAssertFalse(guarded.didFallback)
        XCTAssertEqual(guarded.text, enhanced)
    }

    func testTranslationTruncationGuardStillFallsBackForVeryShortNonPrefixOutput() {
        let original = String(repeating: "这是一段比较长的原始文本，需要完整保留主要信息。", count: 24)
        let enhanced = "这是一段比较长的原始文本。"
        let strategy = TaskLLMStrategyResolver.resolve(
            taskKind: .translation,
            rawText: original,
            promptCharacterCount: 100,
            baseGlossarySelectionPolicy: DictionaryGlossaryPurpose.translation.selectionPolicy,
            capabilities: .unknown
        )

        let guarded = TaskLLMStrategyResolver.applyTruncationGuard(
            outputText: enhanced,
            originalText: original,
            strategy: strategy
        )

        XCTAssertTrue(guarded.didFallback)
        XCTAssertEqual(guarded.text, original)
    }

    func testLongRewriteWithLooseOrUnknownModelLimitStaysSinglePass() {
        let strategy = TaskLLMStrategyResolver.resolve(
            taskKind: .rewrite,
            rawText: String(repeating: "长", count: 360),
            promptCharacterCount: 100,
            baseGlossarySelectionPolicy: DictionaryGlossaryPurpose.rewrite.selectionPolicy,
            capabilities: LLMProviderModelCapabilities(maxContextTokens: 8192, maxOutputTokens: 800)
        )

        XCTAssertEqual(strategy.mode, .singlePass)
        XCTAssertEqual(strategy.outputTokenBudgetHint, 487)
        XCTAssertTrue(strategy.truncationGuard.isEnabled)
    }
}
