// FeaturePromptPresetTests.swift
// Verifies built-in feature prompt preset catalogs and resources.

import XCTest
@testable import Voxt

final class FeaturePromptPresetTests: XCTestCase {
    func testCatalogProvidesLocalizedPresetsForEachEditableFeature() {
        for language in [
            AppInterfaceLanguage.english,
            .chineseSimplified,
            .japanese
        ] {
            XCTAssertEqual(
                FeaturePromptPresetCatalog.presets(for: .enhancement, language: language).count,
                2
            )
            XCTAssertEqual(
                FeaturePromptPresetCatalog.presets(for: .translation, language: language).count,
                2
            )
            XCTAssertEqual(
                FeaturePromptPresetCatalog.presets(for: .rewrite, language: language).count,
                1
            )
        }
    }

    func testDefaultPresetIDsMatchProductDefaults() {
        XCTAssertEqual(FeaturePromptPresetCatalog.defaultPresetID(for: .enhancement), "precise")
        XCTAssertEqual(FeaturePromptPresetCatalog.defaultPresetID(for: .translation), "precise")
        XCTAssertEqual(FeaturePromptPresetCatalog.defaultPresetID(for: .rewrite), "strict")
        XCTAssertNil(FeaturePromptPresetCatalog.defaultPresetID(for: .transcriptSummary))
    }

    func testDefaultPresetContentUsesCurrentDefaultPromptResources() {
        for language in [
            AppInterfaceLanguage.english,
            .chineseSimplified,
            .japanese
        ] {
            for kind in [AppPromptKind.enhancement, .translation, .rewrite] {
                let defaultID = FeaturePromptPresetCatalog.defaultPresetID(for: kind)
                let preset = FeaturePromptPresetCatalog.preset(
                    id: defaultID,
                    for: kind,
                    language: language
                )
                XCTAssertEqual(preset?.prompt, AppPromptDefaults.text(for: kind, language: language))
            }
        }
    }

    func testPresetTemplatesContainRequiredRuntimeVariables() {
        for language in [
            AppInterfaceLanguage.english,
            .chineseSimplified,
            .japanese
        ] {
            let enhancementPresets = FeaturePromptPresetCatalog.presets(
                for: .enhancement,
                language: language
            )
            XCTAssertTrue(enhancementPresets.allSatisfy { $0.prompt.contains("{{USER_MAIN_LANGUAGE}}") })

            let translationPresets = FeaturePromptPresetCatalog.presets(
                for: .translation,
                language: language
            )
            XCTAssertTrue(translationPresets.allSatisfy { preset in
                preset.prompt.contains("{{USER_MAIN_LANGUAGE}}") &&
                    preset.prompt.contains("{{TARGET_LANGUAGE}}")
            })
        }
    }

    func testPreciseCleanupDoesNotRequestListRestructuring() {
        let prompt = FeaturePromptPresetCatalog.preset(
            id: "precise",
            for: .enhancement,
            language: .chineseSimplified
        )?.prompt ?? ""

        XCTAssertTrue(prompt.contains("不要把连续叙述改成标题、列表或新的写作结构"))
        XCTAssertFalse(prompt.contains("使用序号列表方式整理"))
    }

    func testClearStructureIncludesFullStructureRulesAndExamples() {
        let prompt = FeaturePromptPresetCatalog.preset(
            id: "structured",
            for: .enhancement,
            language: .chineseSimplified
        )?.prompt ?? ""

        XCTAssertTrue(prompt.contains("使用序号列表方式整理"))
        XCTAssertTrue(prompt.contains("Markdown 嵌套列表格式"))
        XCTAssertTrue(prompt.contains("任务分三步"))
    }

    func testTranslationCatalogOnlyKeepsPreciseAndNaturalPresets() {
        let presetIDs = FeaturePromptPresetCatalog.presets(
            for: .translation,
            language: .chineseSimplified
        ).map(\.id)

        XCTAssertEqual(presetIDs, ["precise", "natural"])
    }

    func testRewriteCatalogOnlyKeepsDefaultPreset() {
        let presetIDs = FeaturePromptPresetCatalog.presets(
            for: .rewrite,
            language: .chineseSimplified
        ).map(\.id)

        XCTAssertEqual(presetIDs, ["strict"])
    }

    func testBuiltInPromptFollowsInterfaceLanguageWhileCustomPromptIsPreserved() {
        let englishPrompt = FeaturePromptPresetCatalog.preset(
            id: "structured",
            for: .enhancement,
            language: .english
        )!.prompt
        let chinesePrompt = FeaturePromptPresetCatalog.preset(
            id: "structured",
            for: .enhancement,
            language: .chineseSimplified
        )!.prompt

        XCTAssertEqual(
            FeaturePromptPresetCatalog.resolvedPrompt(
                storedID: "structured",
                prompt: englishPrompt,
                kind: .enhancement,
                language: .chineseSimplified
            ),
            chinesePrompt
        )

        let customPrompt = englishPrompt + "\nKeep my product terminology unchanged."
        XCTAssertEqual(
            FeaturePromptPresetCatalog.resolvedPrompt(
                storedID: "structured",
                prompt: customPrompt,
                kind: .enhancement,
                language: .chineseSimplified
            ),
            customPrompt
        )
    }

    func testEnglishAndJapanesePromptsUseLocalizedSpokenPunctuationNames() {
        let english = FeaturePromptPresetCatalog.preset(
            id: "precise",
            for: .enhancement,
            language: .english
        )!.prompt
        XCTAssertTrue(english.contains("comma"))
        XCTAssertTrue(english.contains("exclamation mark"))
        XCTAssertFalse(english.contains("感叹号"))

        let japanese = FeaturePromptPresetCatalog.preset(
            id: "precise",
            for: .enhancement,
            language: .japanese
        )!.prompt
        XCTAssertTrue(japanese.contains("読点"))
        XCTAssertTrue(japanese.contains("感嘆符"))
        XCTAssertFalse(japanese.contains("感叹号"))
    }
}
