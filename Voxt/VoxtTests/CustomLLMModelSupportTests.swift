// CustomLLMModelSupportTests.swift
// Provides Custom LLMModel Support Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class CustomLLMModelSupportTests: XCTestCase {
    func testRepetitionGuardTruncatesShortRepeatedSuffix() {
        let repeated = String(repeating: "好", count: 24)
        let result = LLMOutputRepetitionGuard().repeatedSuffix(in: "结论：\(repeated)")

        XCTAssertEqual(result?.repeatedUnit, "好")
        XCTAssertEqual(result?.truncatedText, "结论：好")
    }

    func testRepetitionGuardTruncatesRepeatedPhraseSuffix() {
        let repeated = String(repeating: " thank you", count: 8)
        let result = LLMOutputRepetitionGuard().repeatedSuffix(in: "Done.\(repeated)")

        XCTAssertEqual(result?.repeatedUnit, " thank you")
        XCTAssertEqual(result?.truncatedText, "Done. thank you")
    }

    func testRepetitionGuardIgnoresSmallNaturalRepetitions() {
        let result = LLMOutputRepetitionGuard().repeatedSuffix(in: "谢谢，谢谢，谢谢。")

        XCTAssertNil(result)
    }

    func testCatalogRecognizesSupportedRepoAndFallbackTitle() {
        XCTAssertTrue(CustomLLMModelCatalog.isSupportedModelRepo("mlx-community/Qwen3-4B-4bit"))
        XCTAssertTrue(CustomLLMModelCatalog.isSupportedModelRepo("Qwen/Qwen3-8B-4bit"))
        XCTAssertFalse(CustomLLMModelCatalog.isSupportedModelRepo("unsupported/repo"))
        XCTAssertEqual(
            CustomLLMModelCatalog.displayTitle(for: "mlx-community/Qwen3-4B-4bit"),
            "Qwen3.5 4B OptiQ (4bit)"
        )
        XCTAssertEqual(
            CustomLLMModelCatalog.canonicalModelRepo("Qwen/Qwen3-8B-4bit"),
            "mlx-community/Qwen3.5-9B-OptiQ-4bit"
        )
        XCTAssertEqual(
            CustomLLMModelCatalog.canonicalModelRepo("mlx-community/Qwen3.5-0.8B-4bit-OptiQ"),
            "mlx-community/Qwen3.5-2B-4bit"
        )
        XCTAssertEqual(
            CustomLLMModelCatalog.displayTitle(for: "custom/repo"),
            "custom/repo"
        )
        XCTAssertEqual(
            CustomLLMModelCatalog.displayTitle(for: "mlx-community/GLM-4-9B-0414-4bit"),
            "GLM 4 9B"
        )
        XCTAssertEqual(
            CustomLLMModelCatalog.displayTitle(for: "mlx-community/Ministral-3-3B-Instruct-2512-4bit"),
            "Mistral 3 3B"
        )
        XCTAssertEqual(
            CustomLLMModelCatalog.displayTitle(for: "mlx-community/LFM2-1.2B-4bit"),
            "LFM2 1.2B (4bit)"
        )
    }

    func testCatalogUsesKnownRemoteSizeFallbacksForLegacyCuratedRepos() {
        XCTAssertNotNil(CustomLLMModelCatalog.fallbackRemoteSizeText(repo: "Qwen/Qwen2-1.5B-Instruct"))
        XCTAssertNotNil(CustomLLMModelCatalog.fallbackRemoteSizeText(repo: "mlx-community/Qwen3-4B-4bit"))
        XCTAssertEqual(
            CustomLLMModelCatalog.fallbackRemoteSizeText(repo: "mlx-community/Qwen3.5-2B-4bit")?
                .replacingOccurrences(of: "\u{2006}", with: " "),
            "1.74 GB"
        )
    }


    func testAllSupportedCustomLLMModelsHaveFixedSizeFallbacks() {
        let missingRepos = CustomLLMModelCatalog.supportedModels
            .map(\.id)
            .filter { CustomLLMModelCatalog.fallbackRemoteSizeText(repo: $0) == nil }

        XCTAssertEqual(missingRepos, [])
    }


    func testRetiredLLMModelsNeverReappearFromLocalState() {
        let hiddenRepo = "mlx-community/gemma-2-2b-it-4bit"

        XCTAssertFalse(
            CustomLLMModelCatalog.displayModels(includingInstalled: []).contains { $0.id == hiddenRepo }
        )
        XCTAssertFalse(
            CustomLLMModelCatalog.displayModels(includingInstalled: [hiddenRepo]).contains { $0.id == hiddenRepo }
        )
    }

    func testCatalogIncludesNewRecommendedHomeMacModels() {
        let modelIDs = Set(CustomLLMModelCatalog.availableModels.map(\.id))

        XCTAssertTrue(modelIDs.contains("lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit"))
        XCTAssertTrue(modelIDs.contains("mlx-community/Qwen3.5-2B-4bit"))
        XCTAssertTrue(modelIDs.contains("mlx-community/Qwen3.5-4B-OptiQ-4bit"))
        XCTAssertTrue(modelIDs.contains("mlx-community/Qwen3.5-9B-OptiQ-4bit"))
        XCTAssertTrue(modelIDs.contains("mlx-community/GLM-4-9B-0414-4bit"))
        XCTAssertTrue(modelIDs.contains("mlx-community/Ministral-3-3B-Instruct-2512-4bit"))
        XCTAssertTrue(modelIDs.contains("mlx-community/LFM2-1.2B-4bit"))
        XCTAssertTrue(modelIDs.contains("mlx-community/LFM2-8B-A1B-3bit-MLX"))
        XCTAssertTrue(modelIDs.contains("mlx-community/Qwen3.6-27B-4bit"))
        XCTAssertTrue(modelIDs.contains("mlx-community/gemma-4-e2b-it-4bit"))
        XCTAssertTrue(modelIDs.contains("mlx-community/gemma-4-e4b-it-4bit"))
        XCTAssertTrue(modelIDs.contains("mlx-community/gemma-4-12B-it-OptiQ-4bit"))
        XCTAssertFalse(modelIDs.contains("Qwen/Qwen2-1.5B-Instruct"))
        XCTAssertFalse(modelIDs.contains("Qwen/Qwen2.5-3B-Instruct"))
        XCTAssertFalse(modelIDs.contains("mlx-community/Qwen2.5-VL-3B-Instruct-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/Qwen3-0.6B-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/Qwen3-1.7B-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/Qwen3-4B-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/Qwen3-8B-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/Qwen3.5-0.8B-OptiQ-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/Qwen3.5-4B-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/gemma-2-2b-it-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/gemma-2-9b-it-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/gemma-3n-E4B-it-lm-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/Qwen3-30B-A3B-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/glm-4-9b-chat-1m-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/GLM-Z1-9B-0414-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/GLM-4.7-Flash-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/Llama-3.2-1B-Instruct-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/Llama-3.2-3B-Instruct-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/Meta-Llama-3-8B-Instruct-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/Mistral-7B-Instruct-v0.3-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/Mistral-Nemo-Instruct-2407-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/Phi-3.5-mini-instruct-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/internlm2_5-7b-chat-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/MiniCPM4-8B-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/granite-3.3-2b-instruct-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/MiMo-7B-SFT-4bit"))
        XCTAssertFalse(modelIDs.contains("mlx-community/AceReason-Nemotron-7B-4bit"))
    }

    func testTextOnlyRequestsKeepArchitectureAppropriateLoadingFactory() {
        XCTAssertTrue(CustomLLMModelCatalog.requiresVLMFactory(repo: "mlx-community/gemma-4-e2b-it-4bit"))
        XCTAssertTrue(CustomLLMModelCatalog.requiresVLMFactory(repo: "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"))
        XCTAssertTrue(CustomLLMModelCatalog.requiresVLMFactory(repo: "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit"))
        XCTAssertTrue(CustomLLMModelCatalog.requiresVLMFactory(repo: "mlx-community/Ministral-3-3B-Instruct-2512-4bit"))
        XCTAssertFalse(CustomLLMModelCatalog.requiresVLMFactory(repo: "mlx-community/paligemma-3b-mix-448-8bit"))
        XCTAssertFalse(CustomLLMModelCatalog.requiresVLMFactory(repo: "mlx-community/Qwen3-8B-4bit"))
        for model in CustomLLMModelCatalog.availableModels {
            XCTAssertFalse(CustomLLMModelCatalog.catalogTagKeys(for: model.id).contains("Vision"))
        }
    }

    func testStorageSupportBuildsExpectedCacheDirectory() {
        let rootDirectory = URL(fileURLWithPath: "/tmp/voxt-tests", isDirectory: true)
        let directory = CustomLLMModelStorageSupport.cacheDirectory(
            for: "mlx-community/Qwen3-4B-4bit",
            rootDirectory: rootDirectory
        )
        XCTAssertEqual(
            directory?.path,
            "/tmp/voxt-tests/mlx-llm/mlx-community_Qwen3-4B-4bit"
        )
    }

    func testChatTemplateDetectionRecognizesDownloadedTemplateSidecars() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let jinjaURL = root.appendingPathComponent("chat_template.jinja")
        try "{% for m in messages %}{{ m.content }}{% endfor %}".write(
            to: jinjaURL,
            atomically: true,
            encoding: .utf8
        )

        XCTAssertTrue(CustomLLMModelDownloadSupport.hasUsableChatTemplate(in: root))
    }

    func testChatTemplateDetectionRecognizesInlineTokenizerTemplate() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let tokenizerConfigURL = root.appendingPathComponent("tokenizer_config.json")
        let json = """
        {
          "chat_template": "{{ bos_token }}{{ messages[0]['content'] }}"
        }
        """
        try json.write(to: tokenizerConfigURL, atomically: true, encoding: .utf8)

        XCTAssertTrue(CustomLLMModelDownloadSupport.hasUsableChatTemplate(in: root))
    }

    func testPartialCustomLLMDirectoryIsNotTreatedAsInstalled() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let modelDirectory = root
            .appendingPathComponent("mlx-llm")
            .appendingPathComponent("mlx-community_Qwen3-4B-4bit")

        try FileManager.default.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
        try Data("partial".utf8).write(to: modelDirectory.appendingPathComponent("model.safetensors"))
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertFalse(CustomLLMModelStorageSupport.isModelDirectoryValid(modelDirectory))
        XCTAssertTrue(FileManager.default.directoryContainsRegularFiles(at: modelDirectory))
    }
}
