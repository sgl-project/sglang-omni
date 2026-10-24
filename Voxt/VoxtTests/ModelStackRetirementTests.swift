import XCTest
@testable import Voxt

@MainActor
final class ModelStackRetirementTests: XCTestCase {
    func testASRCatalogHasNoInstallOnlyCompatibilityModels() {
        let retired: Set<String> = [
            "mlx-community/whisper-tiny-mlx", "mlx-community/whisper-base-mlx",
            "mlx-community/Qwen3-ASR-0.6B-6bit", "mlx-community/Qwen3-ASR-0.6B-8bit",
            "mlx-community/Qwen3-ASR-0.6B-bf16", "mlx-community/Qwen3-ASR-1.7B-4bit",
            "mlx-community/Qwen3-ASR-1.7B-bf16",
            "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit",
            "mlx-community/Voxtral-Mini-4B-Realtime-6bit",
            "mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16",
            "Mediform/canary-1b-v2-mlx-q8", "UsefulSensors/moonshine-tiny",
            "facebook/wav2vec2-base-960h", "facebook/mms-1b-fl102",
            "mlx-community/parakeet-tdt_ctc-110m", "mlx-community/parakeet-tdt-0.6b-v2",
            "mlx-community/parakeet-ctc-0.6b", "mlx-community/parakeet-rnnt-0.6b",
            "mlx-community/parakeet-tdt-1.1b", "mlx-community/parakeet-tdt_ctc-1.1b",
            "mlx-community/parakeet-ctc-1.1b", "mlx-community/parakeet-rnnt-1.1b",
            "mlx-community/GLM-ASR-Nano-2512-4bit",
            "mlx-community/granite-4.0-1b-speech-5bit", "mlx-community/FireRedASR2-AED-mlx"
        ]
        let active = Set(MLXModelCatalog.availableModels.map(\.id))
        XCTAssertEqual(active.count, 11)
        XCTAssertEqual(MLXModelCatalog.supportedModels, MLXModelCatalog.availableModels)
        XCTAssertEqual(Set(MLXModelCatalog.displayModels(includingInstalled: retired).map(\.id)), active)
        XCTAssertTrue(active.isDisjoint(with: retired))
        for repo in retired {
            let target = MLXModelCatalog.canonicalModelRepo(repo)
            XCTAssertTrue(active.contains(target), repo)
            XCTAssertEqual(MLXModelCatalog.canonicalModelRepo(target), target)
            XCTAssertTrue(MLXModelCatalog.hasRegisteredCapability(for: target))
        }
    }

    func testLLMCatalogHasNoInstallOnlyCompatibilityModels() {
        let retired: Set<String> = [
            "Qwen/Qwen2-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct",
            "mlx-community/Qwen2.5-VL-3B-Instruct-4bit", "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "mlx-community/Qwen3-0.6B-4bit", "mlx-community/Qwen3-1.7B-4bit",
            "mlx-community/Qwen3-4B-4bit", "mlx-community/Qwen3-8B-4bit",
            "mlx-community/Qwen3.5-4B-4bit", "mlx-community/Qwen3.5-0.8B-OptiQ-4bit",
            "mlx-community/gemma-2-2b-it-4bit", "mlx-community/gemma-2-9b-it-4bit",
            "mlx-community/gemma-3-1b-it-qat-4bit", "mlx-community/gemma-3n-E2B-it-lm-4bit",
            "mlx-community/gemma-3n-E4B-it-lm-4bit", "mlx-community/Qwen3-30B-A3B-4bit"
        ]
        let active = Set(CustomLLMModelCatalog.availableModels.map(\.id))
        XCTAssertEqual(active.count, 12)
        XCTAssertEqual(CustomLLMModelCatalog.supportedModels, CustomLLMModelCatalog.availableModels)
        XCTAssertEqual(Set(CustomLLMModelCatalog.displayModels(includingInstalled: retired).map(\.id)), active)
        XCTAssertTrue(active.isDisjoint(with: retired))
        for repo in retired {
            let target = CustomLLMModelCatalog.canonicalModelRepo(repo)
            XCTAssertTrue(active.contains(target), repo)
            XCTAssertEqual(CustomLLMModelCatalog.canonicalModelRepo(target), target)
        }
    }

    func testUnknownASRDoesNotInferArchitectureFromName() {
        let capability = MLXModelCatalog.capability(for: "untrusted/qwen3-asr-checkpoint")
        XCTAssertEqual(capability.family, .generic)
        XCTAssertTrue(capability.supportedPurposes.isEmpty)
        XCTAssertFalse(MLXModelCatalog.hasRegisteredCapability(for: "untrusted/qwen3-asr-checkpoint"))
    }

    func testRetiredSelectionsArePersistedAsActiveModelsWithoutChangingPrompts() {
        let testName = "ModelStackRetirementTests.\(UUID().uuidString)"
        let defaults = TestDoubles.makeUserDefaults(testName: testName)
        defer {
            defaults.removePersistentDomain(forName: "VoxtTests.\(testName)")
        }

        var settings = FeatureSettingsStore.load(defaults: defaults)
        settings.transcription.asrSelectionID = .init(rawValue: "mlx:mlx-community/whisper-base-mlx")
        settings.transcription.llmSelectionID = .init(rawValue: "local-llm:mlx-community/Qwen3-4B-4bit")
        settings.rewrite.llmSelectionID = .init(rawValue: "local-llm:mlx-community/gemma-2-2b-it-4bit")
        settings.meeting.summaryModelSelectionID = .init(rawValue: "local-llm:mlx-community/Qwen3-8B-4bit")
        settings.transcription.prompt = "Preserve my custom instructions."
        FeatureSettingsStore.save(settings, defaults: defaults)
        let loaded = FeatureSettingsStore.load(defaults: defaults)
        XCTAssertEqual(loaded.transcription.asrSelectionID, .mlx("mlx-community/whisper-small-mlx"))
        XCTAssertEqual(loaded.transcription.llmSelectionID, .localLLM("mlx-community/Qwen3.5-4B-OptiQ-4bit"))
        XCTAssertEqual(loaded.rewrite.llmSelectionID, .localLLM("mlx-community/gemma-4-e2b-it-4bit"))
        XCTAssertEqual(loaded.meeting.summaryModelSelectionID, .localLLM("mlx-community/Qwen3.5-9B-OptiQ-4bit"))
        XCTAssertEqual(loaded.transcription.prompt, settings.transcription.prompt)
        FeatureSettingsStore.migrateIfNeeded(defaults: defaults)
        XCTAssertEqual(FeatureSettingsStore.load(defaults: defaults).transcription.asrSelectionID, loaded.transcription.asrSelectionID)
    }

    func testRetiredTuningIsDroppedWithoutResettingSurvivingFamily() throws {
        let raw = """
        {"canary":{"preset":"balanced"},"whisper":{"preset":"balanced","whisperTemperature":0.3,"granitePromptBias":"old"}}
        """
        let settings = MLXLocalTuningSettingsStore.load(from: raw)
        XCTAssertNil(settings["canary"])
        XCTAssertEqual(settings["whisper"]?.whisperTemperature, 0.3)
        let stored = MLXLocalTuningSettingsStore.storageValue(for: settings)
        XCTAssertFalse(stored.contains("granitePromptBias"))
        XCTAssertFalse(stored.contains("canary"))
    }

    func testRetiredLLMSettingsCannotOverwriteReplacementAndSameModelAliasIsPreserved() throws {
        let active = "mlx-community/Qwen3.5-2B-4bit"
        let alias = "mlx-community/Qwen3.5-2B-MLX-4bit"
        let retired = "mlx-community/Qwen3-0.6B-4bit"
        let encoded = try JSONEncoder().encode([
            active: LLMGenerationSettings(temperature: 0.2),
            alias: LLMGenerationSettings(temperature: 0.5),
            retired: LLMGenerationSettings(temperature: 0.9)
        ])
        let values = CustomLLMGenerationSettingsStore.resolvedByRepo(from: String(decoding: encoded, as: UTF8.self))
        XCTAssertEqual(values.count, 1)
        XCTAssertEqual(values[active]?.temperature, 0.2)
        let renamedOnly = CustomLLMGenerationSettingsStore.storageValue(forByRepo: [
            alias: LLMGenerationSettings(temperature: 0.5)
        ])
        XCTAssertEqual(CustomLLMGenerationSettingsStore.resolvedByRepo(from: renamedOnly)[active]?.temperature, 0.5)
    }

    func testLocalPreviewThrottlePublishesFirstChunkAndBoundsUpdates() {
        var delivery = LocalLLMPartialDelivery()
        XCTAssertTrue(delivery.shouldPublish(at: 10))
        XCTAssertFalse(delivery.shouldPublish(at: 10.01))
        XCTAssertTrue(delivery.shouldPublish(at: 10.06))
        XCTAssertFalse(delivery.shouldPublish(at: 10.061))
    }
}
