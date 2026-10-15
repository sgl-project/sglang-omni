// MLXModelSupportTests.swift
// Provides MLXModel Support Tests for Voxt test coverage.

import XCTest
@testable import Voxt

final class MLXModelSupportTests: XCTestCase {
    func testDownloadValidationSizeIsBoundToRequestedRepo() throws {
        let smallRepo = "mlx-community/Qwen3-ASR-0.6B-4bit"
        let largeRepo = "beshkenadze/cohere-transcribe-03-2026-mlx-fp16"
        let small = try XCTUnwrap(MLXModelCatalog.fallbackRemoteSizeInfo(repo: smallRepo))
        let large = try XCTUnwrap(MLXModelCatalog.fallbackRemoteSizeInfo(repo: largeRepo))
        XCTAssertNotEqual(small.bytes, large.bytes)
        XCTAssertEqual(MLXModelDownloadSupport.validationSizeState(for: smallRepo), .ready(bytes: small.bytes, text: small.text))
        XCTAssertEqual(MLXModelDownloadSupport.validationSizeState(for: largeRepo), .ready(bytes: large.bytes, text: large.text))
        XCTAssertEqual(MLXModelDownloadSupport.validationSizeState(for: smallRepo), .ready(bytes: small.bytes, text: small.text))
    }

    func testDownloadValidationSizeResolvesLegacyAlias() {
        XCTAssertEqual(
            MLXModelDownloadSupport.validationSizeState(for: "mlx-community/Qwen3-ASR-0.6B-bf16"),
            MLXModelDownloadSupport.validationSizeState(for: "mlx-community/Qwen3-ASR-0.6B-4bit")
        )
    }

    func testCanonicalModelRepoMapsLegacyRepos() {
        XCTAssertEqual(
            MLXModelCatalog.canonicalModelRepo("mlx-community/Parakeet-0.6B"),
            "mlx-community/parakeet-tdt-0.6b-v3"
        )
        XCTAssertEqual(
            MLXModelCatalog.canonicalModelRepo("mlx-community/FireRedASR2"),
            MLXModelCatalog.defaultModelRepo
        )
        XCTAssertEqual(
            MLXModelCatalog.canonicalModelRepo("mlx-community/Qwen3-ASR-0.6B-4bit"),
            "mlx-community/Qwen3-ASR-0.6B-4bit"
        )
    }

    func testRealtimeCapabilityUsesCanonicalizedRepo() {
        XCTAssertFalse(
            MLXModelCatalog.isRealtimeCapableModelRepo("mlx-community/Voxtral-Mini-4B-Realtime-2602")
        )
        XCTAssertFalse(
            MLXModelCatalog.isRealtimeCapableModelRepo("mlx-community/Voxtral-Mini-4B-Realtime-6bit")
        )
        XCTAssertFalse(
            MLXModelCatalog.isRealtimeCapableModelRepo("mlx-community/Qwen3-ASR-0.6B-4bit")
        )
    }

    func testLiveModeUsesNativeSessionForSupportedRealtimeFamilies() {
        XCTAssertEqual(
            MLXModelCatalog.liveMode(for: "mlx-community/Qwen3-ASR-0.6B-4bit"),
            .nativeQwenLive
        )
        XCTAssertEqual(
            MLXModelCatalog.liveMode(for: "mlx-community/Qwen3-ASR-1.7B-6bit"),
            .nativeQwenLive
        )
        XCTAssertEqual(
            MLXModelCatalog.liveMode(for: "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"),
            .nativeNemotronLive
        )
        XCTAssertEqual(
            MLXModelCatalog.liveMode(for: "beshkenadze/cohere-transcribe-03-2026-mlx-fp16"),
            .nativeStreamingLive
        )
        XCTAssertEqual(
            MLXModelCatalog.liveMode(for: "OpenMOSS-Team/MOSS-Transcribe-Diarize"),
            .nativeStreamingLive
        )
        XCTAssertEqual(
            MLXModelCatalog.liveMode(for: "mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"),
            .nativeQwenLive
        )
    }

    func testQwen3CatalogTagsExposeRealtimeBadge() {
        XCTAssertTrue(
            MLXModelCatalog.catalogTagKeys(for: "mlx-community/Qwen3-ASR-0.6B-4bit").contains("Realtime")
        )
        XCTAssertTrue(
            MLXModelCatalog.catalogTagKeys(for: "mlx-community/Qwen3-ASR-1.7B-6bit").contains("Realtime")
        )
    }

    func testNemotronCatalogTagsExposeMultilingualBadge() {
        let tagKeys = MLXModelCatalog.catalogTagKeys(for: "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit")
        XCTAssertTrue(tagKeys.contains("Multilingual"))
        XCTAssertTrue(tagKeys.contains("Realtime"))
    }

    func testCapabilityRegistryUsesExactLanguageMatrices() {
        let qwen = MLXModelCatalog.capability(for: "mlx-community/Qwen3-ASR-0.6B-4bit")
        XCTAssertTrue(qwen.supportsLanguage(code: "zh"))
        XCTAssertTrue(qwen.supportsLanguage(code: "yue"))
        XCTAssertFalse(qwen.supportsLanguage(code: "sw"))
        XCTAssertEqual(qwen.kvCachePolicy, .conservativeQwen)

        let parakeetV3 = MLXModelCatalog.capability(for: "mlx-community/parakeet-tdt-0.6b-v3")
        XCTAssertEqual(parakeetV3.family, .parakeet)
        XCTAssertEqual(parakeetV3.supportedLanguageCodes.count, 25)
        XCTAssertTrue(parakeetV3.supportsLanguage(code: "de"))
        XCTAssertFalse(parakeetV3.supportsLanguage(code: "zh"))

        let legacyParakeet = MLXModelCatalog.capability(for: "mlx-community/parakeet-tdt-0.6b-v2")
        XCTAssertEqual(legacyParakeet, parakeetV3)

        let nemotron = MLXModelCatalog.capability(for: "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit")
        XCTAssertTrue(nemotron.supportsLanguage(code: "zh"))
        XCTAssertTrue(nemotron.supportsLanguage(code: "ja"))
        XCTAssertFalse(nemotron.supportsLanguage(code: "el"))

        XCTAssertEqual(
            MLXModelCatalog.capability(for: "mlx-community/Voxtral-Mini-4B-Realtime-6bit"),
            qwen
        )
    }

    func testEveryCatalogModelHasAnExplicitCapability() {
        let missingRepos = MLXModelCatalog.supportedModels
            .map(\.id)
            .filter { !MLXModelCatalog.hasRegisteredCapability(for: $0) }

        XCTAssertEqual(missingRepos, [])
    }

    func testCapabilityRegistryDrivesFormsAndStructuredOutput() {
        let parakeet = MLXModelCatalog.capability(for: "mlx-community/parakeet-tdt-0.6b-v3")
        XCTAssertTrue(parakeet.configurationCapabilities.isEmpty)
        XCTAssertEqual(parakeet.languageRouting, .automatic)
        XCTAssertEqual(parakeet.timingGranularity, .sentence)
        XCTAssertTrue(parakeet.timingGranularity.providesReliableSegments)
        XCTAssertTrue(parakeet.outputCapabilities.contains(.timestamps))

        let nemotron = MLXModelCatalog.capability(for: "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit")
        XCTAssertEqual(nemotron.timingGranularity, .sentence)
        XCTAssertTrue(nemotron.outputCapabilities.contains(.timestamps))

        let moss = MLXModelCatalog.capability(for: "OpenMOSS-Team/MOSS-Transcribe-Diarize")
        XCTAssertTrue(moss.configurationCapabilities.contains(.mossPromptAndOutput))
        XCTAssertTrue(moss.outputCapabilities.contains(.timestamps))
        XCTAssertTrue(moss.outputCapabilities.contains(.speakerLabels))
        XCTAssertEqual(moss.timingGranularity, .sentence)
        XCTAssertEqual(moss.vadPolicy, .preserveTimeline)

        let whisper = MLXModelCatalog.capability(for: "mlx-community/whisper-large-v3-turbo")
        XCTAssertEqual(whisper.timingGranularity, .chunk)
        XCTAssertFalse(whisper.timingGranularity.providesReliableSegments)
        XCTAssertFalse(whisper.configurationCapabilities.contains(.recognitionPreset))
        XCTAssertNil(whisper.kvCachePolicy)

        let senseVoice = MLXModelCatalog.capability(for: "mlx-community/SenseVoiceSmall")
        XCTAssertTrue(senseVoice.configurationCapabilities.contains(.senseVoiceITN))
        XCTAssertTrue(senseVoice.outputCapabilities.contains(.emotion))
        XCTAssertTrue(senseVoice.outputCapabilities.contains(.audioEvents))
        // Catalog stays `.standard` so meeting external final-speech validation remains enabled.
        XCTAssertEqual(senseVoice.vadPolicy, .standard)
        XCTAssertTrue(senseVoice.vadPolicy.usesExternalFinalSpeechValidation)
    }

    func testBatchStandardFamiliesAllowExternalFinalSpeechTrim() {
        let standardRepos = [
            "mlx-community/whisper-large-v3-turbo",
            "mlx-community/parakeet-tdt-0.6b-v3",
        ]
        for repo in standardRepos {
            let policy = MLXModelCatalog.capability(for: repo).vadPolicy
            XCTAssertEqual(policy, .standard, repo)
            XCTAssertTrue(policy.allowsExternalFinalSpeechTrim, repo)
        }
    }

    func testRecognitionPresetFamiliesLiftFinalChunkDuration() {
        let presetRepos = [
            "beshkenadze/cohere-transcribe-03-2026-mlx-fp16",
        ]
        for repo in presetRepos {
            XCTAssertTrue(
                MLXModelCatalog.capability(for: repo).configurationCapabilities.contains(.recognitionPreset),
                repo
            )
        }
        XCTAssertEqual(MLXTranscriptionPlanning.postStopFinalChunkDuration(presetChunkDuration: 90), 1200)
    }

    func testModelManagedNativeFamiliesForbidExternalFinalSpeechTrim() {
        let managedRepos = [
            "beshkenadze/cohere-transcribe-03-2026-mlx-fp16",
            "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit",
        ]
        for repo in managedRepos {
            let policy = MLXModelCatalog.capability(for: repo).vadPolicy
            XCTAssertEqual(policy, .modelManaged, repo)
            XCTAssertFalse(policy.allowsExternalFinalSpeechTrim, repo)
            XCTAssertFalse(policy.usesExternalFinalSpeechValidation, repo)
        }
    }

    func testSenseVoiceDictationDisablesPCMTrimWhileMeetingValidationStaysEnabled() {
        let capability = MLXModelCatalog.capability(for: "mlx-community/SenseVoiceSmall")
        XCTAssertEqual(capability.vadPolicy, .standard)
        XCTAssertTrue(capability.vadPolicy.usesExternalFinalSpeechValidation)
        XCTAssertTrue(capability.vadPolicy.allowsExternalFinalSpeechTrim)
        XCTAssertFalse(
            MLXTranscriptionPlanning.allowsExternalFinalSpeechTrim(
                vadPolicy: capability.vadPolicy,
                family: .senseVoice
            )
        )
        XCTAssertEqual(
            MeetingFinalSpeechValidator.vadPolicy(
                transcriptionEngine: .mlxAudio,
                mlxModelRepo: "mlx-community/SenseVoiceSmall"
            ),
            .standard
        )
    }


    func testFallbackRemoteSizeSupportsLegacyAndCuratedRepos() {
        XCTAssertEqual(
            MLXModelCatalog.fallbackRemoteSizeText(repo: "mlx-community/FireRedASR2"),
            MLXModelCatalog.fallbackRemoteSizeText(repo: "mlx-community/FireRedASR2-AED-mlx")
        )
        XCTAssertNotNil(
            MLXModelCatalog.fallbackRemoteSizeText(repo: "mlx-community/Qwen3-ASR-0.6B-4bit")
        )
        XCTAssertNotNil(
            MLXModelCatalog.fallbackRemoteSizeText(repo: "mlx-community/whisper-large-v3-turbo")
        )
        XCTAssertNotNil(
            MLXModelCatalog.fallbackRemoteSizeText(repo: "mlx-community/whisper-small-mlx")
        )
    }

    func testWhisperMigrationMapsLegacyModelIDsToMLXRepos() {
        XCTAssertEqual(
            MLXWhisperMigrationSupport.repo(forLegacyWhisperModelID: "tiny"),
            "mlx-community/whisper-small-mlx"
        )
        XCTAssertEqual(
            MLXWhisperMigrationSupport.repo(forLegacyWhisperModelID: "base"),
            "mlx-community/whisper-small-mlx"
        )
        XCTAssertEqual(
            MLXWhisperMigrationSupport.repo(forLegacyWhisperModelID: "medium"),
            "mlx-community/whisper-large-v3-turbo"
        )
        XCTAssertTrue(
            MLXWhisperMigrationSupport.isWhisperRepo("mlx-community/whisper-large-v3-turbo")
        )
    }


}
