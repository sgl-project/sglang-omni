import XCTest
@testable import Voxt
import HuggingFace
import MLX
import MLXAudioSTT

@MainActor
final class MLXModelManagerTests: MLXModelManagerTestCase {
    func testCanonicalModelRepoMapsLegacyReposToCurrentIdentifiers() {
        XCTAssertEqual(
            MLXModelManager.canonicalModelRepo("mlx-community/Parakeet-0.6B"),
            "mlx-community/parakeet-tdt-0.6b-v3"
        )
        XCTAssertEqual(
            MLXModelManager.canonicalModelRepo("mlx-community/GLM-ASR-Nano-4bit"),
            MLXModelCatalog.defaultModelRepo
        )
        XCTAssertEqual(
            MLXModelManager.canonicalModelRepo("mlx-community/Voxtral-Mini-4B-Realtime-2602"),
            MLXModelCatalog.defaultModelRepo
        )
        XCTAssertEqual(
            MLXModelManager.canonicalModelRepo("mlx-community/Voxtral-Mini-4B-Realtime-2602-6bit"),
            MLXModelCatalog.defaultModelRepo
        )
        XCTAssertEqual(
            MLXModelManager.canonicalModelRepo("mlx-community/FireRedASR2"),
            MLXModelCatalog.defaultModelRepo
        )
    }

    func testRetiredVoxtralUsesDefaultCapabilities() {
        XCTAssertFalse(MLXModelManager.isRealtimeCapableModelRepo("mlx-community/Voxtral-Mini-4B-Realtime-2602"))
        XCTAssertFalse(MLXModelManager.isRealtimeCapableModelRepo("mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit"))
        XCTAssertFalse(MLXModelManager.isRealtimeCapableModelRepo("mlx-community/Voxtral-Mini-4B-Realtime-2602-6bit"))
        XCTAssertFalse(MLXModelManager.isRealtimeCapableModelRepo("mlx-community/Voxtral-Mini-4B-Realtime-6bit"))
        XCTAssertFalse(MLXModelManager.isRealtimeCapableModelRepo("mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16"))
        XCTAssertTrue(MLXModelManager.isRealtimeCapableModelRepo("beshkenadze/cohere-transcribe-03-2026-mlx-fp16"))
        XCTAssertTrue(MLXModelManager.isRealtimeCapableModelRepo("OpenMOSS-Team/MOSS-Transcribe-Diarize"))
        XCTAssertTrue(MLXModelManager.isRealtimeCapableModelRepo("mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"))
        XCTAssertFalse(MLXModelManager.isRealtimeCapableModelRepo("mlx-community/Qwen3-ASR-0.6B-4bit"))
    }

    func testLiveModeRoutesRealtimeFamiliesToNativeSessions() {
        XCTAssertEqual(
            MLXModelManager.liveMode(for: "mlx-community/Qwen3-ASR-0.6B-4bit"),
            .nativeQwenLive
        )
        XCTAssertEqual(
            MLXModelManager.liveMode(for: "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"),
            .nativeNemotronLive
        )
        XCTAssertEqual(
            MLXModelManager.liveMode(for: "beshkenadze/cohere-transcribe-03-2026-mlx-fp16"),
            .nativeStreamingLive
        )
        XCTAssertEqual(
            MLXModelManager.liveMode(for: "OpenMOSS-Team/MOSS-Transcribe-Diarize"),
            .nativeStreamingLive
        )
        XCTAssertEqual(
            MLXModelManager.liveMode(for: "mlx-community/Voxtral-Mini-4B-Realtime-6bit"),
            .nativeQwenLive
        )
    }

    func testTranscriptionBehaviorUsesIncrementalModeForDefaultModels() {
        let behavior = MLXModelManager.transcriptionBehavior(for: "mlx-community/Qwen3-ASR-0.6B-4bit")

        XCTAssertEqual(behavior.correctionMode, .incremental)
        XCTAssertTrue(behavior.runsIntermediateCorrections)
        XCTAssertTrue(behavior.allowsQuickStopPass)
        XCTAssertTrue(behavior.preloadsOnRecordingStart)
    }

    func testIntermediateCorrectionDecisionReturnsContextWindowForIncrementalBehavior() {
        let behavior = MLXModelManager.transcriptionBehavior(for: "mlx-community/Qwen3-ASR-0.6B-4bit")

        let decision = MLXTranscriptionPlanning.intermediateCorrectionDecision(
            sampleCount: 16000 * 8,
            sampleRate: 16000,
            nextCorrectionAtSeconds: 6,
            behavior: behavior,
            firstCorrectionMinimumSeconds: 3.5,
            contextWindowSeconds: 18
        )

        XCTAssertNotNil(decision)
        XCTAssertEqual(decision?.elapsedSeconds ?? 0, 8, accuracy: 0.0001)
        XCTAssertEqual(decision?.contextSampleCount, 16000 * 18)
    }

    func testFinalizationPlanUsesQuickPassForLongIncrementalAudio() {
        let behavior = MLXModelManager.transcriptionBehavior(for: "mlx-community/Qwen3-ASR-0.6B-4bit")
        let plan = MLXTranscriptionPlanning.finalizationPlan(
            sampleCount: 16000 * 30,
            sampleRate: 16000,
            behavior: behavior,
            quickPassMinimumDurationSeconds: 14,
            quickPassContextWindowSeconds: 30
        )

        XCTAssertEqual(plan.durationSeconds, 30, accuracy: 0.0001)
        XCTAssertTrue(plan.shouldRunQuickPass)
        XCTAssertEqual(plan.quickPassSampleCount, 16000 * 30)
    }

    func testAvailableModelsIncludeLatestSupportedSTTRepos() {
        let modelIDs = Set(MLXModelManager.availableModels.map(\.id))

        XCTAssertTrue(modelIDs.contains("mlx-community/Qwen3-ASR-0.6B-4bit"))
        XCTAssertTrue(modelIDs.contains("mlx-community/Qwen3-ASR-1.7B-6bit"))
        XCTAssertTrue(modelIDs.contains("mlx-community/Qwen3-ASR-1.7B-8bit"))
        XCTAssertTrue(modelIDs.contains("mlx-community/SenseVoiceSmall"))
        XCTAssertFalse(modelIDs.contains("mlx-community/Voxtral-Mini-4B-Realtime-6bit"))
        XCTAssertTrue(modelIDs.contains("mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit"))
        XCTAssertTrue(modelIDs.contains("mlx-community/parakeet-tdt-0.6b-v3"))
        XCTAssertTrue(modelIDs.contains("beshkenadze/cohere-transcribe-03-2026-mlx-fp16"))
        XCTAssertTrue(modelIDs.contains("OpenMOSS-Team/MOSS-Transcribe-Diarize"))
        XCTAssertFalse(modelIDs.contains("Mediform/canary-1b-v2-mlx-q8"))
        XCTAssertFalse(modelIDs.contains("UsefulSensors/moonshine-tiny"))
        XCTAssertFalse(modelIDs.contains("facebook/wav2vec2-base-960h"))
        XCTAssertFalse(modelIDs.contains("facebook/mms-1b-fl102"))
        XCTAssertFalse(modelIDs.contains("mlx-community/FireRedASR2-AED-mlx"))
        XCTAssertFalse(modelIDs.contains("mlx-community/parakeet-tdt-0.6b-v2"))
        XCTAssertFalse(modelIDs.contains("mlx-community/granite-4.0-1b-speech-5bit"))
    }

    func testRetiredASRModelsDoNotReappearWhenInstalled() {
        let hiddenRepo = "mlx-community/GLM-ASR-Nano-2512-4bit"

        XCTAssertFalse(
            MLXModelCatalog.displayModels(includingInstalled: []).contains { $0.id == hiddenRepo }
        )
        XCTAssertFalse(
            MLXModelCatalog.displayModels(includingInstalled: [hiddenRepo]).contains { $0.id == hiddenRepo }
        )
    }

    func testKnownRemoteSizeFallbacksCoverCuratedLocalModels() {
        XCTAssertEqual(
            MLXModelManager.fallbackRemoteSizeText(repo: "mlx-community/FireRedASR2"),
            MLXModelManager.fallbackRemoteSizeText(repo: "mlx-community/FireRedASR2-AED-mlx")
        )
        XCTAssertNotNil(MLXModelManager.fallbackRemoteSizeText(repo: "beshkenadze/cohere-transcribe-03-2026-mlx-fp16"))
        XCTAssertNotNil(MLXModelManager.fallbackRemoteSizeText(repo: "mlx-community/Qwen3-ASR-0.6B-4bit"))
        XCTAssertNotNil(MLXModelManager.fallbackRemoteSizeText(repo: "mlx-community/whisper-base-mlx"))
        XCTAssertNotNil(CustomLLMModelManager.fallbackRemoteSizeText(repo: "mlx-community/Qwen3-4B-4bit"))
    }

    func testAllCuratedMLXModelsHaveRemoteSizeFallbacks() {
        let missingRepos = MLXModelManager.supportedModels
            .map(\.id)
            .filter { MLXModelManager.fallbackRemoteSizeText(repo: $0) == nil }

        XCTAssertEqual(missingRepos, [])
    }

    func testAllCuratedMLXWhisperModelsHaveRemoteSizeFallbacks() {
        let missingRepos = MLXModelManager.supportedModels
            .map(\.id)
            .filter(MLXWhisperMigrationSupport.isWhisperRepo(_:))
            .filter { MLXModelManager.fallbackRemoteSizeText(repo: $0) == nil }

        XCTAssertEqual(missingRepos, [])
    }
}
