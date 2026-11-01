import XCTest
@testable import Voxt

@MainActor
final class MLXInferenceConfigurationTests: XCTestCase {
    func testWhisperIgnoresUnsupportedPresetButKeepsTemperatureAndLanguage() {
        let plan = resolve("mlx-community/whisper-large-v3-turbo", settings: .init(preset: .accuracyFirst, whisperTemperature: 0.4))
        XCTAssertEqual(plan.family, .whisper)
        XCTAssertEqual(plan.generationParameters, ASRGenerationOptions(
            maximumTokenCount: 1024,
            temperature: 0.4,
            topProbabilityThreshold: 0.95,
            topTokenCount: 0,
            verbose: false,
            language: "en",
            usePunctuation: nil,
            chunkDurationSeconds: 1200,
            minimumChunkDurationSeconds: 1,
            kvCacheBitWidth: nil,
            kvCacheGroupSize: 64,
            kvCacheQuantizedStartTokenCount: 0
        ))
        XCTAssertEqual(plan.languageHint, "en")
    }

    func testQuickPassBudgetDependsOnRealtimeVisibility() {
        let repo = "mlx-community/Qwen3-ASR-0.6B-4bit"
        XCTAssertEqual(resolve(repo, stage: .postStopQuick, visible: true).generationParameters.maximumTokenCount, 1024)
        XCTAssertEqual(resolve(repo, stage: .postStopQuick, visible: false).generationParameters.maximumTokenCount, 512)
        XCTAssertEqual(resolve(repo, stage: .postStopFinal).generationParameters.maximumTokenCount,
                       MLXTranscriptionPlanning.postStopFinalMaxTokens(audioDurationSeconds: 30))
    }

    func testFinalWithoutKnownDurationKeepsFullTokenBudget() {
        let plan = resolve(
            "mlx-community/Qwen3-ASR-0.6B-4bit",
            stage: .postStopFinal,
            audioDurationSeconds: nil
        )
        XCTAssertEqual(plan.generationParameters.maximumTokenCount, 8192)
    }

    func testQwenAccuracyPresetUsesShortWindowsUntilFinalAndKeepsKVPolicy() {
        let repo = "mlx-community/Qwen3-ASR-0.6B-4bit"
        let settings = MLXLocalTuningSettings(preset: .accuracyFirst)
        for stage: MLXCorrectionPassKind in [.intermediate, .postStopQuick, .postStopFinal] {
            let options = resolve(repo, stage: stage, settings: settings).generationParameters
            XCTAssertEqual(options.chunkDurationSeconds, stage == .postStopFinal ? 1200 : 90)
            XCTAssertEqual(options.minimumChunkDurationSeconds, stage == .postStopFinal ? 1 : 2.5)
            XCTAssertEqual(options.kvCacheBitWidth, 8)
            XCTAssertEqual(options.kvCacheGroupSize, 64)
            XCTAssertEqual(options.kvCacheQuantizedStartTokenCount, 256)
            XCTAssertEqual(options.language, "en")
        }
    }

    func testCohereKeepsExplicitTuningEvenForFinal() {
        let plan = resolve(
            "beshkenadze/cohere-transcribe-03-2026-mlx-fp16", stage: .postStopFinal,
            settings: .init(cohereUsePunctuation: false, cohereMaxTokens: 2048, cohereTemperature: 0.2)
        )
        XCTAssertEqual(plan.generationParameters.maximumTokenCount, 2048)
        XCTAssertEqual(plan.generationParameters.temperature, Float(0.2))
        XCTAssertEqual(plan.generationParameters.usePunctuation, false)
    }

    func testMeetingMOSSUsesStructuredOutputAndAddsHotwordsOnlyAfterStop() {
        let settings = MLXLocalTuningSettings(mossMeetingHotwords: "UniqueHotword")
        let intermediate = resolve("OpenMOSS-Team/MOSS-Transcribe-Diarize", settings: settings, purpose: .meeting)
        let final = resolve("OpenMOSS-Team/MOSS-Transcribe-Diarize", stage: .postStopFinal, settings: settings, purpose: .meeting)
        XCTAssertNil(final.languageHint)
        XCTAssertNil(final.generationParameters.language)
        XCTAssertEqual(final.mossOutputMode, .timestampedDiarization)
        XCTAssertFalse(intermediate.mossPrompt?.contains("UniqueHotword") ?? false)
        XCTAssertTrue(final.mossPrompt?.contains("UniqueHotword") ?? false)
    }

    func testParakeetDoesNotReceiveAnUnsupportedLanguageHint() {
        let plan = resolve("mlx-community/parakeet-tdt-0.6b-v3")
        XCTAssertNil(plan.languageHint)
        XCTAssertNil(plan.generationParameters.language)
    }

    func testFinalInferenceRejectsIdentityOnlyModelWithoutFallback() async {
        do {
            _ = try await MLXTranscriber.runFinalInference(
                model: .modelUseFixture(ASRModelUseFixture()),
                audioSamples: [0],
                inferenceConfiguration: resolve("mlx-community/Qwen3-ASR-0.6B-4bit"),
                targetSampleRate: 16_000,
                inferenceTaskPriority: .userInitiated
            )
            XCTFail("An identity-only model must not run inference.")
        } catch ASRModelUseFixture.InferenceError.unavailable {
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testCancelledFinalInferenceStopsBeforeResolvingRuntime() async {
        let configuration = resolve("mlx-community/Qwen3-ASR-0.6B-4bit")
        let inferenceTask = Task { @MainActor in
            try await MLXTranscriber.runFinalInference(
                model: .modelUseFixture(ASRModelUseFixture()),
                audioSamples: [0],
                inferenceConfiguration: configuration,
                targetSampleRate: 16_000,
                inferenceTaskPriority: .userInitiated
            )
        }
        inferenceTask.cancel()
        do {
            _ = try await inferenceTask.value
            XCTFail("Cancelled inference must not return a result.")
        } catch is CancellationError {
        } catch {
            XCTFail("Cancellation must precede runtime resolution: \(error)")
        }
    }

    private func resolve(
        _ repo: String,
        stage: MLXCorrectionPassKind = .intermediate,
        settings: MLXLocalTuningSettings = .init(),
        purpose: MLXTranscriptionPurpose = .dictation,
        visible: Bool = true,
        audioDurationSeconds: Double? = 30
    ) -> MLXTranscriber.ResolvedInferenceConfiguration {
        MLXInferenceConfiguration.resolve(
            for: stage, audioDurationSeconds: audioDurationSeconds,
            hintPayload: .init(language: "en"), tuningSettings: settings,
            transcriptionPurpose: purpose, userLanguageCodes: ["en"],
            capability: MLXModelCatalog.capability(for: repo), dictionaryTerms: "Voxt",
            sessionAllowsRealtimeTextDisplay: visible
        )
    }
}
