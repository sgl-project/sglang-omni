import XCTest
@testable import Voxt

@MainActor
final class MLXInferenceConfigurationTests: XCTestCase {
    func testWhisperIgnoresUnsupportedPresetButKeepsTemperatureAndLanguage() {
        let plan = resolve("mlx-community/whisper-large-v3-turbo", settings: .init(preset: .accuracyFirst, whisperTemperature: 0.4))
        XCTAssertEqual(plan.family, .whisper)
        XCTAssertEqual(plan.generationParameters.chunkDuration, 1200)
        XCTAssertEqual(plan.generationParameters.minChunkDuration, 1)
        XCTAssertEqual(plan.generationParameters.temperature, Float(0.4))
        XCTAssertEqual(plan.languageHint, "en")
    }

    func testQuickPassBudgetDependsOnRealtimeVisibility() {
        let repo = "mlx-community/Qwen3-ASR-0.6B-4bit"
        XCTAssertEqual(resolve(repo, stage: .postStopQuick, visible: true).generationParameters.maxTokens, 1024)
        XCTAssertEqual(resolve(repo, stage: .postStopQuick, visible: false).generationParameters.maxTokens, 512)
        XCTAssertEqual(resolve(repo, stage: .postStopFinal).generationParameters.maxTokens,
                       MLXTranscriptionPlanning.postStopFinalMaxTokens(audioDurationSeconds: 30))
    }

    func testCohereKeepsExplicitTuningEvenForFinal() {
        let plan = resolve(
            "beshkenadze/cohere-transcribe-03-2026-mlx-fp16", stage: .postStopFinal,
            settings: .init(cohereUsePunctuation: false, cohereMaxTokens: 2048, cohereTemperature: 0.2)
        )
        XCTAssertEqual(plan.generationParameters.maxTokens, 2048)
        XCTAssertEqual(plan.generationParameters.temperature, Float(0.2))
        XCTAssertEqual(plan.generationParameters.usePunctuation, false)
    }

    func testMeetingMOSSUsesStructuredOutputAndAddsHotwordsOnlyAfterStop() {
        let settings = MLXLocalTuningSettings(mossMeetingHotwords: "UniqueHotword")
        let intermediate = resolve("OpenMOSS-Team/MOSS-Transcribe-Diarize", settings: settings, purpose: .meeting)
        let final = resolve("OpenMOSS-Team/MOSS-Transcribe-Diarize", stage: .postStopFinal, settings: settings, purpose: .meeting)
        XCTAssertNil(final.languageHint)
        XCTAssertEqual(final.mossOutputMode, .timestampedDiarization)
        XCTAssertFalse(intermediate.mossPrompt?.contains("UniqueHotword") ?? false)
        XCTAssertTrue(final.mossPrompt?.contains("UniqueHotword") ?? false)
    }

    func testParakeetDoesNotReceiveAnUnsupportedLanguageHint() {
        XCTAssertNil(resolve("mlx-community/parakeet-tdt-0.6b-v3").languageHint)
    }

    private func resolve(
        _ repo: String,
        stage: MLXCorrectionPassKind = .intermediate,
        settings: MLXLocalTuningSettings = .init(),
        purpose: MLXTranscriptionPurpose = .dictation,
        visible: Bool = true
    ) -> MLXTranscriber.ResolvedInferenceConfiguration {
        MLXInferenceConfiguration.resolve(
            for: stage, audioDurationSeconds: 30,
            hintPayload: .init(language: "en"), tuningSettings: settings,
            transcriptionPurpose: purpose, userLanguageCodes: ["en"],
            capability: MLXModelCatalog.capability(for: repo), dictionaryTerms: "Voxt",
            sessionAllowsRealtimeTextDisplay: visible
        )
    }
}
