import XCTest
@testable import Voxt

@MainActor
final class OnboardingGuideTests: XCTestCase {
    func testSixStepNavigation() {
        let steps: [OnboardingGuideStep] = [
            .permissions, .models, .transcriptionShortcut,
            .translationShortcut, .translationSelection, .finish
        ]
        XCTAssertEqual(OnboardingGuideStep.allCases, steps)
        for (index, step) in steps.enumerated() {
            XCTAssertEqual(step.stepNumber, index + 1)
            XCTAssertEqual(step.previous, index == 0 ? nil : steps[index - 1])
            XCTAssertEqual(step.next, index == steps.count - 1 ? nil : steps[index + 1])
        }
    }

    func testRetiredStepsResumeAtRelevantShortGuideStep() {
        let defaults = TestDoubles.makeUserDefaults()
        let migrations: [String: OnboardingGuideStep] = [
            "microphone": .permissions, "language": .permissions, "model": .models,
            "transcriptionEnhancement": .transcriptionShortcut, "translation": .translationShortcut,
            "rewriteShortcut": .finish, "rewriteSelection": .finish,
            "appEnhancement": .finish, "meeting": .finish
        ]
        for (rawValue, expected) in migrations {
            defaults.set(rawValue, forKey: AppPreferenceKey.onboardingLastStepID)
            XCTAssertEqual(OnboardingPreferenceManager.savedLastGuideStep(defaults: defaults), expected)
        }
        defaults.set("unknown-future-step", forKey: AppPreferenceKey.onboardingLastStepID)
        XCTAssertNil(OnboardingPreferenceManager.savedLastGuideStep(defaults: defaults))
    }

    func testOpeningAndDiscardingModelDraftPreservesMixedConfiguration() {
        let original = mixedSettings()
        let draft = OnboardingModelDraft()
        XCTAssertFalse(draft.hasChanges)
        XCTAssertEqual(draft.applying(to: original), original)
    }

    func testSpeechChoiceDoesNotOverwriteOtherFeaturesOrEnableEnhancement() {
        let original = mixedSettings()
        var draft = OnboardingModelDraft()
        draft.speech = .mlx(MLXModelManager.defaultModelRepo)
        let updated = draft.applying(to: original)
        var expected = original
        expected.transcription.asrSelectionID = draft.speech!
        XCTAssertEqual(updated, expected)
        XCTAssertFalse(updated.transcription.llmEnabled)
    }

    func testTranslationChoiceDoesNotChangeSpeechOrOtherTextModels() {
        let original = mixedSettings()
        var draft = OnboardingModelDraft()
        draft.translation = .localLLM(CustomLLMModelManager.defaultModelRepo)
        var expected = original
        expected.translation.modelSelectionID = draft.translation!
        XCTAssertEqual(draft.applying(to: original), expected)
    }

    func testDraftAppliesToLatestSettingsInsteadOfRestoringStaleSnapshot() {
        var latest = mixedSettings()
        let draft = OnboardingModelDraft(speech: .dictation)
        latest.rewrite.prompt = "Changed in another settings window"
        latest.transcription.notes.enabled = false
        let updated = draft.applying(to: latest)
        XCTAssertEqual(updated.rewrite, latest.rewrite)
        XCTAssertEqual(updated.transcription.notes, latest.transcription.notes)
    }

    func testLoadingGuideConfigurationDoesNotWritePreferences() {
        let defaults = TestDoubles.makeUserDefaults()
        defaults.set(TranscriptionEngine.remote.rawValue, forKey: AppPreferenceKey.transcriptionEngine)
        let before = defaults.dictionaryRepresentation() as NSDictionary
        _ = FeatureSettingsStore.load(defaults: defaults)
        _ = OnboardingModelDraft()
        XCTAssertEqual(defaults.dictionaryRepresentation() as NSDictionary, before)
    }

    func testPracticeRequiresDeliveryNotJustSessionStart() {
        var state = OnboardingPracticeState()
        let id = UUID()
        receive(.started(id: id, kind: .transcription, windowNumber: 7), into: &state)
        XCTAssertEqual(state.phase, .starting)
        state.startedListening(sessionID: id)
        XCTAssertEqual(state.phase, .listening)
        receive(.processing(id: id), into: &state)
        XCTAssertEqual(state.phase, .processing)
        state.startedListening(sessionID: id)
        XCTAssertEqual(state.phase, .processing, "Late recording callbacks must not regress progress")
        receive(.delivered(id: id, text: "My own words", succeeded: true), into: &state)
        XCTAssertEqual(state.phase, .succeeded)
        XCTAssertEqual(state.result, "My own words")
        receive(.ended(id: id, message: ""), into: &state)
        XCTAssertEqual(state.phase, .succeeded)
    }

    func testWrongWorkflowOrWindowCannotCompletePractice() {
        for event in [
            OnboardingSessionEvent.started(id: UUID(), kind: .voiceTranslation, windowNumber: 7),
            .started(id: UUID(), kind: .transcription, windowNumber: 8),
            .started(id: UUID(), kind: .transcription, windowNumber: nil),
            .delivered(id: UUID(), text: "Unrelated output", succeeded: true)
        ] {
            var state = OnboardingPracticeState()
            receive(event, into: &state)
            XCTAssertEqual(state.phase, .ready)
            XCTAssertNil(state.sessionID)
        }
    }

    func testMissingGuideWindowCannotAdoptSession() {
        var state = OnboardingPracticeState()
        state.receive(.started(id: UUID(), kind: .transcription, windowNumber: nil), expectedKind: .transcription, windowNumber: nil)
        XCTAssertEqual(state.phase, .ready)
    }

    func testEmptyOrUndeliveredResultsDoNotSucceed() {
        for (text, delivered) in [(" ", true), ("Recognized, but not inserted", false)] {
            var state = OnboardingPracticeState()
            let id = UUID()
            receive(.started(id: id, kind: .transcription, windowNumber: 7), into: &state)
            receive(.delivered(id: id, text: text, succeeded: delivered), into: &state)
            XCTAssertEqual(state.phase, .failed)
        }
    }

    func testCancellationOrFailureAllowsRetryAndRejectsStaleCallbacks() {
        var state = OnboardingPracticeState()
        let first = UUID()
        receive(.started(id: first, kind: .transcription, windowNumber: 7), into: &state)
        receive(.ended(id: first, message: "Microphone disconnected"), into: &state)
        XCTAssertEqual(state.phase, .failed)
        XCTAssertFalse(state.isBusy)
        XCTAssertEqual(state.message, "Microphone disconnected")
        let retry = UUID()
        receive(.started(id: retry, kind: .transcription, windowNumber: 7), into: &state)
        receive(.delivered(id: first, text: "Stale result", succeeded: true), into: &state)
        XCTAssertEqual(state.phase, .starting)
        XCTAssertTrue(state.result.isEmpty)
        XCTAssertTrue(state.message.isEmpty)
        receive(.delivered(id: retry, text: "Retry result", succeeded: true), into: &state)
        XCTAssertEqual(state.phase, .succeeded)
    }

    func testSelectedTextTranslationRequiresActualResult() {
        var state = OnboardingPracticeState()
        let id = UUID()
        state.receive(.started(id: id, kind: .selectedTextTranslation, windowNumber: 7), expectedKind: .selectedTextTranslation, windowNumber: 7)
        XCTAssertEqual(state.phase, .processing)
        state.receive(.delivered(id: id, text: "Translated text", succeeded: true), expectedKind: .selectedTextTranslation, windowNumber: 7)
        XCTAssertEqual(state.phase, .succeeded)
    }

    private func receive(_ event: OnboardingSessionEvent, into state: inout OnboardingPracticeState) {
        state.receive(event, expectedKind: .transcription, windowNumber: 7)
    }

    private func mixedSettings() -> FeatureSettings {
        var settings = FeatureSettings.placeholder
        settings.transcription.asrSelectionID = .remoteASR(.openAIWhisper)
        settings.transcription.llmEnabled = false
        settings.translation.asrSelectionID = .dictation
        settings.translation.modelSelectionID = .remoteLLM(.openAI)
        settings.rewrite.llmSelectionID = .remoteLLM(.aliyunBailian)
        settings.meeting.asrSelectionID = .mlx(MLXModelManager.defaultModelRepo)
        settings.meeting.summaryModelSelectionID = .remoteLLM(.stepFun)
        return settings
    }
}
