import XCTest
@testable import Voxt

final class SettingsPermissionSupportTests: XCTestCase {
    private func makeFeatureSettings(
        transcriptionASR: FeatureModelSelectionID = .mlx(MLXModelManager.defaultModelRepo),
        translationASR: FeatureModelSelectionID = .mlx(MLXModelManager.defaultModelRepo),
        rewriteASR: FeatureModelSelectionID = .mlx(MLXModelManager.defaultModelRepo),
        notesEnabled: Bool = true,
        remindersEnabled: Bool = false
    ) -> FeatureSettings {
        FeatureSettings(
            transcription: .init(
                asrSelectionID: transcriptionASR,
                llmEnabled: false,
                llmSelectionID: .localLLM(CustomLLMModelManager.defaultModelRepo),
                prompt: AppPreferenceKey.defaultEnhancementPrompt,
                notes: .init(
                    enabled: notesEnabled,
                    titleModelSelectionID: .localLLM(CustomLLMModelManager.defaultModelRepo),
                    remindersSync: .init(enabled: remindersEnabled)
                )
            ),
            translation: .init(
                asrSelectionID: translationASR,
                modelSelectionID: .localLLM(CustomLLMModelManager.defaultModelRepo),
                targetLanguageRawValue: TranslationTargetLanguage.english.rawValue,
                prompt: AppPreferenceKey.defaultTranslationPrompt
            ),
            rewrite: .init(
                asrSelectionID: rewriteASR,
                llmSelectionID: .localLLM(CustomLLMModelManager.defaultModelRepo),
                prompt: AppPreferenceKey.defaultRewritePrompt,
                appEnhancementEnabled: false
            )
        )
    }

    func testSidebarRequirementContextPreservesFeatureSettingsSelections() {
        let settings = makeFeatureSettings(remindersEnabled: true)
        let context = SettingsPermissionRequirementResolver.sidebarRequirementContext(
            selectedEngine: .remote, featureSettings: settings
        )
        XCTAssertEqual(context.selectedEngine, .remote)
        XCTAssertEqual(context.featureSettings?.translation.asrSelectionID, settings.translation.asrSelectionID)
        XCTAssertTrue(context.featureSettings?.transcription.notes.remindersSync.enabled == true)
    }

    func testBasicAndSidebarPermissionsDoNotRequireSystemCapture() {
        for engine in [TranscriptionEngine.remote, .mlxAudio] {
            let basic = SettingsPermissionRequirementContext(selectedEngine: engine, featureSettings: nil)
            let sidebar = SettingsPermissionRequirementResolver.sidebarRequirementContext(
                selectedEngine: engine, featureSettings: makeFeatureSettings()
            )
            XCTAssertEqual(SettingsPermissionRequirementResolver.requiredPermissions(context: basic), [.microphone, .accessibility])
            XCTAssertEqual(SettingsPermissionRequirementResolver.requiredPermissions(context: sidebar), [.microphone, .accessibility])
        }
        // Removed authorizations cannot accidentally reappear as permission rows.
        XCTAssertEqual(Set(SettingsPermissionKind.allCases.map(\.rawValue)),
                       ["microphone", "speechRecognition", "accessibility", "reminders"])
    }

    func testSpeechRecognitionForSelectedEngineOrEnabledFeature() {
        let contexts = [
            SettingsPermissionRequirementContext(selectedEngine: .dictation, featureSettings: nil),
            SettingsPermissionRequirementContext(selectedEngine: .remote, featureSettings: makeFeatureSettings(transcriptionASR: .dictation)),
            SettingsPermissionRequirementContext(selectedEngine: .remote, featureSettings: makeFeatureSettings(translationASR: .dictation))
        ]
        for context in contexts {
            XCTAssertEqual(SettingsPermissionRequirementResolver.requiredPermissions(context: context),
                           [.microphone, .accessibility, .speechRecognition])
        }
    }

    func testDisabledFeaturesDoNotAddSpeechRecognitionRequirement() {
        var settings = makeFeatureSettings(translationASR: .dictation, rewriteASR: .dictation)
        settings.availability.translationEnabled = false
        settings.availability.rewriteEnabled = false
        let context = SettingsPermissionRequirementContext(selectedEngine: .remote, featureSettings: settings)
        XCTAssertEqual(SettingsPermissionRequirementResolver.requiredPermissions(context: context), [.microphone, .accessibility])
    }

    func testRemindersOnlyRequiredForEnabledNotesSync() {
        for notesEnabled in [false, true] {
            let context = SettingsPermissionRequirementContext(
                selectedEngine: .remote,
                featureSettings: makeFeatureSettings(notesEnabled: notesEnabled, remindersEnabled: true)
            )
            XCTAssertEqual(SettingsPermissionRequirementResolver.requiredPermissions(context: context),
                           notesEnabled ? [.microphone, .accessibility, .reminders] : [.microphone, .accessibility])
        }
    }
}
