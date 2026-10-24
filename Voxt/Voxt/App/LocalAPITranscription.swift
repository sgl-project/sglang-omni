// LocalAPITranscription.swift
// Provides the only Local API code that knows the transcription engines: a recorded file in,
// the selected engine's normalized text out, plus model readiness and the names saved to history.

import Foundation

extension AppDelegate {
    struct LocalAPITranscriptionModelMetadata {
        var title: String
        var remoteProvider: String?
        var remoteModel: String?
    }

    func transcribeLocalAPIAudio(at url: URL, language: ASRLanguageSelection = .userSettings) async throws -> String {
        prepareLegacySettingsForSession(outputMode: .transcription)
        synchronizeRuntimeASRStateForSession(outputMode: .transcription)

        let text: String
        switch transcriptionEngine {
        case .mlxAudio:
            let context = currentLocalASRStartContext()
            let decision = RecordingStartPlanner.resolve(
                selectedEngine: .mlxAudio,
                selectedMLXRepo: context.selectedMLXRepo,
                activeMLXDownloadRepo: context.activeMLXDownloadRepo,
                isSelectedMLXModelDownloaded: context.isSelectedMLXModelDownloaded,
                mlxModelState: context.mlxModelState
            )
            if case let .blocked(reason) = decision {
                throw LocalAPIError(message: reason.userMessage)
            }
            let transcriber = mlxTranscriber ?? MLXTranscriber(modelManager: mlxModelManager)
            mlxTranscriber = transcriber
            text = try await transcriber.transcribeAudioFile(url, languageSelection: language)
        case .remote:
            text = try await remoteASRTranscriber.transcribeMeetingAudioFile(url, languageSelection: language)
        case .dictation:
            throw LocalAPIError(message: AppLocalization.localizedString(
                "Apple Speech can't transcribe recorded audio. Choose a local or remote model in Voxt Settings > Transcription."
            ))
        }

        return RecordingSessionSupport.normalizedTranscriptionDisplayText(
            text,
            transcriptionEngine: transcriptionEngine,
            remoteProvider: remoteASRSelectedProvider,
            userMainLanguage: userMainLanguage
        )
    }

    func localAPITranscriptionModelMetadata() -> LocalAPITranscriptionModelMetadata {
        switch transcriptionEngine {
        case .dictation:
            return LocalAPITranscriptionModelMetadata(title: "Apple Speech Recognition")
        case .mlxAudio:
            let repo = mlxModelManager.currentModelRepo
            return LocalAPITranscriptionModelMetadata(title: "\(mlxModelManager.displayTitle(for: repo)) (\(repo))")
        case .remote:
            let provider = remoteASRSelectedProvider
            let model = remoteASRConfigurations[provider.rawValue]?.model
            return LocalAPITranscriptionModelMetadata(
                title: model.map { "\(provider.title) (\($0))" } ?? provider.title,
                remoteProvider: provider.title,
                remoteModel: model
            )
        }
    }

    /// Where the models of the next request run. A request first copies the feature settings onto
    /// the engine and text-model keys, so this does too; read before that, a key left from an earlier
    /// session reported a text model the request then did not have.
    func localAPIModelLocations() -> LocalAPIModelLocations {
        prepareLegacySettingsForSession(outputMode: .transcription)
        let speech = ASRHintSettingsStore.resolvedSettings(
            for: .dictation, rawValue: UserDefaults.standard.string(forKey: AppPreferenceKey.asrHintSettings))
        return .resolve(transcriptionEngine: transcriptionEngine.rawValue,
                        enhancementMode: enhancementMode.rawValue,
                        appleSpeechOnDevice: speech.prefersOnDeviceRecognition)
    }

    /// `localASR` stands in for the model manager's view of the selected local model.
    func localAPIModelStatus(localASR: LocalASRStartContext? = nil) -> LocalAPIModelStatus {
        prepareLegacySettingsForSession(outputMode: .transcription)
        let transcription: LocalAPIModelReadiness
        switch transcriptionEngine {
        case .dictation:
            transcription = .unsupported
        case .mlxAudio:
            let context = localASR ?? currentLocalASRStartContext()
            if OmniASRBackend.unavailability(for: context.selectedMLXRepo) != nil {
                // A request would be refused before transcribing, downloaded or not.
                transcription = .unsupported
            } else {
                transcription = context.isSelectedMLXModelDownloaded ? .ready : .missing
            }
        case .remote:
            let provider = remoteASRSelectedProvider
            let isConfigured = remoteASRConfigurations[provider.rawValue] != nil
                && RemoteModelConfigurationStore.resolvedASRConfiguration(
                    provider: provider,
                    stored: remoteASRConfigurations
                ).isConfigured
            transcription = isConfigured ? .ready : .missing
        }

        let text: LocalAPIModelReadiness
        switch enhancementMode {
        case .off:
            text = .off
        case .customLLM:
            let isInstalled = resolvedTranscriptionEnhancementLocalRepo().map { repo in
                customLLMManager.canAttemptInference(repo: repo)
            } ?? false
            text = isInstalled ? .ready : .missing
        case .remoteLLM:
            let isConfigured = RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
                provider: resolvedRemoteLLMContext(forTranslation: false).provider,
                stored: remoteLLMConfigurations
            )
            text = isConfigured ? .ready : .missing
        case .appleIntelligence:
            text = .ready
        }
        return LocalAPIModelStatus(transcription: transcription, text: text)
    }
}
