// LocalAPIFlow.swift
// Provides the Local API: audio sent by another app or device over loopback is run through the
// user's selected ASR model, enhancement and dictionary. The result is returned to the caller and
// saved to history; nothing is pasted on the Mac.

import Foundation

extension AppDelegate {
    static let localAPISourceName = "Local API"

    /// Starts or stops the server so it matches the stored settings.
    func applyLocalAPIServerLifecycle() {
        let shouldRun = LocalAPISettings.isEnabled()
        let port = LocalAPISettings.port()
        if !shouldRun || localAPIServerPort != port {
            stopLocalAPIServer()
            localAPIServerStatus = shouldRun ? .starting : .off
        }
        guard shouldRun, localAPIServer == nil else { return }
        // A napping menu bar app answers tens of seconds late; stay awake while listening,
        // but let the Mac itself sleep.
        localAPIActivity = ProcessInfo.processInfo.beginActivity(
            options: .userInitiatedAllowingIdleSystemSleep,
            reason: "Voxt Local API is waiting for dictations"
        )

        let pipeline = LocalAPIDictationPipeline(
            temporaryDirectory: FileManager.default.temporaryDirectory,
            busyReason: { [weak self] in self?.localAPIBusyReason() },
            transcribe: { [weak self] url, language in
                guard let self else { throw CancellationError() }
                return try await self.transcribeLocalAPIAudio(at: url, language: language)
            },
            enhance: { [weak self] text in
                guard let self else { return text }
                return try await self.enhanceLocalAPITranscript(text)
            },
            finalize: { [weak self] text in
                self?.finalizeLocalAPIText(text) ?? LocalAPIFinalText(text: text)
            },
            record: { [weak self] outcome in
                self?.recordLocalAPIHistory(outcome)
            }
        )
        pipeline.composeHandler = { [weak self] text, options in
            guard let self else { throw CancellationError() }
            return try await self.composeLocalAPITranscript(text, options: options)
        }
        let router = LocalAPIRouter(token: LocalAPISettings.token(), maxBodyBytes: LocalAPISettings.maxBodyBytes)
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? ""
        let handler = LocalAPIRequestHandler.make(
            router: router,
            version: version,
            modelStatus: { [weak self] in
                self?.localAPIModelStatus() ?? LocalAPIModelStatus(transcription: .missing, text: .missing)
            },
            dictate: { dictation in
                await pipeline.run(dictation)
            },
            compose: { composeRequest in
                await pipeline.runCompose(composeRequest)
            }
        )
        let server = LocalAPIServer(port: UInt16(port), maxBodyBytes: LocalAPISettings.maxBodyBytes, handler: handler)
        localAPIServer = server
        localAPIServerPort = port
        localAPIServerStatus = .starting
        Task { @MainActor [weak self] in
            do {
                let bound = try await server.start()
                VoxtLog.info("Local API server listening on 127.0.0.1:\(bound)")
                guard let self, self.localAPIServer === server else { return }
                self.localAPIServerStatus = .listening(port: Int(bound))
            } catch {
                VoxtLog.warning("Local API server failed to start: \(error.localizedDescription)")
                guard let self, self.localAPIServer === server else { return }
                self.localAPIServer = nil
                self.localAPIServerPort = nil
                self.localAPIServerStatus = .failed(error.localizedDescription)
            }
        }
    }

    func stopLocalAPIServer() {
        localAPIServer?.stop()
        localAPIServer = nil
        localAPIServerPort = nil
        if let localAPIActivity {
            ProcessInfo.processInfo.endActivity(localAPIActivity)
            self.localAPIActivity = nil
        }
    }

    private func localAPIBusyReason() -> String? {
        if isSessionActive {
            return AppLocalization.localizedString("Voxt is recording on the Mac. Try again in a moment.")
        }
        if meetingSessionCoordinator.isActive {
            return AppLocalization.localizedString("Voxt is recording a meeting on the Mac.")
        }
        return nil
    }

    private func enhanceLocalAPITranscript(_ text: String) async throws -> String {
        switch enhancementMode {
        case .off:
            return text
        case .customLLM:
            guard let repo = resolvedTranscriptionEnhancementLocalRepo(),
                  customLLMManager.canAttemptInference(repo: repo) else {
                VoxtLog.asrWarning("Local API: custom LLM selected but not installed. Returning raw transcription.")
                return text
            }
        case .appleIntelligence, .remoteLLM:
            break
        }
        // The Mac's frontmost app says nothing about where the caller will use the text, so App
        // Branch prompts are skipped and the global transcription prompt is used.
        return try await enhanceTextForCurrentMode(text, sessionID: nil, usesAppBranch: false)
    }

    /// The selected text model as the compose prompt sees it: whether it can run at all.
    func localAPITextModel() -> LocalAPIComposePrompt.TextModel {
        switch enhancementMode {
        case .off:
            return .disabled
        case .customLLM:
            return .customLLM(ready: resolvedTranscriptionEnhancementLocalRepo().map { repo in
                customLLMManager.canAttemptInference(repo: repo)
            } ?? false)
        case .remoteLLM:
            let context = resolvedRemoteLLMContext(forTranslation: false)
            return .remoteLLM(configured: RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
                provider: context.provider,
                stored: remoteLLMConfigurations
            ))
        case .appleIntelligence:
            return .appleIntelligence
        }
    }

    /// Translate, edit, write and dictate with options reuse the selected text model with a
    /// mode-specific prompt for this one request; the user's settings are read, never changed.
    /// Unlike a plain dictation this path fails closed: without a working model it throws
    /// instead of returning raw speech as if the operation had succeeded.
    private func composeLocalAPITranscript(_ text: String, options: LocalAPIComposeOptions) async throws -> String {
        if let reason = LocalAPIComposePrompt.unavailabilityReason(for: localAPITextModel()) {
            throw LocalAPIError(message: AppLocalization.localizedString(reason))
        }
        return try await LocalAPIComposePrompt.compose(rawTranscription: text, options: options) { instruction in
            try await self.composeTextStrictly(text, instruction: instruction)
        }
    }

    private func finalizeLocalAPIText(_ text: String) -> LocalAPIFinalText {
        let context = Self.preparedDeliveryContext(
            originalText: text,
            llmDurationSeconds: nil,
            sessionOutputMode: .transcription,
            userMainLanguage: userMainLanguage,
            matcher: dictionaryStore.makeMatcherIfEnabled(for: text, activeGroupID: activeDictionaryGroupID()),
            usesConservativeEvidence: false,
            automaticReplacementEnabled: UserDefaults.standard.bool(
                forKey: AppPreferenceKey.dictionaryHighConfidenceCorrectionEnabled
            )
        )
        dictionaryStore.recordMatches(context.dictionaryMatches)
        return LocalAPIFinalText(
            text: context.outputText,
            dictionaryHitTerms: Self.orderedUniqueDictionaryTerms(from: context.dictionaryMatches.map(\.term)),
            dictionaryCorrectedTerms: Self.orderedUniqueDictionaryTerms(from: context.dictionaryCorrectedTerms)
        )
    }

    private func recordLocalAPIHistory(_ outcome: LocalAPIDictationOutcome) {
        guard historyEnabled else { return }
        let textModel = outcome.usedEnhancement
            ? resolvedHistoryTextModelMetadata(for: .normal)
            : HistoryTextModelMetadata(
                modeTitle: EnhancementMode.off.title,
                modelTitle: "None",
                remoteProviderTitle: nil,
                remoteModelTitle: nil,
                remoteEndpoint: nil
            )
        let transcriptionModel = localAPITranscriptionModelMetadata()

        _ = historyStore.append(
            entryID: outcome.id,
            text: outcome.text,
            transcriptionEngine: transcriptionEngine.title,
            transcriptionModel: transcriptionModel.title,
            enhancementMode: textModel.modeTitle,
            enhancementModel: textModel.modelTitle,
            kind: .normal,
            isTranslation: false,
            audioDurationSeconds: nil,
            transcriptionProcessingDurationSeconds: outcome.transcriptionSeconds,
            llmDurationSeconds: outcome.enhancementSeconds,
            focusedAppName: Self.localAPISourceName,
            focusedAppBundleID: nil,
            matchedGroupID: nil,
            matchedGroupName: nil,
            matchedAppGroupName: nil,
            matchedURLGroupName: nil,
            remoteASRProvider: transcriptionModel.remoteProvider,
            remoteASRModel: transcriptionModel.remoteModel,
            remoteASREndpoint: nil,
            remoteLLMProvider: textModel.remoteProviderTitle,
            remoteLLMModel: textModel.remoteModelTitle,
            remoteLLMEndpoint: textModel.remoteEndpoint,
            whisperWordTimings: nil,
            dictionaryHitTerms: outcome.dictionaryHitTerms,
            dictionaryCorrectedTerms: outcome.dictionaryCorrectedTerms,
            dictionarySuggestedTerms: []
        )
    }
}

enum LocalAPIServerStatus: Equatable {
    case off
    case starting
    case listening(port: Int)
    case failed(String)
}
