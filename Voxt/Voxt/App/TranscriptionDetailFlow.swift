// TranscriptionDetailFlow.swift
// Provides Transcription Detail Flow for app lifecycle and routing.

import Foundation

extension AppDelegate {
    private enum TranscriptionFollowUpModel {
        case appleIntelligence
        case customLLM(repo: String)
        case remoteLLM(provider: RemoteLLMProvider, configuration: RemoteProviderConfiguration)
    }

    func showTranscriptionDetailWindow(for entryID: UUID) {
        guard let entry = historyStore.entry(id: entryID) else {
            VoxtLog.historyWarning("Transcription detail open skipped: history entry \(entryID) was unavailable.")
            return
        }
        showTranscriptionDetailWindow(for: entry)
    }

    func showTranscriptionDetailWindow(for entry: TranscriptionHistoryEntry) {
        TranscriptionDetailWindowManager.shared.present(
            entry: entry,
            audioURL: historyStore.audioURL(for: entry),
            followUpStatusProvider: { @MainActor entry in
                self.transcriptionFollowUpProviderStatus(for: entry)
            },
            followUpAnswerer: { @MainActor entry, history, question in
                try await self.answerTranscriptionFollowUp(
                    entry: entry,
                    history: history,
                    question: question
                )
            },
            followUpPersistence: { @MainActor entryID, messages in
                self.persistTranscriptionChatMessages(messages, for: entryID)
            }
        )
    }

    func transcriptionFollowUpProviderStatus(for entry: TranscriptionHistoryEntry) -> TranscriptionFollowUpProviderStatus {
        guard let model = resolvedTranscriptionFollowUpModel(for: entry.kind) else {
            return TranscriptionFollowUpProviderStatus(
                isAvailable: false,
                message: unavailableTranscriptionFollowUpProviderMessage(for: entry.kind)
            )
        }

        switch model {
        case .appleIntelligence:
            return TranscriptionFollowUpProviderStatus(
                isAvailable: true,
                message: AppLocalization.localizedString("Using Apple Intelligence for follow-up.")
            )
        case .customLLM(let repo):
            return TranscriptionFollowUpProviderStatus(
                isAvailable: true,
                message: AppLocalization.format(
                    "Using local model: %@",
                    customLLMManager.displayTitle(for: repo)
                )
            )
        case .remoteLLM(let provider, let configuration):
            return TranscriptionFollowUpProviderStatus(
                isAvailable: true,
                message: AppLocalization.format(
                    "Using remote model: %@ · %@",
                    provider.title,
                    configuration.model
                )
            )
        }
    }

    func answerTranscriptionFollowUp(
        entry: TranscriptionHistoryEntry,
        history: [TranscriptSummaryChatMessage],
        question: String
    ) async throws -> String {
        let trimmedQuestion = question.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedText = entry.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedQuestion.isEmpty else {
            throw NSError(
                domain: "Voxt.TranscriptionDetail",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Please enter a follow-up question.")]
            )
        }
        guard !trimmedText.isEmpty else {
            throw NSError(
                domain: "Voxt.TranscriptionDetail",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("No saved transcription result is available yet.")]
            )
        }
        guard let model = resolvedTranscriptionFollowUpModel(for: entry.kind) else {
            throw NSError(
                domain: "Voxt.TranscriptionDetail",
                code: -3,
                userInfo: [NSLocalizedDescriptionKey: unavailableTranscriptionFollowUpProviderMessage(for: entry.kind)]
            )
        }

        let prompt = TranscriptionDetailSupport.followUpPrompt(
            entry: entry,
            history: history,
            question: trimmedQuestion,
            userMainLanguage: userMainLanguagePromptValue
        )
        let modelLabel = transcriptionFollowUpModelLogLabel(model)
        let startedAt = Date()

        VoxtLog.history(
            "Transcription detail follow-up started. model=\(modelLabel), entryID=\(entry.id), kind=\(entry.kind.rawValue), questionChars=\(trimmedQuestion.count), historyCount=\(history.count)"
        )
        VoxtLog.llm(
            """
            Transcription detail follow-up content. model=\(modelLabel)
            [question]
            \(VoxtLog.llmPreview(trimmedQuestion, limit: 2000))
            [saved-result]
            \(VoxtLog.llmPreview(trimmedText, limit: 4000))
            [prompt]
            \(VoxtLog.llmPreview(prompt, limit: 4000))
            """
        )

        do {
            let output = try await runTranscriptionFollowUpPrompt(prompt, model: model)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !output.isEmpty else {
                throw NSError(
                    domain: "Voxt.TranscriptionDetail",
                    code: -4,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Unable to generate answer.")]
                )
            }
            let elapsedMs = Int(Date().timeIntervalSince(startedAt) * 1000)
            VoxtLog.llm(
                """
                Transcription detail follow-up output. model=\(modelLabel)
                [output]
                \(VoxtLog.llmPreview(output, limit: 4000))
                """
            )
            VoxtLog.history(
                "Transcription detail follow-up succeeded. model=\(modelLabel), elapsedMs=\(elapsedMs), outputChars=\(output.count)"
            )
            return output
        } catch {
            VoxtLog.historyWarning(
                "Transcription detail follow-up failed. model=\(modelLabel), error=\(error.localizedDescription)"
            )
            throw error
        }
    }

    @discardableResult
    func persistTranscriptionChatMessages(_ messages: [TranscriptSummaryChatMessage], for entryID: UUID) -> TranscriptionHistoryEntry? {
        historyStore.updateTranscriptionChatMessages(messages, for: entryID)
    }

    private func unavailableTranscriptionFollowUpProviderMessage(for kind: TranscriptionHistoryKind) -> String {
        switch kind {
        case .translation:
            return unavailableTextSelectionMessage(
                resolvedTranscriptionFollowUpTextSelection(for: kind)
            )
        default:
            return unavailableTextSelectionMessage(
                resolvedTranscriptionFollowUpTextSelection(for: kind)
            )
        }
    }

    private func resolvedTranscriptionFollowUpModel(for kind: TranscriptionHistoryKind) -> TranscriptionFollowUpModel? {
        switch resolvedTranscriptionFollowUpTextSelection(for: kind) {
        case .appleIntelligence:
            guard let enhancer else { return nil }
            if #available(macOS 26.0, *) {
                guard TextEnhancer.isAvailable else { return nil }
                _ = enhancer
                return .appleIntelligence
            }
            return nil
        case .localLLM(let repo):
            guard customLLMManager.canAttemptInference(repo: repo) else { return nil }
            return .customLLM(repo: repo)
        case .remoteLLM(let provider):
            guard RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
                provider: provider,
                stored: remoteLLMConfigurations
            ) else {
                return nil
            }
            let configuration = RemoteModelConfigurationStore.resolvedLLMConfiguration(
                provider: provider,
                stored: remoteLLMConfigurations
            )
            return .remoteLLM(provider: provider, configuration: configuration)
        case .none:
            return nil
        }
    }

    private func resolvedTranscriptionFollowUpTextSelection(
        for kind: TranscriptionHistoryKind
    ) -> FeatureModelSelectionID.TextSelection? {
        switch kind {
        case .normal:
            guard transcriptionFeatureSettings.llmEnabled else { return nil }
            return transcriptionFeatureSettings.llmSelectionID.textSelection
        case .translation:
            switch translationFeatureSettings.modelSelectionID.translationSelection {
            case .localLLM(let repo):
                return .localLLM(repo: repo)
            case .localGGUF:
                return nil
            case .remoteLLM(let provider):
                return .remoteLLM(provider: provider)
            case .none:
                return nil
            }
        case .rewrite:
            return rewriteFeatureSettings.llmSelectionID.textSelection
        case .transcript:
            return nil
        }
    }

    private func unavailableTextSelectionMessage(
        _ selection: FeatureModelSelectionID.TextSelection?
    ) -> String {
        switch selection {
        case .appleIntelligence:
            return AppLocalization.localizedString("Apple Intelligence is currently unavailable.")
        case .localLLM:
            return AppLocalization.localizedString("The selected local model is not downloaded yet.")
        case .remoteLLM:
            return AppLocalization.localizedString("The selected remote model is not configured yet.")
        case .none:
            return AppLocalization.localizedString("Enable an enhancement model to ask follow-up questions.")
        }
    }

    private func runTranscriptionFollowUpPrompt(
        _ prompt: String,
        model: TranscriptionFollowUpModel
    ) async throws -> String {
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else { return "" }

        switch model {
        case .appleIntelligence:
            guard let enhancer else {
                throw NSError(
                    domain: "Voxt.TranscriptionDetail",
                    code: -5,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Apple Intelligence is unavailable.")]
                )
            }
            if #available(macOS 26.0, *) {
                return try await enhancer.enhance(userPrompt: trimmedPrompt)
            }
            throw NSError(
                domain: "Voxt.TranscriptionDetail",
                code: -6,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Apple Intelligence requires macOS 26 or later.")]
            )
        case .customLLM(let repo):
            return try await customLLMManager.enhance(userPrompt: trimmedPrompt, repo: repo)
        case .remoteLLM(let provider, let configuration):
            return try await RemoteLLMRuntimeClient().enhance(
                userPrompt: trimmedPrompt,
                provider: provider,
                configuration: configuration
            )
        }
    }

    private func transcriptionFollowUpModelLogLabel(_ model: TranscriptionFollowUpModel) -> String {
        switch model {
        case .appleIntelligence:
            return "apple-intelligence"
        case .customLLM(let repo):
            return "custom-llm:\(repo)"
        case .remoteLLM(let provider, let configuration):
            return "remote-llm:\(provider.rawValue):\(configuration.model)"
        }
    }
}
