// MeetingSummaryFlow.swift
// Provides Meeting Summary Flow for app lifecycle and routing.

import Foundation

extension AppDelegate {
    private enum MeetingSummaryModelID {
        static let appleIntelligence = "apple-intelligence"

        static func customLLM(_ repo: String) -> String {
            "custom-llm:\(repo)"
        }

        static func remoteLLM(_ provider: RemoteLLMProvider) -> String {
            "remote-llm:\(provider.rawValue)"
        }
    }

    private enum MeetingSummaryModel {
        case appleIntelligence
        case customLLM(repo: String)
        case remoteLLM(provider: RemoteLLMProvider, configuration: RemoteProviderConfiguration)
    }

    func translateMeetingRealtimeText(
        _ text: String,
        targetLanguage: TranslationTargetLanguage
    ) async throws -> String {
        try await translateText(text, targetLanguage: targetLanguage)
    }

    func makeMeetingTranslationOperation(
        _ text: String,
        targetLanguage: TranslationTargetLanguage
    ) -> MeetingTranslationOperation {
        let resolution = resolvedTranslationProviderResolution(
            targetLanguage: targetLanguage,
            isSelectedTextTranslation: false
        )
        let executionScope: MeetingTranslationExecutionScope = resolution.provider == .remoteLLM
            ? .externalRequest
            : .localInference
        return MeetingTranslationOperation(executionScope: executionScope) { [weak self] in
            guard let self else { throw CancellationError() }
            return try await self.translateText(
                text,
                targetLanguage: targetLanguage,
                providerResolution: resolution
            )
        }
    }

    var meetingSummaryAutoGenerateEnabled: Bool {
        meetingFeatureSettings.summaryAutoGenerate
    }

    var meetingSummaryPromptTemplatePreference: String {
        AppPromptDefaults.resolvedStoredText(
            meetingFeatureSettings.summaryPrompt,
            kind: .transcriptSummary
        )
    }

    var meetingSummaryModelSelectionPreference: String? {
        switch meetingFeatureSettings.summaryModelSelectionID.textSelection {
        case .appleIntelligence:
            return MeetingSummaryModelID.appleIntelligence
        case .localLLM(let repo):
            return MeetingSummaryModelID.customLLM(repo)
        case .remoteLLM(let provider):
            return MeetingSummaryModelID.remoteLLM(provider)
        case .none:
            return nil
        }
    }

    func meetingSummaryModelOptions() -> [MeetingSummaryModelOption] {
        var options: [MeetingSummaryModelOption] = []

        if #available(macOS 26.0, *), enhancer != nil, TextEnhancer.isAvailable {
            options.append(
                MeetingSummaryModelOption(
                    id: MeetingSummaryModelID.appleIntelligence,
                    title: AppLocalization.localizedString("Apple Intelligence"),
                    subtitle: AppLocalization.localizedString("Built-in on-device summary generation")
                )
            )
        }

        let selectedLocalRepo: String?
        if case .localLLM(let repo) = meetingFeatureSettings.summaryModelSelectionID.textSelection {
            selectedLocalRepo = CustomLLMModelManager.canonicalModelRepo(repo)
        } else {
            selectedLocalRepo = nil
        }
        let downloadedCustomOptions: [MeetingSummaryModelOption] = CustomLLMModelManager.displayModels(including: customLLMManager.currentModelRepo).compactMap { model -> MeetingSummaryModelOption? in
            // Keep the user's configured local model during its initial scan;
            // do not silently switch to another provider while status is unknown.
            guard customLLMManager.isModelDownloaded(repo: model.id)
                || (model.id == selectedLocalRepo && customLLMManager.isCheckingInstallation(repo: model.id)) else {
                return nil
            }
            return MeetingSummaryModelOption(
                id: MeetingSummaryModelID.customLLM(model.id),
                title: customLLMManager.displayTitle(for: model.id),
                subtitle: AppLocalization.localizedString("Local Custom LLM")
            )
        }
        options.append(contentsOf: downloadedCustomOptions)

        let currentCustomRepo = customLLMManager.currentModelRepo
        if customLLMManager.isModelDownloaded(repo: currentCustomRepo) {
            let currentCustomOption = MeetingSummaryModelOption(
                id: MeetingSummaryModelID.customLLM(currentCustomRepo),
                title: customLLMManager.displayTitle(for: currentCustomRepo),
                subtitle: AppLocalization.localizedString("Local Custom LLM")
            )
            if !options.contains(where: { $0.id == currentCustomOption.id }) {
                options.append(currentCustomOption)
            }
        }

        let configuredRemoteOptions: [MeetingSummaryModelOption] = RemoteLLMProvider.allCases.compactMap { provider -> MeetingSummaryModelOption? in
            let configuration = RemoteModelConfigurationStore.resolvedLLMConfiguration(
                provider: provider,
                stored: remoteLLMConfigurations
            )
            guard configuration.isConfigured, configuration.hasUsableModel else {
                return nil
            }
            return MeetingSummaryModelOption(
                id: MeetingSummaryModelID.remoteLLM(provider),
                title: "\(provider.title) · \(configuration.model)",
                subtitle: AppLocalization.localizedString("Configured Remote LLM")
            )
        }
        options.append(contentsOf: configuredRemoteOptions)

        return options
    }

    func currentMeetingSummarySettingsSnapshot() -> MeetingSummarySettingsSnapshot {
        MeetingSummarySettingsSnapshot(
            autoGenerate: meetingSummaryAutoGenerateEnabled,
            promptTemplate: meetingSummaryPromptTemplatePreference,
            modelSelectionID: resolvedMeetingSummaryModelSelectionID(
                preferredID: meetingSummaryModelSelectionPreference,
                availableOptions: meetingSummaryModelOptions()
            )
        )
    }

    func updateMeetingSummaryPreference(
        autoGenerate: Bool? = nil,
        promptTemplate: String? = nil,
        modelSelectionID: String? = nil
    ) {
        FeatureSettingsStore.update(defaults: .standard) { settings in
            if let autoGenerate {
                settings.meeting.summaryAutoGenerate = autoGenerate
            }
            if let promptTemplate {
                settings.meeting.summaryPrompt = AppPromptDefaults.canonicalStoredText(promptTemplate, kind: .transcriptSummary)
            }
            if let modelSelectionID {
                settings.meeting.summaryModelSelectionID = FeatureModelSelectionID.fromTranscriptSummaryModelSelection(modelSelectionID)
                    ?? FeatureModelSelectionID(rawValue: modelSelectionID)
            }
        }
    }

    func meetingSummaryProviderStatus(settings: MeetingSummarySettingsSnapshot? = nil) -> MeetingSummaryProviderStatus {
        let options = meetingSummaryModelOptions()
        guard !options.isEmpty else {
            return MeetingSummaryProviderStatus(
                isAvailable: false,
                message: AppLocalization.localizedString("No downloaded or configured summary model is currently available.")
            )
        }

        let preferredID = settings?.modelSelectionID ?? meetingSummaryModelSelectionPreference
        guard let model = resolvedMeetingSummaryModel(
            preferredID: preferredID,
            availableOptions: options
        ) else {
            return MeetingSummaryProviderStatus(
                isAvailable: false,
                message: AppLocalization.localizedString("The selected summary model is currently unavailable.")
            )
        }

        switch model {
        case .appleIntelligence:
            return MeetingSummaryProviderStatus(
                isAvailable: true,
                message: AppLocalization.localizedString("Using Apple Intelligence for meeting summaries.")
            )
        case .customLLM(let repo):
            return MeetingSummaryProviderStatus(
                isAvailable: true,
                message: AppLocalization.format("Using local model: %@", customLLMManager.displayTitle(for: repo))
            )
        case .remoteLLM(let provider, let configuration):
            return MeetingSummaryProviderStatus(
                isAvailable: true,
                message: AppLocalization.format("Using remote model: %@ · %@", provider.title, configuration.model)
            )
        }
    }

    func generateMeetingSummary(
        transcript: String,
        settings: MeetingSummarySettingsSnapshot
    ) async throws -> MeetingSummarySnapshot {
        let trimmedTranscript = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedTranscript.isEmpty else {
            throw NSError(
                domain: "Voxt.MeetingSummary",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("No meeting transcript is available yet.")]
            )
        }
        let options = meetingSummaryModelOptions()
        guard let model = resolvedMeetingSummaryModel(
            preferredID: settings.modelSelectionID,
            availableOptions: options
        ) else {
            let status = meetingSummaryProviderStatus(settings: settings)
            throw NSError(
                domain: "Voxt.MeetingSummary",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: status.message]
            )
        }

        let modelLabel = meetingSummaryModelLogLabel(model)
        let startedAt = Date()
        VoxtLog.meeting(
            "Meeting summary generation started. model=\(modelLabel), transcriptChars=\(trimmedTranscript.count), templateChars=\(settings.promptTemplate?.count ?? 0)"
        )
        do {
            let output = try await runHierarchicalMeetingSummary(
                transcript: trimmedTranscript,
                settings: settings,
                resolvedModel: model
            )
            let elapsedMs = Int(Date().timeIntervalSince(startedAt) * 1000)

            guard let summary = MeetingSummarySupport.decodeSummary(from: output, settings: settings) else {
                VoxtLog.meetingWarning(
                    "Meeting summary generation parse failed. model=\(modelLabel), outputChars=\(output.count), elapsedMs=\(elapsedMs)"
                )
                throw NSError(
                    domain: "Voxt.MeetingSummary",
                    code: -5,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Meeting summary output could not be parsed.")]
                )
            }
            VoxtLog.meeting(
                "Meeting summary generation succeeded. model=\(modelLabel), elapsedMs=\(elapsedMs), titleChars=\(summary.title.count), todoCount=\(summary.todoItems.count), bodyChars=\(summary.body.count)"
            )
            return summary
        } catch {
            VoxtLog.meetingWarning(
                "Meeting summary generation failed. model=\(modelLabel), error=\(error.localizedDescription)"
            )
            throw error
        }
    }

    func answerMeetingSummaryFollowUp(
        transcript: String,
        summary: MeetingSummarySnapshot?,
        history: [MeetingSummaryChatMessage],
        question: String,
        settings: MeetingSummarySettingsSnapshot
    ) async throws -> String {
        let trimmedTranscript = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedQuestion = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedTranscript.isEmpty else {
            throw NSError(
                domain: "Voxt.MeetingSummaryChat",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("No meeting transcript is available yet.")]
            )
        }
        guard !trimmedQuestion.isEmpty else {
            throw NSError(
                domain: "Voxt.MeetingSummaryChat",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Please enter a follow-up question.")]
            )
        }

        let options = meetingSummaryModelOptions()
        guard let model = resolvedMeetingSummaryModel(
            preferredID: settings.modelSelectionID,
            availableOptions: options
        ) else {
            let status = meetingSummaryProviderStatus(settings: settings)
            throw NSError(
                domain: "Voxt.MeetingSummaryChat",
                code: -3,
                userInfo: [NSLocalizedDescriptionKey: status.message]
            )
        }

        let prompt = MeetingSummarySupport.followUpPrompt(
            transcript: trimmedTranscript,
            summary: summary,
            history: history,
            question: trimmedQuestion,
            userMainLanguage: userMainLanguagePromptValue
        )
        let modelLabel = meetingSummaryModelLogLabel(model)
        let startedAt = Date()
        VoxtLog.meeting(
            "Meeting summary follow-up started. model=\(modelLabel), transcriptChars=\(trimmedTranscript.count), questionChars=\(trimmedQuestion.count), historyCount=\(history.count), hasSummary=\(summary != nil)"
        )
        do {
            let output = try await runMeetingSummaryPromptCoordinated(
                prompt,
                transcript: trimmedTranscript,
                resolvedModel: model
            ).trimmingCharacters(in: .whitespacesAndNewlines)
            let elapsedMs = Int(Date().timeIntervalSince(startedAt) * 1000)
            VoxtLog.meeting(
                "Meeting summary follow-up succeeded. model=\(modelLabel), elapsedMs=\(elapsedMs), outputChars=\(output.count)"
            )
            return output
        } catch {
            VoxtLog.meetingWarning(
                "Meeting summary follow-up failed. model=\(modelLabel), error=\(error.localizedDescription)"
            )
            throw error
        }
    }

    @discardableResult
    func persistMeetingSummary(_ summary: MeetingSummarySnapshot?, for entryID: UUID) -> TranscriptionHistoryEntry? {
        historyStore.updateTranscriptSummary(summary, for: entryID)
    }

    @discardableResult
    func persistMeetingSummaryChatMessages(_ messages: [MeetingSummaryChatMessage], for entryID: UUID) -> TranscriptionHistoryEntry? {
        historyStore.updateSummaryChatMessages(messages, for: entryID)
    }

    private func runMeetingSummaryPrompt(
        _ prompt: String,
        transcript: String,
        resolvedModel: MeetingSummaryModel
    ) async throws -> String {
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPrompt.isEmpty else { return "" }
        let trimmedTranscript = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        let modelLabel = meetingSummaryModelLogLabel(resolvedModel)
        let provider = try meetingSummaryExecutionProvider(for: resolvedModel)
        let strategy = meetingSummaryExecutionStrategy(
            transcript: trimmedTranscript,
            prompt: trimmedPrompt
        )
        let plan = LLMExecutionPlan(
            task: .transcriptSummary(
                transcript: trimmedTranscript,
                request: trimmedPrompt
            ),
            provider: provider,
            delivery: .userMessage,
            promptContent: trimmedPrompt,
            fallbackText: "",
            executionStrategy: strategy,
            outputTokenBudgetHint: strategy.outputTokenBudgetHint,
            contextBlocks: [
                LLMContextBlock(
                    kind: .input,
                    title: "Transcript",
                    content: trimmedTranscript,
                    isStablePrefixCandidate: false
                )
            ],
            conversationHistory: [],
            previousResponseID: nil,
            responseFormat: nil
        )
        VoxtLog.llmDebug("Meeting summary runtime dispatch. model=\(modelLabel), runtime=llm-execution-plan")
        return try await executeLLMExecutionPlan(plan)
    }

    private func runHierarchicalMeetingSummary(
        transcript: String,
        settings: MeetingSummarySettingsSnapshot,
        resolvedModel: MeetingSummaryModel
    ) async throws -> String {
        var units = MeetingSummaryContextPlanning.windows(for: transcript)
        guard !units.isEmpty else { return "" }

        while true {
            let groups = groupedSummaryUnits(units)
            var outputs: [String] = []
            outputs.reserveCapacity(groups.count)
            for group in groups {
                try Task.checkCancellation()
                let boundedInput = group.joined(separator: "\n\n")
                let prompt = MeetingSummarySupport.summaryPrompt(
                    transcript: boundedInput,
                    settings: settings,
                    userMainLanguage: userMainLanguagePromptValue
                )
                let output = try await runMeetingSummaryPromptCoordinated(
                    prompt,
                    transcript: boundedInput,
                    resolvedModel: resolvedModel
                )
                outputs.append(output)
            }
            if outputs.count == 1 {
                return outputs[0]
            }
            units = outputs.map { output in
                let trimmed = output.trimmingCharacters(in: .whitespacesAndNewlines)
                return String(trimmed.prefix(4_000))
            }
        }
    }

    private func groupedSummaryUnits(_ units: [String]) -> [[String]] {
        let limit = MeetingSummaryContextPlanning.summaryWindowCharacterLimit
        var groups: [[String]] = []
        var current: [String] = []
        var currentCount = 0
        for unit in units {
            let bounded = String(unit.prefix(limit))
            let addedCount = bounded.count + (current.isEmpty ? 0 : 2)
            if !current.isEmpty, currentCount + addedCount > limit {
                groups.append(current)
                current = []
                currentCount = 0
            }
            current.append(bounded)
            currentCount += addedCount
        }
        if !current.isEmpty {
            groups.append(current)
        }
        return groups
    }

    private func runMeetingSummaryPromptCoordinated(
        _ prompt: String,
        transcript: String,
        resolvedModel: MeetingSummaryModel
    ) async throws -> String {
        switch resolvedModel {
        case .remoteLLM:
            return try await runMeetingSummaryPrompt(
                prompt,
                transcript: transcript,
                resolvedModel: resolvedModel
            )
        case .appleIntelligence, .customLLM:
            return try await MeetingLocalInferenceCoordinator.shared.withPermit(.summary) {
                try await self.runMeetingSummaryPrompt(
                    prompt,
                    transcript: transcript,
                    resolvedModel: resolvedModel
                )
            }
        }
    }

    private func meetingSummaryExecutionProvider(
        for resolvedModel: MeetingSummaryModel
    ) throws -> LLMExecutionProvider {
        switch resolvedModel {
        case .appleIntelligence:
            guard let enhancer else {
                throw NSError(
                    domain: "Voxt.MeetingSummary",
                    code: -3,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Apple Intelligence is unavailable.")]
                )
            }
            if #available(macOS 26.0, *) {
                _ = enhancer
                return .appleIntelligence
            }
            throw NSError(
                domain: "Voxt.MeetingSummary",
                code: -4,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Apple Intelligence requires macOS 26 or later.")]
            )
        case .customLLM(let repo):
            return .customLLM(repo: repo)
        case .remoteLLM(let provider, let configuration):
            return .remote(provider: provider, configuration: configuration)
        }
    }

    private func meetingSummaryExecutionStrategy(
        transcript: String,
        prompt: String
    ) -> TaskLLMExecutionStrategy {
        let transcriptCharacterCount = transcript.trimmingCharacters(in: .whitespacesAndNewlines).count
        let isLongTranscript = transcriptCharacterCount > TaskLLMStrategyResolver.longTextThreshold
        return TaskLLMExecutionStrategy(
            taskKind: .transcriptSummary,
            rawTextCharacterCount: transcriptCharacterCount,
            promptCharacterCount: prompt.count,
            mode: .singlePass,
            contextBudgetPolicy: isLongTranscript ? .reducedForLongInput : .standard,
            glossarySelectionPolicy: DictionaryGlossarySelectionPolicy(maxTerms: 0, maxCharacters: 0),
            outputTokenBudgetHint: nil,
            segmentationCharacterLimit: nil,
            truncationGuard: .disabled
        )
    }

    private func resolvedMeetingSummaryModelSelectionID(
        preferredID: String?,
        availableOptions: [MeetingSummaryModelOption]
    ) -> String? {
        let trimmedPreferredID = preferredID?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let trimmedPreferredID,
           availableOptions.contains(where: { $0.id == trimmedPreferredID }) {
            return trimmedPreferredID
        }

        return availableOptions.first?.id
    }

    private func resolvedMeetingSummaryModel(
        preferredID: String?,
        availableOptions: [MeetingSummaryModelOption]
    ) -> MeetingSummaryModel? {
        guard let selectionID = resolvedMeetingSummaryModelSelectionID(
            preferredID: preferredID,
            availableOptions: availableOptions
        ) else {
            return nil
        }

        if selectionID == MeetingSummaryModelID.appleIntelligence {
            guard #available(macOS 26.0, *), enhancer != nil, TextEnhancer.isAvailable else {
                return nil
            }
            return .appleIntelligence
        }

        if let repo = value(in: selectionID, forPrefix: "custom-llm:") {
            guard customLLMManager.canAttemptInference(repo: repo) else { return nil }
            return .customLLM(repo: repo)
        }

        if let providerRawValue = value(in: selectionID, forPrefix: "remote-llm:"),
           let provider = RemoteLLMProvider(rawValue: providerRawValue) {
            let configuration = RemoteModelConfigurationStore.resolvedLLMConfiguration(
                provider: provider,
                stored: remoteLLMConfigurations
            )
            guard configuration.isConfigured, configuration.hasUsableModel else { return nil }
            return .remoteLLM(provider: provider, configuration: configuration)
        }

        return nil
    }

    private func value(in selectionID: String, forPrefix prefix: String) -> String? {
        guard selectionID.hasPrefix(prefix) else { return nil }
        let value = String(selectionID.dropFirst(prefix.count))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return value.isEmpty ? nil : value
    }

    private func meetingSummaryModelLogLabel(_ model: MeetingSummaryModel) -> String {
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
