// DictionaryFlow.swift
// Provides Dictionary Flow for app lifecycle and routing.

import Foundation
import AppKit

extension AppDelegate {
    @discardableResult
    func addSelectedShortTextToDictionaryIfPossible() -> Bool {
        guard !isSessionActive, pendingTranscriptionStartTask == nil else { return false }
        let frontmostBundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier
        guard !SelectedTextSystemSelectionSupport.ignoresSelectedTextFlow(bundleID: frontmostBundleID) else {
            VoxtLog.dictionary("Selected text dictionary flow skipped for Obsidian.")
            return false
        }
        guard let term = selectedDictionaryTermFromSystemSelection() else {
            return false
        }

        let scope = currentDictionaryScope()
        let category = dictionaryStore.ensureCategory(id: scope.groupID, name: scope.groupName)

        do {
            let result = try dictionaryStore.createOrReinforceAutoEntry(
                term: term,
                categoryID: category.id,
                categoryNameSnapshot: category.name,
                groupID: scope.groupID,
                groupNameSnapshot: scope.groupName
            )
            let message = result.added
                ? AppLocalization.format("Added to Dictionary: %@", result.term)
                : AppLocalization.format("Already in Dictionary: %@", result.term)
            showFloatingToast(message)
            VoxtLog.dictionary(
                "Selected text dictionary hotkey handled. added=\(result.added), termChars=\(term.count), groupID=\(scope.groupID?.uuidString ?? "nil")"
            )
            return true
        } catch DictionaryStoreError.emptyTerm {
            // Defensive: invalid selection must fall through to transcription wake.
            VoxtLog.dictionaryWarning("Selected text dictionary hotkey skipped: emptyTerm")
            return false
        } catch {
            showFloatingToast(
                AppLocalization.localizedString("Failed to add to Dictionary."),
                kind: .warning
            )
            VoxtLog.dictionaryWarning("Selected text dictionary hotkey failed: \(error)")
            return true
        }
    }

    func dictionaryGlossaryText(
        for sourceText: String,
        purpose: DictionaryGlossaryPurpose,
        selectionPolicy: DictionaryGlossarySelectionPolicy? = nil
    ) -> String? {
        let effectivePolicy = selectionPolicy ?? purpose.selectionPolicy
        let glossary = dictionaryStore.glossaryContext(
            for: sourceText,
            activeGroupID: activeDictionaryGroupID()
        )?.glossaryText(policy: effectivePolicy)
        let trimmed = glossary?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !trimmed.isEmpty else { return nil }

        let selectedTermCount = trimmed.split(separator: "\n").count
        VoxtLog.dictionary(
            "Dictionary glossary selected. purpose=\(purpose), selectedTerms=\(selectedTermCount), glossaryChars=\(trimmed.count), maxTerms=\(effectivePolicy.maxTerms), maxChars=\(effectivePolicy.maxCharacters)"
        )
        return trimmed
    }

    private enum DictionaryHistoryScanModel {
        case appleIntelligence
        case customLLM(repo: String)
        case remoteLLM(provider: RemoteLLMProvider, configuration: RemoteProviderConfiguration)
    }

    func activeDictionaryGroupID() -> UUID? {
        if let matchedGroupID = lastEnhancementPromptContext?.matchedGroupID {
            return matchedGroupID
        }
        return currentDictionaryScope().groupID
    }

    func startDictionaryHistorySuggestionScan() {
        startDictionaryHistorySuggestionScan(request: nil, persistSettings: false)
    }

    func availableDictionaryHistoryScanModelOptions() -> [DictionaryHistoryScanModelOption] {
        var options: [DictionaryHistoryScanModelOption] = []

        let localRepos = [customLLMManager.currentModelRepo, translationCustomLLMRepo, rewriteCustomLLMRepo]
        let uniqueLocalRepos = Array(Set(localRepos)).sorted {
            customLLMManager.displayTitle(for: $0).localizedCaseInsensitiveCompare(customLLMManager.displayTitle(for: $1)) == .orderedAscending
        }

        for repo in uniqueLocalRepos where customLLMManager.canAttemptInference(repo: repo) {
            options.append(
                DictionaryHistoryScanModelOption(
                    id: "local:\(repo)",
                    source: .local,
                    title: AppLocalization.format(
                        "Local · %@",
                        customLLMManager.displayTitle(for: repo)
                    ),
                    detail: repo
                )
            )
        }

        for provider in RemoteLLMProvider.allCases {
            guard RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
                provider: provider,
                stored: remoteLLMConfigurations
            ) else {
                continue
            }
            let configuration = RemoteModelConfigurationStore.resolvedLLMConfiguration(
                provider: provider,
                stored: remoteLLMConfigurations
            )
            options.append(
                DictionaryHistoryScanModelOption(
                    id: "remote:\(provider.rawValue)",
                    source: .remote,
                    title: AppLocalization.format("Remote · %@", provider.title),
                    detail: configuration.model
                )
            )
        }

        return options
    }

    func startDictionaryHistorySuggestionScan(
        request: DictionaryHistoryScanRequest?,
        persistSettings: Bool
    ) {
        guard !dictionarySuggestionStore.historyScanProgress.isRunning else { return }

        let pendingEntries = dictionarySuggestionStore.pendingHistoryEntries(in: historyStore)
        guard !pendingEntries.isEmpty else {
            dictionarySuggestionStore.finishHistoryScan(
                processedCount: 0,
                newSuggestionCount: 0,
                duplicateCount: 0,
                checkpointEntry: nil
            )
            return
        }

        if persistSettings, let request {
            dictionarySuggestionStore.saveFilterSettings(request.filterSettings)
        }

        dictionarySuggestionStore.beginHistoryScan(totalCount: pendingEntries.count)
        pendingDictionaryHistoryScanTask?.cancel()
        pendingDictionaryHistoryScanTask = Task { @MainActor [weak self] in
            guard let self else { return }
            defer { self.pendingDictionaryHistoryScanTask = nil }
            await self.runDictionaryHistorySuggestionScan(entries: pendingEntries, request: request)
        }
    }

    func cancelDictionaryHistorySuggestionScan() {
        guard dictionarySuggestionStore.historyScanProgress.isRunning else { return }
        dictionarySuggestionStore.requestHistoryScanCancellation()
        pendingDictionaryHistoryScanTask?.cancel()
    }

    private func runDictionaryHistorySuggestionScan(
        entries: [TranscriptionHistoryEntry],
        request: DictionaryHistoryScanRequest?
    ) async {
        let groups = DictionaryHistoryScanSupport.loadGroups()
        let groupsByID = Dictionary(uniqueKeysWithValues: groups.map { ($0.id, $0) })
        let groupsByLowercasedName = groups.reduce(into: [String: AppBranchGroup]()) { partialResult, group in
            partialResult[group.name.lowercased()] = group
        }
        let filterSettings = request?.filterSettings.sanitized() ?? dictionarySuggestionStore.filterSettings
        let batchSize = filterSettings.batchSize

        var processedCount = 0
        var newSuggestionCount = 0
        var duplicateCount = 0
        var lastProcessedEntry: TranscriptionHistoryEntry?

        do {
            let model = try resolvedDictionaryHistoryScanModel(for: request)
            for start in stride(from: 0, to: entries.count, by: batchSize) {
                try Task.checkCancellation()
                let batch = Array(entries[start..<min(start + batchSize, entries.count)])
                let prompt = try DictionaryHistoryScanSupport.buildPrompt(
                    for: batch,
                    filterSettings: filterSettings,
                    groupsByID: groupsByID,
                    groupsByLowercasedName: groupsByLowercasedName,
                    userMainLanguage: userMainLanguagePromptValue,
                    userOtherLanguages: userOtherMainLanguagesPromptValue
                )
                let scannedTerms = try await runDictionaryHistoryScanPrompt(prompt, model: model)
                let boundedTerms = Array(scannedTerms.prefix(filterSettings.maxCandidatesPerBatch))
                try Task.checkCancellation()
                let parsedCandidates = try DictionaryHistoryScanSupport.parseCandidates(
                    terms: boundedTerms,
                    batch: batch,
                    groupsByID: groupsByID,
                    groupsByLowercasedName: groupsByLowercasedName
                )
                try Task.checkCancellation()

                let applyResult = dictionarySuggestionStore.applyHistoryScanCandidates(
                    parsedCandidates,
                    dictionaryStore: dictionaryStore
                )
                try Task.checkCancellation()

                processedCount += batch.count
                newSuggestionCount += applyResult.newSuggestionCount
                duplicateCount += applyResult.duplicateCount
                lastProcessedEntry = batch.last

                if let lastProcessedEntry {
                    dictionarySuggestionStore.advanceHistoryScanCheckpoint(to: lastProcessedEntry)
                }
                dictionarySuggestionStore.updateHistoryScan(
                    processedCount: processedCount,
                    newSuggestionCount: newSuggestionCount,
                    duplicateCount: duplicateCount
                )
            }

            try Task.checkCancellation()
            dictionarySuggestionStore.finishHistoryScan(
                processedCount: processedCount,
                newSuggestionCount: newSuggestionCount,
                duplicateCount: duplicateCount,
                checkpointEntry: lastProcessedEntry
            )
        } catch is CancellationError {
            VoxtLog.dictionary("Dictionary history scan cancelled.")
            dictionarySuggestionStore.cancelHistoryScan(
                processedCount: processedCount,
                totalCount: entries.count,
                newSuggestionCount: newSuggestionCount,
                duplicateCount: duplicateCount,
                message: AppLocalization.localizedString("Dictionary ingestion canceled.")
            )
        } catch {
            VoxtLog.dictionaryWarning("Dictionary history scan failed: \(error)")
            dictionarySuggestionStore.failHistoryScan(
                processedCount: processedCount,
                totalCount: entries.count,
                newSuggestionCount: newSuggestionCount,
                duplicateCount: duplicateCount,
                errorMessage: error.localizedDescription
            )
        }
    }

    private func runDictionaryHistoryScanPrompt(
        _ prompt: String,
        model: DictionaryHistoryScanModel
    ) async throws -> [String] {
        switch model {
        case .appleIntelligence:
            guard let enhancer else {
                throw NSError(
                    domain: "Voxt.DictionaryHistoryScan",
                    code: -1,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Apple Intelligence is unavailable.")]
                )
            }
            if #available(macOS 26.0, *) {
                return try await enhancer.dictionaryHistoryScanTerms(userPrompt: prompt)
            }
            throw NSError(
                domain: "Voxt.DictionaryHistoryScan",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Apple Intelligence requires macOS 26 or later.")]
            )
        case .customLLM(let repo):
            return try await customLLMManager.dictionaryHistoryScanTerms(userPrompt: prompt, repo: repo)
        case .remoteLLM(let provider, let configuration):
            return try await RemoteLLMRuntimeClient().dictionaryHistoryScanTerms(
                userPrompt: prompt,
                provider: provider,
                configuration: configuration
            )
        }
    }

    private func resolvedDictionaryHistoryScanModel(for request: DictionaryHistoryScanRequest?) throws -> DictionaryHistoryScanModel {
        if let request {
            return try dictionaryHistoryScanModel(for: request.modelOptionID)
        }
        return try resolvedDictionaryHistoryScanModel()
    }

    private func resolvedDictionaryHistoryScanModel() throws -> DictionaryHistoryScanModel {
        if let saved = savedDictionaryHistoryScanModel() {
            return saved
        }
        if let preferred = preferredDictionaryHistoryScanModel() {
            return preferred
        }
        if let fallback = fallbackDictionaryHistoryScanModel() {
            return fallback
        }
        throw NSError(
            domain: "Voxt.DictionaryHistoryScan",
            code: -3,
            userInfo: [
                NSLocalizedDescriptionKey: AppLocalization.localizedString(
                    "No usable LLM is available for dictionary ingestion. Configure Apple Intelligence, a local custom LLM, or a remote LLM first."
                )
            ]
        )
    }

    private func savedDictionaryHistoryScanModel() -> DictionaryHistoryScanModel? {
        let optionID = UserDefaults.standard.string(
            forKey: AppPreferenceKey.dictionarySuggestionIngestModelOptionID
        ) ?? ""
        guard !optionID.isEmpty else { return nil }
        return try? dictionaryHistoryScanModel(for: optionID)
    }

    private func preferredDictionaryHistoryScanModel() -> DictionaryHistoryScanModel? {
        switch enhancementMode {
        case .appleIntelligence:
            return appleDictionaryHistoryScanModel()
        case .customLLM:
            return customLLMDictionaryHistoryScanModel()
        case .remoteLLM:
            return remoteDictionaryHistoryScanModel()
        case .off:
            return nil
        }
    }

    private func fallbackDictionaryHistoryScanModel() -> DictionaryHistoryScanModel? {
        appleDictionaryHistoryScanModel()
            ?? customLLMDictionaryHistoryScanModel()
            ?? remoteDictionaryHistoryScanModel()
    }

    private func appleDictionaryHistoryScanModel() -> DictionaryHistoryScanModel? {
        guard #available(macOS 26.0, *), enhancer != nil, TextEnhancer.isAvailable else {
            return nil
        }
        return .appleIntelligence
    }

    private func customLLMDictionaryHistoryScanModel() -> DictionaryHistoryScanModel? {
        guard customLLMManager.canAttemptInference(repo: customLLMManager.currentModelRepo) else {
            return nil
        }
        return .customLLM(repo: customLLMManager.currentModelRepo)
    }

    private func remoteDictionaryHistoryScanModel() -> DictionaryHistoryScanModel? {
        let context = resolvedRemoteLLMContext(forTranslation: false)
        guard isStoredRemoteLLMConfigured(context.provider) else { return nil }
        return .remoteLLM(provider: context.provider, configuration: context.configuration)
    }

    private func dictionaryHistoryScanModel(for optionID: String) throws -> DictionaryHistoryScanModel {
        if optionID.hasPrefix("local:") {
            let repo = String(optionID.dropFirst("local:".count))
            guard customLLMManager.canAttemptInference(repo: repo) else {
                throw NSError(
                    domain: "Voxt.DictionaryHistoryScan",
                    code: -5,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Selected local model is not available.")]
                )
            }
            return .customLLM(repo: repo)
        }

        if optionID.hasPrefix("remote:") {
            let rawProvider = String(optionID.dropFirst("remote:".count))
            guard let provider = RemoteLLMProvider(rawValue: rawProvider) else {
                throw NSError(
                    domain: "Voxt.DictionaryHistoryScan",
                    code: -6,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Selected remote model is invalid.")]
                )
            }
            let configuration = RemoteModelConfigurationStore.resolvedLLMConfiguration(
                provider: provider,
                stored: remoteLLMConfigurations
            )
            guard RemoteModelConfigurationStore.isStoredLLMConfigurationConfigured(
                provider: provider,
                stored: remoteLLMConfigurations
            ) else {
                throw NSError(
                    domain: "Voxt.DictionaryHistoryScan",
                    code: -7,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Selected remote model is not configured.")]
                )
            }
            return .remoteLLM(provider: provider, configuration: configuration)
        }

        throw NSError(
            domain: "Voxt.DictionaryHistoryScan",
            code: -8,
            userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("No model was selected for dictionary ingestion.")]
        )
    }

}
