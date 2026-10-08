// DictionaryLearningAnalysisFlow.swift
// Provides Dictionary Learning Analysis Flow for app lifecycle and routing.

import Foundation

extension AppDelegate {
    private struct AutomaticDictionaryLearningPersistResult {
        let addedTerms: [String]
        let reinforcedTerms: [String]

        var isEmpty: Bool {
            addedTerms.isEmpty && reinforcedTerms.isEmpty
        }

        var effectiveTerms: [String] {
            Self.orderedUnique(addedTerms + reinforcedTerms)
        }

        private static func orderedUnique(_ terms: [String]) -> [String] {
            var seen = Set<String>()
            return terms.filter { term in
                let normalized = DictionaryStore.normalizeTerm(term)
                guard !normalized.isEmpty, seen.insert(normalized).inserted else { return false }
                return true
            }
        }
    }

    private enum AutomaticDictionaryLearningModel {
        case appleIntelligence
        case customLLM(repo: String)
        case remoteLLM(provider: RemoteLLMProvider, configuration: RemoteProviderConfiguration)
    }

    func analyzeAutomaticDictionaryLearningRequest(
        _ request: AutomaticDictionaryLearningRequest,
        groupID: UUID?,
        groupNameSnapshot: String?,
        historyEntryID: UUID?
    ) async throws {
        let model = try resolvedAutomaticDictionaryLearningModel()
        VoxtLog.dictionary(
            "Automatic dictionary learning analysis started. model=\(automaticDictionaryLearningModelDescription(model)), historyEntryID=\(historyEntryID?.uuidString ?? "nil")"
        )
        let promptExistingTerms = dictionaryStore.allTerms(limit: 20)
        let prompt = AutomaticDictionaryLearningMonitor.buildPrompt(
            template: dictionaryAutoLearningPrompt,
            for: request,
            existingTerms: promptExistingTerms,
            userMainLanguage: userMainLanguagePromptValue,
            userOtherLanguages: userOtherMainLanguagesPromptValue
        )
        let directCandidateTerms = AutomaticDictionaryLearningMonitor.directCandidateTerms(
            for: request,
            existingTerms: []
        )
        let scannedTerms = try await runAutomaticDictionaryLearningPrompt(prompt, model: model)
        let mergedTerms = mergeDictionaryLearningTerms(
            directCandidateTerms,
            scannedTerms,
            excludeExistingTerms: false,
            activeGroupID: groupID
        )
        VoxtLog.dictionary(
            "Automatic dictionary learning candidate terms merged. direct=\(directCandidateTerms.joined(separator: ", ")), model=\(scannedTerms.joined(separator: ", ")), final=\(mergedTerms.joined(separator: ", "))"
        )

        let persistResult = persistAutomaticDictionaryLearningTerms(
            mergedTerms,
            groupID: groupID,
            groupNameSnapshot: groupNameSnapshot
        )

        guard !persistResult.isEmpty else {
            VoxtLog.dictionary("Automatic dictionary learning finished with no new dictionary entries.")
            return
        }
        if let historyEntryID, !persistResult.addedTerms.isEmpty {
            applyAutomaticDictionaryLearningHistoryUpdate(
                request,
                historyEntryID: historyEntryID,
                addedTerms: persistResult.addedTerms
            )
            VoxtLog.dictionary(
                "Automatic dictionary learning recorded correction into history. historyEntryID=\(historyEntryID.uuidString), added=\(persistResult.addedTerms.joined(separator: ", ")), reinforced=\(persistResult.reinforcedTerms.joined(separator: ", "))"
            )
        } else if !persistResult.addedTerms.isEmpty {
            VoxtLog.dictionary("Automatic dictionary learning added terms but history entry is unavailable.")
        }
        showOverlayStatus(automaticDictionaryLearningSuccessMessage(for: persistResult), clearAfter: 3.2)
        VoxtLog.dictionary(
            "Automatic dictionary learning persisted terms. added=\(persistResult.addedTerms.joined(separator: ", ")), reinforced=\(persistResult.reinforcedTerms.joined(separator: ", "))"
        )
    }

    func applyManualDictionaryCorrection(
        entry: TranscriptionHistoryEntry,
        correctedText rawCorrectedText: String
    ) async throws -> TranscriptionHistoryEntry? {
        let correctedText = rawCorrectedText.trimmingCharacters(in: .whitespacesAndNewlines)
        let baselineText = HistoryCorrectionPresentation.correctedText(
            for: entry.text,
            snapshots: entry.dictionaryCorrectionSnapshots
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        guard entry.kind == .normal else {
            return historyStore.entry(id: entry.id) ?? entry
        }
        guard !correctedText.isEmpty else {
            throw NSError(
                domain: "Voxt.ManualDictionaryCorrection",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Corrected text cannot be empty.")]
            )
        }
        guard correctedText != baselineText else {
            return historyStore.entry(id: entry.id) ?? entry
        }

        let promptExistingTerms = dictionaryStore.allTerms(limit: 20)
        let matchedGroupID = entry.matchedGroupID ?? currentDictionaryScope().groupID
        let matchedGroupName = entry.matchedGroupName ?? currentDictionaryScope().groupName
        var correctedTerms: [String] = []
        var correctionSnapshots: [DictionaryCorrectionSnapshot] = []

        let requestOutcome = AutomaticDictionaryLearningMonitor.makeLearningRequest(
            insertedText: baselineText,
            baselineText: baselineText,
            finalText: correctedText
        )
        switch requestOutcome {
        case .ready(let request):
            let directCandidateTerms = AutomaticDictionaryLearningMonitor.directCandidateTerms(
                for: request,
                existingTerms: []
            )
            var scannedTerms: [String] = []
            if let model = try? resolvedAutomaticDictionaryLearningModel() {
                let prompt = AutomaticDictionaryLearningMonitor.buildPrompt(
                    template: dictionaryAutoLearningPrompt,
                    for: request,
                    existingTerms: promptExistingTerms,
                    userMainLanguage: userMainLanguagePromptValue,
                    userOtherLanguages: userOtherMainLanguagesPromptValue
                )
                do {
                    scannedTerms = try await runAutomaticDictionaryLearningPrompt(prompt, model: model)
                } catch {
                    VoxtLog.dictionaryWarning("Manual dictionary correction term analysis failed: \(error)")
                }
            }

            correctedTerms = mergeDictionaryLearningTerms(
                directCandidateTerms,
                scannedTerms,
                excludeExistingTerms: false,
                activeGroupID: matchedGroupID
            )
            correctionSnapshots = automaticDictionaryLearningHistorySnapshots(
                request: request,
                updatedText: correctedText
            )
        case .skipped(let reason):
            VoxtLog.dictionary("Manual dictionary correction skipped term analysis: \(reason)")
        }

        let persistResult = persistAutomaticDictionaryLearningTerms(
            correctedTerms,
            groupID: matchedGroupID,
            groupNameSnapshot: matchedGroupName
        )

        historyStore.replaceDictionaryCorrectionResult(
            historyID: entry.id,
            updatedText: correctedText,
            correctedTerms: correctedTerms,
            correctionSnapshots: correctionSnapshots
        )

        if !persistResult.isEmpty {
            showOverlayStatus(automaticDictionaryLearningSuccessMessage(for: persistResult), clearAfter: 3.2)
        }

        return historyStore.entry(id: entry.id)
    }

    private func persistAutomaticDictionaryLearningTerms(
        _ scannedTerms: [String],
        groupID: UUID?,
        groupNameSnapshot: String?
    ) -> AutomaticDictionaryLearningPersistResult {
        let category = dictionaryStore.ensureCategory(id: groupID, name: groupNameSnapshot)
        var addedTerms: [String] = []
        var reinforcedTerms: [String] = []
        for term in scannedTerms {
            let normalized = DictionaryStore.normalizeTerm(term)
            guard !normalized.isEmpty else {
                VoxtLog.dictionary("Automatic dictionary learning ignored blank/invalid term candidate.")
                continue
            }
            do {
                let result = try dictionaryStore.createOrReinforceAutoEntry(
                    term: term,
                    categoryID: category.id,
                    categoryNameSnapshot: category.name,
                    groupID: groupID,
                    groupNameSnapshot: groupNameSnapshot
                )
                if result.added {
                    addedTerms.append(result.term)
                } else {
                    reinforcedTerms.append(result.term)
                }
            } catch DictionaryStoreError.duplicateTerm {
                continue
            } catch {
                VoxtLog.dictionaryWarning("Automatic dictionary learning skipped term due to store error: \(error)")
            }
        }
        return AutomaticDictionaryLearningPersistResult(
            addedTerms: addedTerms,
            reinforcedTerms: reinforcedTerms
        )
    }

    private func mergeDictionaryLearningTerms(
        _ directTerms: [String],
        _ modelTerms: [String],
        excludeExistingTerms: Bool,
        activeGroupID: UUID?
    ) -> [String] {
        var merged: [String] = []
        var seen: Set<String> = []

        for term in directTerms + modelTerms {
            let normalized = DictionaryStore.normalizeTerm(term)
            guard !normalized.isEmpty,
                  !seen.contains(normalized) else {
                continue
            }
            if excludeExistingTerms,
               dictionaryStore.hasEntry(normalizedTerm: normalized, activeGroupID: activeGroupID) {
                continue
            }
            seen.insert(normalized)
            merged.append(term)
        }

        return merged
    }

    private func applyAutomaticDictionaryLearningHistoryUpdate(
        _ request: AutomaticDictionaryLearningRequest,
        historyEntryID: UUID,
        addedTerms: [String]
    ) {
        guard let existingEntry = historyStore.entry(id: historyEntryID) else {
            historyStore.applyDictionaryCorrectedTerms([historyEntryID: addedTerms])
            return
        }

        let updatedText = automaticDictionaryLearningUpdatedHistoryText(
            request: request,
            originalText: existingEntry.text
        )
        let correctionSnapshots = automaticDictionaryLearningHistorySnapshots(
            request: request,
            updatedText: updatedText
        )

        historyStore.applyDictionaryCorrectionResult(
            historyID: historyEntryID,
            updatedText: updatedText,
            correctedTerms: addedTerms,
            correctionSnapshots: correctionSnapshots
        )
    }

    private func automaticDictionaryLearningUpdatedHistoryText(
        request: AutomaticDictionaryLearningRequest,
        originalText: String
    ) -> String {
        let trimmedOriginal = originalText.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedBaselineContext = request.baselineContext.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedFinalContext = request.finalContext.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !trimmedFinalContext.isEmpty else {
            return trimmedOriginal
        }
        if trimmedOriginal == trimmedBaselineContext || trimmedOriginal == request.insertedText {
            return trimmedFinalContext
        }
        if let range = trimmedOriginal.range(of: trimmedBaselineContext) {
            var updated = trimmedOriginal
            updated.replaceSubrange(range, with: trimmedFinalContext)
            return updated
        }
        return trimmedFinalContext
    }

    private func automaticDictionaryLearningHistorySnapshots(
        request: AutomaticDictionaryLearningRequest,
        updatedText: String
    ) -> [DictionaryCorrectionSnapshot] {
        let originalFragment = request.baselineChangedFragment.trimmingCharacters(in: .whitespacesAndNewlines)
        let correctedFragment = request.finalChangedFragment.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !originalFragment.isEmpty,
              !correctedFragment.isEmpty,
              originalFragment != correctedFragment,
              let range = updatedText.range(of: correctedFragment)
        else {
            return []
        }

        let nsRange = NSRange(range, in: updatedText)
        return [
            DictionaryCorrectionSnapshot(
                originalText: originalFragment,
                correctedText: correctedFragment,
                finalLocation: nsRange.location,
                finalLength: nsRange.length
            )
        ]
    }

    private func automaticDictionaryLearningSuccessMessage(
        for result: AutomaticDictionaryLearningPersistResult
    ) -> String {
        if result.addedTerms.isEmpty {
            return automaticDictionaryLearningReinforcedMessage(for: result.reinforcedTerms)
        }
        if result.reinforcedTerms.isEmpty {
            return automaticDictionaryLearningAddedMessage(for: result.addedTerms)
        }
        return AppLocalization.format(
            "Added %d corrected terms and reinforced %d existing dictionary terms: %@",
            result.addedTerms.count,
            result.reinforcedTerms.count,
            result.effectiveTerms.joined(separator: ", ")
        )
    }

    private func automaticDictionaryLearningAddedMessage(for addedTerms: [String]) -> String {
        if addedTerms.count == 1 {
            return AppLocalization.format(
                "Added 1 corrected term to the dictionary: %@",
                addedTerms[0]
            )
        }
        return AppLocalization.format(
            "Added %d corrected terms to the dictionary: %@",
            addedTerms.count,
            addedTerms.joined(separator: ", ")
        )
    }

    private func automaticDictionaryLearningReinforcedMessage(for reinforcedTerms: [String]) -> String {
        if reinforcedTerms.count == 1 {
            return AppLocalization.format(
                "Reinforced existing dictionary term: %@",
                reinforcedTerms[0]
            )
        }
        return AppLocalization.format(
            "Reinforced %d existing dictionary terms: %@",
            reinforcedTerms.count,
            reinforcedTerms.joined(separator: ", ")
        )
    }

    private func runAutomaticDictionaryLearningPrompt(
        _ prompt: String,
        model: AutomaticDictionaryLearningModel
    ) async throws -> [String] {
        switch model {
        case .appleIntelligence:
            guard let enhancer else {
                throw NSError(
                    domain: "Voxt.AutomaticDictionaryLearning",
                    code: -1,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Apple Intelligence is unavailable.")]
                )
            }
            if #available(macOS 26.0, *) {
                return try await enhancer.dictionaryHistoryScanTerms(userPrompt: prompt)
            }
            throw NSError(
                domain: "Voxt.AutomaticDictionaryLearning",
                code: -2,
                userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Apple Intelligence requires macOS 26 or later.")]
            )
        case .customLLM(let repo):
            return try await customLLMManager.dictionaryHistoryScanTerms(
                userPrompt: prompt,
                repo: repo
            )
        case .remoteLLM(let provider, let configuration):
            return try await RemoteLLMRuntimeClient().dictionaryHistoryScanTerms(
                userPrompt: prompt,
                provider: provider,
                configuration: configuration
            )
        }
    }

    private func automaticDictionaryLearningModelDescription(
        _ model: AutomaticDictionaryLearningModel
    ) -> String {
        switch model {
        case .appleIntelligence:
            return "apple-intelligence"
        case .customLLM(let repo):
            return "local:\(repo)"
        case .remoteLLM(let provider, let configuration):
            return "remote:\(provider.rawValue):\(configuration.model)"
        }
    }

    private func resolvedAutomaticDictionaryLearningModel() throws -> AutomaticDictionaryLearningModel {
        if let saved = savedAutomaticDictionaryLearningModel() {
            return saved
        }

        if let firstOption = availableDictionaryHistoryScanModelOptions().first {
            return try automaticDictionaryLearningModel(for: firstOption.id)
        }

        if #available(macOS 26.0, *), enhancer != nil, TextEnhancer.isAvailable {
            return .appleIntelligence
        }

        throw NSError(
            domain: "Voxt.AutomaticDictionaryLearning",
            code: -3,
            userInfo: [
                NSLocalizedDescriptionKey: AppLocalization.localizedString(
                    "No usable LLM is available for automatic dictionary learning."
                )
            ]
        )
    }

    private func savedAutomaticDictionaryLearningModel() -> AutomaticDictionaryLearningModel? {
        let optionID = UserDefaults.standard.string(
            forKey: AppPreferenceKey.dictionarySuggestionIngestModelOptionID
        ) ?? ""
        guard !optionID.isEmpty else { return nil }
        return try? automaticDictionaryLearningModel(for: optionID)
    }

    private func automaticDictionaryLearningModel(
        for optionID: String
    ) throws -> AutomaticDictionaryLearningModel {
        if optionID.hasPrefix("local:") {
            let repo = String(optionID.dropFirst("local:".count))
            guard customLLMManager.canAttemptInference(repo: repo) else {
                throw NSError(
                    domain: "Voxt.AutomaticDictionaryLearning",
                    code: -4,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Selected local model is not available.")]
                )
            }
            return .customLLM(repo: repo)
        }

        if optionID.hasPrefix("remote:") {
            let rawProvider = String(optionID.dropFirst("remote:".count))
            guard let provider = RemoteLLMProvider(rawValue: rawProvider) else {
                throw NSError(
                    domain: "Voxt.AutomaticDictionaryLearning",
                    code: -5,
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
                    domain: "Voxt.AutomaticDictionaryLearning",
                    code: -6,
                    userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("Selected remote model is not configured.")]
                )
            }
            return .remoteLLM(provider: provider, configuration: configuration)
        }

        throw NSError(
            domain: "Voxt.AutomaticDictionaryLearning",
            code: -7,
            userInfo: [NSLocalizedDescriptionKey: AppLocalization.localizedString("No model was selected for automatic dictionary learning.")]
        )
    }
}
