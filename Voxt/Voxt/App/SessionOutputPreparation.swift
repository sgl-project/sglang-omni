// Normalizes and corrects the delivery snapshot without performing text injection.

import Foundation

extension AppDelegate {
    nonisolated static func resolveDictionaryOutput(
        text: String,
        matcher: DictionaryMatcher?,
        usesConservativeEvidence: Bool,
        automaticReplacementEnabled: Bool
    ) -> DictionaryCorrectionResult {
        guard let matcher else {
            return DictionaryCorrectionResult(
                text: text,
                candidates: [],
                correctedTerms: [],
                correctionSnapshots: []
            )
        }

        if usesConservativeEvidence {
            let candidates = matcher.recallCandidates(in: text)
            return DictionaryCorrectionResult(
                text: text,
                candidates: candidates,
                correctedTerms: [],
                correctionSnapshots: []
            )
        }

        return matcher.applyCorrections(
            to: text,
            automaticReplacementEnabled: automaticReplacementEnabled
        )
    }

    nonisolated static func preparedDeliveryContext(
        originalText: String,
        llmDurationSeconds: TimeInterval?,
        sessionOutputMode: SessionOutputMode,
        userMainLanguage: UserMainLanguageOption,
        matcher: DictionaryMatcher?,
        usesConservativeEvidence: Bool,
        automaticReplacementEnabled: Bool
    ) -> SessionFinalizeContext {
        let normalized = normalizedOutputText(
            originalText,
            sessionOutputMode: sessionOutputMode,
            userMainLanguage: userMainLanguage
        )

        let extractedRewriteAnswerPayload = RewriteAnswerPayloadParser.extract(from: normalized)
        let rewriteContent = extractedRewriteAnswerPayload?.content ?? normalized
        let dictionaryCorrection = resolveDictionaryOutput(
            text: rewriteContent,
            matcher: matcher,
            usesConservativeEvidence: usesConservativeEvidence,
            automaticReplacementEnabled: automaticReplacementEnabled
        )
        let uniqueDictionaryMatches = orderedUniqueDictionaryMatches(dictionaryCorrection.candidates)
        let rewriteAnswerPayload = extractedRewriteAnswerPayload.map {
            RewriteAnswerPayload(title: $0.title, content: dictionaryCorrection.text)
        }

        return SessionFinalizeContext(
            outputText: dictionaryCorrection.text,
            llmDurationSeconds: llmDurationSeconds,
            dictionaryMatches: uniqueDictionaryMatches,
            dictionaryCorrectedTerms: dictionaryCorrection.correctedTerms,
            dictionaryCorrectionSnapshots: dictionaryCorrection.correctionSnapshots,
            rewriteAnswerPayload: rewriteAnswerPayload
        )
    }

    nonisolated private static func orderedUniqueDictionaryMatches(
        _ candidates: [DictionaryMatchCandidate]
    ) -> [DictionaryMatchCandidate] {
        var seen = Set<String>()
        var ordered: [DictionaryMatchCandidate] = []
        for candidate in candidates {
            let normalized = DictionaryStore.normalizeTerm(candidate.term)
            guard !normalized.isEmpty, seen.insert(normalized).inserted else { continue }
            ordered.append(candidate)
        }
        return ordered
    }

    nonisolated static func orderedUniqueDictionaryTerms(from values: [String]) -> [String] {
        var seen = Set<String>()
        var ordered: [String] = []
        for value in values {
            let normalized = DictionaryStore.normalizeTerm(value)
            guard !normalized.isEmpty, seen.insert(normalized).inserted else { continue }
            ordered.append(value)
        }
        return ordered
    }

    func shouldUseConservativeDictionaryEvidenceForCurrentSession() -> Bool {
        let featureSettings = FeatureSettingsStore.load(defaults: .standard)
        let selectionID: FeatureModelSelectionID
        switch sessionOutputMode {
        case .translation:
            selectionID = featureSettings.translation.asrSelectionID
        case .rewrite:
            selectionID = featureSettings.rewrite.asrSelectionID
        case .transcription:
            selectionID = featureSettings.transcription.asrSelectionID
        }

        guard case .remote(let provider)? = selectionID.asrSelection,
              provider == .doubaoASR
        else {
            return false
        }

        let raw = UserDefaults.standard.string(forKey: AppPreferenceKey.remoteASRProviderConfigurations) ?? ""
        let stored = RemoteModelConfigurationStore.loadConfigurations(
            from: raw,
            sensitiveValueLoading: .metadataOnly
        )
        let configuration = RemoteModelConfigurationStore.resolvedASRConfiguration(provider: provider, stored: stored)

        switch configuration.doubaoDictionaryModeValue {
        case .off:
            return false
        case .requestScoped:
            return configuration.doubaoEnableRequestCorrections
        }
    }

    nonisolated private static func normalizedOutputText(
        _ text: String,
        sessionOutputMode: SessionOutputMode,
        userMainLanguage: UserMainLanguageOption
    ) -> String {
        var value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard value.count >= 2 else { return value }

        // Remove paired wrapping quotes from some model outputs.
        let left = value.first
        let right = value.last
        let isWrappedByDoubleQuotes =
            (left == "\"" && right == "\"") ||
            (left == "“" && right == "”")

        if isWrappedByDoubleQuotes {
            value.removeFirst()
            value.removeLast()
            value = value.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        switch sessionOutputMode {
        case .translation:
            break
        case .transcription, .rewrite:
            let normalizedChineseScript = ChineseScriptNormalizer.normalize(
                value,
                preferredMainLanguage: userMainLanguage
            )
            value = normalizedChineseScript
        }

        return value
    }
}
