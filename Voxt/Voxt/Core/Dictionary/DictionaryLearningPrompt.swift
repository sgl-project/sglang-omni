import Foundation

extension AutomaticDictionaryLearningMonitor {
    private struct PromptContext {
        let request: AutomaticDictionaryLearningRequest
        let existingTerms: [String]
        let userMainLanguage: String
        let userOtherLanguages: String
    }

    private static let latinOrNumberRegex = try! NSRegularExpression(
        pattern: #"[A-Za-z0-9]+(?:[._+\-'][A-Za-z0-9]+)*"#
    )

    private static let hanRegex = try! NSRegularExpression(pattern: #"\p{Han}{2,12}"#)

    private static let templateReplacements: [(token: String, value: (PromptContext) -> String)] = [
        (
            AppPreferenceKey.automaticDictionaryLearningMainLanguageTemplateVariable,
            { $0.userMainLanguage }
        ),
        (
            AppPreferenceKey.automaticDictionaryLearningOtherLanguagesTemplateVariable,
            { $0.userOtherLanguages }
        ),
        (
            AppPreferenceKey.automaticDictionaryLearningInsertedTextTemplateVariable,
            { $0.request.insertedText }
        ),
        (
            AppPreferenceKey.automaticDictionaryLearningBaselineContextTemplateVariable,
            { $0.request.baselineContext }
        ),
        (
            AppPreferenceKey.automaticDictionaryLearningFinalContextTemplateVariable,
            { $0.request.finalContext }
        ),
        (
            AppPreferenceKey.automaticDictionaryLearningBaselineFragmentTemplateVariable,
            { $0.request.baselineChangedFragment }
        ),
        (
            AppPreferenceKey.automaticDictionaryLearningFinalFragmentTemplateVariable,
            { $0.request.finalChangedFragment }
        ),
        (
            AppPreferenceKey.automaticDictionaryLearningExistingTermsTemplateVariable,
            { context in
                if context.existingTerms.isEmpty {
                    return "(empty)"
                }
                return context.existingTerms
                    .prefix(20)
                    .map { "- \($0)" }
                    .joined(separator: "\n")
            }
        )
    ]

    private static func normalizedDirectCandidateFragment(_ fragment: String) -> String {
        fragment
            .replacingOccurrences(of: #"^\s*>+\s*"#, with: "", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: ".,;:!?，。；：！？\"'()[]{}<>"))
    }

    private static func areEquivalentTerms(_ lhs: String, _ rhs: String) -> Bool {
        DictionaryStore.normalizeTerm(lhs) == DictionaryStore.normalizeTerm(rhs)
    }

    private static func isDirectCandidateTermLike(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty,
              trimmed.count >= 2,
              trimmed.count <= 48,
              !trimmed.contains("\n"),
              !trimmed.contains("。"),
              !trimmed.contains("！"),
              !trimmed.contains("？"),
              !trimmed.contains("；") else {
            return false
        }

        let parts = trimmed.split(whereSeparator: \.isWhitespace)
        guard !parts.isEmpty, parts.count <= 4 else {
            return false
        }

        let hasASCIIWord = trimmed.contains { isASCIIWordCharacter($0) }
        let hasIdeographic = trimmed.contains { isIdeographicCharacter($0) }

        if hasASCIIWord {
            return true
        }
        if hasIdeographic, parts.count == 1, trimmed.count <= 8 {
            return true
        }
        return hasASCIIWord && hasIdeographic
    }

    static func buildPrompt(
        template rawTemplate: String,
        for request: AutomaticDictionaryLearningRequest,
        existingTerms: [String],
        userMainLanguage: String,
        userOtherLanguages: String
    ) -> String {
        let template = AppPromptDefaults.resolvedStoredText(
            rawTemplate,
            kind: .dictionaryAutoLearning
        )
        let context = PromptContext(
            request: request,
            existingTerms: existingTerms,
            userMainLanguage: userMainLanguage,
            userOtherLanguages: userOtherLanguages
        )
        let resolvedTemplate = templateReplacements.reduce(template) { partial, item in
            let value = item.value(context)
            return partial.replacingOccurrences(of: item.token, with: value)
        }
        let runtimeConstraints = AppPromptResourceStore.requiredText(
            for: .dictionaryAutoLearningRuntimeConstraints,
            language: AppLocalization.language
        )
            .replacingOccurrences(
                of: "{{CANDIDATE_TERMS}}",
                with: candidateTermsSummary(for: request)
            )
        return "\(resolvedTemplate)\n\n\(runtimeConstraints)"
    }

    static func directCandidateTerms(
        for request: AutomaticDictionaryLearningRequest,
        existingTerms: [String]
    ) -> [String] {
        let baselineCandidate = normalizedDirectCandidateFragment(request.baselineChangedFragment)
        let finalCandidate = normalizedDirectCandidateFragment(request.finalChangedFragment)

        guard !baselineCandidate.isEmpty, !finalCandidate.isEmpty else {
            return []
        }
        guard !areEquivalentTerms(baselineCandidate, finalCandidate) else {
            return []
        }
        guard isDirectCandidateTermLike(baselineCandidate),
              isDirectCandidateTermLike(finalCandidate) else {
            return []
        }

        let normalizedExisting = Set(existingTerms.map(DictionaryStore.normalizeTerm))
        let normalizedFinal = DictionaryStore.normalizeTerm(finalCandidate)
        guard !normalizedFinal.isEmpty,
              !normalizedExisting.contains(normalizedFinal) else {
            return []
        }

        return [finalCandidate]
    }

    private static func candidateTermsSummary(for request: AutomaticDictionaryLearningRequest) -> String {
        let terms = candidateTerms(
            oldFragment: request.baselineChangedFragment,
            newFragment: request.finalChangedFragment
        )
        return terms.isEmpty ? "(empty)" : terms.joined(separator: ", ")
    }

    static func candidateTerms(oldFragment: String, newFragment: String) -> [String] {
        guard !newFragment.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return []
        }

        let previousTermSurfacesByNormalized = Dictionary(
            grouping: tokenizeVocabularyCandidates(oldFragment),
            by: normalizeVocabularyCandidate
        ).mapValues { Set($0) }

        return tokenizeVocabularyCandidates(newFragment)
            .filter { term in
                let previousSurfaces = previousTermSurfacesByNormalized[normalizeVocabularyCandidate(term)] ?? []
                return !previousSurfaces.contains(term)
            }
            .uniquedPreservingOrder(by: normalizeVocabularyCandidate)
            .prefix(8)
            .map { $0 }
    }

    private static func tokenizeVocabularyCandidates(_ text: String) -> [String] {
        let nsRange = NSRange(text.startIndex..<text.endIndex, in: text)
        let latinMatches: [String] = latinOrNumberRegex.matches(in: text, range: nsRange)
            .compactMap { match -> String? in
                guard let range = Range(match.range, in: text) else { return nil }
                let token = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
                return isValidLatinOrNumberToken(token) ? token : nil
            }
        let hanMatches: [String] = hanRegex.matches(in: text, range: nsRange)
            .compactMap { match -> String? in
                guard let range = Range(match.range, in: text) else { return nil }
                let token = String(text[range]).trimmingCharacters(in: .whitespacesAndNewlines)
                return isValidHanToken(token) ? token : nil
            }

        return (latinMatches + hanMatches).uniquedPreservingOrder(by: normalizeVocabularyCandidate)
    }

    private static func isValidLatinOrNumberToken(_ token: String) -> Bool {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 3, trimmed.count <= 32 else { return false }
        return trimmed.rangeOfCharacter(from: .letters) != nil
    }

    private static func isValidHanToken(_ token: String) -> Bool {
        let trimmed = token.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.count >= 2 && trimmed.count <= 12
    }

    nonisolated static func normalizeVocabularyCandidate(_ term: String) -> String {
        term.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }
}

private extension [String] {
    func uniquedPreservingOrder(by transform: (String) -> String) -> [String] {
        var seen = Set<String>()
        return filter { value in
            let key = transform(value)
            guard !key.isEmpty, seen.insert(key).inserted else { return false }
            return true
        }
    }
}
