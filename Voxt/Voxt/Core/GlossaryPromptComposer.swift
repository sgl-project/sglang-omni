// GlossaryPromptComposer.swift
// Provides Glossary Prompt Composer for core app behavior.

import Foundation

enum DictionaryGlossaryPurpose {
    case enhancement
    case translation
    case rewrite

    var selectionPolicy: DictionaryGlossarySelectionPolicy {
        switch self {
        case .enhancement:
            return DictionaryGlossarySelectionPolicy(maxTerms: 8, maxCharacters: 220)
        case .translation:
            return DictionaryGlossarySelectionPolicy(maxTerms: 10, maxCharacters: 280)
        case .rewrite:
            return DictionaryGlossarySelectionPolicy(maxTerms: 8, maxCharacters: 220)
        }
    }

}

struct DictionaryGlossarySelectionPolicy: Equatable {
    let maxTerms: Int
    let maxCharacters: Int

    func reducedForLongInput() -> DictionaryGlossarySelectionPolicy {
        DictionaryGlossarySelectionPolicy(
            maxTerms: max(1, Int((Double(maxTerms) * 0.625).rounded(.down))),
            maxCharacters: max(72, Int((Double(maxCharacters) * 0.65).rounded(.down)))
        )
    }
}

enum DictionaryGlossaryPromptComposer {
    nonisolated static func body(
        glossary: String?,
        purpose: DictionaryGlossaryPurpose
    ) -> String? {
        let trimmedGlossary = glossary?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !trimmedGlossary.isEmpty else { return nil }

        switch purpose {
        case .enhancement:
            return """
            Prefer these exact spellings when the transcript context indicates the user meant them:
            \(trimmedGlossary)

            If a nearby phrase looks like one of these terms, prefer the exact spelling above.
            """
        case .translation:
            return """
            When the source text refers to these proper nouns or product terms, preserve their exact spelling unless translation clearly requires otherwise:
            \(trimmedGlossary)
            """
        case .rewrite:
            return """
            Prefer these exact term spellings in the final output when relevant:
            \(trimmedGlossary)
            """
        }
    }

    nonisolated static func append(
        prompt: String,
        glossary: String?,
        purpose: DictionaryGlossaryPurpose
    ) -> String {
        guard let body = body(glossary: glossary, purpose: purpose) else { return prompt }
        let instruction = """
        ### Dictionary Guidance
        \(body)
        """

        return "\(prompt)\n\n\(instruction)"
    }
}
