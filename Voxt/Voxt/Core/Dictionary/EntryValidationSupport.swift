// EntryValidationSupport.swift
// Provides Entry Validation Support for dictionary matching and learning.

import Foundation

struct DictionaryPreparedEntryInput {
    let display: String
    let normalized: String
    let replacementTerms: [DictionaryReplacementTerm]
}

struct DictionaryScopeKey: Hashable {
    let groupID: UUID?
}

struct DictionaryValidationIndex {
    private(set) var usedMatchKeysByScope: [DictionaryScopeKey: Set<String>] = [:]

    init(entries: [DictionaryEntry], excluding excludedID: UUID? = nil) {
        for entry in entries where entry.id != excludedID {
            insert(entry)
        }
    }

    func contains(_ normalizedKey: String, groupID: UUID?) -> Bool {
        usedMatchKeysByScope[DictionaryScopeKey(groupID: groupID)]?.contains(normalizedKey) == true
    }

    mutating func insert(_ entry: DictionaryEntry) {
        var keys = usedMatchKeysByScope[DictionaryScopeKey(groupID: entry.groupID)] ?? []
        keys.formUnion(entry.matchKeys)
        usedMatchKeysByScope[DictionaryScopeKey(groupID: entry.groupID)] = keys
    }
}

enum DictionaryTermNormalizer {
    nonisolated static func normalize(_ input: String) -> String {
        let folded = input.folding(
            options: [.caseInsensitive, .diacriticInsensitive, .widthInsensitive],
            locale: .current
        )
        var output = ""
        var previousWasWhitespace = false

        for scalar in folded.unicodeScalars {
            if dictionaryIsWordScalar(scalar) {
                output.unicodeScalars.append(scalar)
                previousWasWhitespace = false
            } else if CharacterSet.whitespacesAndNewlines.contains(scalar)
                        || CharacterSet.punctuationCharacters.contains(scalar)
                        || CharacterSet.symbols.contains(scalar) {
                if !previousWasWhitespace && !output.isEmpty {
                    output.append(" ")
                    previousWasWhitespace = true
                }
            }
        }

        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

enum DictionaryEntryInputPreparer {
    static func prepare(
        term: String,
        replacementTerms: [String],
        groupID: UUID?,
        excluding excludedID: UUID? = nil,
        entries: [DictionaryEntry]? = nil,
        validationIndex providedValidationIndex: DictionaryValidationIndex? = nil
    ) throws -> DictionaryPreparedEntryInput {
        let display = term.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalized = DictionaryTermNormalizer.normalize(display)
        guard !display.isEmpty, !normalized.isEmpty else {
            throw DictionaryStoreError.emptyTerm
        }

        let resolvedValidationIndex: DictionaryValidationIndex
        if let providedValidationIndex {
            resolvedValidationIndex = providedValidationIndex
        } else {
            resolvedValidationIndex = DictionaryValidationIndex(entries: entries ?? [], excluding: excludedID)
        }

        var preparedReplacementTerms: [DictionaryReplacementTerm] = []
        var seenReplacementKeys = Set<String>()

        for rawValue in replacementTerms {
            let displayValue = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
            let normalizedValue = DictionaryTermNormalizer.normalize(displayValue)
            guard !displayValue.isEmpty, !normalizedValue.isEmpty else { continue }

            if normalizedValue == normalized {
                throw DictionaryStoreError.replacementMatchesDictionaryTerm
            }

            guard seenReplacementKeys.insert(normalizedValue).inserted else { continue }

            if resolvedValidationIndex.contains(normalizedValue, groupID: groupID) {
                throw DictionaryStoreError.duplicateReplacementTerm(displayValue)
            }

            preparedReplacementTerms.append(
                DictionaryReplacementTerm(
                    text: displayValue,
                    normalizedText: normalizedValue
                )
            )
        }

        return DictionaryPreparedEntryInput(
            display: display,
            normalized: normalized,
            replacementTerms: preparedReplacementTerms
        )
    }
}
