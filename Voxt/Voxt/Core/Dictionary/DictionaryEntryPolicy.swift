import Foundation

enum DictionaryEntryPolicy {
    static func prepareEntryInput(
        term: String,
        replacementTerms: [String],
        groupID: UUID?,
        excluding excludedID: UUID? = nil,
        entries: [DictionaryEntry],
        validationIndex: DictionaryValidationIndex
    ) throws -> DictionaryPreparedEntryInput {
        return try DictionaryEntryInputPreparer.prepare(
            term: term,
            replacementTerms: replacementTerms,
            groupID: groupID,
            excluding: excludedID,
            entries: excludedID == nil ? nil : entries,
            validationIndex: excludedID == nil ? validationIndex : nil
        )
    }

    static func prepareTerm(_ term: String) throws -> (display: String, normalized: String) {
        let display = term.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalized = DictionaryTermNormalizer.normalize(display)
        guard !display.isEmpty, !normalized.isEmpty else {
            throw DictionaryStoreError.emptyTerm
        }
        return (display, normalized)
    }

    static func prepareCategoryName(
        _ name: String,
        categories: [DictionaryCategory],
        excluding excludedID: UUID? = nil
    ) throws -> (display: String, normalized: String) {
        let display = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalized = DictionaryTermNormalizer.normalize(display)
        guard !display.isEmpty, !normalized.isEmpty else {
            throw DictionaryStoreError.emptyCategoryName
        }
        let duplicate = categories.contains {
            $0.id != excludedID && $0.normalizedName == normalized
        }
        if duplicate {
            throw DictionaryStoreError.duplicateCategory
        }
        return (display, normalized)
    }

    static func unreservedVariants(
        from variants: [ObservedVariant],
        normalizedTerm: String,
        replacementTerms: [DictionaryReplacementTerm]
    ) -> [ObservedVariant] {
        let reservedKeys = Set([normalizedTerm] + replacementTerms.map(\.normalizedText))
        return variants.filter { !reservedKeys.contains($0.normalizedText) }
    }

    static func confidence(for candidate: DictionaryMatchCandidate) -> DictionaryVariantConfidence {
        if candidate.score >= 0.985 {
            return .high
        }
        if candidate.score >= 0.92 {
            return .medium
        }
        return .low
    }

    static func higherConfidence(
        lhs: DictionaryVariantConfidence,
        rhs: DictionaryVariantConfidence
    ) -> DictionaryVariantConfidence {
        let rank: [DictionaryVariantConfidence: Int] = [
            .low: 0,
            .medium: 1,
            .high: 2
        ]
        return (rank[lhs] ?? 0) >= (rank[rhs] ?? 0) ? lhs : rhs
    }

    static func sourceContainsNeedle(_ needle: String, normalizedSource: String) -> Bool {
        let normalizedNeedle = DictionaryTermNormalizer.normalize(needle)
        guard !normalizedNeedle.isEmpty else { return false }

        var searchRange: Range<String.Index>? = normalizedSource.startIndex..<normalizedSource.endIndex
        while let range = normalizedSource.range(of: normalizedNeedle, options: [], range: searchRange) {
            if hasValidBoundary(
                before: range.lowerBound,
                after: range.upperBound,
                needle: normalizedNeedle,
                source: normalizedSource
            ) {
                return true
            }
            searchRange = range.upperBound..<normalizedSource.endIndex
        }
        return false
    }

    private static func hasValidBoundary(
        before lowerBound: String.Index,
        after upperBound: String.Index,
        needle: String,
        source: String
    ) -> Bool {
        let needsLeadingBoundary = needle.unicodeScalars.first.map(Self.isASCIIAlphaNumeric) ?? false
        let needsTrailingBoundary = needle.unicodeScalars.last.map(Self.isASCIIAlphaNumeric) ?? false

        if needsLeadingBoundary,
           lowerBound > source.startIndex,
           let previous = source[..<lowerBound].unicodeScalars.last,
           Self.isASCIIAlphaNumeric(previous) {
            return false
        }

        if needsTrailingBoundary,
           upperBound < source.endIndex,
           let next = source[upperBound...].unicodeScalars.first,
           Self.isASCIIAlphaNumeric(next) {
            return false
        }

        return true
    }

    private nonisolated static func isASCIIAlphaNumeric(_ scalar: UnicodeScalar) -> Bool {
        (65...90).contains(Int(scalar.value))
            || (97...122).contains(Int(scalar.value))
            || (48...57).contains(Int(scalar.value))
    }
}
