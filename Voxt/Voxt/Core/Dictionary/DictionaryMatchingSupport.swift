// DictionaryMatchingSupport.swift
// Provides Dictionary Matching Support for dictionary matching and learning.

import Foundation

private struct DictionaryToken {
    let raw: String
    let normalized: String
    let range: NSRange
}

private struct DictionaryScriptProfile {
    var containsLatin = false
    var containsDigit = false
    var containsHan = false
    var containsKana = false
    var containsHangul = false

    nonisolated init() {}

    nonisolated var containsCJKLike: Bool {
        containsHan || containsKana || containsHangul
    }

    nonisolated var isLatinLike: Bool {
        (containsLatin || containsDigit) && !containsCJKLike
    }

    nonisolated var isMixedScript: Bool {
        containsCJKLike && (containsLatin || containsDigit)
    }
}

private struct DictionaryNormalizedMapping {
    var text: String
    var sourceRanges: [NSRange]
}

private enum DictionaryMatchVariantSource {
    case term
    case replacementTerm
    case observedVariant

    nonisolated var source: DictionaryMatchSource {
        switch self {
        case .term:
            return .term
        case .replacementTerm:
            return .replacementTerm
        case .observedVariant:
            return .observedVariant
        }
    }

    nonisolated var allowsFuzzyMatch: Bool {
        switch self {
        case .replacementTerm:
            return false
        case .term, .observedVariant:
            return true
        }
    }
}

private struct DictionaryMatchVariant {
    let text: String
    let normalizedText: String
    let source: DictionaryMatchVariantSource
}

nonisolated func dictionaryIsWordScalar(_ scalar: UnicodeScalar) -> Bool {
    CharacterSet.alphanumerics.contains(scalar)
        || dictionaryIsHanLike(scalar)
        || dictionaryIsKana(scalar)
        || dictionaryIsHangul(scalar)
}

nonisolated func dictionaryIsHanLike(_ scalar: UnicodeScalar) -> Bool {
    switch scalar.value {
    case 0x4E00...0x9FFF,
         0x3400...0x4DBF,
         0x20000...0x2A6DF,
         0x2A700...0x2B73F,
         0x2B740...0x2B81F,
         0x2B820...0x2CEAF:
        return true
    default:
        return false
    }
}

nonisolated func dictionaryIsKana(_ scalar: UnicodeScalar) -> Bool {
    switch scalar.value {
    case 0x3040...0x309F,
         0x30A0...0x30FF,
         0x31F0...0x31FF,
         0xFF66...0xFF9F:
        return true
    default:
        return false
    }
}

nonisolated func dictionaryIsHangul(_ scalar: UnicodeScalar) -> Bool {
    switch scalar.value {
    case 0x1100...0x11FF,
         0x3130...0x318F,
         0xA960...0xA97F,
         0xAC00...0xD7AF,
         0xD7B0...0xD7FF:
        return true
    default:
        return false
    }
}

private nonisolated func dictionaryScriptProfile(for text: String) -> DictionaryScriptProfile {
    var profile = DictionaryScriptProfile()
    for scalar in text.unicodeScalars {
        if CharacterSet.decimalDigits.contains(scalar) {
            profile.containsDigit = true
        } else if CharacterSet.letters.contains(scalar),
                  !dictionaryIsHanLike(scalar),
                  !dictionaryIsKana(scalar),
                  !dictionaryIsHangul(scalar) {
            profile.containsLatin = true
        }

        if dictionaryIsHanLike(scalar) {
            profile.containsHan = true
        }
        if dictionaryIsKana(scalar) {
            profile.containsKana = true
        }
        if dictionaryIsHangul(scalar) {
            profile.containsHangul = true
        }
    }
    return profile
}

private nonisolated func dictionaryNormalizedMapping(for text: String) -> DictionaryNormalizedMapping {
    var output = ""
    var sourceRanges: [NSRange] = []
    var previousWasWhitespace = false
    var index = text.startIndex

    while index < text.endIndex {
        let nextIndex = text.index(after: index)
        let fragment = String(text[index..<nextIndex]).folding(
            options: [.caseInsensitive, .diacriticInsensitive, .widthInsensitive],
            locale: .current
        )
        let sourceRange = NSRange(index..<nextIndex, in: text)

        for scalar in fragment.unicodeScalars {
            if dictionaryIsWordScalar(scalar) {
                output.unicodeScalars.append(scalar)
                sourceRanges.append(sourceRange)
                previousWasWhitespace = false
            } else if CharacterSet.whitespacesAndNewlines.contains(scalar)
                        || CharacterSet.punctuationCharacters.contains(scalar)
                        || CharacterSet.symbols.contains(scalar) {
                if !previousWasWhitespace && !output.isEmpty {
                    output.append(" ")
                    sourceRanges.append(sourceRange)
                    previousWasWhitespace = true
                }
            }
        }

        index = nextIndex
    }

    while output.first == " " {
        output.removeFirst()
        sourceRanges.removeFirst()
    }

    while output.last == " " {
        output.removeLast()
        sourceRanges.removeLast()
    }

    return DictionaryNormalizedMapping(text: output, sourceRanges: sourceRanges)
}

private nonisolated func dictionaryExactNormalizedMatchRanges(
    in text: String,
    normalizedNeedle: String,
    requireTokenBoundaries: Bool
) -> [NSRange] {
    let mapping = dictionaryNormalizedMapping(for: text)
    return dictionaryExactNormalizedMatchRanges(
        in: mapping,
        normalizedNeedle: normalizedNeedle,
        requireTokenBoundaries: requireTokenBoundaries
    )
}

private nonisolated func dictionaryExactNormalizedMatchRanges(
    in mapping: DictionaryNormalizedMapping,
    normalizedNeedle: String,
    requireTokenBoundaries: Bool
) -> [NSRange] {
    guard !normalizedNeedle.isEmpty else { return [] }
    guard !mapping.text.isEmpty, !mapping.sourceRanges.isEmpty else { return [] }

    var matches: [NSRange] = []
    var searchStart = mapping.text.startIndex

    while searchStart < mapping.text.endIndex,
          let matchRange = mapping.text.range(
            of: normalizedNeedle,
            options: [],
            range: searchStart..<mapping.text.endIndex
          ) {
        let lowerIsBoundary =
            matchRange.lowerBound == mapping.text.startIndex
            || mapping.text[mapping.text.index(before: matchRange.lowerBound)] == " "
        let upperIsBoundary =
            matchRange.upperBound == mapping.text.endIndex
            || mapping.text[matchRange.upperBound] == " "

        if !requireTokenBoundaries || (lowerIsBoundary && upperIsBoundary) {
            let lowerOffset = mapping.text.distance(from: mapping.text.startIndex, to: matchRange.lowerBound)
            let upperOffset = mapping.text.distance(from: mapping.text.startIndex, to: matchRange.upperBound)

            if lowerOffset < mapping.sourceRanges.count, upperOffset > lowerOffset {
                let startRange = mapping.sourceRanges[lowerOffset]
                let endRange = mapping.sourceRanges[upperOffset - 1]
                let combined = NSRange(
                    location: startRange.location,
                    length: (endRange.location + endRange.length) - startRange.location
                )
                if !matches.contains(combined) {
                    matches.append(combined)
                }
            }
        }

        if matchRange.lowerBound < mapping.text.endIndex {
            searchStart = mapping.text.index(after: matchRange.lowerBound)
        } else {
            break
        }
    }

    return matches
}

struct DictionaryMatcher {
    let entries: [DictionaryEntry]
    let blockedGlobalMatchKeys: Set<String>

    nonisolated func promptContext(for text: String) -> DictionaryPromptContext {
        let candidates = recallCandidates(in: text)
        return DictionaryPromptContext(entries: entries, candidates: candidates)
    }

    nonisolated func recallCandidates(in text: String) -> [DictionaryMatchCandidate] {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }

        let normalizedMapping = dictionaryNormalizedMapping(for: text)
        let rawTokens = tokenize(text)
        var bestByID: [String: DictionaryMatchCandidate] = [:]

        for entry in entries where entry.status == .active {
            for variant in matchVariants(for: entry) {
                guard !shouldBlock(variant: variant, entry: entry) else { continue }

                for candidate in exactCandidates(
                    for: entry,
                    variant: variant,
                    text: text,
                    normalizedMapping: normalizedMapping
                ) {
                    let key = candidate.id
                    if let existing = bestByID[key], existing.score >= candidate.score {
                        continue
                    }
                    bestByID[key] = candidate
                }

                guard variant.source.allowsFuzzyMatch,
                      let candidate = bestFuzzyCandidate(
                        for: entry,
                        variant: variant,
                        text: text,
                        rawTokens: rawTokens
                      ) else {
                    continue
                }

                let key = candidate.id
                if let existing = bestByID[key], existing.score >= candidate.score {
                    continue
                }
                bestByID[key] = candidate
            }
        }

        return bestByID.values.sorted {
            if $0.score == $1.score {
                if ($0.matchRange?.location ?? 0) == ($1.matchRange?.location ?? 0) {
                    return $0.term < $1.term
                }
                return ($0.matchRange?.location ?? 0) < ($1.matchRange?.location ?? 0)
            }
            return $0.score > $1.score
        }
    }

    nonisolated func applyCorrections(to text: String, automaticReplacementEnabled: Bool) -> DictionaryCorrectionResult {
        let candidates = recallCandidates(in: text)
        let replacementCandidates = candidates
            .filter { shouldApplyReplacement(for: $0, automaticReplacementEnabled: automaticReplacementEnabled) }
            .sorted(by: replacementSortComparator)

        guard !replacementCandidates.isEmpty else {
            return DictionaryCorrectionResult(
                text: text,
                candidates: candidates,
                correctedTerms: [],
                correctionSnapshots: []
            )
        }

        let output = NSMutableString(string: text)
        var correctedTerms: [String] = []
        var appliedRanges: [NSRange] = []
        var appliedSnapshots: [(originalRange: NSRange, snapshot: DictionaryCorrectionSnapshot)] = []

        for candidate in replacementCandidates {
            guard let matchRange = candidate.matchRange, matchRange.length > 0 else { continue }
            guard candidate.matchedText.trimmingCharacters(in: .whitespacesAndNewlines) != candidate.term else { continue }
            guard !appliedRanges.contains(where: { NSIntersectionRange($0, matchRange).length > 0 }) else { continue }

            output.replaceCharacters(in: matchRange, with: candidate.term)
            correctedTerms.append(candidate.term)
            appliedRanges.append(matchRange)
            appliedSnapshots.append((
                originalRange: matchRange,
                snapshot: DictionaryCorrectionSnapshot(
                    originalText: candidate.matchedText,
                    correctedText: candidate.term,
                    finalLocation: 0,
                    finalLength: 0
                )
            ))
        }

        let correctionSnapshots = finalizedCorrectionSnapshots(from: appliedSnapshots)

        return DictionaryCorrectionResult(
            text: output as String,
            candidates: candidates,
            correctedTerms: correctedTerms,
            correctionSnapshots: correctionSnapshots
        )
    }

    private nonisolated func finalizedCorrectionSnapshots(
        from appliedSnapshots: [(originalRange: NSRange, snapshot: DictionaryCorrectionSnapshot)]
    ) -> [DictionaryCorrectionSnapshot] {
        var delta = 0
        return appliedSnapshots
            .sorted { lhs, rhs in
                lhs.originalRange.location < rhs.originalRange.location
            }
            .map { item in
                let originalLength = item.originalRange.length
                let correctedLength = (item.snapshot.correctedText as NSString).length
                let finalized = DictionaryCorrectionSnapshot(
                    originalText: item.snapshot.originalText,
                    correctedText: item.snapshot.correctedText,
                    finalLocation: item.originalRange.location + delta,
                    finalLength: correctedLength
                )
                delta += correctedLength - originalLength
                return finalized
            }
    }

    private nonisolated func shouldApplyReplacement(
        for candidate: DictionaryMatchCandidate,
        automaticReplacementEnabled: Bool
    ) -> Bool {
        if candidate.source == .replacementTerm {
            return true
        }
        guard automaticReplacementEnabled else { return false }
        return candidate.allowsAutomaticReplacement
    }

    private nonisolated func replacementSortComparator(
        lhs: DictionaryMatchCandidate,
        rhs: DictionaryMatchCandidate
    ) -> Bool {
        let lhsLocation = lhs.matchRange?.location ?? -1
        let rhsLocation = rhs.matchRange?.location ?? -1
        if lhsLocation != rhsLocation {
            return lhsLocation > rhsLocation
        }

        let lhsPriority = replacementPriority(for: lhs)
        let rhsPriority = replacementPriority(for: rhs)
        if lhsPriority != rhsPriority {
            return lhsPriority > rhsPriority
        }

        let lhsLength = lhs.matchRange?.length ?? 0
        let rhsLength = rhs.matchRange?.length ?? 0
        if lhsLength != rhsLength {
            return lhsLength > rhsLength
        }

        return lhs.score > rhs.score
    }

    private nonisolated func replacementPriority(for candidate: DictionaryMatchCandidate) -> Int {
        if candidate.source == .replacementTerm {
            return 4
        }

        switch candidate.reason {
        case .exactVariant:
            return 3
        case .exactWindow:
            return 2
        case .fuzzyWindow:
            return 1
        case .exactTerm:
            return 0
        }
    }

    private nonisolated func matchVariants(for entry: DictionaryEntry) -> [DictionaryMatchVariant] {
        var variants = [
            DictionaryMatchVariant(
                text: entry.term,
                normalizedText: entry.normalizedTerm,
                source: .term
            )
        ]

        variants.append(
            contentsOf: entry.replacementTerms.map {
                DictionaryMatchVariant(
                    text: $0.text,
                    normalizedText: $0.normalizedText,
                    source: .replacementTerm
                )
            }
        )

        variants.append(
            contentsOf: entry.observedVariants.map {
                DictionaryMatchVariant(
                    text: $0.text,
                    normalizedText: $0.normalizedText,
                    source: .observedVariant
                )
            }
        )

        return variants
    }

    private nonisolated func shouldBlock(variant: DictionaryMatchVariant, entry: DictionaryEntry) -> Bool {
        guard entry.groupID == nil else { return false }
        return blockedGlobalMatchKeys.contains(variant.normalizedText)
    }

    private nonisolated func exactCandidates(
        for entry: DictionaryEntry,
        variant: DictionaryMatchVariant,
        text: String,
        normalizedMapping: DictionaryNormalizedMapping
    ) -> [DictionaryMatchCandidate] {
        guard !variant.normalizedText.isEmpty else { return [] }
        guard normalizedMapping.text.contains(variant.normalizedText) else { return [] }
        let profile = dictionaryScriptProfile(for: variant.text)
        let requireTokenBoundaries = !profile.containsCJKLike || profile.containsLatin || profile.containsDigit
        let ranges = dictionaryExactNormalizedMatchRanges(
            in: normalizedMapping,
            normalizedNeedle: variant.normalizedText,
            requireTokenBoundaries: requireTokenBoundaries
        )
        let fullText = text as NSString

        return ranges.map { range in
            let matchedText = fullText.substring(with: range)
            let reason = exactReason(for: entry, variant: variant, matchedText: matchedText)
            return DictionaryMatchCandidate(
                entryID: entry.id,
                term: entry.term,
                matchedText: matchedText,
                normalizedMatchedText: variant.normalizedText,
                score: reason == .exactTerm ? 1.0 : 0.995,
                reason: reason,
                source: variant.source.source,
                matchRange: range
            )
        }
    }

    private nonisolated func exactReason(
        for entry: DictionaryEntry,
        variant: DictionaryMatchVariant,
        matchedText: String
    ) -> DictionaryMatchReason {
        switch variant.source {
        case .replacementTerm, .observedVariant:
            return .exactVariant
        case .term:
            return matchedText == entry.term ? .exactTerm : .exactWindow
        }
    }

    private nonisolated func bestFuzzyCandidate(
        for entry: DictionaryEntry,
        variant: DictionaryMatchVariant,
        text: String,
        rawTokens: [DictionaryToken]
    ) -> DictionaryMatchCandidate? {
        guard !variant.normalizedText.isEmpty else { return nil }
        let scriptProfile = dictionaryScriptProfile(for: variant.text)
        guard allowsFuzzyMatch(for: scriptProfile) else { return nil }

        let variantTokenCount = max(1, tokenize(variant.text).count)
        let windowSizes = Array(
            Set([
                variantTokenCount,
                max(1, variantTokenCount - 1),
                variantTokenCount + 1
            ])
        ).sorted()
        var best: DictionaryMatchCandidate?

        for windowSize in windowSizes {
            guard windowSize <= rawTokens.count else { continue }
            for start in 0...(rawTokens.count - windowSize) {
                let window = Array(rawTokens[start..<(start + windowSize)])
                let rawWindow = (text as NSString).substring(
                    with: NSRange(
                        location: window[0].range.location,
                        length: (window[window.count - 1].range.location + window[window.count - 1].range.length) - window[0].range.location
                    )
                )
                let normalizedWindow = window.map(\.normalized).joined(separator: " ")
                guard !normalizedWindow.isEmpty else { continue }
                guard normalizedWindow != variant.normalizedText else { continue }

                let distance = levenshteinDistance(lhs: normalizedWindow, rhs: variant.normalizedText)
                let maxLength = max(normalizedWindow.count, variant.normalizedText.count)
                guard maxLength >= minimumFuzzyLength(for: scriptProfile) else { continue }
                let threshold = fuzzyThreshold(for: scriptProfile, maxLength: maxLength)
                guard distance <= threshold else { continue }

                let score = 1.0 - (Double(distance) / Double(maxLength))
                guard score >= minimumFuzzyScore(for: scriptProfile) else { continue }

                let candidate = DictionaryMatchCandidate(
                    entryID: entry.id,
                    term: entry.term,
                    matchedText: rawWindow,
                    normalizedMatchedText: normalizedWindow,
                    score: score,
                    reason: .fuzzyWindow,
                    source: variant.source.source,
                    matchRange: NSRange(
                        location: window[0].range.location,
                        length: (window[window.count - 1].range.location + window[window.count - 1].range.length) - window[0].range.location
                    )
                )

                if best == nil || best!.score < candidate.score {
                    best = candidate
                }
            }
        }

        return best
    }

    private nonisolated func tokenize(_ text: String) -> [DictionaryToken] {
        var tokens: [DictionaryToken] = []
        var currentStart: String.Index?
        var current = ""

        func flush(until endIndex: String.Index) {
            let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
            defer {
                current = ""
                currentStart = nil
            }

            guard let start = currentStart, !trimmed.isEmpty else { return }
            let normalized = DictionaryStore.normalizeTerm(trimmed)
            guard !normalized.isEmpty else { return }

            tokens.append(
                DictionaryToken(
                    raw: trimmed,
                    normalized: normalized,
                    range: NSRange(start..<endIndex, in: text)
                )
            )
        }

        var index = text.startIndex
        while index < text.endIndex {
            let nextIndex = text.index(after: index)
            let scalar = text[index].unicodeScalars.first

            if let scalar, dictionaryIsWordScalar(scalar) {
                if currentStart == nil {
                    currentStart = index
                }
                current.append(text[index])
            } else {
                flush(until: index)
            }

            index = nextIndex
        }

        flush(until: text.endIndex)
        return tokens
    }

    private nonisolated func allowsFuzzyMatch(for profile: DictionaryScriptProfile) -> Bool {
        profile.isLatinLike || profile.isMixedScript
    }

    private nonisolated func minimumFuzzyLength(for profile: DictionaryScriptProfile) -> Int {
        profile.isMixedScript ? 5 : 4
    }

    private nonisolated func minimumFuzzyScore(for profile: DictionaryScriptProfile) -> Double {
        profile.isMixedScript ? 0.96 : 0.90
    }

    private nonisolated func fuzzyThreshold(for profile: DictionaryScriptProfile, maxLength: Int) -> Int {
        if profile.isMixedScript {
            return maxLength >= 10 ? 2 : 1
        }
        return max(1, min(2, maxLength / 6))
    }

    private nonisolated func levenshteinDistance(lhs: String, rhs: String) -> Int {
        let lhsChars = Array(lhs)
        let rhsChars = Array(rhs)
        guard !lhsChars.isEmpty else { return rhsChars.count }
        guard !rhsChars.isEmpty else { return lhsChars.count }

        var previous = Array(0...rhsChars.count)
        for (i, lhsChar) in lhsChars.enumerated() {
            var current = [i + 1]
            for (j, rhsChar) in rhsChars.enumerated() {
                let cost = lhsChar == rhsChar ? 0 : 1
                current.append(
                    min(
                        current[j] + 1,
                        previous[j + 1] + 1,
                        previous[j] + cost
                    )
                )
            }
            previous = current
        }
        return previous[rhsChars.count]
    }
}
