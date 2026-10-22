import Foundation

// The live path uses prefix/suffix change windows. Token/LCS comparison remains
// only for detecting deletion-only edits that need more observation time.
extension AutomaticDictionaryLearningMonitor {
    struct ChangeWindow: Equatable {
        let baselineRange: NSRange
        let finalRange: NSRange
        let baselineFragment: String
        let finalFragment: String
        let editRatio: Double
        let hasMeaningfulChange: Bool
        let containsDeletionOnlyChangeGroup: Bool
    }

    private struct SemanticToken: Equatable {
        let text: String
        let normalizedText: String
        let start: Int
        let end: Int
    }

    private struct SemanticChangeGroup: Equatable {
        let baselineStartToken: Int?
        let baselineEndToken: Int?
        let finalStartToken: Int?
        let finalEndToken: Int?
    }

    private struct SemanticChangeSummary: Equatable {
        let baselineRange: NSRange
        let finalRange: NSRange
        let baselineFragment: String
        let finalFragment: String
        let baselineChangedCharacterCount: Int
        let finalChangedCharacterCount: Int
        let containsDeletionOnlyChangeGroup: Bool
    }

    private enum TokenKind: Equatable {
        case none
        case latinOrNumber
        case han
    }

    static func changeIntersectsInsertedText(
        baselineChangeRange: NSRange,
        insertedRange: NSRange
    ) -> Bool {
        if baselineChangeRange.length == 0 {
            return baselineChangeRange.location > insertedRange.location
                && baselineChangeRange.location < insertedRange.location + insertedRange.length
        }
        return NSIntersectionRange(baselineChangeRange, insertedRange).length > 0
    }

    static func changedRangeWindow(
        baselineText: String,
        finalText: String
    ) -> ChangeWindow {
        let summary = typefluxChangeSummary(
            baselineText: baselineText,
            finalText: finalText
        )
        let baselineFragment = summary.baselineFragment
        let finalFragment = summary.finalFragment
        let hasMeaningfulChange = !candidateTerms(
            oldFragment: baselineFragment,
            newFragment: finalFragment
        ).isEmpty
        let baselineChars = Array(baselineText)
        let finalChars = Array(finalText)
        let editRatio = Double(max(summary.baselineChangedCharacterCount, summary.finalChangedCharacterCount))
            / Double(Swift.max(Swift.max(baselineChars.count, finalChars.count), 1))

        return ChangeWindow(
            baselineRange: summary.baselineRange,
            finalRange: summary.finalRange,
            baselineFragment: baselineFragment,
            finalFragment: finalFragment,
            editRatio: editRatio,
            hasMeaningfulChange: hasMeaningfulChange,
            containsDeletionOnlyChangeGroup: summary.containsDeletionOnlyChangeGroup
                || containsSemanticDeletionOnlyChangeGroup(baselineText: baselineText, finalText: finalText)
        )
    }

    private static func typefluxChangeSummary(
        baselineText: String,
        finalText: String
    ) -> SemanticChangeSummary {
        guard baselineText != finalText else {
            return SemanticChangeSummary(
                baselineRange: NSRange(location: 0, length: 0),
                finalRange: NSRange(location: 0, length: 0),
                baselineFragment: "",
                finalFragment: "",
                baselineChangedCharacterCount: 0,
                finalChangedCharacterCount: 0,
                containsDeletionOnlyChangeGroup: false
            )
        }

        let baselineCharacters = Array(baselineText)
        let finalCharacters = Array(finalText)
        let sharedPrefixCount = commonPrefixCount(baselineCharacters, finalCharacters)
        let sharedSuffixCount = commonSuffixCount(
            baselineCharacters,
            finalCharacters,
            excludingSharedPrefix: sharedPrefixCount
        )

        let baselineEnd = max(sharedPrefixCount, baselineCharacters.count - sharedSuffixCount)
        let finalEnd = max(sharedPrefixCount, finalCharacters.count - sharedSuffixCount)
        let baselineRange = expandChangedRange(
            in: baselineCharacters,
            start: sharedPrefixCount,
            end: baselineEnd
        )
        let finalRange = expandChangedRange(
            in: finalCharacters,
            start: sharedPrefixCount,
            end: finalEnd
        )

        let baselineFragment = String(baselineCharacters[baselineRange])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let finalFragment = String(finalCharacters[finalRange])
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let containsDeletionOnlyChangeGroup = !baselineFragment.isEmpty && finalFragment.isEmpty

        return SemanticChangeSummary(
            baselineRange: NSRange(location: baselineRange.lowerBound, length: baselineRange.count),
            finalRange: NSRange(location: finalRange.lowerBound, length: finalRange.count),
            baselineFragment: baselineFragment,
            finalFragment: finalFragment,
            baselineChangedCharacterCount: baselineRange.count,
            finalChangedCharacterCount: finalRange.count,
            containsDeletionOnlyChangeGroup: containsDeletionOnlyChangeGroup
        )
    }

    static func editRatio(
        inserted: String,
        baseline: String,
        final: String,
        maxLengthForExactComputation: Int = 2000
    ) -> Double {
        let insertedNorm = normalizeVocabularyCandidate(inserted)
        guard !insertedNorm.isEmpty else { return 0 }
        let baselineNorm = normalizeVocabularyCandidate(baseline)
        let finalNorm = normalizeVocabularyCandidate(final)
        if baselineNorm == finalNorm { return 0 }
        if max(baselineNorm.count, finalNorm.count) > maxLengthForExactComputation {
            return 1
        }
        let distance = levenshteinDistance(Array(baselineNorm), Array(finalNorm))
        return Double(distance) / Double(insertedNorm.count)
    }

    private static func levenshteinDistance(_ lhs: [Character], _ rhs: [Character]) -> Int {
        if lhs.isEmpty { return rhs.count }
        if rhs.isEmpty { return lhs.count }

        var previous = Array(0...rhs.count)
        var current = Array(repeating: 0, count: rhs.count + 1)

        for i in 1...lhs.count {
            current[0] = i
            for j in 1...rhs.count {
                let cost = lhs[i - 1] == rhs[j - 1] ? 0 : 1
                current[j] = Swift.min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost
                )
            }
            swap(&previous, &current)
        }

        return previous[rhs.count]
    }

    static func isPureAppendAfterBaseline(baseline: String, final: String) -> Bool {
        guard final.hasPrefix(baseline), final.count > baseline.count else {
            return false
        }
        guard let lastBaselineCharacter = baseline.last,
              let firstSuffixCharacter = final.dropFirst(baseline.count).first else {
            return false
        }

        let baselineKind = tokenKind(for: lastBaselineCharacter)
        let suffixKind = tokenKind(for: firstSuffixCharacter)
        if baselineKind != .none, baselineKind == suffixKind {
            return false
        }
        return true
    }

    private static func containsSemanticDeletionOnlyChangeGroup(
        baselineText: String,
        finalText: String
    ) -> Bool {
        let baselineTokens = semanticTokens(in: baselineText)
        let finalTokens = semanticTokens(in: finalText)
        let matches = longestCommonSubsequenceMatches(
            baselineTokens: baselineTokens,
            finalTokens: finalTokens
        )
        return semanticChangeGroups(
            baselineCount: baselineTokens.count,
            finalCount: finalTokens.count,
            matches: matches
        ).contains { group in
            group.baselineStartToken != nil && group.finalStartToken == nil
        }
    }

    private static func commonPrefixCount(_ lhs: [Character], _ rhs: [Character]) -> Int {
        let limit = min(lhs.count, rhs.count)
        var count = 0
        while count < limit, lhs[count] == rhs[count] {
            count += 1
        }
        return count
    }

    private static func commonSuffixCount(
        _ lhs: [Character],
        _ rhs: [Character],
        excludingSharedPrefix sharedPrefixCount: Int
    ) -> Int {
        let lhsRemaining = lhs.count - sharedPrefixCount
        let rhsRemaining = rhs.count - sharedPrefixCount
        let limit = min(lhsRemaining, rhsRemaining)
        guard limit > 0 else { return 0 }

        var count = 0
        while count < limit,
              lhs[lhs.count - 1 - count] == rhs[rhs.count - 1 - count] {
            count += 1
        }
        return count
    }

    private static func expandChangedRange(
        in characters: [Character],
        start: Int,
        end: Int
    ) -> Range<Int> {
        guard !characters.isEmpty else { return start..<end }

        var lowerBound = max(0, min(start, characters.count))
        var upperBound = max(lowerBound, min(end, characters.count))

        let anchorIndex: Int?
        if lowerBound < upperBound {
            anchorIndex = lowerBound
        } else if lowerBound < characters.count, tokenKind(for: characters[lowerBound]) != .none {
            anchorIndex = lowerBound
            upperBound = lowerBound + 1
        } else if lowerBound > 0, tokenKind(for: characters[lowerBound - 1]) != .none {
            anchorIndex = lowerBound - 1
            lowerBound -= 1
            upperBound = max(upperBound, lowerBound + 1)
        } else {
            anchorIndex = nil
        }

        guard let anchorIndex else { return lowerBound..<upperBound }
        let kind = tokenKind(for: characters[anchorIndex])
        guard kind != .none else { return lowerBound..<upperBound }

        while lowerBound > 0, tokenKind(for: characters[lowerBound - 1]) == kind {
            lowerBound -= 1
        }

        while upperBound < characters.count, tokenKind(for: characters[upperBound]) == kind {
            upperBound += 1
        }

        return lowerBound..<upperBound
    }

    private static func tokenKind(for character: Character) -> TokenKind {
        if isLatinOrNumberTokenCharacter(character) {
            return .latinOrNumber
        }
        if isHanCharacter(character) {
            return .han
        }
        return .none
    }

    private static func isLatinOrNumberTokenCharacter(_ character: Character) -> Bool {
        guard character.unicodeScalars.count == 1, let scalar = character.unicodeScalars.first else {
            return false
        }

        if CharacterSet.alphanumerics.contains(scalar) {
            return true
        }

        return "._+-'".unicodeScalars.contains(scalar)
    }

    private static func isHanCharacter(_ character: Character) -> Bool {
        character.unicodeScalars.contains { (0x4E00...0x9FFF).contains($0.value) }
    }

    private static func semanticTokens(in text: String) -> [SemanticToken] {
        let characters = Array(text)
        var index = 0
        var tokens: [SemanticToken] = []

        while index < characters.count {
            let character = characters[index]

            if character.isWhitespace || isPunctuationCharacter(character) {
                index += 1
                continue
            }

            let start = index

            if isASCIIWordCharacter(character) {
                index += 1
                while index < characters.count, isASCIIWordCharacter(characters[index]) {
                    index += 1
                }
            } else if isIdeographicCharacter(character) {
                index += 1
            } else {
                index += 1
                while index < characters.count,
                      !characters[index].isWhitespace,
                      !isPunctuationCharacter(characters[index]),
                      !isASCIIWordCharacter(characters[index]),
                      !isIdeographicCharacter(characters[index]) {
                    index += 1
                }
            }

            let tokenText = String(characters[start..<index])
            tokens.append(
                SemanticToken(
                    text: tokenText,
                    normalizedText: tokenText,
                    start: start,
                    end: index
                )
            )
        }

        return tokens
    }

    private static func longestCommonSubsequenceMatches(
        baselineTokens: [SemanticToken],
        finalTokens: [SemanticToken]
    ) -> [(Int, Int)] {
        let baselineCount = baselineTokens.count
        let finalCount = finalTokens.count
        var dp = Array(
            repeating: Array(repeating: 0, count: finalCount + 1),
            count: baselineCount + 1
        )

        if baselineCount > 0, finalCount > 0 {
            for baselineIndex in stride(from: baselineCount - 1, through: 0, by: -1) {
                for finalIndex in stride(from: finalCount - 1, through: 0, by: -1) {
                    if baselineTokens[baselineIndex].normalizedText == finalTokens[finalIndex].normalizedText {
                        dp[baselineIndex][finalIndex] = dp[baselineIndex + 1][finalIndex + 1] + 1
                    } else {
                        dp[baselineIndex][finalIndex] = max(
                            dp[baselineIndex + 1][finalIndex],
                            dp[baselineIndex][finalIndex + 1]
                        )
                    }
                }
            }
        }

        var matches: [(Int, Int)] = []
        var baselineIndex = 0
        var finalIndex = 0

        while baselineIndex < baselineCount, finalIndex < finalCount {
            if baselineTokens[baselineIndex].normalizedText == finalTokens[finalIndex].normalizedText {
                matches.append((baselineIndex, finalIndex))
                baselineIndex += 1
                finalIndex += 1
            } else if dp[baselineIndex + 1][finalIndex] >= dp[baselineIndex][finalIndex + 1] {
                baselineIndex += 1
            } else {
                finalIndex += 1
            }
        }

        return matches
    }

    private static func semanticChangeGroups(
        baselineCount: Int,
        finalCount: Int,
        matches: [(Int, Int)]
    ) -> [SemanticChangeGroup] {
        var groups: [SemanticChangeGroup] = []
        var previousBaselineIndex = -1
        var previousFinalIndex = -1
        let sentinelMatches = matches + [(baselineCount, finalCount)]

        for (nextBaselineIndex, nextFinalIndex) in sentinelMatches {
            let baselineStart = previousBaselineIndex + 1
            let baselineEnd = nextBaselineIndex - 1
            let finalStart = previousFinalIndex + 1
            let finalEnd = nextFinalIndex - 1

            if baselineStart <= baselineEnd || finalStart <= finalEnd {
                groups.append(
                    SemanticChangeGroup(
                        baselineStartToken: baselineStart <= baselineEnd ? baselineStart : nil,
                        baselineEndToken: baselineStart <= baselineEnd ? baselineEnd : nil,
                        finalStartToken: finalStart <= finalEnd ? finalStart : nil,
                        finalEndToken: finalStart <= finalEnd ? finalEnd : nil
                    )
                )
            }

            previousBaselineIndex = nextBaselineIndex
            previousFinalIndex = nextFinalIndex
        }

        return groups
    }

    static func isASCIIWordCharacter(_ character: Character) -> Bool {
        character.unicodeScalars.allSatisfy { scalar in
            scalar.isASCII && CharacterSet.alphanumerics.contains(scalar)
                || CharacterSet(charactersIn: "_-").contains(scalar)
        }
    }

    static func isIdeographicCharacter(_ character: Character) -> Bool {
        character.unicodeScalars.allSatisfy { $0.properties.isIdeographic }
    }

    private static func isPunctuationCharacter(_ character: Character) -> Bool {
        character.unicodeScalars.allSatisfy {
            CharacterSet.punctuationCharacters.contains($0)
                || CharacterSet.symbols.contains($0)
        }
    }
}
