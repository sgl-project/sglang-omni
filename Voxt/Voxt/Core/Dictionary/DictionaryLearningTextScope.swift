import Foundation

extension AutomaticDictionaryLearningMonitor {
    private struct WhitespaceCollapsedProjection {
        let text: String
        let originalStarts: [Int]
        let originalEnds: [Int]
    }

    static func observationScopedText(
        insertedText rawInsertedText: String,
        baselineText rawBaselineText: String,
        currentText rawCurrentText: String
    ) -> String {
        let insertedText = rawInsertedText.trimmingCharacters(in: .whitespacesAndNewlines)
        let baselineText = rawBaselineText.trimmingCharacters(in: .whitespacesAndNewlines)
        let currentText = rawCurrentText.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !insertedText.isEmpty, !baselineText.isEmpty else {
            return sanitizeScopedLineText(currentText)
        }

        guard let insertedRange = insertedRange(of: insertedText, in: baselineText) else {
            return sanitizeObservationScopedText(
                insertedText: insertedText,
                baselineScopedText: insertedText,
                currentText: currentText
            )
        }

        let baselineScopedText = scopedTextForInsertedRange(
            in: baselineText,
            insertedRange: insertedRange,
            fallback: insertedText
        )
        return sanitizeObservationScopedText(
            insertedText: insertedText,
            baselineScopedText: baselineScopedText,
            currentText: currentText
        )
    }

    static func scopedTextForInsertedRange(
        in text: String,
        insertedRange: NSRange,
        fallback: String
    ) -> String {
        let textNSString = text as NSString
        let lineRange = textNSString.lineRange(for: insertedRange)
        let lineText = sanitizeScopedLineText(
            textNSString.substring(with: lineRange)
        )
        return lineText.isEmpty ? fallback : lineText
    }

    static func extractFinalScopedText(
        insertedText: String,
        baselineScopedText: String,
        finalText: String
    ) -> String {
        if !finalText.contains("\n") {
            return sanitizeScopedLineText(finalText)
        }

        if let bestLine = bestMatchingLine(
            primaryTarget: insertedText,
            secondaryTarget: baselineScopedText,
            within: finalText
        ) {
            return bestLine
        }

        return sanitizeScopedLineText(finalText)
    }

    private static func sanitizeObservationScopedText(
        insertedText: String,
        baselineScopedText: String,
        currentText: String
    ) -> String {
        if !currentText.contains("\n") {
            return sanitizeScopedLineText(currentText)
        }

        if let bestLine = bestMatchingLine(
            primaryTarget: insertedText,
            secondaryTarget: baselineScopedText,
            within: currentText
        ) {
            return bestLine
        }

        return ""
    }

    static func insertedRange(of insertedText: String, in baselineText: String) -> NSRange? {
        let searchRange = NSRange(location: 0, length: (baselineText as NSString).length)
        let match = NSRegularExpression.escapedPattern(for: insertedText)
        guard let regex = try? NSRegularExpression(pattern: match, options: [.caseInsensitive]) else {
            return relaxedInsertedRange(of: insertedText, in: baselineText)
        }
        if let exact = regex.firstMatch(in: baselineText, options: [], range: searchRange)?.range {
            return exact
        }
        return relaxedInsertedRange(of: insertedText, in: baselineText)
    }

    private static func relaxedInsertedRange(of insertedText: String, in baselineText: String) -> NSRange? {
        let baselineProjection = collapseWhitespace(in: baselineText)
        let insertedProjection = collapseWhitespace(in: insertedText)

        guard !baselineProjection.text.isEmpty, !insertedProjection.text.isEmpty else {
            return nil
        }
        guard let matchedRange = baselineProjection.text.range(
            of: insertedProjection.text,
            options: [.caseInsensitive]
        ) else {
            return nil
        }

        let lowerBound = baselineProjection.text.distance(
            from: baselineProjection.text.startIndex,
            to: matchedRange.lowerBound
        )
        let upperBound = baselineProjection.text.distance(
            from: baselineProjection.text.startIndex,
            to: matchedRange.upperBound
        )
        guard lowerBound < baselineProjection.originalStarts.count,
              upperBound > 0,
              upperBound - 1 < baselineProjection.originalEnds.count else {
            return nil
        }

        let location = baselineProjection.originalStarts[lowerBound]
        let end = baselineProjection.originalEnds[upperBound - 1]
        return NSRange(location: location, length: max(0, end - location))
    }

    private static func collapseWhitespace(in text: String) -> WhitespaceCollapsedProjection {
        var characters: [Character] = []
        var originalStarts: [Int] = []
        var originalEnds: [Int] = []
        var utf16Location = 0

        for character in text {
            let scalarString = String(character)
            let utf16Length = (scalarString as NSString).length
            defer { utf16Location += utf16Length }

            if scalarString.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                continue
            }

            characters.append(character)
            originalStarts.append(utf16Location)
            originalEnds.append(utf16Location + utf16Length)
        }

        return WhitespaceCollapsedProjection(
            text: String(characters),
            originalStarts: originalStarts,
            originalEnds: originalEnds
        )
    }

    static func union(lhs: NSRange, rhs: NSRange, upperBound: Int) -> NSRange {
        let start = min(lhs.location, rhs.location)
        let end = min(
            max(lhs.location + lhs.length, rhs.location + rhs.length),
            upperBound
        )
        return NSRange(location: start, length: max(end - start, 0))
    }

    static func contextualSnippet(
        in text: String,
        focusRange: NSRange,
        radius: Int
    ) -> String {
        let characters = Array(text)
        guard !characters.isEmpty else { return "" }

        let start = max(0, focusRange.location - radius)
        let end = min(characters.count, focusRange.location + max(focusRange.length, 1) + radius)
        guard start < end else { return text }

        var snippet = String(characters[start..<end]).trimmingCharacters(in: .whitespacesAndNewlines)
        if start > 0 {
            snippet = "…" + snippet
        }
        if end < characters.count {
            snippet += "…"
        }
        return snippet
    }

    static func bestMatchingLine(
        primaryTarget: String,
        secondaryTarget: String?,
        within finalText: String
    ) -> String? {
        let finalNSString = finalText as NSString
        let searchRange = NSRange(location: 0, length: finalNSString.length)
        var bestLine: String?
        var bestScore = Int.min

        finalText.enumerateSubstrings(in: Range(searchRange, in: finalText)!, options: [.byLines, .substringNotRequired]) {
            _, substringRange, _, _ in
            let rawLine = String(finalText[substringRange])
            let line = sanitizeScopedLineText(rawLine)
            guard !line.isEmpty else { return }

            let score = bestLineScore(
                line: line,
                primaryTarget: primaryTarget,
                secondaryTarget: secondaryTarget
            )
            if score > bestScore {
                bestScore = score
                bestLine = line
            }
        }

        let minimumUsefulScore = minimumUsefulLineScore(
            primaryTarget: primaryTarget,
            secondaryTarget: secondaryTarget
        )
        guard bestScore >= minimumUsefulScore else {
            return nil
        }
        return bestLine
    }

    private static func bestLineScore(
        line: String,
        primaryTarget: String,
        secondaryTarget: String?
    ) -> Int {
        let primaryScore = lineSimilarityScore(primaryTarget, line)
        let secondaryScore = secondaryTarget.map { lineSimilarityScore($0, line) } ?? Int.min
        let score = max(primaryScore, secondaryScore)
        if isObservationNoiseLine(line) {
            return score - max(6, line.count / 4)
        }
        return score
    }

    private static func minimumUsefulLineScore(
        primaryTarget: String,
        secondaryTarget: String?
    ) -> Int {
        let primaryCount = normalizedLineMatchingText(primaryTarget).count
        let secondaryCount = secondaryTarget.map { normalizedLineMatchingText($0).count } ?? 0
        return max(6, max(primaryCount, secondaryCount) / 4)
    }

    private static func lineSimilarityScore(_ lhs: String, _ rhs: String) -> Int {
        let lhsChars = Array(normalizedLineMatchingText(lhs))
        let rhsChars = Array(normalizedLineMatchingText(rhs))
        guard !lhsChars.isEmpty, !rhsChars.isEmpty else {
            return 0
        }

        var prefix = 0
        while prefix < lhsChars.count,
              prefix < rhsChars.count,
              lhsChars[prefix] == rhsChars[prefix] {
            prefix += 1
        }

        var suffix = 0
        while suffix < lhsChars.count - prefix,
              suffix < rhsChars.count - prefix,
              lhsChars[lhsChars.count - 1 - suffix] == rhsChars[rhsChars.count - 1 - suffix] {
            suffix += 1
        }

        let commonSubstring = longestCommonSubstringLength(lhsChars, rhsChars)
        let lhsText = String(lhsChars)
        let rhsText = String(rhsChars)
        let containmentBonus: Int
        if lhsText.contains(rhsText) || rhsText.contains(lhsText) {
            containmentBonus = min(lhsChars.count, rhsChars.count)
        } else {
            containmentBonus = 0
        }

        return prefix + suffix + (commonSubstring * 2) + containmentBonus
    }

    private static func longestCommonSubstringLength(
        _ lhsChars: [Character],
        _ rhsChars: [Character]
    ) -> Int {
        guard !lhsChars.isEmpty, !rhsChars.isEmpty else {
            return 0
        }

        var previous = Array(repeating: 0, count: rhsChars.count + 1)
        var longest = 0

        for lhsIndex in 1...lhsChars.count {
            var current = Array(repeating: 0, count: rhsChars.count + 1)
            for rhsIndex in 1...rhsChars.count {
                if lhsChars[lhsIndex - 1] == rhsChars[rhsIndex - 1] {
                    current[rhsIndex] = previous[rhsIndex - 1] + 1
                    longest = max(longest, current[rhsIndex])
                }
            }
            previous = current
        }

        return longest
    }

    private static func normalizedLineMatchingText(_ text: String) -> String {
        sanitizeScopedLineText(text)
            .lowercased()
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func isObservationNoiseLine(_ text: String) -> Bool {
        let line = sanitizeScopedLineText(text)
        guard !line.isEmpty else { return true }

        if line.hasPrefix("zsh:") || line.hasPrefix("bash:") || line.hasPrefix("fish:") {
            return true
        }

        if line.contains("command not found:") || line.contains("no such file or directory") {
            return true
        }

        if line.hasPrefix("~/") || line.hasPrefix("/") {
            return true
        }

        return false
    }

    private static func sanitizeScopedLineText(_ text: String) -> String {
        var line = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if line.hasPrefix("> ") {
            line.removeFirst(2)
        } else if line == ">" {
            line = ""
        }
        return line.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
