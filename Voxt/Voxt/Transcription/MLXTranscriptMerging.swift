// Pure transcription decisions; no capture, task, or model ownership.

import Foundation

extension MLXTranscriptionPlanning {
    nonisolated static func mergeSequentialTranscript(base: String, next: String) -> String {
        sequentialTranscriptMergeResult(base: base, next: next).text
    }

    nonisolated static func sequentialTranscriptMergeResult(base: String, next: String) -> MLXSequentialTranscriptMergeResult {
        let left = base.trimmingCharacters(in: .whitespacesAndNewlines)
        let right = next.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !left.isEmpty else {
            return MLXSequentialTranscriptMergeResult(text: right, overlapCount: 0)
        }
        guard !right.isEmpty else {
            return MLXSequentialTranscriptMergeResult(text: left, overlapCount: 0)
        }
        if left.hasSuffix(right) {
            return MLXSequentialTranscriptMergeResult(text: left, overlapCount: right.count)
        }
        if right.hasPrefix(left) {
            return MLXSequentialTranscriptMergeResult(text: right, overlapCount: left.count)
        }

        let leftLast = left.unicodeScalars.last
        let rightFirst = right.unicodeScalars.first
        let minimumOverlapCount: Int
        if let leftLast, let rightFirst,
           CharacterSet.alphanumerics.contains(leftLast),
           CharacterSet.alphanumerics.contains(rightFirst) {
            minimumOverlapCount = 3
        } else {
            minimumOverlapCount = 2
        }

        let overlapCount = suffixPrefixOverlapCount(left, right)
        if overlapCount >= minimumOverlapCount {
            let rightChars = Array(right)
            return MLXSequentialTranscriptMergeResult(
                text: left + String(rightChars.dropFirst(overlapCount)),
                overlapCount: overlapCount
            )
        }

        let shouldInsertSpace: Bool
        if let leftLast, let rightFirst {
            shouldInsertSpace =
                CharacterSet.alphanumerics.contains(leftLast) &&
                CharacterSet.alphanumerics.contains(rightFirst)
        } else {
            shouldInsertSpace = true
        }
        return MLXSequentialTranscriptMergeResult(
            text: shouldInsertSpace ? "\(left) \(right)" : left + right,
            overlapCount: 0
        )
    }

    nonisolated static func mergedHiddenPostStopPreview(base: String, candidate: String) -> String {
        let stableBase = base.trimmingCharacters(in: .whitespacesAndNewlines)
        let stableCandidate = candidate.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !stableBase.isEmpty else { return stableCandidate }
        guard !stableCandidate.isEmpty else { return stableBase }
        if stableBase == stableCandidate {
            return stableCandidate
        }
        let maxTrustedCandidateCount = stableBase.count + max(48, stableBase.count / 3)
        if stableCandidate.count > maxTrustedCandidateCount,
           !stableCandidate.contains(stableBase) {
            return stableBase
        }
        if stableBase.contains(stableCandidate) {
            return stableBase
        }
        if stableCandidate.contains(stableBase) {
            return stableCandidate
        }
        if endsWithSentenceBoundary(stableBase),
           let stitched = stitchedSentenceContinuation(
               base: stableBase,
               candidate: stableCandidate,
               minimumOverlap: 10,
               maximumCandidatePrefixNoise: 4
           ) {
            return stitched
        }

        let sharedPrefix = longestCommonPrefix(stableBase, stableCandidate).count
        let suffixPrefixOverlap = suffixPrefixOverlapCount(stableBase, stableCandidate)
        if suffixPrefixOverlap == 0 && sharedPrefix < 8 {
            if !hasSharedWindow(stableBase, stableCandidate, minLength: 12),
               endsWithSentenceBoundary(stableBase) {
                let combined = stableBase + stableCandidate
                let maxSafeCombinedCount = stableBase.count + stableCandidate.count + 4
                if combined.count <= maxSafeCombinedCount {
                    return combined
                }
            }
            return stableBase.count >= stableCandidate.count ? stableBase : stableCandidate
        }

        let merged = mergeStablePrefix(stableBase, candidate: stableCandidate)
        let growthBudget = max(16, min(stableBase.count, stableCandidate.count) / 4)
        let maxSafeCount = max(stableBase.count, stableCandidate.count) + growthBudget
        if merged.count > maxSafeCount {
            return stableBase.count >= stableCandidate.count ? stableBase : stableCandidate
        }

        return merged
    }

    nonisolated static func mergeStablePrefix(_ stable: String, candidate: String) -> String {
        guard !stable.isEmpty else { return candidate }
        guard !candidate.isEmpty else { return stable }
        if candidate.hasPrefix(stable) {
            return candidate
        }

        let stableChars = Array(stable)
        let candidateChars = Array(candidate)
        let maxOverlap = min(stableChars.count, candidateChars.count)

        for overlap in stride(from: maxOverlap, through: 1, by: -1) {
            let stableSuffix = String(stableChars.suffix(overlap))
            let candidatePrefix = String(candidateChars.prefix(overlap))
            if stableSuffix == candidatePrefix {
                return stable + String(candidateChars.dropFirst(overlap))
            }
        }

        return stable + " " + candidate
    }

    nonisolated static func longestCommonPrefix(_ lhs: String, _ rhs: String) -> String {
        var leftIndex = lhs.startIndex
        var rightIndex = rhs.startIndex

        while leftIndex < lhs.endIndex, rightIndex < rhs.endIndex, lhs[leftIndex] == rhs[rightIndex] {
            leftIndex = lhs.index(after: leftIndex)
            rightIndex = rhs.index(after: rightIndex)
        }

        return String(lhs[..<leftIndex])
    }

    private nonisolated static func suffixPrefixOverlapCount(_ lhs: String, _ rhs: String) -> Int {
        let left = Array(lhs)
        let right = Array(rhs)
        let maxOverlap = min(left.count, right.count)

        for overlap in stride(from: maxOverlap, through: 1, by: -1) {
            if Array(left.suffix(overlap)) == Array(right.prefix(overlap)) {
                return overlap
            }
        }

        return 0
    }

    private nonisolated static func hasSharedWindow(_ lhs: String, _ rhs: String, minLength: Int) -> Bool {
        guard min(lhs.count, rhs.count) >= minLength else { return false }
        let shorter = lhs.count <= rhs.count ? lhs : rhs
        let longer = lhs.count <= rhs.count ? rhs : lhs
        let chars = Array(shorter)
        let upperBound = chars.count - minLength
        guard upperBound >= 0 else { return false }

        for start in 0...upperBound {
            let window = String(chars[start..<(start + minLength)])
            if longer.contains(window) {
                return true
            }
        }
        return false
    }

    private nonisolated static func stitchedSentenceContinuation(
        base: String,
        candidate: String,
        minimumOverlap: Int,
        maximumCandidatePrefixNoise: Int
    ) -> String? {
        let baseChars = Array(base)
        let candidateChars = Array(candidate)
        guard baseChars.count >= minimumOverlap, candidateChars.count >= minimumOverlap else {
            return nil
        }

        let maxNoise = min(maximumCandidatePrefixNoise, max(candidateChars.count - minimumOverlap, 0))
        for prefixNoise in 0...maxNoise {
            let remaining = candidateChars.count - prefixNoise
            guard remaining >= minimumOverlap else { continue }
            let maxOverlap = min(baseChars.count, remaining)
            for overlap in stride(from: maxOverlap, through: minimumOverlap, by: -1) {
                let baseSuffix = Array(baseChars.suffix(overlap))
                let candidateSlice = Array(candidateChars[prefixNoise..<(prefixNoise + overlap)])
                if baseSuffix == candidateSlice {
                    let continuationStart = prefixNoise + overlap
                    let continuation = continuationStart < candidateChars.count
                        ? String(candidateChars[continuationStart...])
                        : ""
                    return base + continuation
                }
            }
        }

        return nil
    }

    private nonisolated static func endsWithSentenceBoundary(_ text: String) -> Bool {
        guard let last = text.trimmingCharacters(in: .whitespacesAndNewlines).last else {
            return false
        }
        return "。！？!?；;：:）)]」』\"”".contains(last)
    }
}
