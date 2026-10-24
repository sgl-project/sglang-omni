// LiveTranscriptSegmentationState.swift
// Provides Live Transcript Segmentation State for transcription processing.

import Foundation

struct LiveTranscriptSegmentationState: Equatable, Sendable {
    struct BoundaryMarker: Equatable, Sendable {
        let canonicalCharacterCount: Int
        let label: String
    }

    private(set) var frozenTranscriptPrefix = ""
    private(set) var latestRawTranscript = ""
    private(set) var boundaryMarkers: [BoundaryMarker] = []

    mutating func visibleText(for incomingRawText: String) -> String {
        let trimmed = incomingRawText.trimmingCharacters(in: .whitespacesAndNewlines)
        latestRawTranscript = trimmed
        guard !trimmed.isEmpty else { return "" }
        return visibleText(from: trimmed)
    }

    mutating func displayText(for incomingRawText: String) -> String {
        let trimmed = incomingRawText.trimmingCharacters(in: .whitespacesAndNewlines)
        latestRawTranscript = trimmed
        guard !trimmed.isEmpty else { return "" }
        guard !boundaryMarkers.isEmpty else { return trimmed }
        return annotatedText(from: trimmed)
    }

    mutating func freezeCurrentSegment(
        using incomingRawText: String? = nil,
        markerLabel: String? = nil
    ) -> String? {
        if let incomingRawText {
            _ = visibleText(for: incomingRawText)
        }

        let visible = currentVisibleText
        guard !visible.isEmpty else { return nil }

        if let markerLabel {
            let canonicalCharacterCount = canonicalTranscriptPrefix(latestRawTranscript).count
            boundaryMarkers.append(
                BoundaryMarker(
                    canonicalCharacterCount: canonicalCharacterCount,
                    label: markerLabel
                )
            )
        }

        frozenTranscriptPrefix = latestRawTranscript
        return visible
    }

    var currentVisibleText: String {
        guard !latestRawTranscript.isEmpty else { return "" }
        return visibleText(from: latestRawTranscript)
    }

    mutating func reset() {
        frozenTranscriptPrefix = ""
        latestRawTranscript = ""
        boundaryMarkers = []
    }

    private func visibleText(from rawText: String) -> String {
        guard !frozenTranscriptPrefix.isEmpty else { return rawText }

        if rawText.hasPrefix(frozenTranscriptPrefix) {
            let suffixStart = rawText.index(rawText.startIndex, offsetBy: frozenTranscriptPrefix.count)
            return trimmedDuplicatePrefix(from: String(rawText[suffixStart...]))
        }

        let canonicalFrozen = canonicalTranscriptPrefix(frozenTranscriptPrefix)
        let canonicalIncoming = canonicalTranscriptPrefix(rawText)
        guard !canonicalFrozen.isEmpty,
              canonicalIncoming.hasPrefix(canonicalFrozen),
              let suffix = suffixAfterCanonicalPrefix(in: rawText, canonicalPrefix: canonicalFrozen)
        else {
            return trimmedDuplicatePrefix(
                from: suffixAfterCanonicalCharacterCount(
                    in: rawText,
                    canonicalCharacterCount: canonicalFrozen.count
                ) ?? rawText
            )
        }

        return trimmedDuplicatePrefix(from: suffix)
    }

    private func trimmedDuplicatePrefix(from rawSuffix: String) -> String {
        rawSuffix.trimmingCharacters(
            in: CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: "，。！？；：,.!?;:、"))
        )
    }

    private func canonicalTranscriptPrefix(_ text: String) -> String {
        let disallowed = CharacterSet.whitespacesAndNewlines.union(
            CharacterSet(charactersIn: "，。！？；：,.!?;:、\"'“”‘’()[]{}<>《》【】")
        )
        return text.unicodeScalars
            .filter { !disallowed.contains($0) }
            .map { CharacterSet.uppercaseLetters.contains($0) ? String($0).lowercased() : String($0) }
            .joined()
    }

    private func suffixAfterCanonicalPrefix(
        in text: String,
        canonicalPrefix: String
    ) -> String? {
        var matchedCount = 0
        let targetCount = canonicalPrefix.count
        guard targetCount > 0 else { return text }

        for index in text.indices {
            let character = String(text[index])
            let canonicalCharacter = canonicalTranscriptPrefix(character)
            if !canonicalCharacter.isEmpty {
                matchedCount += canonicalCharacter.count
            }
            if matchedCount >= targetCount {
                let nextIndex = text.index(after: index)
                return String(text[nextIndex...])
            }
        }

        return ""
    }

    private func suffixAfterCanonicalCharacterCount(
        in text: String,
        canonicalCharacterCount: Int
    ) -> String? {
        guard canonicalCharacterCount > 0 else { return text }

        var matchedCount = 0
        for index in text.indices {
            let canonicalCharacter = canonicalTranscriptPrefix(String(text[index]))
            if !canonicalCharacter.isEmpty {
                matchedCount += canonicalCharacter.count
            }
            if matchedCount >= canonicalCharacterCount {
                let nextIndex = text.index(after: index)
                return String(text[nextIndex...])
            }
        }

        return ""
    }

    private func annotatedText(from rawText: String) -> String {
        var result = ""
        var matchedCanonicalCount = 0
        var nextBoundaryIndex = 0
        var pendingLabels: [String] = []

        for character in rawText {
            let characterString = String(character)
            let canonicalCharacter = canonicalTranscriptPrefix(characterString)

            if !pendingLabels.isEmpty, !canonicalCharacter.isEmpty {
                result.append(markerText(for: pendingLabels))
                pendingLabels.removeAll()
            }

            result.append(character)

            if !canonicalCharacter.isEmpty {
                matchedCanonicalCount += canonicalCharacter.count
            }

            while nextBoundaryIndex < boundaryMarkers.count,
                  matchedCanonicalCount >= boundaryMarkers[nextBoundaryIndex].canonicalCharacterCount {
                pendingLabels.append(boundaryMarkers[nextBoundaryIndex].label)
                nextBoundaryIndex += 1
            }
        }

        if !pendingLabels.isEmpty {
            result.append(markerText(for: pendingLabels))
        }

        return result
    }

    private func markerText(for labels: [String]) -> String {
        labels
            .map { " • \($0) " }
            .joined()
    }
}
