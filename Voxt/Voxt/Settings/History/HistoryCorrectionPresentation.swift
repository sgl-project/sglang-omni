// HistoryCorrectionPresentation.swift
// Provides History Correction Presentation for history settings.

import Foundation

enum HistoryCorrectionSegment: Equatable {
    case plain(String)
    case original(String)
    case corrected(String)
}

enum HistoryCorrectionPresentation {
    static func correctedText(
        for text: String,
        snapshots: [DictionaryCorrectionSnapshot]
    ) -> String {
        segments(for: text, snapshots: snapshots).reduce(into: "") { partial, segment in
            switch segment {
            case .plain(let value), .corrected(let value):
                partial += value
            case .original:
                break
            }
        }
    }

    static func segments(
        for text: String,
        snapshots: [DictionaryCorrectionSnapshot]
    ) -> [HistoryCorrectionSegment] {
        let validSnapshots = normalizedSnapshots(snapshots, text: text)
        guard !validSnapshots.isEmpty else {
            return [.plain(text)]
        }

        let nsText = text as NSString
        var segments: [HistoryCorrectionSegment] = []
        var cursor = 0

        for snapshot in validSnapshots {
            let correctedRange = NSRange(location: snapshot.finalLocation, length: snapshot.finalLength)
            guard correctedRange.location >= cursor else { continue }

            if correctedRange.location > cursor {
                segments.append(.plain(nsText.substring(with: NSRange(location: cursor, length: correctedRange.location - cursor))))
            }

            segments.append(.original(snapshot.originalText))
            segments.append(.corrected(nsText.substring(with: correctedRange)))
            cursor = correctedRange.location + correctedRange.length
        }

        if cursor < nsText.length {
            segments.append(.plain(nsText.substring(from: cursor)))
        }

        return coalescedSegments(segments)
    }

    private static func normalizedSnapshots(
        _ snapshots: [DictionaryCorrectionSnapshot],
        text: String
    ) -> [DictionaryCorrectionSnapshot] {
        let nsText = text as NSString
        var normalized: [DictionaryCorrectionSnapshot] = []
        var consumed: [NSRange] = []

        for snapshot in snapshots.sorted(by: { lhs, rhs in
            if lhs.finalLocation == rhs.finalLocation {
                return lhs.finalLength < rhs.finalLength
            }
            return lhs.finalLocation < rhs.finalLocation
        }) {
            guard !snapshot.originalText.isEmpty,
                  !snapshot.correctedText.isEmpty,
                  snapshot.originalText != snapshot.correctedText,
                  snapshot.finalLocation >= 0,
                  snapshot.finalLength > 0
            else {
                continue
            }

            let correctedRange = NSRange(location: snapshot.finalLocation, length: snapshot.finalLength)
            guard NSMaxRange(correctedRange) <= nsText.length else { continue }
            guard !consumed.contains(where: { NSIntersectionRange($0, correctedRange).length > 0 }) else { continue }
            let correctedText = nsText.substring(with: correctedRange)

            normalized.append(
                DictionaryCorrectionSnapshot(
                    originalText: snapshot.originalText,
                    correctedText: correctedText,
                    finalLocation: snapshot.finalLocation,
                    finalLength: correctedRange.length
                )
            )
            consumed.append(correctedRange)
        }

        return normalized
    }

    private static func coalescedSegments(_ segments: [HistoryCorrectionSegment]) -> [HistoryCorrectionSegment] {
        var output: [HistoryCorrectionSegment] = []

        for segment in segments {
            switch (output.last, segment) {
            case let (.plain(existing)?, .plain(incoming)):
                output[output.count - 1] = .plain(existing + incoming)
            case let (.original(existing)?, .original(incoming)):
                output[output.count - 1] = .original(existing + incoming)
            case let (.corrected(existing)?, .corrected(incoming)):
                output[output.count - 1] = .corrected(existing + incoming)
            default:
                output.append(segment)
            }
        }

        return output
    }
}
