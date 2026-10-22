// Lightweight, cached presentation for the status menu; no storage reads on hover.

import AppKit
import Foundation

enum StatusMenuHistorySupport {
    static let recentLimit = 5
    static let recentPreviewMaxWidth: CGFloat = 100

    private static let recentPreviewAttributes: [NSAttributedString.Key: Any] = [
        .font: NSFont.menuFont(ofSize: NSFont.systemFontSize)
    ]

    static func filters(availability: FeatureAvailabilitySettings) -> [HistoryFilterTab] {
        HistoryFilterTab.allCases.filter {
            $0.correspondingFeatureTab.isEnabled(in: availability)
        }
    }

    static func recentEntries(
        from candidates: [TranscriptionHistoryListEntry],
        availability: FeatureAvailabilitySettings
    ) -> [TranscriptionHistoryListEntry] {
        Array(candidates.filter {
            $0.kind == .normal || ($0.kind == .translation && availability.translationEnabled)
        }.sorted {
            if $0.createdAt != $1.createdAt { return $0.createdAt > $1.createdAt }
            return $0.id.uuidString < $1.id.uuidString
        }.prefix(recentLimit))
    }

    static func previewTitle(for entry: TranscriptionHistoryListEntry) -> String {
        let preview = entry.previewText.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
        // SQLite length()/substr() count Unicode code points, not Swift graphemes.
        let wasTruncatedByRepository = entry.textLength > entry.previewText.unicodeScalars.count
        let needsEllipsis = wasTruncatedByRepository || renderedWidth(of: preview) > recentPreviewMaxWidth
        guard needsEllipsis else { return preview }

        var result = ""
        for character in preview {
            let candidate = result + String(character) + "…"
            guard renderedWidth(of: candidate) <= recentPreviewMaxWidth else { break }
            result.append(character)
        }
        return result.isEmpty ? "…" : result + "…"
    }

    private static func renderedWidth(of text: String) -> CGFloat {
        (text as NSString).size(withAttributes: recentPreviewAttributes).width
    }
}
