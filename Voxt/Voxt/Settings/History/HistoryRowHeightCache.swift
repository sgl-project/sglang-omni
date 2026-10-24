// HistoryRowHeightCache.swift
// Width-aware height estimates for history rows.

import AppKit
import Foundation

final class HistoryRowHeightCache {
    static let minimumHeight: CGFloat = 48
    static let maximumTextLines = 3
    static let textWidthReserve: CGFloat = 145
    // HistoryRow uses 7pt of vertical padding on each side in the history list.
    static let rowVerticalPadding: CGFloat = 14
    static let textLineHeight: CGFloat = 17
    static let textLineSpacing: CGFloat = 2

    private var heights: [String: CGFloat] = [:]

    func height(
        for entry: TranscriptionHistoryEntry,
        width: CGFloat,
        verticalInset: CGFloat
    ) -> CGFloat {
        let bucket = Self.widthBucket(width)
        let displayText = HistoryRow.displayText(for: entry)
        let key = "w\(bucket)|\(entry.id.uuidString)|\(displayText.hashValue)"
        if let cached = heights[key] {
            return cached
        }

        let value = Self.estimate(
            text: displayText,
            width: width,
            verticalInset: verticalInset
        )
        heights[key] = value
        return value
    }

    func invalidateAll() {
        heights.removeAll(keepingCapacity: true)
    }

    static func estimate(
        text: String,
        width: CGFloat,
        verticalInset: CGFloat
    ) -> CGFloat {
        let availableTextWidth = max(80, width - textWidthReserve)
        let naturalTextHeight = textHeight(text, width: availableTextWidth)
        let lineCount = max(1, Int(ceil(naturalTextHeight / textLineHeight)))
        let visibleLineCount = min(maximumTextLines, lineCount)
        let visibleTextHeight = textLineHeight * CGFloat(visibleLineCount)
            + textLineSpacing * CGFloat(max(0, visibleLineCount - 1))

        return max(
            minimumHeight,
            ceil(visibleTextHeight + rowVerticalPadding + (verticalInset * 2))
        )
    }

    static func widthBucket(_ width: CGFloat) -> Int {
        max(1, Int((max(width, 1) / 8).rounded()) * 8)
    }

    private static func textHeight(_ text: String, width: CGFloat) -> CGFloat {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return textLineHeight }

        let bounds = (trimmed as NSString).boundingRect(
            with: CGSize(width: width, height: .greatestFiniteMagnitude),
            options: [.usesLineFragmentOrigin, .usesFontLeading],
            attributes: [.font: NSFont.systemFont(ofSize: 13)],
            context: nil
        )
        return max(textLineHeight, ceil(bounds.height))
    }
}
