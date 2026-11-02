// MeetingTranscriptRowHeightCache.swift
// Width-aware heights for meeting transcript virtual rows.
// Initial heights use text boundingRect estimates; on-screen GeometryReader
// reports can refine them (including shrinking overestimated rows).

import AppKit
import Foundation

struct MeetingTranscriptRowHeightCache {
    private var heights: [String: CGFloat] = [:]
    private var reported: Set<String> = []

    /// Widths below this are treated as provisional and are not cached.
    static let minimumReliableWidth: CGFloat = 120

    mutating func invalidateAll() {
        heights.removeAll(keepingCapacity: true)
        reported.removeAll(keepingCapacity: true)
    }

    mutating func invalidateWidth(_ width: CGFloat) {
        let bucket = Self.widthBucket(width)
        let prefix = "w\(bucket)|"
        let removedKeys = heights.keys.filter { $0.hasPrefix(prefix) }
        for key in removedKeys {
            heights.removeValue(forKey: key)
            reported.remove(key)
        }
    }

    func height(
        for row: MeetingTranscriptVirtualRow,
        width: CGFloat,
        showsTranslation: Bool
    ) -> CGFloat {
        let key = Self.cacheKey(for: row, width: width, showsTranslation: showsTranslation)
        if let cached = heights[key] {
            return cached
        }
        return Self.estimate(for: row, width: width, showsTranslation: showsTranslation)
    }

    mutating func cachedHeight(
        for row: MeetingTranscriptVirtualRow,
        width: CGFloat,
        showsTranslation: Bool
    ) -> CGFloat {
        let key = Self.cacheKey(for: row, width: width, showsTranslation: showsTranslation)
        if let cached = heights[key] {
            return cached
        }
        let value = Self.estimate(for: row, width: width, showsTranslation: showsTranslation)
        if width >= Self.minimumReliableWidth {
            heights[key] = value
        }
        return value
    }

    func hasReportedHeight(
        for row: MeetingTranscriptVirtualRow,
        width: CGFloat,
        showsTranslation: Bool
    ) -> Bool {
        let key = Self.cacheKey(for: row, width: width, showsTranslation: showsTranslation)
        return reported.contains(key)
    }

    /// Stores an on-screen layout height. Unlike hosting-view fittingSize, this may
    /// shrink below the initial estimate so inter-row gaps stay visually even.
    @discardableResult
    mutating func storeReportedHeight(
        _ height: CGFloat,
        for row: MeetingTranscriptVirtualRow,
        width: CGFloat,
        showsTranslation: Bool
    ) -> Bool {
        guard width >= Self.minimumReliableWidth else { return false }
        let rounded = ceil(height)
        guard rounded >= 24 else { return false }

        let key = Self.cacheKey(for: row, width: width, showsTranslation: showsTranslation)
        if let existing = heights[key], abs(existing - rounded) < 1 {
            reported.insert(key)
            return false
        }
        heights[key] = rounded
        reported.insert(key)
        return true
    }

    /// Legacy name kept for tests; reported heights are the authoritative measured values.
    @discardableResult
    mutating func storeMeasuredHeight(
        _ height: CGFloat,
        for row: MeetingTranscriptVirtualRow,
        width: CGFloat,
        showsTranslation: Bool
    ) -> Bool {
        storeReportedHeight(height, for: row, width: width, showsTranslation: showsTranslation)
    }

    func hasMeasuredHeight(
        for row: MeetingTranscriptVirtualRow,
        width: CGFloat,
        showsTranslation: Bool
    ) -> Bool {
        hasReportedHeight(for: row, width: width, showsTranslation: showsTranslation)
    }

    static func estimate(
        for row: MeetingTranscriptVirtualRow,
        width: CGFloat,
        showsTranslation: Bool
    ) -> CGFloat {
        switch row {
        case .speakerHeader:
            return 28
        case .segment(let presentation):
            return estimateSegmentHeight(
                text: presentation.segment.text,
                translatedText: presentation.segment.translatedText,
                isTranslationPending: presentation.segment.isTranslationPending,
                width: max(width, minimumReliableWidth),
                showsTranslation: showsTranslation || presentation.showsTranslation
            )
        }
    }

    static func estimateSegmentHeight(
        text: String,
        translatedText: String?,
        isTranslationPending: Bool,
        width: CGFloat,
        showsTranslation: Bool
    ) -> CGFloat {
        // Match MeetingDetailSegmentRow: 14pt horizontal padding on each side.
        let contentWidth = max(80, width - 28)
        let headerHeight: CGFloat = 15
        let spacing: CGFloat = 9
        let verticalPadding: CGFloat = 28

        let bodyHeight = textHeight(
            text,
            width: contentWidth,
            font: .systemFont(ofSize: 14, weight: .medium)
        )

        var translationHeight: CGFloat = 0
        if showsTranslation {
            let trimmed = translatedText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if !trimmed.isEmpty {
                translationHeight = spacing + textHeight(
                    trimmed,
                    width: contentWidth,
                    font: .systemFont(ofSize: 13, weight: .medium)
                )
            } else if isTranslationPending {
                translationHeight = spacing + 15
            }
        }

        // Slightly generous first paint so text is never clipped before GeometryReader refines.
        return max(60, verticalPadding + headerHeight + spacing + bodyHeight + translationHeight + 2)
    }

    static func widthBucket(_ width: CGFloat) -> Int {
        max(1, Int((width / 8).rounded()) * 8)
    }

    static func cacheKey(
        for row: MeetingTranscriptVirtualRow,
        width: CGFloat,
        showsTranslation: Bool
    ) -> String {
        let bucket = widthBucket(width)
        switch row {
        case .speakerHeader(let id, let title, let count):
            return "w\(bucket)|h|\(id)|\(title)|\(count)"
        case .segment(let presentation):
            let translated = presentation.segment.translatedText ?? ""
            let pending = presentation.segment.isTranslationPending ? "1" : "0"
            let translationFlag = (showsTranslation || presentation.showsTranslation) ? "1" : "0"
            return "w\(bucket)|s|\(presentation.id.uuidString)|\(translationFlag)|\(pending)|\(presentation.segment.text.hashValue)|\(translated.hashValue)"
        }
    }

    private static func textHeight(_ text: String, width: CGFloat, font: NSFont) -> CGFloat {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return 17 }
        let constraint = CGSize(width: width, height: .greatestFiniteMagnitude)
        let bounds = (trimmed as NSString).boundingRect(
            with: constraint,
            options: [.usesLineFragmentOrigin, .usesFontLeading],
            attributes: [.font: font]
        )
        return max(17, ceil(bounds.height))
    }
}
