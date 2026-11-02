// MeetingTranscriptVirtualList.swift
// Lazy SwiftUI transcript list with stable spacing and scroll requests.
//
// NSTableView + NSHostingView automatic/manual row heights repeatedly mis-measured
// multi-line SwiftUI Text (extra blank space under wrapped cards). LazyVStack keeps
// visually correct spacing while still deferring off-screen row bodies. Playback and
// search CPU wins come from ViewModel caching + activeSegment-only scroll, not AppKit
// cell reuse.

import SwiftUI

struct MeetingTranscriptVirtualList: View {
    let rows: [MeetingTranscriptVirtualRow]
    let showsTranslation: Bool
    let rowSpacing: CGFloat
    let scrollRequest: MeetingTranscriptScrollRequest?
    let canEditTranscript: Bool
    let editingSegmentID: UUID?
    let editingText: String
    let onSelectSegment: (MeetingTranscriptSegment) -> Void
    let onBeginEditing: (MeetingTranscriptSegment) -> Void
    let onEditingTextChanged: (String) -> Void
    let onSaveEditing: () -> Void
    let onCancelEditing: () -> Void
    let onDeleteSegment: (MeetingTranscriptSegment) -> Void
    let onToggleHighlight: (MeetingTranscriptSegment) -> Void

    init(
        rows: [MeetingTranscriptVirtualRow],
        showsTranslation: Bool,
        rowSpacing: CGFloat = 12,
        scrollRequest: MeetingTranscriptScrollRequest? = nil,
        canEditTranscript: Bool = false,
        editingSegmentID: UUID? = nil,
        editingText: String = "",
        onSelectSegment: @escaping (MeetingTranscriptSegment) -> Void = { _ in },
        onBeginEditing: @escaping (MeetingTranscriptSegment) -> Void = { _ in },
        onEditingTextChanged: @escaping (String) -> Void = { _ in },
        onSaveEditing: @escaping () -> Void = {},
        onCancelEditing: @escaping () -> Void = {},
        onDeleteSegment: @escaping (MeetingTranscriptSegment) -> Void = { _ in },
        onToggleHighlight: @escaping (MeetingTranscriptSegment) -> Void = { _ in }
    ) {
        self.rows = rows
        self.showsTranslation = showsTranslation
        self.rowSpacing = rowSpacing
        self.scrollRequest = scrollRequest
        self.canEditTranscript = canEditTranscript
        self.editingSegmentID = editingSegmentID
        self.editingText = editingText
        self.onSelectSegment = onSelectSegment
        self.onBeginEditing = onBeginEditing
        self.onEditingTextChanged = onEditingTextChanged
        self.onSaveEditing = onSaveEditing
        self.onCancelEditing = onCancelEditing
        self.onDeleteSegment = onDeleteSegment
        self.onToggleHighlight = onToggleHighlight
    }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: rowSpacing) {
                    ForEach(rows) { row in
                        rowView(for: row)
                            .id(row.id)
                    }
                }
            }
            .onChange(of: scrollRequest?.generation) { _, _ in
                fulfillScrollRequest(using: proxy)
            }
            .onAppear {
                fulfillScrollRequest(using: proxy)
            }
        }
    }

    @ViewBuilder
    private func rowView(for row: MeetingTranscriptVirtualRow) -> some View {
        switch row {
        case .speakerHeader(_, let title, let count):
            MeetingDetailSpeakerHeaderRow(title: title, count: count)
        case .segment(let presentation):
                MeetingDetailSegmentRow(
                    segment: presentation.segment,
                    speakerTitle: presentation.speakerTitle,
                    isActive: presentation.isActive,
                    showsTranslation: presentation.showsTranslation || showsTranslation,
                    isSearchMatch: presentation.isSearchMatch,
                    canEditTranscript: canEditTranscript,
                    isEditing: editingSegmentID == presentation.segment.id,
                    editingText: editingText,
                    onSelect: { onSelectSegment(presentation.segment) },
                    onBeginEditing: { onBeginEditing(presentation.segment) },
                    onEditingTextChanged: onEditingTextChanged,
                    onSaveEditing: onSaveEditing,
                    onCancelEditing: onCancelEditing,
                    onDelete: { onDeleteSegment(presentation.segment) },
                    onToggleHighlight: { onToggleHighlight(presentation.segment) }
                )
            .equatable()
        }
    }

    private func fulfillScrollRequest(using proxy: ScrollViewProxy) {
        guard let scrollRequest else { return }
        let anchor: UnitPoint
        switch scrollRequest.anchor {
        case .top:
            anchor = .top
        case .center:
            anchor = .center
        case .bottom:
            anchor = .bottom
        }
        proxy.scrollTo(scrollRequest.rowID, anchor: anchor)
    }
}
