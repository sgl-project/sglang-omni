import SwiftUI

struct HistorySettingsNotesList<NoteRowContent: View>: View {
    let noteViewMode: HistoryNoteViewMode
    let allNotes: [VoxtNoteItem]
    let visibleNoteSections: [VoxtNoteSectionSnapshot]
    let linearNoteStatuses: [VoxtNoteStatus]
    let linearCompletedVisibleLimit: Int
    let hasMoreNotes: Bool
    let noteHistoryRowHeight: CGFloat
    let noteListRowSpacing: CGFloat
    let historyRowSpacing: CGFloat
    let onLoadMoreNotes: () -> Void
    let onLoadMoreCompletedNotes: (Int) -> Void
    let row: (VoxtNoteItem, Int, CGFloat?) -> NoteRowContent

    @ViewBuilder
    var body: some View {
        switch noteViewMode {
        case .list:
            groupedNotesList
        case .linearCard:
            linearNotesList
        }
    }

    private var groupedNotesList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 14) {
                ForEach(visibleNoteSections) { section in
                    VStack(alignment: .leading, spacing: 4) {
                        NoteHistorySectionHeader(
                            status: section.status,
                            count: allNotes.lazy.filter { $0.status == section.status }.count
                        )

                        LazyVStack(spacing: noteListRowSpacing) {
                            ForEach(section.items) { note in
                                row(note, 2, noteHistoryRowHeight)
                            }
                        }
                    }
                }

                if hasMoreNotes {
                    Button(AppLocalization.localizedString("Load More"), action: onLoadMoreNotes)
                    .buttonStyle(SettingsPillButtonStyle())
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
                }
            }
            .padding(.bottom, 8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    private var linearNotesList: some View {
        GeometryReader { geometry in
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(alignment: .top, spacing: 12) {
                    ForEach(linearNoteStatuses) { status in
                        linearNoteColumn(status)
                    }
                }
                .frame(height: geometry.size.height, alignment: .top)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private func linearNoteColumn(_ status: VoxtNoteStatus) -> some View {
        let allStatusNotes = allNotes.filter { $0.status == status }
        let visibleStatusNotes = HistorySettingsData.visibleLinearNotes(
            from: allNotes,
            status: status,
            completedVisibleLimit: linearCompletedVisibleLimit
        )

        return VStack(alignment: .leading, spacing: 4) {
            NoteHistorySectionHeader(status: status, count: allStatusNotes.count)
                .padding(.horizontal, 4)

            ScrollView {
                LazyVStack(spacing: historyRowSpacing) {
                    if visibleStatusNotes.isEmpty {
                        Text(AppLocalization.localizedString("No notes yet"))
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 20)
                    } else {
                        ForEach(visibleStatusNotes) { note in
                            row(note, 5, nil)
                        }
                    }

                    if status == .done, visibleStatusNotes.count < allStatusNotes.count {
                        NoteHistoryMoreButton {
                            onLoadMoreCompletedNotes(allStatusNotes.count)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 4)
                    }
                }
                .padding(.horizontal, 4)
                .padding(.bottom, 4)
            }
        }
        .padding(6)
        .frame(width: 260)
        .frame(maxHeight: .infinity, alignment: .top)
        .background(
            SettingsUIStyle.groupedFillColor,
            in: RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                .strokeBorder(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
        )
    }
}

struct HistorySettingsEntriesList: View {
    let items: [HistoryListItem]
    let totalCount: Int
    let historyRowFallbackHeight: CGFloat
    let historyRowSpacing: CGFloat
    let historyRowVerticalInset: CGFloat
    let isLoading: Bool
    let rowHeightForItemAtWidth: (HistoryListItem, CGFloat) -> CGFloat
    let audioURL: (TranscriptionHistoryEntry) -> URL?
    let onLoadMore: () -> Void
    let onCopyEntry: (TranscriptionHistoryEntry) -> Void
    let onShowEntryInfo: (TranscriptionHistoryEntry) -> Void
    let onDeleteEntry: (TranscriptionHistoryEntry) -> Void
    let onCopyMeetingEntry: (UUID) -> Void
    let onShowMeetingEntryInfo: (TranscriptionHistoryListEntry) -> Void
    let onDeleteMeetingEntry: (UUID) -> Void

    var body: some View {
        let list = PagedVerticalList(
            items: items,
            totalCount: totalCount,
            rowHeight: historyRowFallbackHeight,
            rowSpacing: historyRowSpacing,
            rowHeightForItemAtWidth: rowHeightForItemAtWidth,
            isLoading: isLoading,
            onLoadMore: onLoadMore
        ) { item in
            switch item {
            case .dayHeader(let date):
                HistoryDayHeader(date: date)
            case .entry(let entry):
                HistoryRow(
                    entry: entry,
                    audioURL: audioURL(entry),
                    isCompact: false,
                    onCopy: { onCopyEntry(entry) },
                    onShowInfo: { onShowEntryInfo(entry) },
                    onDelete: { onDeleteEntry(entry) }
                )
                .padding(.vertical, historyRowVerticalInset)
            case .meetingEntry(let entry):
                HistoryListRow(
                    timeText: RelativeNoteTimestampFormatter.historyListTime(for: entry.createdAt),
                    displayText: entry.displayText,
                    onCopy: { onCopyMeetingEntry(entry.id) },
                    onShowInfo: { onShowMeetingEntryInfo(entry) },
                    onDelete: { onDeleteMeetingEntry(entry.id) }
                )
                .padding(.vertical, historyRowVerticalInset)
            }
        }

        list.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }
}
