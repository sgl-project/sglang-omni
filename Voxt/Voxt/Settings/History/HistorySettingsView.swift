// HistorySettingsView.swift
// Provides History Settings View for history settings.

import SwiftUI
import AppKit

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

struct HistorySettingsView: View {
    @Environment(\.locale) private var locale
    @AppStorage(AppPreferenceKey.historyCleanupEnabled) private var historyCleanupEnabled = true
    @AppStorage(AppPreferenceKey.historyRetentionPeriod) private var historyRetentionPeriodRaw = HistoryRetentionPeriod.ninetyDays.rawValue
    @AppStorage(AppPreferenceKey.historyRetentionCount) private var historyRetentionCountRaw = HistoryRetentionCount.unlimited.rawValue
    @AppStorage(AppPreferenceKey.historyAudioStorageEnabled) private var historyAudioStorageEnabled = false

    @ObservedObject var historyStore: TranscriptionHistoryStore
    @ObservedObject var noteStore: VoxtNoteStore
    @ObservedObject var dictionaryStore: DictionaryStore
    @Binding var selectedFilter: HistoryFilterTab
    let navigationRequest: SettingsNavigationRequest?
    @State private var copyToastMessage = ""
    @State private var copyToastDismissTask: Task<Void, Never>?
    @State private var copiedEntryID: UUID?
    @State private var copiedNoteID: UUID?
    @State private var isHistoryAudioSettingsPresented = false
    @State private var historyAudioStorageDisplayPath = ""
    @State private var historyAudioStorageSelectionError: String?
    @State private var historyAudioExportResultMessage: String?
    @State private var historyAudioStorageStats = HistoryAudioStorageStats(storedFileCount: 0, totalBytes: 0)
    @State private var pendingBulkDeletionTarget: HistoryBulkDeletionTarget?
    @State private var selectedHistoryInfoEntry: TranscriptionHistoryEntry?
    @State private var historySearchText = ""
    @State private var showHistorySearchDialog = false
    @State private var visibleHistoryEntries: [TranscriptionHistoryEntry] = []
    @State private var visibleMeetingHistoryEntries: [TranscriptionHistoryListEntry] = []
    @State private var totalHistoryEntryCount = 0
    @State private var isLoadingHistoryEntries = false
    @State private var historyPageGeneration = 0
    @State private var historyAudioStatsGeneration = 0
    @State private var suppressedStoreHistoryReloadCount = 0
    @State private var selectedNoteStatuses = Set(VoxtNoteStatus.allCases)
    @State private var noteVisibleLimit = 80
    @State private var noteViewMode: HistoryNoteViewMode = .linearCard
    @State private var linearCompletedVisibleLimit = 20
    @State private var historyRowHeightCache = HistoryRowHeightCache()

    private let historyPageSize = 80
    private let meetingHistoryPageSize = 40
    private let notePageSize = 80
    private let linearCompletedPageSize = 10
    private let historyRowFallbackHeight: CGFloat = 74
    private let noteHistoryRowHeight: CGFloat = 68
    private let historyRowSpacing: CGFloat = 2
    private let noteListRowSpacing: CGFloat = 6
    private let historyRowVerticalInset: CGFloat = 4

    private var historyRetentionPeriod: HistoryRetentionPeriod {
        HistoryRetentionPeriod(rawValue: historyRetentionPeriodRaw) ?? .ninetyDays
    }

    private var historyRetentionCount: HistoryRetentionCount {
        HistoryRetentionCount(rawValue: historyRetentionCountRaw) ?? .unlimited
    }

    private var allNotes: [VoxtNoteItem] {
        let matchingIDs = Set(HistorySettingsData.filteredNotes(
            noteStore.items,
            statuses: selectedNoteStatuses,
            query: historySearchText
        ).map(\.id))

        return HistorySettingsData.noteSectionOrder.flatMap { status in
            noteStore.orderedItems(for: status).filter { matchingIDs.contains($0.id) }
        }
    }

    private var visibleNotes: [VoxtNoteItem] {
        HistorySettingsData.visibleEntries(from: allNotes, visibleLimit: noteVisibleLimit)
    }

    private var visibleNoteSections: [VoxtNoteSectionSnapshot] {
        HistorySettingsData.noteSections(from: visibleNotes)
    }

    private var visibleEntries: [TranscriptionHistoryEntry] {
        visibleHistoryEntries
    }

    private var visibleHistoryEntryCount: Int {
        selectedFilter == .transcript
            ? visibleMeetingHistoryEntries.count
            : visibleHistoryEntries.count
    }

    private var historyListItems: [HistoryListItem] {
        var items: [HistoryListItem] = []
        var currentDay: Date?
        let calendar = Calendar.current

        let dates: [(Date, HistoryListItem)] = selectedFilter == .transcript
            ? visibleMeetingHistoryEntries.map { ($0.createdAt, .meetingEntry($0)) }
            : visibleEntries.map { ($0.createdAt, .entry($0)) }

        for (createdAt, item) in dates {
            let day = calendar.startOfDay(for: createdAt)
            if currentDay != day {
                items.append(.dayHeader(day))
                currentDay = day
            }
            items.append(item)
        }

        return items
    }

    private var historyListTotalCount: Int {
        historyListItems.count + max(0, totalHistoryEntryCount - visibleHistoryEntryCount)
    }

    private var isNoteTabSelected: Bool {
        selectedFilter == .note
    }

    private var isSearchActive: Bool {
        !historySearchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var isNoteStatusFilterActive: Bool {
        selectedNoteStatuses != Set(VoxtNoteStatus.allCases)
    }

    private var emptyState: HistoryContentEmptyState {
        if selectedFilter == .note {
            return allNotes.isEmpty ? .noNotes : .none
        }
        return totalHistoryEntryCount == 0 ? .noEntriesInCategory : .none
    }

    private var emptyStateTitle: String {
        if isSearchActive || (isNoteTabSelected && isNoteStatusFilterActive) {
            return localized("No matching results")
        }

        switch selectedFilter {
        case .transcription:
            return localized("No transcription history yet")
        case .translation:
            return localized("No translation history yet")
        case .transcript:
            return localized("No meeting transcripts yet")
        case .rewrite:
            return localized("No rewrite history yet")
        case .note:
            return localized("No notes yet")
        }
    }

    private var emptyStateMessage: String {
        if isSearchActive {
            return localized("Try another keyword or clear the search filter.")
        }

        let distinguishSides = HotkeyPreference.loadDistinguishModifierSides()
        switch selectedFilter {
        case .transcription:
            return AppLocalization.format(
                "Press %@ to start dictation. Completed results will appear here.",
                HotkeyPreference.displayString(for: HotkeyPreference.load(), distinguishModifierSides: distinguishSides)
            )
        case .translation:
            return AppLocalization.format(
                "Press %@ to try voice translation. Completed results will appear here.",
                HotkeyPreference.displayString(for: HotkeyPreference.loadTranslation(), distinguishModifierSides: distinguishSides)
            )
        case .transcript:
            return AppLocalization.format(
                "Press %@ to start Meeting Mode. Saved transcripts will appear here.",
                HotkeyPreference.displayString(for: HotkeyPreference.loadMeeting(), distinguishModifierSides: distinguishSides)
            )
        case .rewrite:
            return AppLocalization.format(
                "Press %@ to rewrite selected text or spoken instructions.",
                HotkeyPreference.displayString(for: HotkeyPreference.loadRewrite(), distinguishModifierSides: distinguishSides)
            )
        case .note:
            return localized("Capture key points during recording, then review notes here.")
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .center, spacing: 12) {
                Text(selectedFilter.title)
                    .font(.title3.weight(.semibold))

                Spacer(minLength: 12)

                if isNoteTabSelected {
                    HistoryNoteStatusFilterSelect(selection: $selectedNoteStatuses)
                }

                Button {
                    showHistorySearchDialog = true
                } label: {
                    SettingsSearchIconView()
                }
                .buttonStyle(SettingsCompactIconButtonStyle())
                .help(localized(isNoteTabSelected ? "Search Notes" : "Search History"))

                Button {
                    pendingBulkDeletionTarget = isNoteTabSelected ? .notes : .history(selectedFilter)
                } label: {
                    HistoryActionIcon(kind: .delete, color: .secondary)
                }
                .buttonStyle(HistoryToolbarDeleteButtonStyle())
                .help(localized("Delete All"))
                .disabled(isNoteTabSelected ? noteStore.items.isEmpty : totalHistoryEntryCount == 0)

                if isNoteTabSelected {
                    HistoryNoteViewPicker(selection: $noteViewMode)
                } else {
                    Button {
                        historyAudioStorageSelectionError = nil
                        historyAudioExportResultMessage = nil
                        isHistoryAudioSettingsPresented = true
                    } label: {
                        HistoryToolbarSettingsIcon(size: 16)
                    }
                    .buttonStyle(SettingsCompactIconButtonStyle())
                    .help(localized("History Audio Settings"))
                }
            }

            GroupBox {
                VStack(alignment: .leading, spacing: 12) {
                    if isSearchActive {
                        HStack(spacing: 8) {
                            Text(AppLocalization.format("Filtered by \"%@\"", historySearchText))
                                .font(.caption)
                                .foregroundStyle(.secondary)

                            Button(localized("Clear")) {
                                historySearchText = ""
                            }
                            .buttonStyle(.plain)
                        }
                    }

                    if emptyState != .none {
                        SettingsEmptyStateView(
                            illustration: .history,
                            title: emptyStateTitle,
                            message: emptyStateMessage
                        )
                    } else if isNoteTabSelected {
                        notesList
                    } else {
                        historyList
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            }
            .settingsNavigationAnchor(.historySettings)
            .settingsNavigationAnchor(.historyEntries)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .overlay(alignment: .top) {
            if !copyToastMessage.isEmpty {
                ModelDebugToast(message: copyToastMessage) {
                    dismissCopyToast()
                }
                .padding(.top, 12)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .animation(.easeInOut(duration: 0.16), value: copyToastMessage)
        .sheet(isPresented: $isHistoryAudioSettingsPresented) {
            HistoryAudioSettingsSheet(
                historyCleanupEnabled: $historyCleanupEnabled,
                historyRetentionPeriodRaw: $historyRetentionPeriodRaw,
                historyRetentionCountRaw: $historyRetentionCountRaw,
                historyAudioStorageEnabled: $historyAudioStorageEnabled,
                historyAudioStorageDisplayPath: $historyAudioStorageDisplayPath,
                historyAudioStorageSelectionError: $historyAudioStorageSelectionError,
                historyAudioExportResultMessage: $historyAudioExportResultMessage,
                isPresented: $isHistoryAudioSettingsPresented,
                historyRetentionPeriod: historyRetentionPeriod,
                historyRetentionCount: historyRetentionCount,
                historyAudioStorageStatsSummary: historyAudioStorageStatsSummary,
                onOpenHistoryAudioStorageInFinder: openHistoryAudioStorageInFinder,
                onChooseHistoryAudioStorageDirectory: chooseHistoryAudioStorageDirectory,
                onExportAllHistoryAudio: exportAllHistoryAudio
            )
        }
        .sheet(item: $selectedHistoryInfoEntry) { entry in
            HistoryDetailSheetContent(
                entry: entry,
                audioURL: historyStore.audioURL(for: entry),
                locale: locale
            )
            .frame(minWidth: 520, idealWidth: 620, minHeight: 480, idealHeight: 640)
        }
        .sheet(isPresented: $showHistorySearchDialog) {
            SettingsSearchDialog(
                title: localized(isNoteTabSelected ? "Search Notes" : "Search History"),
                placeholder: localized(
                    isNoteTabSelected
                        ? "Search note titles or content"
                        : "Search history text, titles, or apps"
                ),
                query: $historySearchText,
                isPresented: $showHistorySearchDialog
            )
        }
        .alert(item: $pendingBulkDeletionTarget) { target in
            Alert(
                title: Text(bulkDeletionTitle(for: target)),
                message: Text(bulkDeletionMessage(for: target)),
                primaryButton: .destructive(Text(localized("Delete"))) {
                    confirmBulkDeletion(target)
                },
                secondaryButton: .cancel(Text(localized("Cancel")))
            )
        }
        .onAppear {
            applyNavigationTarget(navigationRequest?.target)
            if !HistoryRetentionPeriod.allCases.contains(where: { $0.rawValue == historyRetentionPeriodRaw }) {
                historyRetentionPeriodRaw = HistoryRetentionPeriod.ninetyDays.rawValue
            }
            if !HistoryRetentionCount.allCases.contains(where: { $0.rawValue == historyRetentionCountRaw }) {
                historyRetentionCountRaw = HistoryRetentionCount.unlimited.rawValue
            }
            refreshHistoryAudioStorageDisplayPath()
            refreshHistoryAudioStorageStats()
            reloadHistoryEntries(reset: true)
        }
        .onChange(of: navigationRequest?.id) { _, _ in
            applyNavigationTarget(navigationRequest?.target)
        }
        .onChange(of: selectedFilter) { _, _ in
            noteVisibleLimit = notePageSize
            reloadHistoryEntries(reset: true)
        }
        .onChange(of: historySearchText) { _, _ in
            noteVisibleLimit = notePageSize
            linearCompletedVisibleLimit = 20
            reloadHistoryEntries(reset: true)
        }
        .onChange(of: selectedNoteStatuses) { _, _ in
            noteVisibleLimit = notePageSize
            linearCompletedVisibleLimit = 20
        }
        .onChange(of: historyCleanupEnabled) { _, _ in
            applyRetentionPolicyAndReload()
        }
        .onChange(of: historyRetentionPeriodRaw) { _, newValue in
            if !HistoryRetentionPeriod.allCases.contains(where: { $0.rawValue == newValue }) {
                historyRetentionPeriodRaw = HistoryRetentionPeriod.ninetyDays.rawValue
            }
            applyRetentionPolicyAndReload()
        }
        .onChange(of: historyRetentionCountRaw) { _, newValue in
            if !HistoryRetentionCount.allCases.contains(where: { $0.rawValue == newValue }) {
                historyRetentionCountRaw = HistoryRetentionCount.unlimited.rawValue
            }
            applyRetentionPolicyAndReload()
        }
        .onReceive(historyStore.$entries) { _ in
            if suppressedStoreHistoryReloadCount > 0 {
                suppressedStoreHistoryReloadCount -= 1
                return
            }
            refreshHistoryAudioStorageStats()
            reloadHistoryEntries(reset: true)
        }
        .onDisappear {
            dismissCopyToast()
        }
    }

    @ViewBuilder
    private var notesList: some View {
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
                                noteHistoryRow(note, fixedHeight: noteHistoryRowHeight)
                            }
                        }
                    }
                }

                if HistorySettingsData.hasMoreItems(in: allNotes, visibleLimit: noteVisibleLimit) {
                    Button(localized("Load More")) {
                        noteVisibleLimit = HistorySettingsData.nextVisibleLimit(
                            currentLimit: noteVisibleLimit,
                            pageSize: notePageSize,
                            totalCount: allNotes.count
                        )
                    }
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

    private var linearNoteStatuses: [VoxtNoteStatus] {
        HistorySettingsData.linearNoteSectionOrder.filter(selectedNoteStatuses.contains)
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
                        Text(localized("No notes yet"))
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 20)
                    } else {
                        ForEach(visibleStatusNotes) { note in
                            noteHistoryRow(note, contentLineLimit: 5)
                        }
                    }

                    if status == .done, visibleStatusNotes.count < allStatusNotes.count {
                        NoteHistoryMoreButton {
                            linearCompletedVisibleLimit = HistorySettingsData.nextVisibleLimit(
                                currentLimit: linearCompletedVisibleLimit,
                                pageSize: linearCompletedPageSize,
                                totalCount: allStatusNotes.count
                            )
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

    @ViewBuilder
    private func noteHistoryRow(
        _ note: VoxtNoteItem,
        contentLineLimit: Int = 2,
        fixedHeight: CGFloat? = nil
    ) -> some View {
        if let fixedHeight {
            noteHistoryRowContent(
                note,
                contentLineLimit: contentLineLimit,
                fixedHeight: fixedHeight
            )
        } else {
            noteHistoryRowContent(note, contentLineLimit: contentLineLimit, fixedHeight: nil)
                .padding(.vertical, historyRowVerticalInset)
        }
    }

    private func noteHistoryRowContent(
        _ note: VoxtNoteItem,
        contentLineLimit: Int,
        fixedHeight: CGFloat?
    ) -> some View {
        NoteHistoryRow(
            item: note,
            layout: noteViewMode == .linearCard ? .linearCard : .list,
            contentLineLimit: contentLineLimit,
            fixedHeight: fixedHeight,
            onCopy: {
                copyStringToPasteboard(note.text)
                copiedNoteID = note.id
                showCopyToast()
                Task {
                    try? await Task.sleep(for: .seconds(1.2))
                    if copiedNoteID == note.id {
                        copiedNoteID = nil
                    }
                }
            },
            onDoubleClick: {
                _ = noteStore.performDoubleClickAction(for: note.id)
            },
            onSetStatus: { status in
                _ = noteStore.setStatus(status, for: note.id)
            },
            onSetPriority: { priority in
                _ = noteStore.setPriority(priority, for: note.id)
            },
            onRename: { title in
                _ = noteStore.rename(note.id, to: title)
            },
            onUpdateDetails: { title, text in
                noteStore.updateDetails(note.id, title: title, text: text)
            },
            onReorder: { draggedNoteID in
                noteStore.reorder(noteID: draggedNoteID, relativeTo: note.id)
            },
            onDelete: {
                copiedNoteID = nil
                noteStore.delete(id: note.id)
            }
        )
    }

    @ViewBuilder
    private var historyList: some View {
        let items = historyListItems
        let list = PagedVerticalList(
            items: items,
            totalCount: historyListTotalCount,
            rowHeight: historyRowFallbackHeight,
            rowSpacing: historyRowSpacing,
            rowHeightForItemAtWidth: { item, width in
                historyRowHeight(for: item, width: width)
            },
            isLoading: isLoadingHistoryEntries,
            onLoadMore: { reloadHistoryEntries(reset: false) }
        ) { item in
            switch item {
            case .dayHeader(let date):
                HistoryDayHeader(date: date)
            case .entry(let entry):
                HistoryRow(
                    entry: entry,
                    audioURL: historyStore.audioURL(for: entry),
                    isCompact: false,
                    onCopy: {
                        copyStringToPasteboard(
                            HistoryCorrectionPresentation.correctedText(
                                for: entry.text,
                                snapshots: entry.dictionaryCorrectionSnapshots
                            )
                        )
                        copiedEntryID = entry.id
                        showCopyToast()
                        Task {
                            try? await Task.sleep(for: .seconds(1.2))
                            if copiedEntryID == entry.id {
                                copiedEntryID = nil
                            }
                        }
                    },
                    onShowInfo: {
                        showHistoryDetail(for: entry)
                    },
                    onDelete: { deleteHistoryEntry(entry) }
                )
                .padding(.vertical, historyRowVerticalInset)
            case .meetingEntry(let entry):
                HistoryListRow(
                    timeText: RelativeNoteTimestampFormatter.historyListTime(for: entry.createdAt),
                    displayText: entry.displayText,
                    onCopy: { copyHistoryEntry(id: entry.id) },
                    onShowInfo: { showHistoryDetail(for: entry) },
                    onDelete: { deleteHistoryEntry(id: entry.id) }
                )
                .padding(.vertical, historyRowVerticalInset)
            }
        }

        list.frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    private func historyRowHeight(for item: HistoryListItem, width: CGFloat) -> CGFloat {
        switch item {
        case .dayHeader:
            return 32
        case .entry(let entry):
            return historyRowHeightCache.height(
                for: entry,
                width: width,
                verticalInset: historyRowVerticalInset
            )
        case .meetingEntry(let entry):
            return HistoryRowHeightCache.estimate(
                text: entry.displayText,
                width: width,
                verticalInset: historyRowVerticalInset
            )
        }
    }

    private var historySearchListHeight: CGFloat {
        let visibleRowCount = max(1, min(visibleHistoryEntryCount, 5))
        let rowsHeight = CGFloat(visibleRowCount) * historyRowFallbackHeight
            + CGFloat(max(0, visibleRowCount - 1)) * historyRowSpacing
        let footerHeight: CGFloat = (isLoadingHistoryEntries || visibleHistoryEntryCount < totalHistoryEntryCount) ? 40 : 0
        return min(max(rowsHeight + footerHeight, historyRowFallbackHeight), 360)
    }

    private func scrollToNavigationTargetIfNeeded(using proxy: ScrollViewProxy) {
        guard let navigationRequest,
              navigationRequest.target.tab == .history,
              let section = navigationRequest.target.section
        else {
            return
        }

        DispatchQueue.main.async {
            withAnimation(.easeInOut(duration: 0.18)) {
                proxy.scrollTo(section.rawValue, anchor: .top)
            }
        }
    }

    private func applyNavigationTarget(_ target: SettingsNavigationTarget?) {
        guard target?.tab == .history,
              let historyFilter = target?.historyFilter
        else {
            return
        }

        selectedFilter = historyFilter
    }

    private func confirmBulkDeletion(_ target: HistoryBulkDeletionTarget) {
        copiedEntryID = nil
        copiedNoteID = nil
        dismissCopyToast()
        switch target {
        case .history(let filter):
            guard let kind = historyKind(for: filter), historyStore.clear(kind: kind) else { return }
            reloadHistoryEntries(reset: true)
        case .notes:
            noteStore.clearAll()
        }
    }

    private func reloadHistoryEntries(reset: Bool) {
        guard !isNoteTabSelected else {
            visibleHistoryEntries = []
            visibleMeetingHistoryEntries = []
            totalHistoryEntryCount = 0
            isLoadingHistoryEntries = false
            return
        }

        let loadedCount = selectedFilter == .transcript
            ? visibleMeetingHistoryEntries.count
            : visibleHistoryEntries.count
        let offset = reset ? 0 : loadedCount
        guard reset || offset < totalHistoryEntryCount else { return }
        guard reset || !isLoadingHistoryEntries else { return }

        let pageSize = selectedFilter == .transcript ? meetingHistoryPageSize : historyPageSize
        loadHistoryEntries(offset: offset, limit: pageSize, reset: reset)
    }

    private func loadHistoryEntries(offset: Int, limit: Int, reset: Bool) {
        historyPageGeneration += 1
        let generation = historyPageGeneration
        let kind = selectedHistoryKind
        let query = historySearchText
        isLoadingHistoryEntries = true

        if kind == .transcript {
            historyStore.loadListEntries(
                kind: kind,
                query: query,
                limit: limit,
                offset: offset
            ) { count, page in
                guard generation == historyPageGeneration else { return }
                totalHistoryEntryCount = count
                visibleMeetingHistoryEntries = reset ? page : visibleMeetingHistoryEntries + page
                visibleHistoryEntries = []
                isLoadingHistoryEntries = false
            }
            return
        }

        historyStore.loadEntries(
            kind: kind,
            query: query,
            limit: limit,
            offset: offset
        ) { count, page in
            guard generation == historyPageGeneration else { return }
            totalHistoryEntryCount = count
            visibleHistoryEntries = reset ? page : visibleHistoryEntries + page
            visibleMeetingHistoryEntries = []
            isLoadingHistoryEntries = false
        }
    }

    private func copyHistoryEntry(id: UUID) {
        historyStore.loadEntry(id: id) { entry in
            guard let entry else { return }
            copyStringToPasteboard(
                HistoryCorrectionPresentation.correctedText(
                    for: entry.text,
                    snapshots: entry.dictionaryCorrectionSnapshots
                )
            )
            copiedEntryID = entry.id
            showCopyToast()
            Task { @MainActor in
                try? await Task.sleep(for: .seconds(1.2))
                if copiedEntryID == entry.id {
                    copiedEntryID = nil
                }
            }
        }
    }

    private func deleteHistoryEntry(id: UUID) {
        historyStore.loadEntry(id: id) { entry in
            guard let entry else { return }
            deleteHistoryEntry(entry)
        }
    }

    private func deleteHistoryEntry(_ entry: TranscriptionHistoryEntry) {
        copiedEntryID = nil
        if selectedHistoryInfoEntry?.id == entry.id {
            selectedHistoryInfoEntry = nil
        }

        suppressedStoreHistoryReloadCount += 1
        guard historyStore.delete(id: entry.id) else {
            suppressedStoreHistoryReloadCount = max(0, suppressedStoreHistoryReloadCount - 1)
            return
        }
        refreshHistoryAudioStorageStats()

        guard let removedIndex = visibleHistoryEntries.firstIndex(where: { $0.id == entry.id }) else {
            if let removedIndex = visibleMeetingHistoryEntries.firstIndex(where: { $0.id == entry.id }) {
                visibleMeetingHistoryEntries.remove(at: removedIndex)
                totalHistoryEntryCount = max(0, totalHistoryEntryCount - 1)
                guard visibleMeetingHistoryEntries.count < totalHistoryEntryCount else { return }
                loadHistoryEntries(offset: visibleMeetingHistoryEntries.count, limit: 1, reset: false)
                return
            }
            reloadHistoryEntries(reset: true)
            return
        }

        visibleHistoryEntries.remove(at: removedIndex)
        totalHistoryEntryCount = max(0, totalHistoryEntryCount - 1)

        guard visibleHistoryEntries.count < totalHistoryEntryCount else { return }
        loadHistoryEntries(offset: visibleHistoryEntries.count, limit: 1, reset: false)
    }

    private func applyRetentionPolicyAndReload() {
        historyStore.updateRetentionPolicy()
        reloadHistoryEntries(reset: true)
        refreshHistoryAudioStorageStats()
    }

    private var selectedHistoryKind: TranscriptionHistoryKind? {
        historyKind(for: selectedFilter)
    }

    private func historyKind(for filter: HistoryFilterTab) -> TranscriptionHistoryKind? {
        switch filter {
        case .transcription:
            return .normal
        case .translation:
            return .translation
        case .transcript:
            return .transcript
        case .rewrite:
            return .rewrite
        case .note:
            return nil
        }
    }

    private func bulkDeletionTitle(for target: HistoryBulkDeletionTarget) -> String {
        switch target {
        case .history(let filter):
            return AppLocalization.format("Delete All %@ History?", filter.title)
        case .notes:
            return localized("Delete All Notes?")
        }
    }

    private func bulkDeletionMessage(for target: HistoryBulkDeletionTarget) -> String {
        switch target {
        case .history(let filter):
            return AppLocalization.format(
                "This will permanently delete all entries in %@ history.",
                filter.title
            )
        case .notes:
            return localized("This will permanently delete all notes.")
        }
    }

    private func openHistoryAudioStorageInFinder() {
        HistoryAudioStorageDirectoryManager.openRootInFinder()
    }

    private func chooseHistoryAudioStorageDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = HistoryAudioStorageDirectoryManager.resolvedRootURL()

        guard panel.runModal() == .OK, let selectedURL = panel.url else { return }

        do {
            try HistoryAudioStorageDirectoryManager.saveUserSelectedRootURL(selectedURL)
            historyAudioStorageSelectionError = nil
            refreshHistoryAudioStorageDisplayPath()
        } catch {
            historyAudioStorageSelectionError = AppLocalization.format(
                "Failed to update history audio storage path: %@",
                error.localizedDescription
            )
        }
    }

    private func refreshHistoryAudioStorageDisplayPath() {
        historyAudioStorageDisplayPath = HistoryAudioStorageDirectoryManager.resolvedRootURL().path
    }

    private func refreshHistoryAudioStorageStats() {
        historyAudioStatsGeneration += 1
        let generation = historyAudioStatsGeneration
        historyStore.currentAudioArchiveStorageStats { stats in
            guard generation == historyAudioStatsGeneration else { return }
            historyAudioStorageStats = stats
        }
    }

    private var historyAudioStorageStatsSummary: String {
        AppLocalization.format(
            "Saved audio: %d files · %@",
            historyAudioStorageStats.storedFileCount,
            formattedByteCount(historyAudioStorageStats.totalBytes)
        )
    }

    private func formattedByteCount(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useBytes, .useKB, .useMB, .useGB]
        formatter.countStyle = .file
        formatter.includesUnit = true
        formatter.isAdaptive = true
        return formatter.string(fromByteCount: bytes)
    }

    private func exportAllHistoryAudio() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = FileManager.default.homeDirectoryForCurrentUser

        guard panel.runModal() == .OK, let destinationURL = panel.url else { return }

        do {
            let summary = try historyStore.exportAllAudioArchives(to: destinationURL)
            historyAudioExportResultMessage = AppLocalization.format(
                "Exported %d audio files. Skipped %d. Failed %d.",
                summary.exportedCount,
                summary.skippedCount,
                summary.failedCount
            )
        } catch {
            historyAudioExportResultMessage = AppLocalization.format(
                "Audio export failed: %@",
                error.localizedDescription
            )
        }
        refreshHistoryAudioStorageStats()
    }

    private func showHistoryDetail(for entry: TranscriptionHistoryEntry) {
        guard entry.kind == .transcript, let appDelegate = AppDelegate.shared else {
            selectedHistoryInfoEntry = entry
            return
        }
        appDelegate.showMeetingDetailWindow(for: entry)
    }

    private func showHistoryDetail(for entry: TranscriptionHistoryListEntry) {
        guard let appDelegate = AppDelegate.shared else { return }
        appDelegate.showMeetingDetailWindow(for: entry.id)
    }

    private func showCopyToast() {
        showCopyToast(localized("Copied to clipboard"))
    }

    private func showCopyToast(_ message: String, duration: TimeInterval = 2.2) {
        copyToastDismissTask?.cancel()
        copyToastMessage = message
        copyToastDismissTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(duration))
            guard !Task.isCancelled else { return }
            copyToastMessage = ""
        }
    }

    private func dismissCopyToast() {
        copyToastDismissTask?.cancel()
        copyToastMessage = ""
    }
}
