import Foundation
import SwiftUI

struct MeetingDetailWindowView: View {
    @AppStorage(AppPreferenceKey.interfaceLanguage) private var interfaceLanguageRaw = AppInterfaceLanguage.system.rawValue
    @ObservedObject var viewModel: MeetingDetailViewModel
    @StateObject private var playbackController: MeetingDetailPlaybackController
    @State private var activeSegmentID: UUID?
    @State private var speakerRenameGroupID: String?
    @State private var speakerRenameDraft = ""
    @State private var isScrubbing = false
    @State private var scrollRequest: MeetingTranscriptScrollRequest?
    @State private var scrollGeneration: UInt64 = 0
    @State private var displayedSegmentIDs: Set<UUID> = []
    @State private var waveformData: MeetingWaveformData?

    init(viewModel: MeetingDetailViewModel) {
        self.viewModel = viewModel
        _playbackController = StateObject(wrappedValue: MeetingDetailPlaybackController(audioURL: viewModel.audioURL))
    }

    var body: some View {
        let _ = interfaceLanguageRaw
        ZStack {
            GeometryReader { proxy in
                let sidebarWidth = max(300, min(proxy.size.width / 3.0, 380))

                HStack(alignment: .top, spacing: 8) {
                    leftPane
                        .frame(maxWidth: .infinity, maxHeight: .infinity)

                    if viewModel.mode != .fileDraft, !viewModel.isSummaryCollapsed {
                        rightSidebar
                            .frame(width: sidebarWidth)
                            .frame(maxHeight: .infinity)
                    }
                }
                .padding(12)
            }
            .frame(minWidth: 980, minHeight: 650)
            .ignoresSafeArea(.container, edges: .top)
            .onAppear {
                viewModel.handleViewAppear()
                displayedSegmentIDs = Set(viewModel.displayedSegments.map(\.id))
                updateActiveSegment(for: playbackController.currentTime)
            }

            if speakerRenameGroupID != nil {
                dialogDimmingMask(opacity: 0.14)

                speakerRenameDialog
            }

            if viewModel.isUndoDeleteAvailable {
                undoDeleteBanner
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
                    .padding(.leading, 28)
                    .padding(.bottom, 24)
            }

        }
        .background(MeetingDetailUIStyle.windowFillColor)
        .ignoresSafeArea(.container, edges: .top)
        .sheet(isPresented: $viewModel.isTranslationLanguagePickerPresented) {
            translationLanguageDialog
        }
        .sheet(isPresented: $viewModel.isSummarySettingsPresented) {
            MeetingDetailSummarySettingsDialog(viewModel: viewModel)
        }
        .onChange(of: viewModel.segmentStructureRevision) { _, _ in
            displayedSegmentIDs = Set(viewModel.displayedSegments.map(\.id))
            updateActiveSegment(for: playbackController.currentTime)
            requestLiveScrollToNewestIfNeeded()
        }
        .onChange(of: viewModel.displayedSegments) { _, newValue in
            displayedSegmentIDs = Set(newValue.map(\.id))
        }
        .onChange(of: playbackController.currentTime) { _, newValue in
            guard viewModel.mode == .history else { return }
            updateActiveSegment(for: newValue)
        }
        .onChange(of: activeSegmentID) { _, newValue in
            requestHistoryScrollToActiveSegment(newValue)
        }
            .onChange(of: isScrubbing) { _, scrubbing in
                guard !scrubbing else { return }
                requestHistoryScrollToActiveSegment(activeSegmentID)
            }
            .task(id: viewModel.audioURL?.standardizedFileURL.path) {
                guard viewModel.mode == .history, let audioURL = viewModel.audioURL else { return }
                waveformData = await MeetingWaveformBuilder.load(from: audioURL)
            }
    }

    private func dialogDimmingMask(opacity: Double) -> some View {
        Color.black.opacity(opacity)
            .ignoresSafeArea()
            .clipShape(
                RoundedRectangle(cornerRadius: MeetingDetailUIStyle.windowCornerRadius, style: .continuous)
            )
    }

    private var leftPane: some View {
        VStack(alignment: .leading, spacing: 12) {
            topToolbar

            if viewModel.mode == .fileDraft {
                VStack(alignment: .leading, spacing: 4) {
                    Text(verbatim: viewModel.title).font(.headline).lineLimit(1)
                    Text(viewModel.subtitle).font(.caption).foregroundStyle(.secondary)
                }
                .padding(.horizontal, 12)
            }

            if viewModel.isSearchPresented {
                transcriptSearchBar
            }

            transcriptPane

            playbackPane
        }
    }

    private var topToolbar: some View {
        HStack(alignment: .center, spacing: 10) {
            Color.clear
                .frame(width: 62, height: 1)

            transcriptTabPicker

            if viewModel.transcriptPresentationMode == .timeline,
               viewModel.showsSpeakerDisplayModePicker {
                transcriptSpeakerDisplayModePicker
            }

            Spacer(minLength: 8)

            HStack(spacing: 6) {
                MeetingDetailSegmentActionButton(
                    action: viewModel.toggleSearchPresentation,
                    tint: Color.accentColor,
                    isActive: viewModel.isSearchPresented,
                    helpText: AppLocalization.localizedString("Search"),
                    accessibilityText: AppLocalization.localizedString("Search")
                ) {
                    MeetingDetailSearchIcon(
                        color: viewModel.isSearchPresented ? Color.accentColor : Color.secondary
                    )
                }

                MeetingDetailSegmentActionButton(
                    action: viewModel.toggleTranslation,
                    tint: Color.accentColor,
                    isActive: viewModel.translationEnabled,
                    helpText: AppLocalization.localizedString("Translate"),
                    accessibilityText: AppLocalization.localizedString("Translate"),
                    isDisabled: viewModel.mode == .fileDraft
                ) {
                    MeetingDetailTranslateIcon(
                        color: viewModel.translationEnabled ? Color.accentColor : Color.secondary
                    )
                }

                MeetingDetailSegmentActionButton(
                    action: { try? viewModel.export() },
                    tint: Color.secondary,
                    isActive: false,
                    helpText: AppLocalization.localizedString("Export"),
                    accessibilityText: AppLocalization.localizedString("Export"),
                    isDisabled: !viewModel.canExport
                ) {
                    MeetingDetailExportIcon(color: .secondary)
                }

                if viewModel.mode == .fileDraft {
                    Button(AppLocalization.localizedString("Copy")) {
                        _ = AppDelegate.shared?.pasteboardTextWriter.write(
                            MeetingTranscriptFormatter.joinedText(for: viewModel.segments),
                            to: .general, restorePrevious: false
                        )
                    }
                    .buttonStyle(MeetingPillButtonStyle())
                }

                Rectangle()
                    .fill(MeetingDetailUIStyle.dividerColor)
                    .frame(width: 1, height: 18)

                MeetingDetailSegmentActionButton(
                    action: viewModel.toggleSummaryCollapsed,
                    tint: Color.accentColor,
                    isActive: viewModel.isSummaryCollapsed,
                    helpText: AppLocalization.localizedString(
                        viewModel.isSummaryCollapsed ? "Expand Summary" : "Collapse Summary"
                    ),
                    accessibilityText: AppLocalization.localizedString(
                        viewModel.isSummaryCollapsed ? "Expand Summary" : "Collapse Summary"
                    ),
                    isDisabled: viewModel.mode == .fileDraft
                ) {
                    MeetingDetailSummaryCollapseIcon(
                        color: viewModel.isSummaryCollapsed ? Color.accentColor : Color.secondary
                    )
                }
            }
        }
    }

    private var transcriptTabPicker: some View {
        HStack(spacing: 2) {
            ForEach(viewModel.availableTranscriptPresentationModes) { mode in
                Button {
                    viewModel.setTranscriptPresentationMode(mode)
                } label: {
                    Text(transcriptTabTitle(for: mode))
                        .font(.system(size: 11.5, weight: .semibold))
                        .padding(.horizontal, 10)
                        .frame(height: 28)
                }
                .buttonStyle(.plain)
                .foregroundStyle(
                    viewModel.transcriptPresentationMode == mode
                        ? Color.accentColor
                        : Color.secondary
                )
                .background(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(
                            viewModel.transcriptPresentationMode == mode
                                ? Color.accentColor.opacity(0.14)
                                : .clear
                        )
                )
                .overlay {
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .stroke(
                            viewModel.transcriptPresentationMode == mode
                                ? Color.accentColor.opacity(0.45)
                                : .clear,
                            lineWidth: 1
                        )
                }
            }
        }
        .padding(2)
        .fixedSize(horizontal: true, vertical: false)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(MeetingDetailUIStyle.controlFillColor)
        )
        .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(MeetingDetailUIStyle.borderColor, lineWidth: 1)
        }
    }

    private func transcriptTabTitle(for mode: MeetingDetailViewModel.TranscriptPresentationMode) -> String {
        switch mode {
        case .timeline:
            return AppLocalization.localizedString("Timeline")
        case .speakerMarks:
            return AppLocalization.localizedString("Speaker Marks")
        }
    }

    private var transcriptSpeakerDisplayModePicker: some View {
        HStack(spacing: 2) {
            ForEach(MeetingDetailViewModel.TranscriptSpeakerDisplayMode.allCases) { mode in
                Button {
                    viewModel.setTranscriptSpeakerDisplayMode(mode)
                } label: {
                    Text(transcriptSpeakerDisplayModeTitle(for: mode))
                        .font(.system(size: 11.5, weight: .semibold))
                        .padding(.horizontal, 9)
                        .frame(height: 26)
                }
                .buttonStyle(.plain)
                .foregroundStyle(
                    viewModel.transcriptSpeakerDisplayMode == mode
                        ? Color.primary
                        : Color.secondary
                )
                .background(
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(
                            viewModel.transcriptSpeakerDisplayMode == mode
                                ? MeetingDetailUIStyle.windowFillColor
                                : .clear
                        )
                )
            }
        }
        .padding(2)
        .fixedSize(horizontal: true, vertical: false)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(MeetingDetailUIStyle.controlFillColor.opacity(0.72))
        )
        .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(MeetingDetailUIStyle.borderColor.opacity(0.72), lineWidth: 1)
        }
    }

    private func transcriptSpeakerDisplayModeTitle(
        for mode: MeetingDetailViewModel.TranscriptSpeakerDisplayMode
    ) -> String {
        switch mode {
        case .source:
            return AppLocalization.localizedString("Audio Source")
        case .speaker:
            return AppLocalization.localizedString("Speaker")
        }
    }

    private var transcriptSearchBar: some View {
        HStack(spacing: 10) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)

            TextField(
                AppLocalization.localizedString("Search transcript"),
                text: Binding(
                    get: { viewModel.searchQuery },
                    set: { viewModel.setSearchQuery($0) }
                )
            )
                .textFieldStyle(.plain)

            if !viewModel.searchQuery.isEmpty {
                Button {
                    viewModel.setSearchQuery("")
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .meetingDetailPanelSurface(cornerRadius: 12)
    }

    private var transcriptPane: some View {
        VStack(alignment: .leading, spacing: 16) {
            transcriptCaption

            if viewModel.isFinalizing {
                meetingFinalizationBanner
            }

            if viewModel.displayedSegments.isEmpty {
                transcriptEmptyState
            } else if viewModel.transcriptPresentationMode == .timeline {
                MeetingDetailTranscriptListPane(
                    rows: timelineVirtualRows,
                    showsTranslation: viewModel.translationEnabled,
                    scrollRequest: scrollRequest,
                    canEditTranscript: viewModel.canEditTranscript,
                    editingSegmentID: viewModel.editingSegmentID,
                    editingText: viewModel.editingText,
                    onSelectSegment: seekToSegment,
                    onBeginEditing: viewModel.beginEditingSegment,
                    onEditingTextChanged: { viewModel.editingText = $0 },
                    onSaveEditing: viewModel.saveEditingSegment,
                    onCancelEditing: viewModel.cancelEditingSegment,
                    onDeleteSegment: viewModel.deleteSegment,
                    onToggleHighlight: viewModel.toggleHighlight
                )
                .equatable()
            } else {
                speakerMarksPane
            }
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .meetingDetailPanelSurface(cornerRadius: 16)
    }

    private var timelineVirtualRows: [MeetingTranscriptVirtualRow] {
        MeetingTranscriptListSupport.timelineRows(
            from: viewModel.displayedSegments,
            activeSegmentID: activeSegmentID,
            showsTranslation: viewModel.translationEnabled,
            searchQuery: viewModel.searchQuery,
            speakerTitle: viewModel.timelineSpeakerTitle(for:)
        )
    }

    private var speakerMarkVirtualRows: [MeetingTranscriptVirtualRow] {
        MeetingTranscriptListSupport.speakerMarkRows(
            from: viewModel.speakerGroups,
            activeSegmentID: activeSegmentID,
            showsTranslation: viewModel.translationEnabled,
            searchQuery: viewModel.searchQuery
        )
    }

    private var transcriptCaption: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(AppLocalization.localizedString("Meeting Transcript"))
                .font(.system(size: 16, weight: .semibold))
                .foregroundStyle(.primary)

            Text(viewModel.subtitle)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(.secondary)
        }
    }

    @ViewBuilder
    private var transcriptEmptyState: some View {
        if viewModel.segments.isEmpty {
            VStack(spacing: 10) {
                if viewModel.isFinalizing {
                    ProgressView()
                        .controlSize(.small)
                }

                Text(
                    viewModel.isFinalizing
                        ? AppLocalization.localizedString("Preparing final transcript…")
                        : transcriptEmptyTitle
                )
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(.secondary)

                Text(
                    viewModel.isFinalizing
                        ? AppLocalization.localizedString("Voxt is finishing audio flushing, final transcription, and speaker analysis.")
                        : transcriptEmptyMessage
                )
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary.opacity(0.85))
            }
            .frame(maxWidth: .infinity, minHeight: 280, alignment: .center)
        } else {
            VStack(spacing: 10) {
                Text(AppLocalization.localizedString("No matching transcript segments."))
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.secondary)

                Text(AppLocalization.localizedString("Try a different keyword or clear the current search."))
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary.opacity(0.85))
            }
            .frame(maxWidth: .infinity, minHeight: 240, alignment: .center)
        }
    }

    private var transcriptEmptyTitle: String {
        if viewModel.mode == .history, viewModel.audioURL != nil {
            return AppLocalization.localizedString("No transcript was produced for this recording.")
        }
        return AppLocalization.localizedString("The transcript timeline for Me / Them will appear here once the meeting starts.")
    }

    private var transcriptEmptyMessage: String {
        if viewModel.mode == .history, viewModel.audioURL != nil {
            return AppLocalization.localizedString("The audio recording is saved and available for playback.")
        }
        return AppLocalization.localizedString("This panel stays focused on the detailed transcript and synced playback.")
    }

    private var meetingFinalizationBanner: some View {
        HStack(alignment: .center, spacing: 10) {
            ProgressView()
                .controlSize(.small)

            VStack(alignment: .leading, spacing: 3) {
                Text(AppLocalization.localizedString("Preparing final meeting details…"))
                    .font(.system(size: 12.5, weight: .semibold))
                    .foregroundStyle(.primary)

                Text(AppLocalization.localizedString("Current transcript is available now. Final text, speaker labels, audio playback, and summary will update when processing finishes."))
                    .font(.system(size: 11.5, weight: .medium))
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color.accentColor.opacity(0.08))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(Color.accentColor.opacity(0.16), lineWidth: 1)
        )
    }

    private var speakerMarksPane: some View {
        VStack(alignment: .leading, spacing: 16) {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 128), spacing: 10)], spacing: 10) {
                ForEach(viewModel.speakerGroups) { group in
                    speakerOverviewCard(for: group)
                }
            }
            .fixedSize(horizontal: false, vertical: true)

            MeetingDetailTranscriptListPane(
                rows: speakerMarkVirtualRows,
                showsTranslation: viewModel.translationEnabled,
                scrollRequest: scrollRequest,
                canEditTranscript: viewModel.canEditTranscript,
                editingSegmentID: viewModel.editingSegmentID,
                editingText: viewModel.editingText,
                onSelectSegment: seekToSegment,
                onBeginEditing: viewModel.beginEditingSegment,
                onEditingTextChanged: { viewModel.editingText = $0 },
                onSaveEditing: viewModel.saveEditingSegment,
                onCancelEditing: viewModel.cancelEditingSegment,
                onDeleteSegment: viewModel.deleteSegment,
                onToggleHighlight: viewModel.toggleHighlight
            )
            .equatable()
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private func speakerOverviewCard(for group: MeetingDetailSpeakerGroup) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Text(group.title)
                    .font(.system(size: 12, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)

                Spacer(minLength: 6)

                if viewModel.canEditSpeakers {
                    Button {
                        presentSpeakerRename(for: group)
                    } label: {
                        Image(systemName: "pencil")
                            .font(.system(size: 10.5, weight: .semibold))
                            .frame(width: 22, height: 22)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                    .help(AppLocalization.localizedString("Rename Speaker"))
                }
            }

            Text(AppLocalization.format("%d", group.segments.count))
                .font(.system(size: 22, weight: .semibold))
                .foregroundStyle(.primary)

            Text(AppLocalization.format("%d words", group.wordCount))
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(MeetingDetailUIStyle.controlFillColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(MeetingDetailUIStyle.softBorderColor, lineWidth: 1)
        )
    }

    private var speakerRenameDialog: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 5) {
                Text(AppLocalization.localizedString("Rename Speaker"))
                    .font(.system(size: 15, weight: .semibold))
                    .foregroundStyle(.primary)

                Text(AppLocalization.localizedString("This name will be applied to all matching transcript segments."))
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
            }

            TextField(AppLocalization.localizedString("Speaker name"), text: $speakerRenameDraft)
                .textFieldStyle(.plain)
                .font(.system(size: 13, weight: .medium))
                .padding(.horizontal, 12)
                .frame(height: 38)
                .background(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .fill(MeetingDetailUIStyle.controlFillColor)
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 10, style: .continuous)
                        .strokeBorder(MeetingDetailUIStyle.borderColor, lineWidth: 1)
                )

            HStack(spacing: 10) {
                Button(AppLocalization.localizedString("Cancel")) {
                    cancelSpeakerRename()
                }
                .buttonStyle(MeetingPillButtonStyle())

                Spacer(minLength: 8)

                Button(AppLocalization.localizedString("Save")) {
                    commitSpeakerRename()
                }
                .buttonStyle(MeetingPrimaryButtonStyle())
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(SettingsUIStyle.dialogPadding)
        .frame(width: 400)
        .background(SettingsUIStyle.windowBackgroundColor)
        .clipShape(
            RoundedRectangle(cornerRadius: SettingsUIStyle.dialogCornerRadius, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: SettingsUIStyle.dialogCornerRadius, style: .continuous)
                .strokeBorder(SettingsUIStyle.dialogBorderColor, lineWidth: 0.7)
        )
    }

    private var playbackPane: some View {
        MeetingDetailPlaybackPane(
            viewModel: viewModel,
            playbackController: playbackController,
            isScrubbing: $isScrubbing,
            waveformData: waveformData
        )
    }

    private var rightSidebar: some View {
        MeetingDetailSummarySidebar(viewModel: viewModel)
    }

    private var undoDeleteBanner: some View {
        HStack(spacing: 10) {
            Image(systemName: "trash")
                .font(.system(size: 11, weight: .semibold))

            Text(AppLocalization.localizedString("Transcript segment deleted"))
                .font(.system(size: 12, weight: .medium))

            Button(AppLocalization.localizedString("Undo")) {
                viewModel.undoDelete()
            }
            .buttonStyle(.plain)
            .font(.system(size: 12, weight: .semibold))
            .foregroundStyle(Color.accentColor)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .background(.regularMaterial, in: Capsule(style: .continuous))
        .overlay(
            Capsule(style: .continuous)
                .strokeBorder(Color.primary.opacity(0.12), lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.16), radius: 12, y: 4)
    }

    private func seekToSegment(_ segment: MeetingTranscriptSegment) {
        guard viewModel.mode == .history, playbackController.isAvailable else { return }
        playbackController.seek(to: segment.startSeconds)
        isScrubbing = false
    }

    private var translationLanguageDialog: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(AppLocalization.localizedString("Choose Translation Language"))
                .font(.title3.weight(.semibold))

            SettingsMenuPicker(
                selection: $viewModel.translationDraftLanguageRaw,
                options: TranslationTargetLanguage.allCases.map { language in
                    SettingsMenuOption(value: language.rawValue, title: language.title)
                },
                selectedTitle: translationDraftLanguage.title,
                width: 320
            )
            .frame(maxWidth: .infinity, alignment: .center)
            .padding(.vertical, 8)
            .accessibilityLabel(AppLocalization.localizedString("Target Language"))

            SettingsDialogActionRow {
                Button(AppLocalization.localizedString("Cancel")) {
                    viewModel.cancelTranslationLanguageSelection()
                }
                .buttonStyle(SettingsPillButtonStyle())
                .keyboardShortcut(.cancelAction)

                Button(AppLocalization.localizedString("Start Translation")) {
                    viewModel.confirmTranslationLanguageSelection()
                }
                .buttonStyle(SettingsPrimaryButtonStyle())
                .keyboardShortcut(.defaultAction)
            }
        }
        .settingsDialogChrome(width: 400, onClose: {
            viewModel.cancelTranslationLanguageSelection()
        })
    }

    private var translationDraftLanguage: TranslationTargetLanguage {
        TranslationTargetLanguage(rawValue: viewModel.translationDraftLanguageRaw) ?? .english
    }

    private func updateActiveSegment(for currentTime: TimeInterval) {
        guard viewModel.mode == .history else {
            activeSegmentID = nil
            return
        }
        guard currentTime > 0.01 || playbackController.isPlaying || isScrubbing else {
            activeSegmentID = nil
            return
        }
        let newActiveSegment = activeSegment(at: currentTime)
        activeSegmentID = newActiveSegment?.id
    }

    private func activeSegment(at currentTime: TimeInterval) -> MeetingTranscriptSegment? {
        let segments = viewModel.segments
        guard !segments.isEmpty else { return nil }

        var low = 0
        var high = segments.count
        while low < high {
            let mid = (low + high) / 2
            if segments[mid].startSeconds <= currentTime {
                low = mid + 1
            } else {
                high = mid
            }
        }

        return low > 0 ? segments[low - 1] : segments.first
    }

    private func requestHistoryScrollToActiveSegment(_ segmentID: UUID?) {
        guard viewModel.mode == .history else { return }
        guard !isScrubbing else { return }
        guard viewModel.transcriptPresentationMode == .timeline else { return }
        guard let segmentID, displayedSegmentIDs.contains(segmentID) else { return }
        scrollGeneration &+= 1
        scrollRequest = MeetingTranscriptScrollRequest(
            rowID: segmentID.uuidString,
            anchor: .center,
            generation: scrollGeneration
        )
    }

    private func requestLiveScrollToNewestIfNeeded() {
        guard viewModel.mode == .live else { return }
        guard viewModel.transcriptPresentationMode == .timeline else { return }
        guard let newest = viewModel.displayedNewestSegmentID() else { return }
        scrollGeneration &+= 1
        scrollRequest = MeetingTranscriptScrollRequest(
            rowID: newest.uuidString,
            anchor: .bottom,
            generation: scrollGeneration
        )
    }

    private func presentSpeakerRename(for group: MeetingDetailSpeakerGroup) {
        speakerRenameGroupID = group.id
        speakerRenameDraft = group.title
    }

    private func cancelSpeakerRename() {
        speakerRenameGroupID = nil
        speakerRenameDraft = ""
    }

    private func commitSpeakerRename() {
        guard let speakerRenameGroupID else { return }
        viewModel.renameSpeaker(identityKey: speakerRenameGroupID, displayName: speakerRenameDraft)
        cancelSpeakerRename()
    }
}
