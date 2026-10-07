// MeetingDetailViewModel.swift
// Provides Meeting Detail View Model for meeting detail windows.

import Combine
import Foundation

@MainActor
enum MeetingSummaryLoadState: Equatable {
    case idle
    case loading
    case unavailable(String)
    case failed(String)
}

@MainActor
final class MeetingDetailViewModel: ObservableObject {
    enum TranscriptPresentationMode: String, CaseIterable, Identifiable {
        case timeline
        case speakerMarks

        var id: String { rawValue }
    }

    enum TranscriptSpeakerDisplayMode: String, CaseIterable, Identifiable {
        case source
        case speaker

        var id: String { rawValue }
    }

    enum Mode {
        case history
        case live
        case fileDraft
    }

    @Published private(set) var title: String
    @Published private(set) var subtitle: String
    @Published private(set) var segments: [MeetingTranscriptSegment]
    @Published private(set) var segmentStructureRevision = 0
    @Published private(set) var displayedSegments: [MeetingTranscriptSegment] = []
    @Published private(set) var speakerGroups: [MeetingDetailSpeakerGroup] = []
    @Published private(set) var isPaused = false
    @Published private(set) var isFinalizing = false
    @Published var translationEnabled: Bool
    @Published var isTranslationLanguagePickerPresented = false
    @Published var translationDraftLanguageRaw: String

    @Published private(set) var summary: MeetingSummarySnapshot?
    @Published private(set) var summaryChatMessages: [MeetingSummaryChatMessage]
    @Published private(set) var summaryState: MeetingSummaryLoadState = .idle
    @Published private(set) var isSummaryStale = false
    @Published private(set) var isSummaryChatLoading = false
    @Published private(set) var summaryChatErrorMessage: String?
    @Published var isSummarySettingsPresented = false
    @Published var summaryAutoGenerate: Bool
    @Published var summaryPromptTemplate: String
    @Published var summaryModelSelectionID: String
    @Published var summaryChatDraft = ""
    @Published var transcriptPresentationModeRaw = TranscriptPresentationMode.timeline.rawValue
    @Published var transcriptSpeakerDisplayModeRaw = TranscriptSpeakerDisplayMode.source.rawValue
    @Published var isSearchPresented = false
    @Published private(set) var searchQuery = ""
    @Published var isSummaryCollapsed = false
    @Published private(set) var editingSegmentID: UUID?
    @Published var editingText = ""
    @Published private(set) var isUndoDeleteAvailable = false

    private(set) var speakerOrdinalByIdentityKey: [String: Int] = [:]

    let mode: Mode
    let audioURL: URL?
    let captureMode: MeetingCaptureMode
    @Published private(set) var summaryModelOptions: [MeetingSummaryModelOption]

    let historyEntryID: UUID?
    private let translationHandler: MeetingDetailWindowManager.TranslationHandler
    private let summarySettingsProvider: MeetingDetailWindowManager.SummarySettingsProvider?
    private let summaryModelOptionsProvider: MeetingDetailWindowManager.SummaryModelOptionsProvider?
    let summaryStatusProvider: MeetingDetailWindowManager.SummaryStatusProvider?
    private let summaryGenerator: MeetingDetailWindowManager.SummaryGenerator?
    private let summaryPersistence: MeetingDetailWindowManager.SummaryPersistence?
    private let summaryStalePersistence: MeetingDetailWindowManager.SummaryStalePersistence?
    private let summaryChatAnswerer: MeetingDetailWindowManager.SummaryChatAnswerer?
    private let summaryChatPersistence: MeetingDetailWindowManager.SummaryChatPersistence?
    let transcriptSegmentsPersistence: MeetingDetailWindowManager.TranscriptSegmentsPersistence?
    private let historySubtitle: String?

    private var cancellables = Set<AnyCancellable>()
    private var isLiveRecording = false
    private let translationScheduler = MeetingRealtimeTranslationScheduler(workClass: .detailTranslation)
    private var failedTranslationRevisions: [UUID: String] = [:]
    private var summaryTask: Task<Void, Never>?
    private var summaryChatTask: Task<Void, Never>?
    private var summaryGeneration = 0
    private var cachedSummaryTranscript: String?
    private var hasHandledInitialAppearance = false
    private var deletedSegmentForUndo: (segment: MeetingTranscriptSegment, index: Int)?
    private var undoDeleteTask: Task<Void, Never>?

    init(
        title: String,
        subtitle: String,
        historyEntryID: UUID,
        initialSummary: MeetingSummarySnapshot?,
        initialSummaryStale: Bool = false,
        initialSummaryChatMessages: [MeetingSummaryChatMessage],
        initialSummarySettings: MeetingSummarySettingsSnapshot,
        summaryModelOptions: [MeetingSummaryModelOption],
        summarySettingsProvider: @escaping MeetingDetailWindowManager.SummarySettingsProvider,
        summaryModelOptionsProvider: @escaping MeetingDetailWindowManager.SummaryModelOptionsProvider,
        segments: [MeetingTranscriptSegment],
        captureMode: MeetingCaptureMode? = nil,
        audioURL: URL?,
        translationHandler: @escaping MeetingDetailWindowManager.TranslationHandler,
        summaryStatusProvider: @escaping MeetingDetailWindowManager.SummaryStatusProvider,
        summaryGenerator: @escaping MeetingDetailWindowManager.SummaryGenerator,
        summaryPersistence: @escaping MeetingDetailWindowManager.SummaryPersistence,
        summaryStalePersistence: @escaping MeetingDetailWindowManager.SummaryStalePersistence = { _, _ in nil },
        summaryChatAnswerer: @escaping MeetingDetailWindowManager.SummaryChatAnswerer,
        summaryChatPersistence: @escaping MeetingDetailWindowManager.SummaryChatPersistence,
        transcriptSegmentsPersistence: @escaping MeetingDetailWindowManager.TranscriptSegmentsPersistence
    ) {
        self.mode = .history
        self.title = title
        self.captureMode = captureMode ?? Self.inferredCaptureMode(from: segments)
        self.subtitle = AppLocalization.format("%@ · %@", self.captureMode.title, subtitle)
        self.historyEntryID = historyEntryID
        self.summary = initialSummary
        self.isSummaryStale = initialSummaryStale
        self.summaryChatMessages = initialSummaryChatMessages
        self.summaryModelOptions = summaryModelOptions
        self.segments = segments
        self.audioURL = audioURL
        self.isPaused = true
        self.isFinalizing = false
        self.translationHandler = translationHandler
        self.summarySettingsProvider = summarySettingsProvider
        self.summaryModelOptionsProvider = summaryModelOptionsProvider
        self.summaryStatusProvider = summaryStatusProvider
        self.summaryGenerator = summaryGenerator
        self.summaryPersistence = summaryPersistence
        self.summaryStalePersistence = summaryStalePersistence
        self.summaryChatAnswerer = summaryChatAnswerer
        self.summaryChatPersistence = summaryChatPersistence
        self.transcriptSegmentsPersistence = transcriptSegmentsPersistence
        self.historySubtitle = subtitle

        self.translationDraftLanguageRaw = Self.initialTranslationLanguageRaw()
        self.translationEnabled = Self.segmentsContainTranslations(segments)

        let resolvedConfiguration = Self.resolveSummaryConfiguration(
            settings: initialSummarySettings,
            modelOptions: summaryModelOptions,
            currentSelectionID: nil
        )
        self.summaryAutoGenerate = resolvedConfiguration.autoGenerate
        self.summaryPromptTemplate = resolvedConfiguration.promptTemplate
        self.summaryModelSelectionID = resolvedConfiguration.modelSelectionID

        if initialSummary != nil {
            summaryState = .idle
        }
        bindInterfaceLanguageChanges()
        refreshTranscriptListCaches()
    }

    /// Completed ASR checkpoint, not a history entry. No writable persistence or
    /// inference handlers are installed until the task finishes successfully.
    init(fileTaskTitle: String, segments: [MeetingTranscriptSegment]) {
        self.mode = .fileDraft
        self.title = fileTaskTitle
        self.subtitle = AppLocalization.localizedString("Transcription ready — speaker analysis incomplete")
        self.captureMode = .meeting
        self.historyEntryID = nil
        self.summary = nil
        self.summaryChatMessages = []
        self.summaryModelOptions = []
        self.segments = segments
        self.audioURL = nil // Do not give the player a deletable queue-cache path.
        self.translationHandler = { _, _ in .cancelled() }
        self.summarySettingsProvider = nil
        self.summaryModelOptionsProvider = nil
        self.summaryStatusProvider = nil
        self.summaryGenerator = nil
        self.summaryPersistence = nil
        self.summaryStalePersistence = nil
        self.summaryChatAnswerer = nil
        self.summaryChatPersistence = nil
        self.transcriptSegmentsPersistence = nil
        self.historySubtitle = nil
        self.translationDraftLanguageRaw = Self.initialTranslationLanguageRaw()
        self.translationEnabled = false
        self.summaryAutoGenerate = false
        self.summaryPromptTemplate = ""
        self.summaryModelSelectionID = ""
        self.isSummaryCollapsed = true
        bindInterfaceLanguageChanges()
        refreshTranscriptListCaches()
    }

    init(
        liveState: MeetingOverlayState,
        initialSummarySettings: MeetingSummarySettingsSnapshot,
        summaryModelOptions: [MeetingSummaryModelOption],
        summarySettingsProvider: @escaping MeetingDetailWindowManager.SummarySettingsProvider,
        summaryModelOptionsProvider: @escaping MeetingDetailWindowManager.SummaryModelOptionsProvider,
        translationHandler: @escaping MeetingDetailWindowManager.TranslationHandler
    ) {
        self.mode = .live
        self.captureMode = liveState.captureMode
        self.title = AppLocalization.localizedString("Meeting Details")
        let liveSubtitle = liveState.isPaused
            ? AppLocalization.localizedString("Meeting Paused")
            : AppLocalization.localizedString("Meeting In Progress")
        self.subtitle = AppLocalization.format("%@ · %@", liveState.captureMode.title, liveSubtitle)
        self.historyEntryID = nil
        self.summary = nil
        self.summaryChatMessages = []
        self.summaryModelOptions = summaryModelOptions
        self.segments = Self.liveDisplaySegments(from: liveState.segments)
        self.audioURL = nil
        self.isPaused = liveState.isPaused
        self.isFinalizing = liveState.isFinalizing
        self.translationHandler = translationHandler
        self.summarySettingsProvider = summarySettingsProvider
        self.summaryModelOptionsProvider = summaryModelOptionsProvider
        self.summaryStatusProvider = nil
        self.summaryGenerator = nil
        self.summaryPersistence = nil
        self.summaryStalePersistence = nil
        self.summaryChatAnswerer = nil
        self.summaryChatPersistence = nil
        self.transcriptSegmentsPersistence = nil
        self.historySubtitle = nil
        self.isLiveRecording = liveState.isRecording

        self.translationDraftLanguageRaw = Self.initialTranslationLanguageRaw()
        self.translationEnabled = liveState.realtimeTranslateEnabled || Self.segmentsContainTranslations(liveState.segments)

        let resolvedConfiguration = Self.resolveSummaryConfiguration(
            settings: initialSummarySettings,
            modelOptions: summaryModelOptions,
            currentSelectionID: nil
        )
        self.summaryAutoGenerate = resolvedConfiguration.autoGenerate
        self.summaryPromptTemplate = resolvedConfiguration.promptTemplate
        self.summaryModelSelectionID = resolvedConfiguration.modelSelectionID

        liveState.$segments
            .receive(on: RunLoop.main)
            .sink { [weak self] segments in
                self?.updateLiveSegments(segments)
            }
            .store(in: &cancellables)

        Publishers.CombineLatest(liveState.$isPaused, liveState.$isRecording)
            .receive(on: RunLoop.main)
            .sink { [weak self] isPaused, isRecording in
                self?.isPaused = isPaused
                self?.isLiveRecording = isRecording
                self?.updateLiveSubtitle(isPaused: isPaused, isRecording: isRecording)
            }
            .store(in: &cancellables)

        bindInterfaceLanguageChanges()
        refreshTranscriptListCaches()

        liveState.$realtimeTranslateEnabled
            .receive(on: RunLoop.main)
            .sink { [weak self] isEnabled in
                guard let self else { return }
                if isEnabled {
                    self.translationEnabled = true
                }
            }
            .store(in: &cancellables)

        liveState.$isFinalizing
            .receive(on: RunLoop.main)
            .sink { [weak self] isFinalizing in
                guard let self else { return }
                self.isFinalizing = isFinalizing
                self.updateLiveSubtitle()
            }
            .store(in: &cancellables)
    }

    deinit {
        summaryTask?.cancel()
        summaryChatTask?.cancel()
        undoDeleteTask?.cancel()
    }

    func handleViewAppear() {
        guard !hasHandledInitialAppearance else { return }
        hasHandledInitialAppearance = true
        refreshSummaryConfigurationFromProviders()

        switch mode {
        case .history:
            guard !segments.isEmpty else {
                summaryState = .idle
                return
            }
            if summary != nil {
                summaryState = .idle
                return
            }
            guard summaryAutoGenerate else {
                summaryState = .idle
                return
            }
            regenerateSummary(isAutomatic: true)
        case .live, .fileDraft:
            summaryState = .idle
        }
    }

    func setTranslationEnabled(_ isEnabled: Bool) {
        guard mode != .fileDraft else { return }
        guard isEnabled else {
            isTranslationLanguagePickerPresented = false
            translationEnabled = false
            cancelTranslationTasks()
            clearPendingTranslationState()
            failedTranslationRevisions.removeAll()
            return
        }

        if Self.segmentsContainTranslations(segments) {
            translationEnabled = true
            failedTranslationRevisions.removeAll()
            translateEligibleSegmentsIfNeeded(targetLanguage: resolvedStoredTranslationLanguage())
            return
        }

        translationDraftLanguageRaw = resolvedStoredTranslationLanguage().rawValue
        isTranslationLanguagePickerPresented = true
        translationEnabled = false
    }

    func confirmTranslationLanguageSelection() {
        guard mode != .fileDraft else { return }
        guard let language = TranslationTargetLanguage(rawValue: translationDraftLanguageRaw) else {
            cancelTranslationLanguageSelection()
            return
        }

        UserDefaults.standard.set(
            language.rawValue,
            forKey: AppPreferenceKey.meetingRealtimeTranslationTargetLanguage
        )
        isTranslationLanguagePickerPresented = false
        translationEnabled = true
        failedTranslationRevisions.removeAll()
        translateEligibleSegmentsIfNeeded(targetLanguage: language)
    }

    func cancelTranslationLanguageSelection() {
        isTranslationLanguagePickerPresented = false
        translationEnabled = false
    }

    func toggleTranslation() {
        setTranslationEnabled(!translationEnabled)
    }

    func setTranscriptPresentationMode(_ mode: TranscriptPresentationMode) {
        guard self.mode != .fileDraft || mode == .timeline else { return }
        guard captureMode.capabilities.allowsSpeakerFeatures || mode == .timeline else { return }
        transcriptPresentationModeRaw = mode.rawValue
    }

    func setTranscriptSpeakerDisplayMode(_ mode: TranscriptSpeakerDisplayMode) {
        guard self.mode != .fileDraft, captureMode.capabilities.allowsSpeakerFeatures else { return }
        transcriptSpeakerDisplayModeRaw = mode.rawValue
        refreshTranscriptListCaches()
    }

    func toggleSearchPresentation() {
        isSearchPresented.toggle()
        if !isSearchPresented {
            setSearchQuery("")
        }
    }

    func setSearchQuery(_ query: String) {
        guard searchQuery != query else { return }
        searchQuery = query
        refreshTranscriptListCaches()
    }

    func toggleSummaryCollapsed() {
        guard mode != .fileDraft else { return }
        isSummaryCollapsed.toggle()
    }

    func renameSpeaker(identityKey: String, displayName: String) {
        guard canEditSpeakers, historyEntryID != nil else { return }

        let trimmedDisplayName = displayName.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedDisplayName = trimmedDisplayName.isEmpty ? nil : trimmedDisplayName

        let updatedSegments = segments.map { segment in
            guard speakerRenameIdentityMatches(segment, identityKey: identityKey) else { return segment }
            return segment.updatingSpeakerDisplayName(normalizedDisplayName)
        }

        guard updatedSegments != segments else { return }
        guard persistTranscriptSegments(updatedSegments) else { return }
        applyTranscriptMutation(updatedSegments, shouldRestartTranslation: false)
    }

    func beginEditingSegment(_ segment: MeetingTranscriptSegment) {
        guard canEditTranscript else { return }
        guard segments.contains(where: { $0.id == segment.id }) else { return }
        editingSegmentID = segment.id
        editingText = segment.text
    }

    func cancelEditingSegment() {
        editingSegmentID = nil
        editingText = ""
    }

    func saveEditingSegment() {
        guard canEditTranscript,
              let editingSegmentID,
              let currentSegment = segments.first(where: { $0.id == editingSegmentID })
        else {
            return
        }

        let trimmedText = editingText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else { return }

        let updatedSegment = currentSegment.updatingText(trimmedText)
        var updatedSegments = segments
        guard let index = updatedSegments.firstIndex(where: { $0.id == editingSegmentID }) else { return }
        updatedSegments[index] = updatedSegment

        guard persistTranscriptSegments(updatedSegments) else { return }
        cancelEditingSegment()
        applyTranscriptMutation(updatedSegments, shouldRestartTranslation: true)
    }

    func deleteSegment(_ segment: MeetingTranscriptSegment) {
        guard canEditTranscript,
              let index = segments.firstIndex(where: { $0.id == segment.id })
        else {
            return
        }

        var updatedSegments = segments
        updatedSegments.remove(at: index)
        guard persistTranscriptSegments(updatedSegments) else { return }

        if editingSegmentID == segment.id {
            cancelEditingSegment()
        }
        deletedSegmentForUndo = (segment: segment, index: index)
        isUndoDeleteAvailable = true
        undoDeleteTask?.cancel()
        undoDeleteTask = Task { [weak self] in
            try? await Task.sleep(for: .seconds(5))
            guard !Task.isCancelled else { return }
            await MainActor.run {
                self?.clearUndoDelete()
            }
        }
        applyTranscriptMutation(updatedSegments, shouldRestartTranslation: true)
    }

    func undoDelete() {
        guard canEditTranscript,
              let deletedSegmentForUndo
        else {
            return
        }

        var restoredSegments = segments
        let insertionIndex = min(max(deletedSegmentForUndo.index, 0), restoredSegments.count)
        restoredSegments.insert(deletedSegmentForUndo.segment, at: insertionIndex)
        guard persistTranscriptSegments(restoredSegments) else { return }

        clearUndoDelete()
        applyTranscriptMutation(restoredSegments, shouldRestartTranslation: true)
    }

    func toggleHighlight(for segment: MeetingTranscriptSegment) {
        guard canEditTranscript,
              let index = segments.firstIndex(where: { $0.id == segment.id })
        else {
            return
        }

        var updatedSegments = segments
        updatedSegments[index] = segment.updatingHighlight(!segment.isHighlighted)
        guard persistTranscriptSegments(updatedSegments) else { return }
        applyTranscriptMutation(
            updatedSegments,
            shouldRestartTranslation: false,
            marksSummaryStale: false
        )
    }

    private func persistTranscriptSegments(_ updatedSegments: [MeetingTranscriptSegment]) -> Bool {
        guard let historyEntryID, let transcriptSegmentsPersistence else { return false }
        _ = transcriptSegmentsPersistence(historyEntryID, updatedSegments)
        return true
    }

    private func applyTranscriptMutation(
        _ updatedSegments: [MeetingTranscriptSegment],
        shouldRestartTranslation: Bool,
        marksSummaryStale: Bool = true
    ) {
        if shouldRestartTranslation {
            cancelTranslationTasks()
        }
        if marksSummaryStale {
            summaryTask?.cancel()
            summaryTask = nil
            summaryGeneration &+= 1
            summaryState = .idle
        }
        segments = updatedSegments
        segmentStructureRevision &+= 1
        cachedSummaryTranscript = nil
        if marksSummaryStale {
            isSummaryStale = true
            if let historyEntryID {
                _ = summaryStalePersistence?(historyEntryID, true)
            }
        }
        refreshTranscriptListCaches()
        if shouldRestartTranslation, translationEnabled {
            translateEligibleSegmentsIfNeeded(targetLanguage: resolvedStoredTranslationLanguage())
        }
    }

    private func clearUndoDelete() {
        undoDeleteTask?.cancel()
        undoDeleteTask = nil
        deletedSegmentForUndo = nil
        isUndoDeleteAvailable = false
    }

    private func speakerRenameIdentityMatches(_ segment: MeetingTranscriptSegment, identityKey: String) -> Bool {
        if segment.speakerIdentityKey == identityKey {
            return true
        }

        let displayPrefix = "display:"
        guard identityKey.hasPrefix(displayPrefix) else { return false }
        let expectedDisplayName = String(identityKey.dropFirst(displayPrefix.count))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !expectedDisplayName.isEmpty else { return false }

        let currentDisplayName = segment.speakerDisplayName?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return currentDisplayName == expectedDisplayName
    }

    func presentSummarySettings() {
        refreshSummaryConfigurationFromProviders()
        isSummarySettingsPresented = true
    }

    func setSummaryAutoGenerate(_ isEnabled: Bool) {
        guard mode != .fileDraft else { return }
        summaryAutoGenerate = isEnabled
        FeatureSettingsStore.update(defaults: .standard) { settings in
            settings.meeting.summaryAutoGenerate = isEnabled
        }
        if !isEnabled && summary == nil {
            summaryState = .idle
        }
    }

    func setSummaryPromptTemplate(_ promptTemplate: String) {
        guard mode != .fileDraft else { return }
        summaryPromptTemplate = promptTemplate
        FeatureSettingsStore.update(defaults: .standard) { settings in
            settings.meeting.summaryPrompt = AppPromptDefaults.canonicalStoredText(
                promptTemplate,
                kind: .transcriptSummary
            )
        }
    }

    func resetSummaryPromptTemplate() {
        setSummaryPromptTemplate(AppPromptDefaults.text(for: .transcriptSummary))
    }

    func setSummaryModelSelectionID(_ selectionID: String) {
        guard mode != .fileDraft else { return }
        summaryModelSelectionID = selectionID
        FeatureSettingsStore.update(defaults: .standard) { settings in
            settings.meeting.summaryModelSelectionID = FeatureModelSelectionID
                .fromTranscriptSummaryModelSelection(selectionID)
                ?? FeatureModelSelectionID(rawValue: selectionID)
        }
    }

    func regenerateSummary(isAutomatic: Bool = false) {
        guard mode == .history, let historyEntryID, let summaryGenerator else {
            summaryState = .unavailable(summaryProviderMessage)
            return
        }

        let providerStatus = summaryProviderStatus
        guard providerStatus.isAvailable else {
            summaryState = .unavailable(providerStatus.message)
            return
        }
        let settings = summarySettingsSnapshot
        let segmentsSnapshot = segments
        summaryGeneration &+= 1
        let generation = summaryGeneration

        if !isAutomatic || summary == nil {
            summaryState = .loading
        }

        summaryTask?.cancel()
        summaryTask = Task { [weak self] in
            guard let self else { return }
            do {
                if isAutomatic {
                    try await Task.sleep(for: .milliseconds(220))
                }
                let transcript = await self.summaryTranscript(for: segmentsSnapshot)
                guard !transcript.isEmpty else {
                    await MainActor.run {
                        self.summaryState = .failed(AppLocalization.localizedString("No meeting transcript is available yet."))
                    }
                    return
                }
                let generated = try await summaryGenerator(transcript, settings)
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    guard self.summaryGeneration == generation, !Task.isCancelled else { return }
                    self.summary = generated
                    self.summaryState = .idle
                    self.isSummaryStale = false
                    _ = self.summaryPersistence?(historyEntryID, generated)
                }
            } catch is CancellationError {
                return
            } catch {
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    self.summaryState = .failed(error.localizedDescription)
                }
            }
        }
    }

    func sendSummaryChat() {
        guard mode == .history,
              let historyEntryID,
              let summaryChatAnswerer
        else {
            return
        }

        let question = summaryChatDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty else { return }

        summaryChatDraft = ""
        summaryChatErrorMessage = nil
        let userMessage = MeetingSummaryChatMessage(role: .user, content: question)
        summaryChatMessages.append(userMessage)
        _ = summaryChatPersistence?(historyEntryID, summaryChatMessages)
        isSummaryChatLoading = true

        let settings = summarySettingsSnapshot
        let existingHistory = summaryChatMessages
        let currentSummary = summary
        let segmentsSnapshot = segments

        summaryChatTask?.cancel()
        summaryChatTask = Task { [weak self] in
            guard let self else { return }
            do {
                let transcript = await self.summaryTranscript(for: segmentsSnapshot)
                let relevantContext = await Task.detached(priority: .utility) {
                    MeetingSummaryContextPlanning.relevantFollowUpContext(
                        transcript: transcript,
                        question: question
                    )
                }.value
                let answer = try await summaryChatAnswerer(
                    relevantContext,
                    currentSummary,
                    existingHistory,
                    question,
                    settings
                )
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    let assistantMessage = MeetingSummaryChatMessage(role: .assistant, content: answer)
                    self.summaryChatMessages.append(assistantMessage)
                    self.isSummaryChatLoading = false
                    self.summaryChatErrorMessage = nil
                    _ = self.summaryChatPersistence?(historyEntryID, self.summaryChatMessages)
                }
            } catch is CancellationError {
                return
            } catch {
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    self.isSummaryChatLoading = false
                    self.summaryChatErrorMessage = error.localizedDescription
                    _ = self.summaryChatPersistence?(historyEntryID, self.summaryChatMessages)
                }
            }
        }
    }

    func updateLiveSegments(_ incomingSegments: [MeetingTranscriptSegment]) {
        cachedSummaryTranscript = nil
        let targetLanguage = resolvedStoredTranslationLanguage()
        let updatedSegments = mergeSegmentsPreservingTranslationState(
            Self.liveDisplaySegments(from: incomingSegments)
        ).map { segment in
            guard failedTranslationRevisions[segment.id] == translationRevision(
                for: segment,
                targetLanguage: targetLanguage
            ) else { return segment }
            return segment.updatingTranslation(
                translatedText: segment.translatedText,
                isTranslationPending: false
            )
        }
        guard updatedSegments != segments else { return }
        segments = updatedSegments
        segmentStructureRevision &+= 1
        refreshTranscriptListCaches()
        if translationEnabled {
            translateEligibleSegmentsIfNeeded(targetLanguage: targetLanguage)
        }
    }

    static func liveDisplaySegments(
        from segments: [MeetingTranscriptSegment]
    ) -> [MeetingTranscriptSegment] {
        MeetingTranscriptPostProcessor.process(
            segments,
            options: .liveOverlay
        )
    }

    private func updateLiveSubtitle(
        isPaused: Bool? = nil,
        isRecording: Bool? = nil
    ) {
        guard mode == .live else { return }
        if isFinalizing {
            subtitle = AppLocalization.localizedString("Preparing final meeting details")
        } else if isPaused ?? self.isPaused {
            subtitle = AppLocalization.localizedString("Meeting Paused")
        } else if isRecording ?? isLiveRecording {
            subtitle = AppLocalization.localizedString("Meeting In Progress")
        } else {
            subtitle = AppLocalization.localizedString("Meeting Ended")
        }
    }

    private func bindInterfaceLanguageChanges() {
        NotificationCenter.default.publisher(for: .voxtInterfaceLanguageDidChange)
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in
                guard let self else { return }
                switch self.mode {
                case .history:
                    guard let historySubtitle = self.historySubtitle else { return }
                    self.subtitle = AppLocalization.format(
                        "%@ · %@",
                        self.captureMode.title,
                        historySubtitle
                    )
                case .live:
                    self.title = AppLocalization.localizedString("Meeting Details")
                    self.updateLiveSubtitle()
                case .fileDraft:
                    self.subtitle = AppLocalization.localizedString("Transcription ready — speaker analysis incomplete")
                }
            }
            .store(in: &cancellables)
    }

    private func summaryTranscript(for snapshot: [MeetingTranscriptSegment]) async -> String {
        if let cachedSummaryTranscript {
            return cachedSummaryTranscript
        }
        let transcript = await Task.detached(priority: .utility) {
            MeetingTranscriptFormatter.llmInputText(for: snapshot)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }.value
        cachedSummaryTranscript = transcript
        return transcript
    }

    func refreshSummaryConfiguration(
        settings: MeetingSummarySettingsSnapshot,
        modelOptions: [MeetingSummaryModelOption]
    ) {
        guard mode != .fileDraft else { return }
        summaryModelOptions = modelOptions
        let resolvedConfiguration = Self.resolveSummaryConfiguration(
            settings: settings,
            modelOptions: modelOptions,
            currentSelectionID: summaryModelSelectionID
        )
        summaryAutoGenerate = resolvedConfiguration.autoGenerate
        summaryPromptTemplate = resolvedConfiguration.promptTemplate
        summaryModelSelectionID = resolvedConfiguration.modelSelectionID
    }

    private func mergeSegmentsPreservingTranslationState(_ incomingSegments: [MeetingTranscriptSegment]) -> [MeetingTranscriptSegment] {
        let existingByID = Dictionary(uniqueKeysWithValues: segments.map { ($0.id, $0) })
        return incomingSegments.map { incoming in
            guard let existing = existingByID[incoming.id] else { return incoming }

            let existingTranslatedText = existing.translatedText?.trimmingCharacters(in: .whitespacesAndNewlines)
            let incomingTranslatedText = incoming.translatedText?.trimmingCharacters(in: .whitespacesAndNewlines)
            let resolvedTranslatedText = incomingTranslatedText?.isEmpty == false
                ? incomingTranslatedText
                : (existingTranslatedText?.isEmpty == false ? existingTranslatedText : nil)
            let textChanged =
                existing.text.trimmingCharacters(in: .whitespacesAndNewlines) !=
                incoming.text.trimmingCharacters(in: .whitespacesAndNewlines)
            let shouldRefreshTranslation = (existingTranslatedText?.isEmpty == false) && textChanged

            return MeetingTranscriptSegment(
                id: incoming.id,
                speaker: incoming.speaker,
                speakerID: incoming.speakerID,
                speakerDisplayName: incoming.speakerDisplayName,
                audioSource: incoming.audioSource,
                speakerConfidence: incoming.speakerConfidence,
                startSeconds: incoming.startSeconds,
                endSeconds: incoming.endSeconds,
                text: incoming.text,
                translatedText: resolvedTranslatedText,
                isTranslationPending: incoming.isTranslationPending || existing.isTranslationPending || shouldRefreshTranslation,
                preventsAdjacentMerge: incoming.preventsAdjacentMerge,
                isHighlighted: existing.isHighlighted || incoming.isHighlighted
            )
        }
    }

    private func translateEligibleSegmentsIfNeeded(targetLanguage: TranslationTargetLanguage) {
        for segment in segments where shouldTranslate(segment: segment, targetLanguage: targetLanguage) {
            markSegment(segment.id) { current in
                current.updatingTranslation(translatedText: current.translatedText, isTranslationPending: true)
            }

            let revision = translationRevision(for: segment, targetLanguage: targetLanguage)
            let operation = translationHandler(segment.text, targetLanguage)
            _ = translationScheduler.submit(
                segmentID: segment.id,
                sourceText: segment.text,
                targetLanguage: targetLanguage,
                operation: operation
            ) { [weak self] result in
                guard let self,
                      let current = self.segments.first(where: { $0.id == segment.id })
                else { return }
                guard self.translationRevision(for: current, targetLanguage: targetLanguage) == revision else {
                    self.translateEligibleSegmentsIfNeeded(targetLanguage: targetLanguage)
                    return
                }
                switch result {
                case .success(let translatedText):
                    let trimmed = translatedText.trimmingCharacters(in: .whitespacesAndNewlines)
                    if trimmed.isEmpty {
                        self.failedTranslationRevisions[segment.id] = revision
                    } else {
                        self.failedTranslationRevisions[segment.id] = nil
                    }
                    self.markSegment(segment.id) { current in
                        current.updatingTranslation(
                            translatedText: trimmed.isEmpty ? nil : trimmed,
                            isTranslationPending: false
                        )
                    }
                case .failure:
                    self.failedTranslationRevisions[segment.id] = revision
                    self.markSegment(segment.id) { current in
                        current.updatingTranslation(
                            translatedText: current.translatedText,
                            isTranslationPending: false
                        )
                    }
                }
                self.translateEligibleSegmentsIfNeeded(targetLanguage: targetLanguage)
            }
        }
    }

    private func shouldTranslate(
        segment: MeetingTranscriptSegment,
        targetLanguage: TranslationTargetLanguage
    ) -> Bool {
        guard segment.speaker == .them else { return false }
        guard failedTranslationRevisions[segment.id] != translationRevision(
            for: segment,
            targetLanguage: targetLanguage
        ) else { return false }
        let translatedText = segment.translatedText?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return translatedText.isEmpty || segment.isTranslationPending
    }

    private func translationRevision(
        for segment: MeetingTranscriptSegment,
        targetLanguage: TranslationTargetLanguage
    ) -> String {
        segment.text.trimmingCharacters(in: .whitespacesAndNewlines) + "\u{0}" + targetLanguage.rawValue
    }

    private func markSegment(_ id: UUID, update: (MeetingTranscriptSegment) -> MeetingTranscriptSegment) {
        guard let index = segments.firstIndex(where: { $0.id == id }) else { return }
        let previous = segments[index]
        let updated = update(previous)
        guard updated != previous else { return }
        segments[index] = updated

        // Translation status/text does not change speaker ordering or word counts.
        // Keep search filtering authoritative when translated text affects membership.
        guard searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              previous.id == updated.id,
              previous.text == updated.text,
              previous.speaker == updated.speaker,
              previous.speakerIdentityKey == updated.speakerIdentityKey,
              previous.startSeconds == updated.startSeconds,
              let displayedIndex = displayedSegments.firstIndex(where: { $0.id == id }),
              let groupIndex = speakerGroups.firstIndex(where: { $0.id == updated.speakerIdentityKey }),
              let segmentIndex = speakerGroups[groupIndex].segments.firstIndex(where: { $0.id == id })
        else {
            refreshTranscriptListCaches()
            return
        }
        displayedSegments[displayedIndex] = updated
        let group = speakerGroups[groupIndex]
        var groupSegments = group.segments
        groupSegments[segmentIndex] = updated
        speakerGroups[groupIndex] = MeetingDetailSpeakerGroup(
            id: group.id,
            title: group.title,
            speaker: group.speaker,
            segments: groupSegments,
            wordCount: group.wordCount
        )
    }

    private func cancelTranslationTasks() {
        translationScheduler.cancelAll()
    }

    private func clearPendingTranslationState() {
        segments = segments.map { segment in
            guard segment.isTranslationPending else { return segment }
            return segment.updatingTranslation(
                translatedText: segment.translatedText,
                isTranslationPending: false
            )
        }
        refreshTranscriptListCaches()
    }

    private var transcriptListCache = MeetingTranscriptListCache()

    private func refreshTranscriptListCaches() {
        transcriptListCache.updateOrdinals(for: segments)
        speakerOrdinalByIdentityKey = transcriptListCache.ordinals
        displayedSegments = MeetingTranscriptListSupport.displayedSegments(
            from: segments,
            searchQuery: searchQuery,
            speakerTitle: { [weak self] segment in
                self?.timelineSpeakerTitle(for: segment) ?? segment.speaker.displayTitle
            }
        )
        speakerGroups = transcriptListCache.groups(
            for: displayedSegments,
            title: { [weak self] segment in
                self?.timelineSpeakerTitle(for: segment) ?? segment.speaker.displayTitle
            }
        )
    }

    private func resolvedStoredTranslationLanguage() -> TranslationTargetLanguage {
        guard let rawValue = UserDefaults.standard.string(forKey: AppPreferenceKey.meetingRealtimeTranslationTargetLanguage),
              let language = TranslationTargetLanguage(rawValue: rawValue)
        else {
            return .english
        }
        return language
    }

    private func refreshSummaryConfigurationFromProviders() {
        guard let summarySettingsProvider, let summaryModelOptionsProvider else { return }
        refreshSummaryConfiguration(
            settings: summarySettingsProvider(),
            modelOptions: summaryModelOptionsProvider()
        )
    }

    private static func segmentsContainTranslations(_ segments: [MeetingTranscriptSegment]) -> Bool {
        segments.contains { segment in
            !(segment.translatedText?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ?? true)
        }
    }

    private static func initialTranslationLanguageRaw() -> String {
        let savedLanguage = UserDefaults.standard.string(forKey: AppPreferenceKey.meetingRealtimeTranslationTargetLanguage)
        return savedLanguage?.isEmpty == false
            ? savedLanguage!
            : TranslationTargetLanguage.english.rawValue
    }


}
