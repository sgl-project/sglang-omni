// MeetingDetailWindow.swift
// Provides Meeting Detail Window for meeting detail windows.

import AppKit
import SwiftUI

@MainActor
final class MeetingDetailWindowManager {
    static let shared = MeetingDetailWindowManager()

    typealias TranslationHandler = @MainActor (String, TranslationTargetLanguage) -> MeetingTranslationOperation
    typealias SummarySettingsProvider = @MainActor () -> MeetingSummarySettingsSnapshot
    typealias SummaryModelOptionsProvider = @MainActor () -> [MeetingSummaryModelOption]
    typealias SummaryStatusProvider = @MainActor (MeetingSummarySettingsSnapshot) -> MeetingSummaryProviderStatus
    typealias SummaryGenerator = @MainActor (String, MeetingSummarySettingsSnapshot) async throws -> MeetingSummarySnapshot
    typealias SummaryPersistence = @MainActor (UUID, MeetingSummarySnapshot?) -> TranscriptionHistoryEntry?
    typealias SummaryStalePersistence = @MainActor (UUID, Bool) -> TranscriptionHistoryEntry?
    typealias SummaryChatAnswerer = @MainActor (String, MeetingSummarySnapshot?, [MeetingSummaryChatMessage], String, MeetingSummarySettingsSnapshot) async throws -> String
    typealias SummaryChatPersistence = @MainActor (UUID, [MeetingSummaryChatMessage]) -> TranscriptionHistoryEntry?
    typealias TranscriptSegmentsPersistence = @MainActor (UUID, [MeetingTranscriptSegment]) -> TranscriptionHistoryEntry?

    private var historyControllers: [UUID: MeetingDetailWindowController] = [:]
    private var fileControllers: [UUID: MeetingDetailWindowController] = [:]
    private var liveController: MeetingDetailWindowController?

    func presentFileTranscript(taskID: UUID, title: String, segments: [MeetingTranscriptSegment]) {
        let controller: MeetingDetailWindowController
        if let existing = fileControllers[taskID] {
            controller = existing
            if !controller.displaysFileTranscript(title: title, segments: segments) {
                let viewModel = MeetingDetailViewModel(fileTaskTitle: title, segments: segments)
                controller.replaceViewModel(viewModel) { [weak self] in self?.fileControllers[taskID] = nil }
            }
        } else {
            let viewModel = MeetingDetailViewModel(fileTaskTitle: title, segments: segments)
            controller = MeetingDetailWindowController(viewModel: viewModel) { [weak self] in
                self?.fileControllers[taskID] = nil
            }
            fileControllers[taskID] = controller
        }
        controller.showWindow(nil)
        AppBehaviorController.bringStandardWindowToFront(controller.window)
    }

    func hasFileTranscriptWindow(taskID: UUID) -> Bool { fileControllers[taskID] != nil }

    func closeFileTranscript(taskID: UUID) {
        fileControllers.removeValue(forKey: taskID)?.close()
    }

    func presentHistoryMeeting(
        entry: TranscriptionHistoryEntry,
        replacingFileTaskID: UUID? = nil,
        activate: Bool = true,
        audioURL: URL?,
        initialSummarySettings: MeetingSummarySettingsSnapshot,
        summaryModelOptionsProvider: @escaping SummaryModelOptionsProvider,
        summarySettingsProvider: @escaping SummarySettingsProvider,
        translationHandler: @escaping TranslationHandler,
        summaryStatusProvider: @escaping SummaryStatusProvider,
        summaryGenerator: @escaping SummaryGenerator,
        summaryPersistence: @escaping SummaryPersistence,
        summaryStalePersistence: @escaping SummaryStalePersistence = { _, _ in nil },
        summaryChatAnswerer: @escaping SummaryChatAnswerer,
        summaryChatPersistence: @escaping SummaryChatPersistence,
        transcriptSegmentsPersistence: @escaping TranscriptSegmentsPersistence
    ) {
        if let controller = historyControllers[entry.id] {
            if let replacingFileTaskID { closeFileTranscript(taskID: replacingFileTaskID) }
            controller.refreshSummaryConfiguration(
                settings: summarySettingsProvider(),
                modelOptions: summaryModelOptionsProvider()
            )
            if activate {
                controller.showWindow(nil)
                AppBehaviorController.bringStandardWindowToFront(controller.window)
            }
            return
        }

        let summaryModelOptions = summaryModelOptionsProvider()

        let viewModel = MeetingDetailViewModel(
            title: AppLocalization.localizedString("Meeting Details"),
            subtitle: entry.createdAt.formatted(date: .abbreviated, time: .shortened),
            historyEntryID: entry.id,
            initialSummary: entry.transcriptSummary,
            initialSummaryStale: entry.transcriptSummaryStale,
            initialSummaryChatMessages: entry.transcriptSummaryChatMessages ?? [],
            initialSummarySettings: initialSummarySettings,
            summaryModelOptions: summaryModelOptions,
            summarySettingsProvider: summarySettingsProvider,
            summaryModelOptionsProvider: summaryModelOptionsProvider,
            segments: entry.transcriptSegments ?? [],
            captureMode: entry.meetingCaptureMode,
            audioURL: audioURL,
            translationHandler: translationHandler,
            summaryStatusProvider: summaryStatusProvider,
            summaryGenerator: summaryGenerator,
            summaryPersistence: summaryPersistence,
            summaryStalePersistence: summaryStalePersistence,
            summaryChatAnswerer: summaryChatAnswerer,
            summaryChatPersistence: summaryChatPersistence,
            transcriptSegmentsPersistence: transcriptSegmentsPersistence
        )
        let onClose: () -> Void = { [weak self] in self?.historyControllers[entry.id] = nil }
        let controller: MeetingDetailWindowController
        if let replacingFileTaskID,
           let draftController = fileControllers.removeValue(forKey: replacingFileTaskID) {
            // Preserve NSWindow identity/frame; replace the view tree so playback
            // StateObjects bind only to the final history-owned archive URL.
            controller = draftController
            controller.replaceViewModel(viewModel, onClose: onClose)
        } else {
            controller = MeetingDetailWindowController(viewModel: viewModel, onClose: onClose)
        }
        historyControllers[entry.id] = controller
        if activate {
            controller.showWindow(nil)
            AppBehaviorController.bringStandardWindowToFront(controller.window)
        }
    }

    func presentLiveMeeting(
        state: MeetingOverlayState,
        initialSummarySettings: MeetingSummarySettingsSnapshot,
        summaryModelOptionsProvider: @escaping SummaryModelOptionsProvider,
        summarySettingsProvider: @escaping SummarySettingsProvider,
        translationHandler: @escaping TranslationHandler
    ) {
        if let controller = liveController {
            controller.refreshSummaryConfiguration(
                settings: summarySettingsProvider(),
                modelOptions: summaryModelOptionsProvider()
            )
            controller.showWindow(nil)
            AppBehaviorController.bringStandardWindowToFront(controller.window)
            return
        }

        let viewModel = MeetingDetailViewModel(
            liveState: state,
            initialSummarySettings: initialSummarySettings,
            summaryModelOptions: summaryModelOptionsProvider(),
            summarySettingsProvider: summarySettingsProvider,
            summaryModelOptionsProvider: summaryModelOptionsProvider,
            translationHandler: translationHandler
        )
        let controller = MeetingDetailWindowController(viewModel: viewModel) { [weak self] in
            self?.liveController = nil
        }
        liveController = controller
        controller.showWindow(nil)
        AppBehaviorController.bringStandardWindowToFront(controller.window)
    }

    func closeLiveWindow() {
        liveController?.close()
        liveController = nil
    }
}

@MainActor
private final class MeetingDetailWindowController: NSWindowController, NSWindowDelegate {
    private static let defaultWindowSize = NSSize(width: 1040, height: 700)
    private static let minimumWindowSize = NSSize(width: 860, height: 560)

    private var onClose: () -> Void

    init(viewModel: MeetingDetailViewModel, onClose: @escaping () -> Void) {
        self.onClose = onClose

        let rootView = MeetingDetailWindowView(viewModel: viewModel)
        let hostingController = NSHostingController(rootView: rootView)
        let window = NSWindow(
            contentRect: NSRect(origin: .zero, size: Self.defaultWindowSize),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.contentViewController = hostingController
        window.title = AppLocalization.localizedString("Meeting Details")
        window.center()
        window.setFrameAutosaveName("VoxtMeetingDetailWindow")
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.toolbar = nil
        window.isMovableByWindowBackground = false
        window.isReleasedWhenClosed = false
        window.isOpaque = true
        window.backgroundColor = SettingsUIStyle.windowBackgroundNSColor
        window.hasShadow = true
        window.collectionBehavior = []
        window.level = .normal
        window.minSize = Self.minimumWindowSize
        window.setContentSize(Self.defaultWindowSize)

        super.init(window: window)
        window.delegate = self
        window.standardWindowButton(.closeButton)?.isHidden = false
        window.standardWindowButton(.miniaturizeButton)?.isHidden = false
        window.standardWindowButton(.zoomButton)?.isHidden = false
        positionWindowTrafficLightButtons(window)
        scheduleTrafficLightButtonPositionUpdate(for: window)
    }

    func displaysFileTranscript(title: String, segments: [MeetingTranscriptSegment]) -> Bool {
        guard let host = window?.contentViewController as? NSHostingController<MeetingDetailWindowView> else { return false }
        let model = host.rootView.viewModel
        return model.mode == .fileDraft && model.title == title && model.segments == segments
    }

    func replaceViewModel(_ viewModel: MeetingDetailViewModel, onClose: @escaping () -> Void) {
        self.onClose = onClose
        guard let window else { return }
        let frame = window.frame
        window.contentViewController = NSHostingController(rootView: MeetingDetailWindowView(viewModel: viewModel))
        window.setFrame(frame, display: true)
        scheduleTrafficLightButtonPositionUpdate(for: window)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func windowWillClose(_ notification: Notification) {
        onClose()
    }

    func windowDidResize(_ notification: Notification) {
        guard let window else { return }
        scheduleTrafficLightButtonPositionUpdate(for: window)
    }

    func windowDidBecomeKey(_ notification: Notification) {
        guard let window else { return }
        scheduleTrafficLightButtonPositionUpdate(for: window)
    }

    func refreshSummaryConfiguration(
        settings: MeetingSummarySettingsSnapshot,
        modelOptions: [MeetingSummaryModelOption]
    ) {
        guard let hostingController = window?.contentViewController as? NSHostingController<MeetingDetailWindowView> else {
            return
        }
        hostingController.rootView.viewModel.refreshSummaryConfiguration(
            settings: settings,
            modelOptions: modelOptions
        )
    }

    private func positionWindowTrafficLightButtons(_ window: NSWindow) {
        guard let closeButton = window.standardWindowButton(.closeButton),
              let miniaturizeButton = window.standardWindowButton(.miniaturizeButton),
              let zoomButton = window.standardWindowButton(.zoomButton),
              let container = closeButton.superview
        else {
            return
        }

        let leftInset: CGFloat = 15
        let topInset: CGFloat = 21
        let spacing: CGFloat = 6

        let buttonSize = closeButton.frame.size
        let y = container.bounds.height - topInset - buttonSize.height
        let closeX = leftInset
        let miniaturizeX = closeX + buttonSize.width + spacing
        let zoomX = miniaturizeX + buttonSize.width + spacing

        closeButton.translatesAutoresizingMaskIntoConstraints = true
        miniaturizeButton.translatesAutoresizingMaskIntoConstraints = true
        zoomButton.translatesAutoresizingMaskIntoConstraints = true

        closeButton.setFrameOrigin(CGPoint(x: closeX, y: y))
        miniaturizeButton.setFrameOrigin(CGPoint(x: miniaturizeX, y: y))
        zoomButton.setFrameOrigin(CGPoint(x: zoomX, y: y))
    }

    private func scheduleTrafficLightButtonPositionUpdate(for window: NSWindow) {
        DispatchQueue.main.async { [weak self, weak window] in
            guard let self, let window else { return }
            self.positionWindowTrafficLightButtons(window)
        }
    }
}
