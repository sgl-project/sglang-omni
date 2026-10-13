// TranscriptionDetailWindow.swift
// Provides Transcription Detail Window for window and overlay UI.

import AppKit
import SwiftUI

@MainActor
final class TranscriptionDetailWindowManager {
    static let shared = TranscriptionDetailWindowManager()

    typealias FollowUpStatusProvider = @MainActor (TranscriptionHistoryEntry) -> TranscriptionFollowUpProviderStatus
    typealias FollowUpAnswerer = @MainActor (TranscriptionHistoryEntry, [TranscriptSummaryChatMessage], String) async throws -> String
    typealias FollowUpPersistence = @MainActor (UUID, [TranscriptSummaryChatMessage]) -> TranscriptionHistoryEntry?

    private var historyControllers: [UUID: TranscriptionDetailWindowController] = [:]

    func present(
        entry: TranscriptionHistoryEntry,
        audioURL: URL?,
        followUpStatusProvider: @escaping FollowUpStatusProvider,
        followUpAnswerer: @escaping FollowUpAnswerer,
        followUpPersistence: @escaping FollowUpPersistence
    ) {
        VoxtLog.history("Transcription detail open requested. entryID=\(entry.id), kind=\(entry.kind.rawValue)")

        if let controller = historyControllers[entry.id] {
            controller.refresh(entry: entry, audioURL: audioURL)
            controller.showWindow(nil)
            AppBehaviorController.bringStandardWindowToFront(controller.window)
            controller.recenterIfNeeded()
            if let window = controller.window {
                VoxtLog.history(
                    "Transcription detail window reused. frame=\(NSStringFromRect(window.frame)), visible=\(window.isVisible), miniaturized=\(window.isMiniaturized)"
                )
            }
            return
        }

        let viewModel = TranscriptionDetailViewModel(
            entry: entry,
            audioURL: audioURL,
            followUpStatusProvider: followUpStatusProvider,
            followUpAnswerer: followUpAnswerer,
            followUpPersistence: followUpPersistence
        )
        let controller = TranscriptionDetailWindowController(viewModel: viewModel) { [weak self] in
            self?.historyControllers[entry.id] = nil
        }
        historyControllers[entry.id] = controller
        controller.showWindow(nil)
        AppBehaviorController.bringStandardWindowToFront(controller.window)
        controller.recenterIfNeeded()
        if let window = controller.window {
            VoxtLog.history(
                "Transcription detail window shown. frame=\(NSStringFromRect(window.frame)), visible=\(window.isVisible), miniaturized=\(window.isMiniaturized)"
            )
        }
    }
}

@MainActor
private final class TranscriptionDetailWindowController: NSWindowController, NSWindowDelegate {
    private static let defaultWindowSize = NSSize(width: 680, height: 560)
    private static let minimumWindowSize = NSSize(width: 500, height: 380)

    private let viewModel: TranscriptionDetailViewModel
    private let onClose: () -> Void

    init(viewModel: TranscriptionDetailViewModel, onClose: @escaping () -> Void) {
        self.viewModel = viewModel
        self.onClose = onClose

        let rootView = TranscriptionDetailWindowView(viewModel: viewModel)
        let hostingController = NSHostingController(rootView: rootView)
        let window = NSWindow(
            contentRect: NSRect(origin: .zero, size: Self.defaultWindowSize),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.contentViewController = hostingController
        window.title = viewModel.title
        window.center()
        window.setFrameAutosaveName("VoxtTranscriptionDetailWindow")
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.toolbar = nil
        window.collectionBehavior = []
        window.isMovableByWindowBackground = false
        window.isReleasedWhenClosed = false
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = true
        window.level = .normal
        window.minSize = Self.minimumWindowSize
        window.setContentSize(Self.defaultWindowSize)
        window.center()

        super.init(window: window)
        window.delegate = self
        window.standardWindowButton(.closeButton)?.isHidden = false
        window.standardWindowButton(.miniaturizeButton)?.isHidden = false
        window.standardWindowButton(.zoomButton)?.isHidden = false
        positionWindowTrafficLightButtons(window)
        scheduleTrafficLightButtonPositionUpdate(for: window)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func refresh(entry: TranscriptionHistoryEntry, audioURL: URL?) {
        VoxtLog.history("Transcription detail refresh requested. entryID=\(entry.id)")
        viewModel.refresh(entry: entry, audioURL: audioURL)
        window?.title = viewModel.title
    }

    func recenterIfNeeded() {
        guard let window else { return }
        if window.isMiniaturized {
            window.deminiaturize(nil)
        }

        let needsSizeRepair =
            window.frame.width < Self.minimumWindowSize.width ||
            window.frame.height < Self.minimumWindowSize.height
        if needsSizeRepair {
            let repairedFrame = NSRect(origin: .zero, size: Self.defaultWindowSize)
            window.setFrame(repairedFrame, display: false)
        }

        if let visibleFrame = window.screen?.visibleFrame ?? NSScreen.main?.visibleFrame {
            let needsPositionRepair =
                !visibleFrame.intersects(window.frame) ||
                window.frame.maxX <= visibleFrame.minX ||
                window.frame.maxY <= visibleFrame.minY
            if needsPositionRepair || needsSizeRepair {
                window.center()
            }
        } else if needsSizeRepair {
            window.center()
        }
    }

    func windowWillClose(_ notification: Notification) {
        VoxtLog.history("Transcription detail window closed.")
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

    private func positionWindowTrafficLightButtons(_ window: NSWindow) {
        guard let closeButton = window.standardWindowButton(.closeButton),
              let miniaturizeButton = window.standardWindowButton(.miniaturizeButton),
              let zoomButton = window.standardWindowButton(.zoomButton),
              let container = closeButton.superview
        else {
            return
        }

        let leftInset: CGFloat = 22
        let topInset: CGFloat = 24
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

private struct TranscriptionDetailWindowView: View {
    @ObservedObject var viewModel: TranscriptionDetailViewModel

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: DetailPanelUIStyle.windowCornerRadius, style: .continuous)
                .fill(DetailPanelUIStyle.windowFillColor)
                .overlay(
                    RoundedRectangle(cornerRadius: DetailPanelUIStyle.windowCornerRadius, style: .continuous)
                        .strokeBorder(DetailPanelUIStyle.borderColor, lineWidth: 1)
                )

            TranscriptionDetailConversationView(viewModel: viewModel)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        }
        .padding(10)
        .frame(minWidth: 520, minHeight: 380)
        .background(Color.clear)
        .ignoresSafeArea(.container, edges: .top)
    }
}
