// VoxtNoteWindow.swift
// Provides the Peekaboo-style Voxt note panel and corner reveal behavior.

import AppKit
import Combine
import QuartzCore
import SwiftUI

nonisolated enum VoxtNotePanelGeometry {
    static let triggerSize: CGFloat = 16
    static let panelWidth: CGFloat = 332
    static let minimumHeight: CGFloat = 380
    static let maximumHeight: CGFloat = 700
    static let screenInset: CGFloat = 12

    static func hotspot(
        in screenFrame: CGRect,
        corner: VoxtNotePanelCorner,
        size: CGFloat = triggerSize
    ) -> CGRect {
        let origin: CGPoint
        switch corner {
        case .topLeft:
            origin = CGPoint(x: screenFrame.minX, y: screenFrame.maxY - size)
        case .topRight:
            origin = CGPoint(x: screenFrame.maxX - size, y: screenFrame.maxY - size)
        case .bottomLeft:
            origin = CGPoint(x: screenFrame.minX, y: screenFrame.minY)
        case .bottomRight:
            origin = CGPoint(x: screenFrame.maxX - size, y: screenFrame.minY)
        }
        return CGRect(origin: origin, size: CGSize(width: size, height: size))
    }

    static func panelFrame(
        in visibleFrame: CGRect,
        size: CGSize,
        corner: VoxtNotePanelCorner,
        inset: CGFloat = screenInset
    ) -> CGRect {
        let x: CGFloat
        let y: CGFloat
        switch corner {
        case .topLeft, .bottomLeft:
            x = visibleFrame.minX + inset
        case .topRight, .bottomRight:
            x = visibleFrame.maxX - size.width - inset
        }
        switch corner {
        case .topLeft, .topRight:
            y = visibleFrame.maxY - size.height - inset
        case .bottomLeft, .bottomRight:
            y = visibleFrame.minY + inset
        }
        return CGRect(origin: CGPoint(x: x, y: y), size: size)
    }

    static func hiddenFrame(
        from visibleFrame: CGRect,
        corner: VoxtNotePanelCorner,
        distance: CGFloat = 18
    ) -> CGRect {
        switch corner {
        case .topLeft:
            return visibleFrame.offsetBy(dx: -distance, dy: distance)
        case .topRight:
            return visibleFrame.offsetBy(dx: distance, dy: distance)
        case .bottomLeft:
            return visibleFrame.offsetBy(dx: -distance, dy: -distance)
        case .bottomRight:
            return visibleFrame.offsetBy(dx: distance, dy: -distance)
        }
    }

    static func preferredHeight(itemCount: Int, sectionCount: Int) -> CGFloat {
        let header: CGFloat = 76
        let itemGaps = max(itemCount - sectionCount, 0)
        let content: CGFloat = itemCount == 0
            ? 90
            : CGFloat(itemCount) * 32
                + CGFloat(itemGaps) * 4
                + CGFloat(sectionCount) * 24
                + 10
        return min(max(header + content + 16, minimumHeight), maximumHeight)
    }
}

nonisolated enum VoxtNoteCompletedPagination {
    static let initialLimit = 5
    static let pageSize = 10

    static func visibleCount(totalCount: Int, limit: Int) -> Int {
        min(max(totalCount, 0), max(limit, 0))
    }

    static func nextLimit(currentLimit: Int, totalCount: Int) -> Int {
        let normalizedCurrentLimit = max(currentLimit, initialLimit)
        guard totalCount > normalizedCurrentLimit else { return normalizedCurrentLimit }
        return min(totalCount, normalizedCurrentLimit + pageSize)
    }
}

nonisolated struct VoxtNoteCornerHoverStateMachine {
    enum Transition: Equatable {
        case none
        case reveal
        case hide
    }

    private(set) var isVisible = false
    private var hotspotEnteredAt: TimeInterval?
    private var revealedAt: TimeInterval?
    private var revealGrace: TimeInterval = 0.8
    private var panelHasBeenEntered = false
    private var leaveBeganAt: TimeInterval?

    mutating func update(
        at timestamp: TimeInterval,
        isInHotspot: Bool,
        isInPanel: Bool,
        isInteractionLocked: Bool,
        revealDelay: TimeInterval,
        hideDelay: TimeInterval
    ) -> Transition {
        if !isVisible {
            guard isInHotspot else {
                hotspotEnteredAt = nil
                return .none
            }
            if hotspotEnteredAt == nil { hotspotEnteredAt = timestamp }
            guard timestamp - (hotspotEnteredAt ?? timestamp) >= revealDelay else { return .none }
            isVisible = true
            revealedAt = timestamp
            revealGrace = 0.8
            panelHasBeenEntered = false
            leaveBeganAt = nil
            hotspotEnteredAt = nil
            return .reveal
        }

        if isInteractionLocked || isInHotspot {
            leaveBeganAt = nil
            return .none
        }
        if isInPanel {
            panelHasBeenEntered = true
            leaveBeganAt = nil
            return .none
        }
        if !panelHasBeenEntered,
           let revealedAt,
           timestamp - revealedAt < revealGrace {
            return .none
        }
        if leaveBeganAt == nil {
            leaveBeganAt = timestamp
            return .none
        }
        guard timestamp - (leaveBeganAt ?? timestamp) >= max(0, hideDelay) else { return .none }
        reset()
        return .hide
    }

    mutating func forceVisible(at timestamp: TimeInterval, grace: TimeInterval = 3) {
        isVisible = true
        revealedAt = timestamp
        revealGrace = grace
        panelHasBeenEntered = false
        leaveBeganAt = nil
        hotspotEnteredAt = nil
    }

    mutating func forceHidden() {
        reset()
    }

    func nextEvaluationDelay(
        at timestamp: TimeInterval,
        revealDelay: TimeInterval,
        hideDelay: TimeInterval
    ) -> TimeInterval? {
        if !isVisible, let hotspotEnteredAt {
            return max(0, revealDelay - (timestamp - hotspotEnteredAt))
        }
        if isVisible,
           !panelHasBeenEntered,
           let revealedAt,
           timestamp - revealedAt < revealGrace {
            return max(0, revealGrace - (timestamp - revealedAt))
        }
        if isVisible, let leaveBeganAt {
            return max(0, max(0, hideDelay) - (timestamp - leaveBeganAt))
        }
        return nil
    }

    private mutating func reset() {
        isVisible = false
        hotspotEnteredAt = nil
        revealedAt = nil
        panelHasBeenEntered = false
        leaveBeganAt = nil
    }
}

@MainActor
final class VoxtNotePanelUIState: ObservableObject {
    @Published var editingNoteID: UUID?
    @Published var detailNoteID: UUID?
    @Published var isMenuTracking = false
    @Published private(set) var selectedScope: VoxtNoteScope = .notes
    @Published private(set) var completedItemLimit = VoxtNoteCompletedPagination.initialLimit
    private(set) var draggedNoteID: UUID?

    var isInteractionLocked: Bool {
        editingNoteID != nil || detailNoteID != nil || isMenuTracking || draggedNoteID != nil
    }

    func selectScope(_ scope: VoxtNoteScope) {
        guard selectedScope != scope else { return }
        editingNoteID = nil
        detailNoteID = nil
        draggedNoteID = nil
        selectedScope = scope
    }

    func loadMoreCompletedItems(totalCount: Int) {
        let nextLimit = VoxtNoteCompletedPagination.nextLimit(
            currentLimit: completedItemLimit,
            totalCount: totalCount
        )
        guard nextLimit != completedItemLimit else { return }
        completedItemLimit = nextLimit
    }

    func beginEditing(_ noteID: UUID) {
        detailNoteID = nil
        editingNoteID = noteID
    }

    func endEditing() {
        editingNoteID = nil
    }

    func showDetail(_ noteID: UUID) {
        editingNoteID = nil
        detailNoteID = noteID
    }

    func hideDetail() {
        detailNoteID = nil
    }

    func beginDragging(_ noteID: UUID) {
        editingNoteID = nil
        detailNoteID = nil
        draggedNoteID = noteID
    }

    func endDragging() {
        draggedNoteID = nil
    }

    func finishDragging(releasedOutsidePanel: Bool) -> UUID? {
        guard let draggedNoteID else { return nil }
        self.draggedNoteID = nil
        return releasedOutsidePanel ? draggedNoteID : nil
    }
}

@MainActor
final class VoxtNotePanelSettingsState: ObservableObject {
    @Published private(set) var value: VoxtNotePanelSettings

    init(_ value: VoxtNotePanelSettings) {
        self.value = value
    }

    func update(_ value: VoxtNotePanelSettings) {
        guard self.value != value else { return }
        self.value = value
    }
}

final class VoxtNotePanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}

@MainActor
final class VoxtNotePanelController {
    private let panel: VoxtNotePanel
    private let store: VoxtNoteStore
    private let uiState: VoxtNotePanelUIState
    private let settingsState: VoxtNotePanelSettingsState
    private var cancellables: Set<AnyCancellable> = []
    private var isShowing = false
    private var needsResizeAfterShowing = false
    private var transitionGeneration = 0
    private(set) var currentScreen: NSScreen?
    private(set) var currentCorner: VoxtNotePanelCorner

    init(
        store: VoxtNoteStore,
        uiState: VoxtNotePanelUIState,
        settingsState: VoxtNotePanelSettingsState,
        onOpenSettings: @escaping () -> Void
    ) {
        self.store = store
        self.uiState = uiState
        self.settingsState = settingsState
        currentCorner = settingsState.value.corner

        panel = VoxtNotePanel(
            contentRect: CGRect(
                x: 0,
                y: 0,
                width: VoxtNotePanelGeometry.panelWidth,
                height: VoxtNotePanelGeometry.minimumHeight
            ),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: true
        )
        let rootView = VoxtNoteWindowView(
            store: store,
            uiState: uiState,
            settingsState: settingsState,
            onOpenSettings: onOpenSettings
        )
        panel.contentView = NSHostingView(rootView: rootView)
        configurePanel()
        bindContentSize()
    }

    var visibleFrame: CGRect? {
        panel.isVisible ? panel.frame : nil
    }

    var isVisible: Bool { panel.isVisible }

    func show(on screen: NSScreen, makeKey: Bool = false) {
        currentScreen = screen
        currentCorner = settingsState.value.corner
        let finalFrame = frame(on: screen, corner: currentCorner)

        if panel.isVisible {
            if makeKey {
                panel.makeKeyAndOrderFront(nil)
            } else {
                panel.orderFrontRegardless()
            }
            panel.alphaValue = 1
            guard !framesMatch(panel.frame, finalFrame) else { return }
            animateShow(to: finalFrame, fadeIn: false)
            return
        }

        panel.setFrame(
            VoxtNotePanelGeometry.hiddenFrame(from: finalFrame, corner: currentCorner),
            display: true
        )
        panel.alphaValue = 0
        if makeKey {
            panel.makeKeyAndOrderFront(nil)
        } else {
            panel.orderFrontRegardless()
        }
        animateShow(to: finalFrame, fadeIn: true)
    }

    func hide() {
        guard panel.isVisible else { return }
        transitionGeneration += 1
        let generation = transitionGeneration
        isShowing = false
        needsResizeAfterShowing = false
        let targetFrame = VoxtNotePanelGeometry.hiddenFrame(from: panel.frame, corner: currentCorner)
        NSAnimationContext.runAnimationGroup { context in
            context.duration = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion ? 0 : 0.12
            context.timingFunction = CAMediaTimingFunction(controlPoints: 0.4, 0, 1, 1)
            panel.animator().setFrame(targetFrame, display: true)
            panel.animator().alphaValue = 0
        } completionHandler: { [weak self] in
            MainActor.assumeIsolated {
                guard let self, generation == self.transitionGeneration else { return }
                self.panel.orderOut(nil)
                self.panel.alphaValue = 1
            }
        }
    }

    private func animateShow(to finalFrame: CGRect, fadeIn: Bool) {
        transitionGeneration += 1
        let generation = transitionGeneration
        isShowing = true
        NSAnimationContext.runAnimationGroup { context in
            context.duration = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion ? 0 : 0.16
            context.timingFunction = CAMediaTimingFunction(controlPoints: 0.16, 1, 0.3, 1)
            panel.animator().setFrame(finalFrame, display: true)
            if fadeIn { panel.animator().alphaValue = 1 }
        } completionHandler: { [weak self] in
            MainActor.assumeIsolated {
                guard let self, generation == self.transitionGeneration else { return }
                self.isShowing = false
                if self.needsResizeAfterShowing {
                    self.needsResizeAfterShowing = false
                    self.resizeAndReanchor()
                }
            }
        }
    }

    private func configurePanel() {
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.level = .floating
        panel.hidesOnDeactivate = false
        panel.isMovable = false
        panel.becomesKeyOnlyIfNeeded = true
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient]
    }

    private func bindContentSize() {
        Publishers.CombineLatest4(
            store.$revision,
            settingsState.$value,
            uiState.$selectedScope,
            uiState.$completedItemLimit
        )
        .receive(on: RunLoop.main)
        .sink { [weak self] _, settings, _, _ in
            guard let self else { return }
            self.currentCorner = settings.corner
            self.resizeAndReanchor()
        }
        .store(in: &cancellables)
    }

    private func resizeAndReanchor() {
        guard let screen = currentScreen else { return }
        if isShowing {
            needsResizeAfterShowing = true
            return
        }
        let targetFrame = frame(on: screen, corner: currentCorner)
        guard !framesMatch(panel.frame, targetFrame) else { return }
        if panel.isVisible {
            panel.setFrame(targetFrame, display: true, animate: true)
        } else {
            panel.setFrame(targetFrame, display: false)
        }
    }

    private func frame(on screen: NSScreen, corner: VoxtNotePanelCorner) -> CGRect {
        let snapshot = store.snapshot(for: uiState.selectedScope)
        let completedCount = snapshot.sections.first(where: { $0.status == .done })?.items.count ?? 0
        let visibleCompletedCount = VoxtNoteCompletedPagination.visibleCount(
            totalCount: completedCount,
            limit: uiState.completedItemLimit
        )
        let visibleItemCount = snapshot.visibleCount
            - completedCount
            + visibleCompletedCount
        return VoxtNotePanelGeometry.panelFrame(
            in: screen.visibleFrame,
            size: CGSize(
                width: VoxtNotePanelGeometry.panelWidth,
                height: VoxtNotePanelGeometry.preferredHeight(
                    itemCount: visibleItemCount,
                    sectionCount: snapshot.sections.count
                )
            ),
            corner: corner
        )
    }

    private func framesMatch(_ lhs: CGRect, _ rhs: CGRect) -> Bool {
        abs(lhs.minX - rhs.minX) < 0.5
            && abs(lhs.minY - rhs.minY) < 0.5
            && abs(lhs.width - rhs.width) < 0.5
            && abs(lhs.height - rhs.height) < 0.5
    }
}

@MainActor
final class VoxtNoteCornerHoverMonitor {
    private let panelController: VoxtNotePanelController
    private let uiState: VoxtNotePanelUIState
    private let settingsState: VoxtNotePanelSettingsState
    private let store: VoxtNoteStore
    private var stateMachine = VoxtNoteCornerHoverStateMachine()
    private var transitionTimer: DispatchSourceTimer?
    private var localMouseMonitor: Any?
    private var globalMouseMonitor: Any?
    private var screenChangeToken: NSObjectProtocol?
    private var stateChangeCancellables: Set<AnyCancellable> = []
    private var isMonitoring = false
    private var hasPendingStateChangeSample = false

    init(
        panelController: VoxtNotePanelController,
        uiState: VoxtNotePanelUIState,
        settingsState: VoxtNotePanelSettingsState,
        store: VoxtNoteStore
    ) {
        self.panelController = panelController
        self.uiState = uiState
        self.settingsState = settingsState
        self.store = store
    }

    func start() {
        guard !isMonitoring else { return }
        isMonitoring = true
        let mouseEvents: NSEvent.EventTypeMask = [
            .mouseMoved,
            .leftMouseDragged,
            .rightMouseDragged,
            .otherMouseDragged,
            .leftMouseUp,
            .rightMouseUp,
            .otherMouseUp
        ]
        localMouseMonitor = NSEvent.addLocalMonitorForEvents(matching: mouseEvents) { [weak self] event in
            MainActor.assumeIsolated { self?.samplePointer() }
            return event
        }
        globalMouseMonitor = NSEvent.addGlobalMonitorForEvents(matching: mouseEvents) { [weak self] _ in
            DispatchQueue.main.async {
                guard let self, self.isMonitoring else { return }
                self.samplePointer()
            }
        }

        screenChangeToken = NotificationCenter.default.addObserver(
            forName: NSApplication.didChangeScreenParametersNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.samplePointer() }
        }
        observeStateChanges()
        samplePointer()
    }

    func stop() {
        isMonitoring = false
        hasPendingStateChangeSample = false
        transitionTimer?.cancel()
        transitionTimer = nil
        if let localMouseMonitor { NSEvent.removeMonitor(localMouseMonitor) }
        if let globalMouseMonitor { NSEvent.removeMonitor(globalMouseMonitor) }
        localMouseMonitor = nil
        globalMouseMonitor = nil
        if let screenChangeToken { NotificationCenter.default.removeObserver(screenChangeToken) }
        screenChangeToken = nil
        stateChangeCancellables.removeAll()
        stateMachine.forceHidden()
        panelController.hide()
    }

    func revealProgrammatically() {
        guard let screen = screen(containing: NSEvent.mouseLocation) ?? NSScreen.main else { return }
        uiState.selectScope(.notes)
        stateMachine.forceVisible(at: ProcessInfo.processInfo.systemUptime, grace: 3)
        panelController.show(on: screen)
        samplePointer()
    }

    private func samplePointer() {
        // Menu tracking owns hover until dismissal. Keep an already-visible note
        // panel in place, but avoid screen queries and timer churn on every move.
        // The UI-state observer schedules a fresh sample when tracking ends.
        guard !uiState.isMenuTracking else {
            transitionTimer?.cancel()
            transitionTimer = nil
            return
        }
        let location = NSEvent.mouseLocation
        let activeScreen = screen(containing: location)
        let corner = settingsState.value.corner
        let isInHotspot = activeScreen.map {
            VoxtNotePanelGeometry.hotspot(in: $0.frame, corner: corner)
                .insetBy(dx: -1, dy: -1)
                .contains(location)
        } ?? false
        let isInPanel = panelController.visibleFrame?.contains(location) ?? false
        let isMouseButtonPressed = NSEvent.pressedMouseButtons != 0
        if !isMouseButtonPressed,
           let noteID = uiState.finishDragging(releasedOutsidePanel: !isInPanel) {
            store.startAfterExternalDrag(noteID: noteID)
        }

        let settings = settingsState.value
        let timestamp = ProcessInfo.processInfo.systemUptime
        let transition = stateMachine.update(
            at: timestamp,
            isInHotspot: isInHotspot,
            isInPanel: isInPanel,
            isInteractionLocked: uiState.isInteractionLocked || isMouseButtonPressed,
            revealDelay: settings.revealDelay,
            hideDelay: settings.hideDelay
        )
        switch transition {
        case .none:
            break
        case .reveal:
            if let activeScreen {
                panelController.show(on: activeScreen)
            }
        case .hide:
            panelController.hide()
        }
        scheduleNextEvaluationIfNeeded(at: timestamp, settings: settings)
    }

    private func observeStateChanges() {
        uiState.objectWillChange
            .sink { [weak self] _ in
                self?.samplePointerAfterStateChange()
            }
            .store(in: &stateChangeCancellables)

        settingsState.$value
            .dropFirst()
            .sink { [weak self] _ in
                self?.samplePointerAfterStateChange()
            }
            .store(in: &stateChangeCancellables)
    }

    private func samplePointerAfterStateChange() {
        guard isMonitoring, !hasPendingStateChangeSample else { return }
        hasPendingStateChangeSample = true
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.hasPendingStateChangeSample = false
            guard self.isMonitoring else { return }
            self.samplePointer()
        }
    }

    private func scheduleNextEvaluationIfNeeded(
        at timestamp: TimeInterval,
        settings: VoxtNotePanelSettings
    ) {
        transitionTimer?.cancel()
        transitionTimer = nil
        guard let delay = stateMachine.nextEvaluationDelay(
            at: timestamp,
            revealDelay: settings.revealDelay,
            hideDelay: settings.hideDelay
        ) else { return }

        let timer = DispatchSource.makeTimerSource(queue: .main)
        timer.schedule(deadline: .now() + max(delay, 0.01), leeway: .milliseconds(10))
        timer.setEventHandler { [weak self] in
            MainActor.assumeIsolated {
                self?.transitionTimer = nil
                self?.samplePointer()
            }
        }
        transitionTimer = timer
        timer.resume()
    }

    private func screen(containing point: CGPoint) -> NSScreen? {
        NSScreen.screens.first { NSMouseInRect(point, $0.frame, false) }
            ?? NSScreen.screens.first { $0.frame.insetBy(dx: -1, dy: -1).contains(point) }
    }
}

@MainActor
final class VoxtNoteWindowManager {
    private let store: VoxtNoteStore
    private let settingsProvider: () -> VoxtNotePanelSettings
    private let settingsState: VoxtNotePanelSettingsState
    private let uiState = VoxtNotePanelUIState()
    private let panelController: VoxtNotePanelController
    private let hoverMonitor: VoxtNoteCornerHoverMonitor
    private var menuNotificationTokens: [NSObjectProtocol] = []
    private var availabilityCancellable: AnyCancellable?
    private var isRunning = false
    private var isFeatureEnabled = false

    init(
        store: VoxtNoteStore,
        settingsProvider: @escaping () -> VoxtNotePanelSettings,
        onOpenSettings: @escaping () -> Void = {}
    ) {
        self.store = store
        self.settingsProvider = settingsProvider
        let settingsState = VoxtNotePanelSettingsState(settingsProvider())
        self.settingsState = settingsState
        panelController = VoxtNotePanelController(
            store: store,
            uiState: uiState,
            settingsState: settingsState,
            onOpenSettings: onOpenSettings
        )
        hoverMonitor = VoxtNoteCornerHoverMonitor(
            panelController: panelController,
            uiState: uiState,
            settingsState: settingsState,
            store: store
        )
        availabilityCancellable = store.$availability
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in
                self?.applyLifecycle()
            }
    }

    func updateLifecycle(isEnabled: Bool) {
        isFeatureEnabled = isEnabled
        settingsState.update(settingsProvider())
        applyLifecycle()
    }

    private func applyLifecycle() {
        if isFeatureEnabled && store.isAvailable {
            startIfNeeded()
        } else {
            stop()
        }
    }

    func show() {
        guard store.isAvailable else { return }
        settingsState.update(settingsProvider())
        hoverMonitor.revealProgrammatically()
    }

    func hide() {
        panelController.hide()
    }

    var isVisible: Bool { panelController.isVisible }

    func stop() {
        guard isRunning else {
            panelController.hide()
            return
        }
        hoverMonitor.stop()
        menuNotificationTokens.forEach(NotificationCenter.default.removeObserver)
        menuNotificationTokens.removeAll()
        isRunning = false
    }

    private func startIfNeeded() {
        guard !isRunning else { return }
        isRunning = true
        observeMenuTracking()
        hoverMonitor.start()
    }

    private func observeMenuTracking() {
        let center = NotificationCenter.default
        menuNotificationTokens = [
            center.addObserver(
                forName: NSMenu.didBeginTrackingNotification,
                object: nil,
                queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated { self?.uiState.isMenuTracking = true }
            },
            center.addObserver(
                forName: NSMenu.didEndTrackingNotification,
                object: nil,
                queue: .main
            ) { [weak self] _ in
                MainActor.assumeIsolated { self?.uiState.isMenuTracking = false }
            }
        ]
    }
}
