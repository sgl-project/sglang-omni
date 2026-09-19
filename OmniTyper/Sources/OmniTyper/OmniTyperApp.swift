// SPDX-License-Identifier: Apache-2.0
import AppKit
import SwiftUI

@main
struct OmniTyperApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    var body: some Scene {
        Settings { EmptyView() }
            .commands {
                CommandGroup(replacing: .appSettings) {
                    Button(Page.settings.title) {
                        delegate.model.consolePage = .settings
                        delegate.openWindow()
                    }.keyboardShortcut(",", modifiers: .command)
                }
            }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, NSMenuDelegate {
    var model: AppModel!
    private var window: NSWindow!
    var panel: RecordingPanel!
    var selectionBanner: NSPanel?
    let modeBarTransition = ModeBarTransition()
    private var panelAnchor: NSPoint?
    private var positioningPanel = false
    private var transitioningMarker: (expandedX: CGFloat, compactX: CGFloat)?
    private var selectionResizeTask: Task<Void, Never>?
    private var selectionMotion = SelectionExpansionMotion(value: 0)
    private var panelWasSelectingMode = false
    private var statusItem: NSStatusItem!

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        model = AppModel()
        let content = RootView(model: model, store: model.store)
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1060, height: 750),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "OmniTyper"
        window.titlebarAppearsTransparent = false
        window.titlebarSeparatorStyle = .line
        window.isMovableByWindowBackground = false
        window.titleVisibility = .visible
        window.contentView = ConsoleHostingView(rootView: content)
        window.minSize = NSSize(width: 920, height: 660)
        window.isReleasedWhenClosed = false
        window.delegate = self
        window.center()
        window.setFrameAutosaveName("OmniTyperMainWindow")
        model.showMainWindow = { [weak self] in self?.openWindow() }
        model.showVoicePanel = { [weak self] in self?.showPanel() }
        model.hideVoicePanel = { [weak self] in
            self?.selectionResizeTask?.cancel(); self?.selectionResizeTask = nil
            self?.selectionBanner?.orderOut(nil)
            self?.panel?.orderOut(nil)
        }
        setupMenu()
        if !ProcessInfo.processInfo.arguments.contains("--background") { openWindow() }
        if let index = ProcessInfo.processInfo.arguments.firstIndex(of: "--snapshot"),
           ProcessInfo.processInfo.arguments.count > index + 1 {
            let path = ProcessInfo.processInfo.arguments[index + 1]
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in self?.saveSnapshot(path) }
        }
    }

    private func setupMenu() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        statusItem.button?.image = NSImage(systemSymbolName: "waveform", accessibilityDescription: "OmniTyper")
        let menu = NSMenu()
        menu.delegate = self
        statusItem.menu = menu
        rebuildMenu()
    }

    /// Rebuilt on open so a language change reaches the menu bar without a restart.
    func menuNeedsUpdate(_ menu: NSMenu) { rebuildMenu() }

    private func rebuildMenu() {
        guard let menu = statusItem.menu else { return }
        statusItem.button?.toolTip = L("menu.tooltip")
        menu.removeAllItems()
        menu.addItem(withTitle: L("menu.open"), action: #selector(openWindow), keyEquivalent: "")
        menu.addItem(.separator())
        for mode in VoiceMode.allCases {
            let item = NSMenuItem(title: mode.title, action: #selector(startFromMenu(_:)), keyEquivalent: "")
            item.representedObject = mode.rawValue; menu.addItem(item)
        }
        menu.addItem(withTitle: L("menu.stop"), action: #selector(stopRecording), keyEquivalent: "")
        menu.addItem(withTitle: L("action.cancel"), action: #selector(cancelRecording), keyEquivalent: "")
        menu.addItem(.separator())
        menu.addItem(withTitle: L("menu.quit"), action: #selector(quit), keyEquivalent: "q")
        for item in menu.items { item.target = self }
    }

    @objc func openWindow() {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
    }

    @objc private func startFromMenu(_ sender: NSMenuItem) {
        guard let raw = sender.representedObject as? String, let mode = VoiceMode(rawValue: raw) else { return }
        model.toggle(mode)
    }
    @objc private func stopRecording() { model.finish() }
    @objc private func cancelRecording() { model.cancel() }
    @objc private func quit() { NSApp.terminate(nil) }

    func windowWillClose(_ notification: Notification) { NSApp.setActivationPolicy(.accessory) }
    func windowDidMove(_ notification: Notification) { updatePanelPosition(notification) }
    func windowDidResize(_ notification: Notification) { updatePanelPosition(notification) }

    private func updatePanelPosition(_ notification: Notification) {
        guard let moved = notification.object as? NSWindow, moved === panel else { return }
        if let marker = transitioningMarker, let width = modeBarTransition.width {
            let progress = ModeSelectionPanel.compactProgress(width: moved.frame.width, from: width, expanded: false)
            modeBarTransition.markerX = marker.expandedX + (marker.compactX - marker.expandedX) * progress - moved.frame.minX
        }
        guard !positioningPanel else { return }
        panelAnchor = NSPoint(x: moved.frame.midX, y: moved.frame.minY)
        updateSelectionBanner()
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool { openWindow(); return true }
    func applicationWillTerminate(_ notification: Notification) { model.shutdown() }

    func showPanel() {
        guard !positioningPanel else { return }
        let startingSelection = model.isSelectingMode && (!panelWasSelectingMode || panel?.isVisible != true)
        let reopeningCompact = startingSelection && model.isReselectingMode && panel?.isVisible == true
            && panel.frame.size == VoicePanel.compactSize
        let initializeContent = panel == nil || (startingSelection && panel?.isVisible != true)
        let finishingSelection = panelWasSelectingMode && !model.isSelectingMode
        let reduceMotion = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
        if startingSelection || !model.isSelectingMode || reduceMotion {
            selectionResizeTask?.cancel(); selectionResizeTask = nil
            selectionMotion = SelectionExpansionMotion(value: model.selectionDisplayTarget)
        }
        if reopeningCompact {
            // Note (Codex): Reopening belongs to the existing marker, not the off-center point where it was pressed.
            model.selectionTrackX = panel.frame.midX - ModeSelectionPanel.centerX(for: model.mode)
            model.selectionPanelBottom = panel.frame.minY
            model.selectionExpansionStartY = panel.frame.maxY
        }
        panelWasSelectingMode = model.isSelectingMode
        let size = VoicePanel.size(for: model)
        if panel == nil {
            panel = RecordingPanel(contentRect: NSRect(origin: .zero, size: size),
                            styleMask: [.nonactivatingPanel, .borderless], backing: .buffered, defer: false)
            panel.level = .floating; panel.isOpaque = false; panel.backgroundColor = .clear
            panel.hasShadow = true; panel.hidesOnDeactivate = false
            panel.isReleasedWhenClosed = false
            panel.isMovableByWindowBackground = false; panel.delegate = self
            panel.becomesKeyOnlyIfNeeded = true
            panel.cancelAction = { [weak self] in
                guard let model = self?.model else { return }
                if model.isReviewingResult && !model.isBusy { model.finishReview() } else { model.cancel() }
            }
            panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        }
        if initializeContent {
            let content = VoicePanelHostingView(rootView: VoicePanel(model: model, store: model.store,
                                                                     transition: modeBarTransition, worker: model.worker))
            content.sizingOptions = []
            panel.contentView = content
        }
        panel.allowsTextInput = model.showsEditor && model.isReviewingResult && !model.isBusy
        panel.windowDrag.enabled = !model.isSelectingMode
        let location = model.isSelectingMode
            ? NSPoint(x: model.selectionPointerX, y: model.selectionPointerY) : panelAnchor ?? NSEvent.mouseLocation
        let screen = (!model.isSelectingMode && panel.isVisible ? panel.screen : nil)
            ?? NSScreen.screens.first(where: { NSMouseInRect(location, $0.frame, false) }) ?? NSScreen.main
        if let visible = screen?.visibleFrame, model.isSelectingMode || !panel.isVisible || panel.frame.size != size {
            if finishingSelection && model.usesCompactPanel {
                let compact = VoicePanel.collapsedFrame(from: panel.frame, mode: model.mode, within: visible)
                panelAnchor = NSPoint(x: compact.midX, y: compact.minY)
            }
            let anchor = panelAnchor ?? NSPoint(x: visible.midX, y: visible.minY + 28)
            let origin = NSPoint(x: min(max(anchor.x - size.width / 2, visible.minX), visible.maxX - size.width),
                                 y: min(max(anchor.y, visible.minY), visible.maxY - size.height))
            let frame = model.isSelectingMode ? VoicePanel.selectionFrame(for: model, within: visible)
                : NSRect(origin: origin, size: size)
            if model.isSelectingMode && !startingSelection && !reduceMotion {
                if selectionResizeTask == nil {
                    selectionResizeTask = Task { [weak self] in
                        var previousTime = ProcessInfo.processInfo.systemUptime
                        while !Task.isCancelled {
                            try? await Task.sleep(nanoseconds: 8_333_333)
                            guard !Task.isCancelled, let self else { return }
                            guard self.model.isSelectingMode, self.panel.isVisible else {
                                self.selectionResizeTask = nil
                                return
                            }
                            let now = ProcessInfo.processInfo.systemUptime
                            let target = self.model.selectionDisplayTarget
                            self.selectionMotion.advance(toward: target, elapsed: now - previousTime)
                            previousTime = now
                            self.positioningPanel = true
                            self.panel.setFrame(VoicePanel.selectionFrame(for: self.model, within: visible,
                                                                         expansion: self.selectionMotion.value), display: true)
                            self.panel.contentView?.layoutSubtreeIfNeeded()
                            self.positioningPanel = false
                            self.panelAnchor = NSPoint(x: self.panel.frame.midX, y: self.panel.frame.minY)
                            self.model.selectionBarOrigin = NSPoint(x: self.panel.frame.midX - ModeSelectionPanel.trackWidth / 2,
                                                                    y: self.panel.frame.minY)
                            if self.selectionMotion.value == target && self.selectionMotion.velocity == 0 {
                                self.selectionResizeTask = nil
                                return
                            }
                        }
                    }
                }
            } else {
                positioningPanel = true
                let animateResize = panel.isVisible && panel.frame.size != size && !reduceMotion
                if finishingSelection && model.usesCompactPanel && animateResize {
                    modeBarTransition.width = panel.frame.width
                    let markerX = panel.frame.midX - ModeSelectionPanel.trackWidth / 2 + ModeSelectionPanel.centerX(for: model.mode)
                    transitioningMarker = (expandedX: markerX, compactX: frame.midX)
                    modeBarTransition.markerX = markerX - panel.frame.minX
                } else if reopeningCompact && animateResize {
                    modeBarTransition.width = frame.width
                    let markerX = frame.midX - ModeSelectionPanel.trackWidth / 2 + ModeSelectionPanel.centerX(for: model.mode)
                    transitioningMarker = (expandedX: markerX, compactX: panel.frame.midX)
                    modeBarTransition.markerX = panel.frame.width / 2
                }
                panel.setFrame(frame, display: true, animate: animateResize && (!startingSelection || reopeningCompact))
                transitioningMarker = nil
                modeBarTransition.markerX = nil
                modeBarTransition.width = nil
                panel.contentView?.layoutSubtreeIfNeeded()
                positioningPanel = false
                panelAnchor = NSPoint(x: panel.frame.midX, y: panel.frame.minY)
            }
        }
        if model.isSelectingMode {
            model.selectionBarOrigin = NSPoint(x: panel.frame.midX - ModeSelectionPanel.trackWidth / 2,
                                               y: panel.frame.minY)
            if startingSelection { model.selectionExpansionStartY = panel.frame.minY + VoicePanel.selectorSize.height }
        }
        if model.showsEditor && model.isReviewingResult && !model.isBusy { panel.makeKeyAndOrderFront(nil) }
        else { panel.orderFrontRegardless() }
        updateSelectionBanner()
    }

    private func updateSelectionBanner() {
        guard let panel, panel.isVisible, let text = model.compactSelectionGuidance,
              let visible = panel.screen?.visibleFrame else {
            selectionBanner?.orderOut(nil)
            return
        }
        let banner: NSPanel
        if let selectionBanner { banner = selectionBanner }
        else {
            banner = NSPanel(contentRect: .zero, styleMask: [.nonactivatingPanel, .borderless],
                             backing: .buffered, defer: false)
            banner.isOpaque = false; banner.backgroundColor = .clear
            banner.hasShadow = true; banner.hidesOnDeactivate = false; banner.isReleasedWhenClosed = false
            banner.becomesKeyOnlyIfNeeded = true
            selectionBanner = banner
            panel.addChildWindow(banner, ordered: .above)
        }
        let view = SelectionGuidanceBanner(text: text, needsSelection: model.needsEditSelection,
                                          appearance: model.store.preferences.appearance)
        let size: NSSize
        if let content = banner.contentView as? NSHostingView<SelectionGuidanceBanner> {
            if content.rootView != view {
                content.rootView = view
                size = content.fittingSize
            } else { size = banner.frame.size }
        } else {
            let content = NSHostingView(rootView: view)
            banner.contentView = content
            size = content.fittingSize
        }
        let gap: CGFloat = 8
        let above = panel.frame.maxY + gap
        let bottom = above + size.height <= visible.maxY ? above : panel.frame.minY - gap - size.height
        let frame = NSRect(x: min(max(panel.frame.midX - size.width / 2, visible.minX), visible.maxX - size.width),
                           y: max(visible.minY, bottom), width: size.width, height: size.height)
        selectionBanner?.setFrame(frame, display: true)
        selectionBanner?.orderFrontRegardless()
    }

    private func saveSnapshot(_ path: String) {
        guard let view = window.contentView,
              let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds) else { return }
        view.cacheDisplay(in: view.bounds, to: bitmap)
        if let png = bitmap.representation(using: .png, properties: [:]) {
            try? png.write(to: URL(fileURLWithPath: path))
        }
        NSApp.terminate(nil)
    }
}

struct SelectionExpansionMotion {
    // Note (Codex): Critical damping settles 98% of a step in about 0.21 seconds without bouncing.
    static let angularFrequency: CGFloat = 28
    var value: CGFloat
    var velocity: CGFloat = 0

    mutating func advance(toward target: CGFloat, elapsed: TimeInterval) {
        let time = CGFloat(elapsed)
        let offset = value - target
        let coefficient = velocity + Self.angularFrequency * offset
        let decay = exp(-Self.angularFrequency * time)
        value = target + (offset + coefficient * time) * decay
        velocity = (velocity - Self.angularFrequency * coefficient * time) * decay
        if value < 0 || value > 1 { value = min(max(value, 0), 1); velocity = 0 }
        if abs(value - target) < 0.0005 && abs(velocity) < 0.01 { value = target; velocity = 0 }
    }
}

final class ConsoleHostingView: NSHostingView<RootView> {
    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(nil)
        super.mouseDown(with: event)
    }
}

final class VoicePanelHostingView: NSHostingView<VoicePanel> {
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    override var needsPanelToBecomeKey: Bool { false }
}

final class RecordingPanel: NSPanel {
    let windowDrag = WindowDrag()
    let selectionDrag = PanelSelectionDrag()
    var allowsTextInput = false
    var cancelAction: (() -> Void)?
    override var canBecomeKey: Bool { allowsTextInput }
    override var canBecomeMain: Bool { false }
    override func animationResizeTime(_ newFrame: NSRect) -> TimeInterval { VoicePanel.transitionDuration }
    override func cancelOperation(_ sender: Any?) { cancelAction?() }
    override func sendEvent(_ event: NSEvent) {
        if !selectionDrag.handle(event, in: self) && !windowDrag.handle(event, in: self) { super.sendEvent(event) }
    }

}
