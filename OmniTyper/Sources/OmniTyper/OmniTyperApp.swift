// SPDX-License-Identifier: Apache-2.0
import AppKit
import SwiftUI

@main
struct OmniTyperApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    var body: some Scene { Settings { EmptyView() } }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, NSMenuDelegate {
    private var model: AppModel!
    private var window: NSWindow!
    private var panel: RecordingPanel!
    private var statusItem: NSStatusItem!

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        model = AppModel()
        let content = RootView(model: model, store: model.store)
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1060, height: 750),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "OmniTyper"
        window.titlebarSeparatorStyle = .line
        window.isMovableByWindowBackground = false
        window.contentView = ConsoleHostingView(rootView: content)
        window.minSize = NSSize(width: 920, height: 660)
        window.isReleasedWhenClosed = false
        window.delegate = self
        window.center()
        window.setFrameAutosaveName("OmniTyperMainWindow")
        model.showMainWindow = { [weak self] in self?.openWindow() }
        model.showVoicePanel = { [weak self] in self?.showPanel() }
        model.hideVoicePanel = { [weak self] in self?.panel?.orderOut(nil) }
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
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool { openWindow(); return true }
    func applicationWillTerminate(_ notification: Notification) { model.shutdown() }

    private func showPanel() {
        let needsPlacement = panel == nil
        if panel == nil {
            panel = RecordingPanel(contentRect: NSRect(x: 0, y: 0, width: 460, height: 190),
                            styleMask: [.nonactivatingPanel, .borderless], backing: .buffered, defer: false)
            panel.level = .floating; panel.isOpaque = false; panel.backgroundColor = .clear
            panel.hasShadow = true; panel.hidesOnDeactivate = false
            panel.isMovableByWindowBackground = false
            panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
            panel.contentView = VoicePanelHostingView(rootView: VoicePanel(model: model, recorder: model.recorder, worker: model.worker))
        }
        if needsPlacement || !NSScreen.screens.contains(where: { $0.visibleFrame.intersects(panel.frame) }) {
            let pointer = NSEvent.mouseLocation
            let screen = NSScreen.screens.first(where: { NSMouseInRect(pointer, $0.frame, false) }) ?? NSScreen.main
            if let visible = screen?.visibleFrame {
                panel.setFrameOrigin(NSPoint(x: visible.midX - panel.frame.width / 2, y: visible.minY + 28))
            }
        }
        panel.orderFrontRegardless()
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
    let dragRegions = NSHashTable<NSView>.weakObjects()
    private var dragStart: (pointer: NSPoint, origin: NSPoint)?

    override func sendEvent(_ event: NSEvent) {
        switch event.type {
        case .leftMouseDown:
            dragStart = nil
            if dragRegions.allObjects.contains(where: { view in
                view.window === self && !view.isHiddenOrHasHiddenAncestor
                    && view.bounds.intersection(view.visibleRect).contains(view.convert(event.locationInWindow, from: nil))
            }) {
                dragStart = (convertPoint(toScreen: event.locationInWindow), frame.origin)
                return
            }
        case .leftMouseDragged:
            if let dragStart {
                let pointer = convertPoint(toScreen: event.locationInWindow)
                setFrameOrigin(NSPoint(x: dragStart.origin.x + pointer.x - dragStart.pointer.x,
                                       y: dragStart.origin.y + pointer.y - dragStart.pointer.y))
                return
            }
        case .leftMouseUp:
            if dragStart != nil { dragStart = nil; return }
        default: break
        }
        super.sendEvent(event)
    }
}

struct WindowDragArea: NSViewRepresentable {
    func makeNSView(context: Context) -> DragView { DragView() }
    func updateNSView(_ view: DragView, context: Context) {}

    final class DragView: NSView {
        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            (window as? RecordingPanel)?.dragRegions.add(self)
        }
        override func hitTest(_ point: NSPoint) -> NSView? { nil }
    }
}
