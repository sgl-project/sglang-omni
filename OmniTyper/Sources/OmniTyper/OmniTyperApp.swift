// SPDX-License-Identifier: Apache-2.0
import AppKit
import SwiftUI

@main
enum OmniTyperLauncher {
    @MainActor static func main() {
        // Note (Codex): AppKit owns startup so a placeholder Settings scene cannot become the launch window.
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        withExtendedLifetime(delegate) { app.run() }
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSWindowDelegate, NSMenuDelegate {
    private var model: AppModel!
    private var window: NSWindow!
    private var panel: RecordingPanel!
    private var statusItem: NSStatusItem!
    private var settingsWindow: NSWindow?

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
        window.titleVisibility = .visible
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
        setupMainMenu()
        setupMenu()
        if !ProcessInfo.processInfo.arguments.contains("--background") { openWindow() }
        if let index = ProcessInfo.processInfo.arguments.firstIndex(of: "--snapshot"),
           ProcessInfo.processInfo.arguments.count > index + 1 {
            let path = ProcessInfo.processInfo.arguments[index + 1]
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in self?.saveSnapshot(path) }
        }
    }

    private func setupMainMenu() {
        func systemTitle(_ key: String) -> String { L10n.string(key, in: nil) }
        let mainMenu = NSMenu()
        let applicationMenu = NSMenu()
        let applicationItem = mainMenu.addItem(withTitle: "OmniTyper", action: nil, keyEquivalent: "")
        applicationItem.submenu = applicationMenu
        applicationMenu.addItem(withTitle: systemTitle("menu.about"), action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)), keyEquivalent: "")
        applicationMenu.addItem(.separator())
        let settingsItem = applicationMenu.addItem(withTitle: systemTitle("nav.Settings") + "…", action: #selector(openSettings), keyEquivalent: ",")
        settingsItem.target = self
        applicationMenu.addItem(.separator())
        let services = NSMenu(title: systemTitle("menu.services"))
        applicationMenu.addItem(withTitle: services.title, action: nil, keyEquivalent: "").submenu = services
        NSApp.servicesMenu = services
        applicationMenu.addItem(.separator())
        applicationMenu.addItem(withTitle: systemTitle("menu.hide"), action: #selector(NSApplication.hide(_:)), keyEquivalent: "h")
        let hideOthers = applicationMenu.addItem(withTitle: systemTitle("menu.hideOthers"), action: #selector(NSApplication.hideOtherApplications(_:)), keyEquivalent: "h")
        hideOthers.keyEquivalentModifierMask = [.command, .option]
        applicationMenu.addItem(withTitle: systemTitle("menu.showAll"), action: #selector(NSApplication.unhideAllApplications(_:)), keyEquivalent: "")
        applicationMenu.addItem(.separator())
        applicationMenu.addItem(withTitle: systemTitle("menu.quit"), action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")

        let editMenu = NSMenu(title: systemTitle("menu.edit"))
        mainMenu.addItem(withTitle: editMenu.title, action: nil, keyEquivalent: "").submenu = editMenu
        editMenu.addItem(withTitle: systemTitle("menu.undo"), action: Selector(("undo:")), keyEquivalent: "z")
        let redo = editMenu.addItem(withTitle: systemTitle("menu.redo"), action: Selector(("redo:")), keyEquivalent: "z")
        redo.keyEquivalentModifierMask = [.command, .shift]
        editMenu.addItem(.separator())
        for (key, action, shortcut) in [("cut", "cut:", "x"), ("copy", "copy:", "c"), ("paste", "paste:", "v"), ("selectAll", "selectAll:", "a")] {
            editMenu.addItem(withTitle: systemTitle("menu." + key), action: Selector(action), keyEquivalent: shortcut)
        }

        let viewMenu = NSMenu(title: systemTitle("menu.view"))
        mainMenu.addItem(withTitle: viewMenu.title, action: nil, keyEquivalent: "").submenu = viewMenu
        let fullScreen = viewMenu.addItem(withTitle: systemTitle("menu.fullScreen"), action: #selector(NSWindow.toggleFullScreen(_:)), keyEquivalent: "f")
        fullScreen.keyEquivalentModifierMask = [.command, .control]

        let windowMenu = NSMenu(title: systemTitle("menu.window"))
        mainMenu.addItem(withTitle: windowMenu.title, action: nil, keyEquivalent: "").submenu = windowMenu
        windowMenu.addItem(withTitle: systemTitle("menu.close"), action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        windowMenu.addItem(withTitle: systemTitle("menu.minimize"), action: #selector(NSWindow.performMiniaturize(_:)), keyEquivalent: "m")
        windowMenu.addItem(withTitle: systemTitle("menu.zoom"), action: #selector(NSWindow.performZoom(_:)), keyEquivalent: "")
        windowMenu.addItem(.separator())
        windowMenu.addItem(withTitle: systemTitle("menu.bringAllToFront"), action: #selector(NSApplication.arrangeInFront(_:)), keyEquivalent: "")
        NSApp.windowsMenu = windowMenu
        NSApp.mainMenu = mainMenu
    }

    @objc private func openSettings() {
        if settingsWindow == nil {
            let settings = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 900, height: 450),
                                    styleMask: [.titled, .closable], backing: .buffered, defer: false)
            settings.title = "OmniTyper " + L10n.string("nav.Settings", in: nil)
            settings.isReleasedWhenClosed = false
            settings.center()
            settingsWindow = settings
        }
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        settingsWindow?.makeKeyAndOrderFront(nil)
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
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool { openWindow(); return false }
    func applicationWillTerminate(_ notification: Notification) { model.shutdown() }

    private func showPanel() {
        let needsPlacement = panel == nil
        if panel == nil {
            panel = RecordingPanel(contentRect: NSRect(x: 0, y: 0, width: 460, height: 190),
                            styleMask: [.nonactivatingPanel, .borderless], backing: .buffered, defer: false)
            panel.level = .floating; panel.isOpaque = false; panel.backgroundColor = .clear
            panel.hasShadow = true; panel.hidesOnDeactivate = false
            panel.isReleasedWhenClosed = false
            panel.isMovableByWindowBackground = false; panel.delegate = self
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
    let windowDrag = WindowDrag()
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
    override func sendEvent(_ event: NSEvent) {
        if !windowDrag.handle(event, in: self) { super.sendEvent(event) }
    }
}
