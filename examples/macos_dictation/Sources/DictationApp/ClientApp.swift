#if canImport(DictationCore)
import DictationCore
#endif
import AppKit
import Carbon
import Combine
import SwiftUI
import UniformTypeIdentifiers

private final class DictationPanel: NSPanel {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }
}

/// Track the whole floating bar even while the user's editor remains the active app.
private final class DictationPanelContent: NSHostingView<ClientFloatingView> {
    var onHover: ((Bool) -> Void)?
    private var hoverArea: NSTrackingArea?

    override func updateTrackingAreas() {
        if let hoverArea { removeTrackingArea(hoverArea) }
        super.updateTrackingAreas()
        let area = NSTrackingArea(rect: .zero,
                                  options: [.mouseEnteredAndExited, .activeAlways, .inVisibleRect],
                                  owner: self, userInfo: nil)
        addTrackingArea(area)
        hoverArea = area
        refreshHover()
    }

    override func mouseEntered(with event: NSEvent) { onHover?(true) }
    override func mouseExited(with event: NSEvent) { onHover?(false) }

    func refreshHover() {
        guard let window, window.isVisible else { return }
        let point = convert(window.convertPoint(fromScreen: NSEvent.mouseLocation), from: nil)
        onHover?(visibleRect.contains(point))
    }
}

@MainActor
final class ClientDelegate: NSObject, NSApplicationDelegate {
    let state = ClientState()
    private var item: NSStatusItem?
    private var resultWindow: NSWindow?
    private var settingsWindow: NSWindow?
    private var panel: DictationPanel?
    private var shortcutObserver: AnyCancellable?
    private var focusObservers: [NSObjectProtocol] = []
    private var recordingMenuItems: [NSMenuItem] = []
    private var phaseObserver: AnyCancellable?
    private var feedbackObserver: AnyCancellable?
    private var escapeMonitor: Any?

    func applicationDidFinishLaunching(_ notification: Notification) {
        state.openResults = { [weak self] in self?.showResults() }
        state.openAudio = { [weak self] in self?.importAudio() }
        makeMenus()
        state.registrar.action = { [weak self] id in
            guard let self, NSApp.modalWindow == nil, self.state.shortcuts.isAvailable,
                  !self.state.shortcuts.isCapturing else { return }
            if id == 1 { self.state.toggleRecording() }
            else { self.state.cancel() }
        }
        state.shortcuts.start()
        shortcutObserver = state.shortcuts.$shortcut.sink { [weak self] shortcut in
            self?.recordingMenuItems.forEach { $0.title = "开始 / 结束录音    \(shortcut.display)" }
        }
        phaseObserver = state.session.$phase.removeDuplicates().sink { [weak self] phase in
            guard let self else { return }
            if phase == .failed { self.state.insertion.abandon("本轮未完成，没有自动回填。") }
            self.item?.button?.image = NSImage(systemSymbolName: phase == .recording ? "mic.fill" : "mic.circle",
                                               accessibilityDescription: "Omni 听写：\(phase.rawValue)")
        }
        feedbackObserver = state.feedback.$opacity.removeDuplicates().sink { [weak self] opacity in
            guard let self else { return }
            if opacity <= 0 { self.panel?.orderOut(nil) }
            else { self.showPanel(opacity: opacity) }
        }
        escapeMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp]) { [weak self] event in
            guard let self else { return event }
            if self.state.shortcuts.isCapturing {
                if event.type == .keyDown, !event.isARepeat {
                    if event.keyCode == UInt16(kVK_Escape),
                       event.modifierFlags.intersection([.control, .option, .command, .shift]).isEmpty {
                        self.state.shortcuts.cancelCapture()
                    } else { self.state.shortcuts.capture(RecordingShortcut(event: event)) }
                } else if event.type == .keyUp {
                    self.state.shortcuts.release(keyCode: UInt32(event.keyCode))
                }
                return nil
            }
            if event.type == .keyDown, event.keyCode == 53, NSApp.modalWindow == nil, self.state.session.isBusy {
                self.state.cancel()
                return nil
            }
            return event
        }
        for name in [NSWindow.didResignKeyNotification, NSWindow.willCloseNotification, NSApplication.didResignActiveNotification] {
            focusObservers.append(NotificationCenter.default.addObserver(forName: name, object: nil, queue: .main) { [weak self] note in
                MainActor.assumeIsolated {
                    guard let self else { return }
                    if note.name == NSApplication.didResignActiveNotification || note.object as? NSWindow === self.settingsWindow {
                        self.state.shortcuts.cancelCapture()
                    }
                }
            })
        }
        state.checkServices(retryWarmup: false)
        if !state.shortcuts.isAvailable { showResults() }
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        showResults()
        return true
    }

    func applicationWillTerminate(_ notification: Notification) {
        state.cancel()
        state.warmup.stop()
        state.registrar.stop()
        focusObservers.forEach { NotificationCenter.default.removeObserver($0) }
        if let escapeMonitor { NSEvent.removeMonitor(escapeMonitor) }
    }

    private func entry(_ title: String, _ action: Selector, key: String = "") -> NSMenuItem {
        let item = NSMenuItem(title: title, action: action, keyEquivalent: key)
        item.target = self
        return item
    }

    private func makeMenus() {
        let main = NSMenu()
        let app = NSMenu()
        app.addItem(entry("设置…", #selector(showSettings), key: ","))
        app.addItem(.separator())
        app.addItem(entry("退出 Omni 听写", #selector(quit), key: "q"))
        let root = NSMenuItem()
        root.submenu = app
        main.addItem(root)
        let file = NSMenu(title: "文件")
        let recording = entry("开始 / 结束录音", #selector(toggleRecording))
        recordingMenuItems.append(recording)
        file.addItem(recording)
        file.addItem(entry("取消本轮", #selector(cancel)))
        file.addItem(entry("打开录音文件…", #selector(importAudio), key: "o"))
        let edit = NSMenu(title: "编辑")
        edit.addItem(NSMenuItem(title: "撤销", action: Selector(("undo:")), keyEquivalent: "z"))
        edit.addItem(NSMenuItem(title: "剪切", action: #selector(NSText.cut(_:)), keyEquivalent: "x"))
        edit.addItem(NSMenuItem(title: "复制", action: #selector(NSText.copy(_:)), keyEquivalent: "c"))
        edit.addItem(NSMenuItem(title: "粘贴", action: #selector(NSText.paste(_:)), keyEquivalent: "v"))
        edit.addItem(NSMenuItem(title: "全选", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a"))
        let windows = NSMenu(title: "窗口")
        windows.addItem(entry("本轮结果", #selector(showResults)))
        windows.addItem(NSMenuItem(title: "关闭窗口", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w"))
        for menu in [file, edit, windows] {
            let root = NSMenuItem(title: menu.title, action: nil, keyEquivalent: "")
            root.submenu = menu
            main.addItem(root)
        }
        NSApp.mainMenu = main
        NSApp.windowsMenu = windows
        let status = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        let menu = NSMenu()
        let statusRecording = entry("开始 / 结束录音", #selector(toggleRecording))
        recordingMenuItems.append(statusRecording)
        menu.addItem(statusRecording)
        menu.addItem(entry("取消本轮    ⌃⇧Esc", #selector(cancel)))
        menu.addItem(entry("本轮结果", #selector(showResults)))
        menu.addItem(entry("打开录音文件…", #selector(importAudio)))
        menu.addItem(.separator())
        menu.addItem(entry("设置…", #selector(showSettings), key: ","))
        menu.addItem(entry("退出 Omni 听写", #selector(quit), key: "q"))
        status.menu = menu
        status.button?.toolTip = "Omni 听写 · 本地 MLX"
        item = status
    }

    @objc private func toggleRecording() { state.toggleRecording() }
    @objc private func cancel() { state.cancel() }
    @objc private func quit() { NSApp.terminate(nil) }

    @objc private func showResults() {
        if resultWindow == nil {
            let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 680, height: 540),
                                  styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
            window.title = "Omni 听写"
            window.titlebarAppearsTransparent = true
            window.isReleasedWhenClosed = false
            window.contentMinSize = NSSize(width: 560, height: 460)
            window.contentView = NSHostingView(rootView: ClientResultView(model: state.session, state: state, insertion: state.insertion))
            window.center()
            resultWindow = window
        }
        NSApp.activate(ignoringOtherApps: true)
        resultWindow?.makeKeyAndOrderFront(nil)
    }

    @objc private func showSettings() {
        state.accessibilityGranted = DictationAccessibility.isTrusted
        if settingsWindow == nil {
            let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 550, height: 650),
                                  styleMask: [.titled, .closable], backing: .buffered, defer: false)
            window.title = "Omni 听写设置"
            window.isReleasedWhenClosed = false
            window.contentView = NSHostingView(rootView: ClientSettingsView(model: state.session, state: state))
            window.center()
            settingsWindow = window
        }
        NSApp.activate(ignoringOtherApps: true)
        settingsWindow?.makeKeyAndOrderFront(nil)
    }

    @objc private func importAudio() {
        guard !state.session.isBusy, !state.insertion.isDelivering else { return }
        let dialog = NSOpenPanel()
        dialog.allowedContentTypes = [.audio]
        dialog.allowsMultipleSelection = false
        dialog.canChooseDirectories = false
        dialog.message = "选择 \(Int(DictationSession.maximumAudioSeconds)) 秒以内的录音，仅发送给本机 Omni ASR。"
        NSApp.activate(ignoringOtherApps: true)
        guard dialog.runModal() == .OK, let url = dialog.url, !state.session.isBusy else { return }
        state.insertion.abandon("文件转写仅供查看和复制，不自动填入其他应用。")
        do { state.session.submitAudio(try AudioEncoder.load(url)) }
        catch { state.session.reportInputFailure("无法读取录音：\(error.localizedDescription)") }
        showResults()
    }

    private func showPanel(opacity: Double) {
        if panel == nil {
            let panel = DictationPanel(contentRect: NSRect(origin: .zero, size: ClientFloatingView.size),
                                       styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
            panel.level = .floating
            panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
            panel.hidesOnDeactivate = false
            panel.isOpaque = false
            panel.backgroundColor = .clear
            panel.hasShadow = true
            panel.isMovableByWindowBackground = true
            let content = DictationPanelContent(rootView: ClientFloatingView(model: state.session, state: state,
                                                                             insertion: state.insertion))
            content.onHover = { [weak self] in self?.state.feedback.setHovered($0) }
            panel.contentView = content
            self.panel = panel
        }
        if panel?.isVisible != true,
           let screen = NSScreen.screens.first(where: { $0.frame.contains(NSEvent.mouseLocation) }) ?? NSScreen.main {
            let frame = screen.visibleFrame
            panel?.setFrameOrigin(NSPoint(x: frame.midX - ClientFloatingView.size.width / 2, y: frame.minY + 28))
        }
        panel?.alphaValue = opacity
        if panel?.isVisible != true {
            panel?.orderFrontRegardless()
            // The bar may appear under a stationary pointer, before an enter event.
            (panel?.contentView as? DictationPanelContent)?.refreshHover()
        }
    }
}

@main
@MainActor
enum ClientMain {
    static func main() {
        let app = NSApplication.shared
        let delegate = ClientDelegate()
        app.delegate = delegate
        app.setActivationPolicy(.accessory)
        withExtendedLifetime(delegate) { app.run() }
    }
}
