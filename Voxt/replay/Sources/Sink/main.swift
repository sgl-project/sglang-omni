import AppKit

// note (luojiaxuan): replay-sink: one plain-text field that records every change of its contents.
//
// Log lines (JSON, one per line, appended to --log):
//   {"event":"start","pid":…,"t_mach_continuous_ns":…}
//   {"event":"text","t_mach_continuous_ns":…,"text":"…"}        every change of the field's characters
//   {"event":"clear","label":"…","key":true,"t_mach_continuous_ns":…}
//   {"event":"mark","label":"…","t_mach_continuous_ns":…}
//   {"event":"focus","key":false,"t_mach_continuous_ns":…}       the window gained or lost key status
//
// Commands (one line each on the Unix socket --socket; each gets a one-line reply):
//   ping            -> ok <pid>
//   clear <label>   -> ok <t> key=<0|1>   empties the field (not logged as a text change), then logs "clear"
//   mark <label>    -> ok <t>
//   type <text>     -> ok                 inserts <text> at the selection, as typing would (for self-tests)
//   text            -> ok <JSON string>   the field's current contents
//   focus           -> ok key=<0|1>       activates the app and focuses the field
//   quit            -> ok

let usage = "usage: replay-sink --log PATH --socket PATH [--no-activate]"

struct SinkOptions {
    var logPath = ""
    var socketPath = ""
    var activate = true
}

func parseOptions() -> SinkOptions {
    var options = SinkOptions()
    var args = CommandLine.arguments.dropFirst().makeIterator()
    while let arg = args.next() {
        switch arg {
        case "--log": options.logPath = args.next() ?? ""
        case "--socket": options.socketPath = args.next() ?? ""
        case "--no-activate": options.activate = false
        default:
            FileHandle.standardError.write(Data("unknown argument \(arg)\n\(usage)\n".utf8))
            exit(64)
        }
    }
    if options.logPath.isEmpty || options.socketPath.isEmpty {
        FileHandle.standardError.write(Data("\(usage)\n".utf8))
        exit(64)
    }
    return options
}

final class SinkController: NSObject, NSApplicationDelegate, NSWindowDelegate, NSTextStorageDelegate {
    private let options: SinkOptions
    private let log: LineAppender
    private var window: NSWindow!
    private var textView: NSTextView!
    private var clearing = false

    init(options: SinkOptions) throws {
        self.options = options
        log = try LineAppender(path: options.logPath)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        buildMenu()
        buildWindow()
        log.append(["event": "start", "pid": Int(getpid()), "t_mach_continuous_ns": MachClock.now()])
        do {
            try startServer()
        } catch {
            FileHandle.standardError.write(Data("replay-sink: \(error)\n".utf8))
            exit(1)
        }
        if options.activate {
            NSApp.activate(ignoringOtherApps: true)
            window.makeKeyAndOrderFront(nil)
        }
        window.makeFirstResponder(textView)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    func applicationWillTerminate(_ notification: Notification) {
        unlink(options.socketPath)
    }

    // note (luojiaxuan): Cmd-V and the other edit shortcuts reach the text view through the Edit menu,
    // so a writer that pastes needs it.
    private func buildMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem()
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Quit Replay Sink", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        appItem.submenu = appMenu
        main.addItem(appItem)

        let editItem = NSMenuItem()
        let edit = NSMenu(title: "Edit")
        edit.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
        edit.addItem(withTitle: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        edit.addItem(.separator())
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        editItem.submenu = edit
        main.addItem(editItem)
        NSApp.mainMenu = main
    }

    private func buildWindow() {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 640, height: 420),
                          styleMask: [.titled, .closable, .resizable, .miniaturizable],
                          backing: .buffered, defer: false)
        window.title = "Replay Sink"
        window.center()
        window.delegate = self
        window.isReleasedWhenClosed = false

        let scroll = NSTextView.scrollableTextView()
        textView = (scroll.documentView as! NSTextView)
        textView.isRichText = false
        textView.importsGraphics = false
        textView.allowsUndo = true
        textView.font = NSFont.systemFont(ofSize: 15)
        // note (luojiaxuan): The field must hold exactly what the writer inserted: no substitutions,
        // corrections or smart spacing.
        textView.isAutomaticQuoteSubstitutionEnabled = false
        textView.isAutomaticDashSubstitutionEnabled = false
        textView.isAutomaticTextReplacementEnabled = false
        textView.isAutomaticSpellingCorrectionEnabled = false
        textView.isAutomaticTextCompletionEnabled = false
        textView.isAutomaticLinkDetectionEnabled = false
        textView.isAutomaticDataDetectionEnabled = false
        textView.isContinuousSpellCheckingEnabled = false
        textView.isGrammarCheckingEnabled = false
        textView.smartInsertDeleteEnabled = false
        textView.textStorage?.delegate = self
        window.contentView = scroll
    }

    private func isFocused() -> Bool {
        NSApp.isActive && window.isKeyWindow && window.firstResponder === textView
    }

    // MARK: change log

    func textStorage(_ textStorage: NSTextStorage, didProcessEditing editedMask: NSTextStorageEditActions,
                     range editedRange: NSRange, changeInLength delta: Int) {
        let t = MachClock.now()
        guard editedMask.contains(.editedCharacters), !clearing else { return }
        log.append(["event": "text", "t_mach_continuous_ns": t, "text": textStorage.string])
    }

    func windowDidBecomeKey(_ notification: Notification) {
        log.append(["event": "focus", "key": true, "t_mach_continuous_ns": MachClock.now()])
    }

    func windowDidResignKey(_ notification: Notification) {
        log.append(["event": "focus", "key": false, "t_mach_continuous_ns": MachClock.now()])
    }

    // MARK: commands

    /// note (luojiaxuan): Runs on the main thread.
    private func handle(_ line: String) -> String {
        let parts = line.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: false)
        let command = parts.first.map(String.init) ?? ""
        let argument = parts.count > 1 ? String(parts[1]) : ""
        switch command {
        case "ping":
            return "ok \(getpid())"
        case "clear":
            clearing = true
            textView.string = ""
            clearing = false
            textView.undoManager?.removeAllActions()
            window.makeFirstResponder(textView)
            let t = MachClock.now()
            let key = isFocused()
            log.append(["event": "clear", "label": argument, "key": key, "t_mach_continuous_ns": t])
            return "ok \(t) key=\(key ? 1 : 0)"
        case "mark":
            let t = MachClock.now()
            log.append(["event": "mark", "label": argument, "t_mach_continuous_ns": t])
            return "ok \(t)"
        case "type":
            textView.insertText(argument, replacementRange: textView.selectedRange())
            return "ok"
        case "text":
            return "ok " + JSONLine.quote(textView.string)
        case "focus":
            NSApp.activate(ignoringOtherApps: true)
            window.makeKeyAndOrderFront(nil)
            window.makeFirstResponder(textView)
            return "ok key=\(isFocused() ? 1 : 0)"
        case "quit":
            DispatchQueue.main.async { NSApp.terminate(nil) }
            return "ok"
        default:
            return "err unknown command \(command)"
        }
    }

    private func startServer() throws {
        unlink(options.socketPath)
        var address = try unixAddress(options.socketPath)
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else { throw ReplayError("socket: \(String(cString: strerror(errno)))") }
        let bound = withUnsafePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) { Darwin.bind(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size)) }
        }
        guard bound == 0 else { throw ReplayError("bind \(options.socketPath): \(String(cString: strerror(errno)))") }
        guard listen(fd, 8) == 0 else { throw ReplayError("listen: \(String(cString: strerror(errno)))") }
        Thread { [weak self] in
            while true {
                let client = accept(fd, nil, nil)
                if client < 0 {
                    if errno == EINTR { continue }
                    return
                }
                Thread { self?.serve(client) }.start()
            }
        }.start()
    }

    private func serve(_ client: Int32) {
        defer { close(client) }
        let reader = LineReader(fd: client)
        while let line = reader.next() {
            let reply = DispatchQueue.main.sync { handle(line) }
            if !writeAll(fd: client, reply + "\n") { return }
        }
    }
}

let options = parseOptions()
let app = NSApplication.shared
app.setActivationPolicy(options.activate ? .regular : .accessory)
let controller: SinkController
do {
    controller = try SinkController(options: options)
} catch {
    FileHandle.standardError.write(Data("replay-sink: \(error)\n".utf8))
    exit(1)
}
app.delegate = controller
app.run()
