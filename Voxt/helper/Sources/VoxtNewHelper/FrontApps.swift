import AppKit
import ApplicationServices
import VoxtNewHelperKit
import Foundation

/// The app in front, kept from NSWorkspace's activation notifications, so
/// any thread reads it without a hop to the main thread (the hotkey tap's
/// context snapshot, textio): app.activated tells the app of each change.
///
/// As an app comes to the front, an Electron app or one of the bundle ids
/// app.configure names (the app's Chromium browsers) gets
/// AXManualAccessibility, once per process (Chromium builds a page's
/// accessibility tree only when asked, and only a moment later: set at
/// activation, the tree is there when a shortcut reads the selection or
/// the tab's address). browser.url sets it too when it finds no web area.
final class FrontApps: @unchecked Sendable {
    let out: Writer
    private let lock = NSLock()
    private var app: App?
    private var started = false
    private var electron: [String: Bool] = [:] // bundle path → is Electron
    let manual = ManualAccessibility()
    private let queue = DispatchQueue(label: "voxtnew.front", qos: .userInitiated)

    init(out: Writer) { self.out = out }

    /// Starts watching (once; on the main thread's run loop).
    func start() {
        let first = lock.withLock { () -> Bool in
            defer { started = true }
            return !started
        }
        guard first else { return }
        DispatchQueue.main.async { [self] in
            if let a = NSWorkspace.shared.frontmostApplication {
                lock.withLock { app = Self.describe(a) }
            }
            NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.didActivateApplicationNotification,
                                                              object: nil, queue: nil) { [weak self] n in
                guard let self, let a = n.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication else { return }
                self.activated(a)
            }
            NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.didTerminateApplicationNotification,
                                                              object: nil, queue: nil) { [weak self] n in
                guard let self, let a = n.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication else { return }
                self.manual.forget(pid: Int(a.processIdentifier))
            }
        }
    }

    /// app.configure: the bundle ids that get AXManualAccessibility besides
    /// Electron apps. The app in front gets it now when it is one of them
    /// (it came to the front before the list did).
    func configure(_ p: AppConfigureParams) {
        manual.configure(p.manualAccessibility ?? [])
        guard let cur = cached, manual.listed(cur.bundleId) else { return }
        queue.async { [self] in
            if AXIsProcessTrusted(), manual.claim(pid: cur.pid, .activation) {
                setManualAccessibility(pid: pid_t(cur.pid), bundleId: cur.bundleId, why: "in front")
            }
        }
    }

    /// Sets AXManualAccessibility on a process, logging the outcome (debug).
    @discardableResult
    func setManualAccessibility(pid: pid_t, bundleId: String?, why: String) -> Bool {
        let r = AXUIElementSetAttributeValue(AXUIElementCreateApplication(pid), "AXManualAccessibility" as CFString, kCFBooleanTrue)
        let what = "AXManualAccessibility on \(bundleId ?? "?") (pid \(pid), \(why))"
        out.logLine("debug", r == .success ? what + ": set" : what + ": failed \(r.rawValue)")
        return r == .success
    }

    static func describe(_ a: NSRunningApplication) -> App {
        App(pid: Int(a.processIdentifier), bundleId: a.bundleIdentifier, name: a.localizedName)
    }

    private func activated(_ a: NSRunningApplication) {
        let desc = Self.describe(a)
        lock.withLock { app = desc }
        out.notify(Method.appActivated, desc)
        let bundle = a.bundleURL
        queue.async { [self] in
            let electron = bundle.map(isElectron) ?? false
            guard manual.wants(bundleId: desc.bundleId, electron: electron), AXIsProcessTrusted(),
                  manual.claim(pid: desc.pid, .activation) else { return }
            setManualAccessibility(pid: a.processIdentifier, bundleId: desc.bundleId, why: electron ? "Electron" : "activated")
        }
    }

    func isElectron(_ bundle: URL) -> Bool {
        let key = bundle.path
        if let known = lock.withLock({ electron[key] }) { return known }
        let is_ = FileManager.default.fileExists(atPath: bundle.appendingPathComponent("Contents/Frameworks/Electron Framework.framework").path)
        lock.withLock { electron[key] = is_ }
        return is_
    }

    /// The app in front now: the cache, else NSWorkspace (on the main thread).
    func current() async -> App {
        if let a = cached { return a }
        return await onMain { NSWorkspace.shared.frontmostApplication.map(FrontApps.describe) ?? App(pid: 0, bundleId: nil, name: nil) }
    }

    /// The app in front as last reported, from any thread.
    var cached: App? { lock.withLock { app } }
}
