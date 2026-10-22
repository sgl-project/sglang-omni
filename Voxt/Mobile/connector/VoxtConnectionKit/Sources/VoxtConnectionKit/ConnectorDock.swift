import AppKit
import Observation
import os

/// Timing of window placement; read with `log show --predicate 'subsystem == "<the Connector's bundle ID>"'`.
let windowLog = Logger(subsystem: ConnectorIdentity.bundleIdentifier, category: "window")

/// The Dock icon: shown by default so the Settings window is easy to find; when hidden, the
/// Connector runs from the menu bar only, like an LSUIElement app.
@MainActor @Observable public final class ConnectorDock {
    /// Leaving the Dock hands activation to the previous app, whose windows cover Settings within
    /// 50 ms. Settings comes back once the Connector has lost activation; this timer only covers the
    /// case where it never had it. Re-activating on the next main-loop turn is too early, and a
    /// fixed 150 ms timer brings Settings back only at about 640 ms.
    public static let reactivationFallback: Duration = .milliseconds(300)

    public var showsInDock: Bool {
        didSet {
            preferences.showsInDock = showsInDock
            apply()
            if !showsInDock { showSettingsAfterLosingActivation() }
        }
    }
    /// Opens the Settings window or brings it forward. The app sets it once a view can open windows.
    @ObservationIgnored public var openSettingsWindow: () -> Void = {}
    @ObservationIgnored private let preferences: any ConnectorPreferences
    @ObservationIgnored private let setPolicy: (NSApplication.ActivationPolicy) -> Void
    @ObservationIgnored private let activate: () -> Void
    @ObservationIgnored private let after: (Duration, @escaping @MainActor () -> Void) -> Void
    @ObservationIgnored private let onNextResign: (@escaping @MainActor () -> Void) -> Void

    public init(preferences: any ConnectorPreferences,
                setPolicy: @escaping (NSApplication.ActivationPolicy) -> Void = { NSApplication.shared.setActivationPolicy($0) },
                activate: @escaping () -> Void = { ConnectorDock.activateApp() },
                after: @escaping (Duration, @escaping @MainActor () -> Void) -> Void = { delay, work in
                    let seconds = Double(delay.components.seconds) + Double(delay.components.attoseconds) / 1e18
                    DispatchQueue.main.asyncAfter(deadline: .now() + seconds) { MainActor.assumeIsolated { work() } }
                },
                onNextResign: @escaping (@escaping @MainActor () -> Void) -> Void = { ConnectorDock.onNextResign($0) }) {
        self.preferences = preferences
        self.setPolicy = setPolicy
        self.activate = activate
        self.after = after
        self.onNextResign = onNextResign
        showsInDock = preferences.showsInDock
    }

    public nonisolated static func activationPolicy(showsInDock: Bool) -> NSApplication.ActivationPolicy {
        showsInDock ? .regular : .accessory
    }

    /// Call before launch finishes, so a hidden icon never appears.
    public func applyAtLaunch() { apply() }

    /// Activates first: a window opened while another app is active stays behind that app's windows.
    public func showSettings() {
        windowLog.notice("show settings: activating")
        activate()
        windowLog.notice("show settings: opening window")
        openSettingsWindow()
        windowLog.notice("show settings: open requested")
    }

    /// `NSApp.activate()` only asks; with another app active the request is refused and the window
    /// stays covered. The deprecated call still takes activation.
    public static func activateApp() {
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    /// A Dock icon click, or opening the app while it runs. Returns false: handled here.
    public func reopen(hasVisibleWindows: Bool) -> Bool {
        showSettings()
        return false
    }

    private func showSettingsAfterLosingActivation() {
        var done = false
        let show: @MainActor () -> Void = { [weak self] in
            guard !done else { return }
            done = true
            windowLog.notice("left the Dock: showing settings again")
            self?.showSettings()
        }
        onNextResign(show)
        after(Self.reactivationFallback, show)
    }

    /// Runs `work` once, the next time this app stops being the active app.
    public static func onNextResign(_ work: @escaping @MainActor () -> Void) {
        let center = NotificationCenter.default
        nonisolated(unsafe) var observer: (any NSObjectProtocol)?
        observer = center.addObserver(forName: NSApplication.didResignActiveNotification, object: nil, queue: .main) { _ in
            if let observer { center.removeObserver(observer) }
            observer = nil
            MainActor.assumeIsolated { work() }
        }
    }

    private func apply() { setPolicy(Self.activationPolicy(showsInDock: showsInDock)) }
}
