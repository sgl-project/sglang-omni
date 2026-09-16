#if canImport(DictationCore)
import DictationCore
#endif
import AppKit
import ApplicationServices
import IOKit.hidsystem

@MainActor
final class DoubleOptionShortcut {
    var action: (() -> Void)?
    var isEnabled: () -> Bool = { true }
    private var gesture = DoubleOptionGesture()
    private var globalMonitor: Any?
    private var localMonitor: Any?
    private var activationObserver: NSObjectProtocol?

    func start() {
        stop()
        let mask: NSEvent.EventTypeMask = [.flagsChanged, .keyDown, .leftMouseDown, .rightMouseDown, .otherMouseDown]
        localMonitor = NSEvent.addLocalMonitorForEvents(matching: mask) { [weak self] event in
            self?.handle(event)
            return event
        }
        if AXIsProcessTrusted() {
            globalMonitor = NSEvent.addGlobalMonitorForEvents(matching: mask) { [weak self] in self?.handle($0) }
        }
        activationObserver = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification, object: nil, queue: .main
        ) { [weak self] _ in MainActor.assumeIsolated { self?.gesture.reset() } }
    }

    func stop() {
        if let localMonitor { NSEvent.removeMonitor(localMonitor) }
        if let globalMonitor { NSEvent.removeMonitor(globalMonitor) }
        if let activationObserver { NSWorkspace.shared.notificationCenter.removeObserver(activationObserver) }
        localMonitor = nil
        globalMonitor = nil
        activationObserver = nil
        gesture.reset()
    }

    private func handle(_ event: NSEvent) {
        guard isEnabled(), AXIsProcessTrusted(), event.type == .flagsChanged,
              [58, 61].contains(event.keyCode) else { gesture.reset(); return }
        // Device-specific bits distinguish overlapping left/right Option presses.
        let left = event.modifierFlags.rawValue & UInt(NX_DEVICELALTKEYMASK) != 0
        let right = event.modifierFlags.rawValue & UInt(NX_DEVICERALTKEYMASK) != 0
        let other = !event.modifierFlags.intersection([.command, .control, .shift, .function]).isEmpty || (left && right)
        let down = event.keyCode == 58 ? left : right
        if gesture.option(key: event.keyCode, down: down, otherModifiers: other, time: event.timestamp) { action?() }
    }
}
