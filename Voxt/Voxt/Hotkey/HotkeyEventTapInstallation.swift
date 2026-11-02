import Foundation
import ApplicationServices

/// One tap, one source and one run loop. Detaching an old installation never
/// stops the replacement's thread. The manager stores this owner under stateLock.
nonisolated final class HotkeyEventTapInstallation: @unchecked Sendable {
    private final class CallbackContext {
        let handle: (CGEventType, CGEvent) -> Bool
        init(handle: @escaping (CGEventType, CGEvent) -> Bool) { self.handle = handle }
    }

    let location: CGEventTapLocation
    private let lock = NSLock()
    private var tap: CFMachPort?
    private var source: CFRunLoopSource?
    private let runLoop: HotkeyEventTapRunLoop
    private let context: CallbackContext

    static func create(
        eventMask: CGEventMask,
        handle: @escaping (CGEventType, CGEvent) -> Bool
    ) -> HotkeyEventTapInstallation? {
        let context = CallbackContext(handle: handle)
        let callback: CGEventTapCallBack = { _, type, event, refcon in
            guard let refcon else { return Unmanaged.passUnretained(event) }
            let context = Unmanaged<CallbackContext>.fromOpaque(refcon).takeUnretainedValue()
            return context.handle(type, event) ? nil : Unmanaged.passUnretained(event)
        }
        // Prefer the session tap for Fn delivery, validating the granted events at either location.
        for location in [CGEventTapLocation.cgSessionEventTap, .cghidEventTap] {
            guard let tap = createEventTap(
                location: location, eventMask: eventMask, callback: callback,
                userInfo: Unmanaged.passUnretained(context).toOpaque()
            ) else { continue }
            // Do not let callbacks enter the manager until it has registered the owner.
            let loop = HotkeyEventTapRunLoop()
            guard let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0),
                  loop.addSource(source) else {
                CFMachPortInvalidate(tap)
                loop.stop()
                return nil
            }
            return HotkeyEventTapInstallation(tap: tap, source: source, runLoop: loop, context: context, location: location)
        }
        return nil
    }

    static func createEventTap(
        location: CGEventTapLocation,
        eventMask: CGEventMask,
        callback: CGEventTapCallBack,
        userInfo: UnsafeMutableRawPointer?
    ) -> CFMachPort? {
        guard let previousTaps = currentEventTaps() else {
            VoxtLog.warning("Cannot inspect global hotkey taps before installation.")
            return nil
        }
        let previousTapIDs = Set(previousTaps.map(\.eventTapID))
        guard let tap = CGEvent.tapCreate(
            tap: location, place: .tailAppendEventTap, options: .defaultTap,
            eventsOfInterest: eventMask, callback: callback, userInfo: userInfo
        ) else { return nil }
        CGEvent.tapEnable(tap: tap, enable: false)
        // note (Codex): macOS can create a nonnil tap after stripping unauthorized keyboard events.
        guard let installedTaps = currentEventTaps(),
              newTapHasRequiredEvents(
                eventMask: eventMask,
                location: location,
                processIdentifier: ProcessInfo.processInfo.processIdentifier,
                previousTapIDs: previousTapIDs,
                installedTaps: installedTaps
              ) else {
            CFMachPortInvalidate(tap)
            VoxtLog.warning("Global hotkey tap could not be verified for all requested events. location=\(location.debugName), \(EventListeningPermissionManager.status().description). Check the app's Input Monitoring authorization in System Settings and relaunch it.")
            return nil
        }
        return tap
    }

    static func currentEventTaps() -> [CGEventTapInformation]? {
        var count: UInt32 = 0
        guard CGGetEventTapList(0, nil, &count) == .success else { return nil }
        guard count > 0 else { return [] }
        var taps = [CGEventTapInformation](repeating: CGEventTapInformation(), count: Int(count))
        guard CGGetEventTapList(count, &taps, &count) == .success else { return nil }
        return Array(taps.prefix(Int(count)))
    }

    static func newTapHasRequiredEvents(
        eventMask: CGEventMask,
        location: CGEventTapLocation,
        processIdentifier: pid_t,
        previousTapIDs: Set<UInt32>,
        installedTaps: [CGEventTapInformation]
    ) -> Bool {
        let candidates = installedTaps.filter {
            $0.tappingProcess == processIdentifier &&
            $0.tapPoint == location &&
            $0.options == .defaultTap &&
            !previousTapIDs.contains($0.eventTapID)
        }
        guard candidates.count == 1, let installedTap = candidates.first else { return false }
        return installedTap.eventsOfInterest & eventMask == eventMask
    }

    private init(tap: CFMachPort, source: CFRunLoopSource, runLoop: HotkeyEventTapRunLoop, context: CallbackContext, location: CGEventTapLocation) {
        self.tap = tap
        self.source = source
        self.runLoop = runLoop
        self.context = context
        self.location = location
    }

    func enable() {
        lock.lock()
        defer { lock.unlock() }
        if let tap { CGEvent.tapEnable(tap: tap, enable: true) }
    }

    func stop() {
        lock.lock()
        let oldTap = tap
        let oldSource = source
        tap = nil
        source = nil
        lock.unlock()
        guard let oldTap else { return }
        CGEvent.tapEnable(tap: oldTap, enable: false)
        CFMachPortInvalidate(oldTap)
        if let oldSource { runLoop.removeSource(oldSource, retaining: context) }
        runLoop.stop()
    }

    deinit { stop() }
}
