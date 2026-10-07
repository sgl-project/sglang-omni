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
        // A modifying session tap is authorized by Accessibility and does not
        // require Input Monitoring. Prefer it over the HID tap: on macOS the
        // HID tap can be created but silently miss Fn/keyboard events when
        // Input Monitoring is disabled.
        for location in [CGEventTapLocation.cgSessionEventTap, .cghidEventTap] {
            guard let tap = CGEvent.tapCreate(
                tap: location, place: .tailAppendEventTap, options: .defaultTap,
                eventsOfInterest: eventMask, callback: callback,
                userInfo: Unmanaged.passUnretained(context).toOpaque()
            ) else { continue }
            // Do not let callbacks enter the manager until it has registered the owner.
            CGEvent.tapEnable(tap: tap, enable: false)
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
