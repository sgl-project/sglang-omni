import ApplicationServices
import VoxtNewHelperKit
import Foundation

/// hotkey.record: KeyRecordSession (VoxtNewHelperKit) holds the state and
/// its locking; each recording gets a short-lived session event tap of its
/// own (RecordTap).
final class HotkeyRecorder: Sendable {
    private let session = KeyRecordSession(trusted: { AXIsProcessTrusted() }, makeTap: { RecordTap.start($0) })

    func record(timeoutSeconds: Double) async throws -> HotkeyRecordResult {
        let keys: String = try await withCheckedThrowingContinuation { cont in
            session.begin(timeoutSeconds: timeoutSeconds) { cont.resume(with: $0) }
        }
        return HotkeyRecordResult(keys: keys)
    }

    func cancel() {
        session.cancel()
    }
}

/// One recording's event tap, on a thread of its own: key downs, modifier
/// changes and other mouse buttons go to the handler, which swallows them
/// while it records.
private final class RecordTap: KeyRecordTap, @unchecked Sendable {
    private let handler: @Sendable (KeyRecordInput) -> Bool
    private let lock = NSLock()
    // Under lock.
    private var port: CFMachPort?
    private var loop: CFRunLoop?
    private var stopped = false

    private init(handler: @escaping @Sendable (KeyRecordInput) -> Bool) {
        self.handler = handler
    }

    /// A running tap, or nil when the tap cannot be created (no
    /// Accessibility).
    static func start(_ handler: @escaping @Sendable (KeyRecordInput) -> Bool) -> RecordTap? {
        let t = RecordTap(handler: handler)
        return t.run() ? t : nil
    }

    private func run() -> Bool {
        let mask: CGEventMask = [CGEventType.keyDown, .flagsChanged, .otherMouseDown]
            .reduce(0) { $0 | (1 << $1.rawValue) }
        let ready = DispatchSemaphore(value: 0)
        let refcon = Unmanaged.passUnretained(self).toOpaque()
        let callback: CGEventTapCallBack = { _, type, event, user in
            guard let user else { return Unmanaged.passUnretained(event) }
            return Unmanaged<RecordTap>.fromOpaque(user).takeUnretainedValue().handle(type, event)
        }
        // The thread keeps the tap (refcon) alive until its run loop stops.
        let thread = Thread { [self] in
            guard let p = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap, options: .defaultTap,
                                            eventsOfInterest: mask, callback: callback, userInfo: refcon) else {
                ready.signal()
                return
            }
            let l = CFRunLoopGetCurrent()
            CFRunLoopAddSource(l, CFMachPortCreateRunLoopSource(kCFAllocatorDefault, p, 0), .commonModes)
            let kept = lock.withLock {
                guard !stopped else { return false } // gave up waiting for it
                port = p
                loop = l
                return true
            }
            guard kept else {
                CFMachPortInvalidate(p)
                ready.signal()
                return
            }
            CGEvent.tapEnable(tap: p, enable: true)
            ready.signal()
            CFRunLoopRun()
        }
        thread.name = "voxtnew.hotkey.record"
        thread.start()
        guard ready.wait(timeout: .now() + 1) == .success, lock.withLock({ port != nil }) else {
            stop()
            return false
        }
        return true
    }

    func stop() {
        let (p, l) = lock.withLock {
            stopped = true
            defer { port = nil; loop = nil }
            return (port, loop)
        }
        if let p {
            CGEvent.tapEnable(tap: p, enable: false)
            CFMachPortInvalidate(p)
        }
        if let l { CFRunLoopStop(l) }
    }

    private func handle(_ type: CGEventType, _ event: CGEvent) -> Unmanaged<CGEvent>? {
        let pass = Unmanaged.passUnretained(event)
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            if let p = lock.withLock({ port }) { CGEvent.tapEnable(tap: p, enable: true) }
            return pass
        }
        let flags = event.flags.rawValue
        let input: KeyRecordInput
        switch type {
        case .flagsChanged:
            input = .flagsChanged(flags: flags)
        case .keyDown:
            input = .keyDown(keyCode: UInt16(event.getIntegerValueField(.keyboardEventKeycode)), flags: flags)
        case .otherMouseDown:
            input = .mouseDown(button: Int(event.getIntegerValueField(.mouseEventButtonNumber)), flags: flags)
        default:
            return pass
        }
        return handler(input) ? nil : pass
    }
}
