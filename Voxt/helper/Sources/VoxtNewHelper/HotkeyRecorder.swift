import ApplicationServices
import VoxtNewHelperKit
import Foundation

/// A one-shot session event tap for hotkey.record.
final class HotkeyRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<String, Error>?
    private var tap: CFMachPort?
    private var runLoop: CFRunLoop?
    private var tapThread: Thread?
    private var timeoutTimer: DispatchWorkItem?
    private var activeMods = Set<ModKey>()
    private var peakMods = Set<ModKey>()
    private var recording = false

    func record(timeoutSeconds: Double) async throws -> HotkeyRecordResult {
        let keys = try await recordKeys(timeoutSeconds: timeoutSeconds)
        return HotkeyRecordResult(keys: keys)
    }

    private func recordKeys(timeoutSeconds: Double) async throws -> String {
        let timeout = timeoutSeconds > 0 ? timeoutSeconds : 15
        return try await withCheckedThrowingContinuation { cont in
            lock.withLock {
                self.cancelLocked()
                guard AXIsProcessTrusted() else {
                    cont.resume(throwing: RPCError.denied(PermissionName.accessibility, "Accessibility is required to record shortcuts"))
                    return
                }
                self.continuation = cont
                self.activeMods = []
                self.peakMods = []
                self.recording = true
                guard self.startTap() else {
                    self.recording = false
                    self.continuation = nil
                    cont.resume(throwing: RPCError.denied(PermissionName.accessibility, "Accessibility is required to record shortcuts"))
                    return
                }
                let work = DispatchWorkItem { [weak self] in
                    self?.lock.withLock { self?.finish(RPCError(Code.timeout, "Recording timed out")) }
                }
                self.timeoutTimer = work
                DispatchQueue.global().asyncAfter(deadline: .now() + timeout, execute: work)
            }
        }
    }

    func cancel() {
        lock.withLock { finish(RPCError(Code.cancelled, "Cancelled")) }
    }

    private func cancelLocked() {
        finish(RPCError(Code.cancelled, "Cancelled"))
    }

    private func finish(_ err: RPCError) {
        timeoutTimer?.cancel()
        timeoutTimer = nil
        recording = false
        stopTap()
        activeMods = []
        peakMods = []
        guard let cont = continuation else { return }
        continuation = nil
        cont.resume(throwing: err)
    }

    private func succeed(_ keys: String) {
        timeoutTimer?.cancel()
        timeoutTimer = nil
        recording = false
        stopTap()
        activeMods = []
        peakMods = []
        guard let cont = continuation else { return }
        continuation = nil
        cont.resume(returning: keys)
    }

    private func startTap() -> Bool {
        let mask: CGEventMask = [CGEventType.keyDown, .flagsChanged, .otherMouseDown]
            .reduce(0) { $0 | (1 << $1.rawValue) }
        let ready = DispatchSemaphore(value: 0)
        let refcon = Unmanaged.passUnretained(self).toOpaque()
        let callback: CGEventTapCallBack = { _, type, event, user in
            guard let user else { return Unmanaged.passUnretained(event) }
            let me = Unmanaged<HotkeyRecorder>.fromOpaque(user).takeUnretainedValue()
            return me.handle(type, event)
        }
        let thread = Thread { [weak self] in
            guard let self else { ready.signal(); return }
            guard let port = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap, options: .defaultTap,
                                               eventsOfInterest: mask, callback: callback, userInfo: refcon) else {
                ready.signal()
                return
            }
            let loop = CFRunLoopGetCurrent()
            let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, port, 0)
            CFRunLoopAddSource(loop, source, .commonModes)
            CGEvent.tapEnable(tap: port, enable: true)
            self.lock.withLock {
                self.tap = port
                self.runLoop = loop
            }
            ready.signal()
            CFRunLoopRun()
            self.lock.withLock {
                self.tap = nil
                self.runLoop = nil
            }
        }
        thread.name = "voxtnew.hotkey.record"
        tapThread = thread
        thread.start()
        return ready.wait(timeout: .now() + 1) == .success && lock.withLock({ tap != nil })
    }

    private func stopTap() {
        let (port, loop) = lock.withLock { (tap, runLoop) }
        if let port {
            CGEvent.tapEnable(tap: port, enable: false)
            CFMachPortInvalidate(port)
        }
        if let loop { CFRunLoopStop(loop) }
        tap = nil
        runLoop = nil
        tapThread = nil
    }

    private func handle(_ type: CGEventType, _ event: CGEvent) -> Unmanaged<CGEvent>? {
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            if let port = lock.withLock({ tap }) { CGEvent.tapEnable(tap: port, enable: true) }
            return Unmanaged.passUnretained(event)
        }
        guard lock.withLock({ recording }) else { return Unmanaged.passUnretained(event) }
        let flags = event.flags.rawValue
        let keyCode = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
        switch type {
        case .flagsChanged:
            lock.withLock {
                let prev = activeMods
                activeMods = WireHotkeyRecord.mods(from: flags)
                peakMods.formUnion(activeMods)
                if !prev.isEmpty && activeMods.isEmpty, let s = WireHotkeyRecord.chordString(mods: peakMods) {
                    succeed(s)
                }
            }
            return nil
        case .keyDown:
            if ModKey.forKeyCode(keyCode) != nil { return nil }
            let mods = WireHotkeyRecord.mods(from: flags)
            if let s = WireHotkeyRecord.chordString(mods: mods, keyCode: keyCode) {
                succeed(s)
            }
            return nil
        case .otherMouseDown:
            let button = Int(event.getIntegerValueField(.mouseEventButtonNumber))
            let mods = WireHotkeyRecord.mods(from: flags)
            if button >= 2, let s = WireHotkeyRecord.chordString(mods: mods, mouseButton: button) {
                succeed(s)
            }
            return nil
        default:
            return Unmanaged.passUnretained(event)
        }
    }
}
