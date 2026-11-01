import ApplicationServices
import VoxtNewHelperKit
import CoreGraphics
import Foundation

// The global hotkey event tap. A modifying (.defaultTap) session tap: macOS
// authorizes it with Accessibility alone (a listen-only or HID tap would
// need Input Monitoring), checked against VoxtNew.app, the helper's
// responsible process. It runs on its own thread with its own run loop;
// the callback decides synchronously, with the recognizer, whether to
// swallow the event, and queues hotkey.event notifications, each with the
// app in front (the session's context snapshot).
//
// Without Accessibility the tap cannot be created: it retries every 2 s
// and sends hotkey.state once in (no restart needed after the user grants
// it). Accessibility taken away removes the tap, and it retries again.
//
// Events VoxtNew posts itself (⌘V, ⌘C, Return) carry InjectedEvent.marker
// ("VOXT") in eventSourceUserData and pass untouched; other apps' posted
// events (Auralink's "AURA") are judged as typing.

final class HotkeyTap: @unchecked Sendable {
    let out: Writer
    let front: FrontApps
    private let lock = NSLock()
    private var recognizer = HotkeyRecognizer(config: HotkeyConfig(bindings: []))
    private var tap: CFMachPort?
    private var runLoop: CFRunLoop?
    private var timer: CFRunLoopTimer?
    private var retry: DispatchSourceTimer?
    private var events = 0
    private var disabledByTimeout = 0
    private var maxCallbackMicros: Double = 0
    private var lastError: String?
    /// hotkey.event goes out in order, off the tap's thread.
    private let sendQueue = DispatchQueue(label: "voxtnew.hotkey.send", qos: .userInteractive)

    init(out: Writer, front: FrontApps) {
        self.out = out
        self.front = front
    }

    func configure(_ p: HotkeyConfigureParams) -> HotkeyStatus {
        let c = HotkeyConfig(bindings: p.bindings, doubleTapMs: p.doubleTapMs ?? systemDoubleClickMs(),
                             longPressMs: p.longPressMs ?? 300, tapMaxMs: p.tapMaxMs ?? 1500,
                             swallowFn: p.swallowFn ?? false, active: p.active ?? false)
        lock.withLock { recognizer.configure(c) }
        if p.bindings.isEmpty {
            uninstall()
        } else if lock.withLock({ tap == nil && retry == nil }) {
            install()
        }
        return status()
    }

    func status() -> HotkeyStatus {
        lock.withLock {
            HotkeyStatus(installed: tap != nil, accessibility: AXIsProcessTrusted(), bindings: recognizer.config.bindings.count,
                         events: events, disabledByTimeout: disabledByTimeout, missedEvents: recognizer.missedEvents,
                         maxCallbackMicros: (maxCallbackMicros * 10).rounded() / 10, error: lastError)
        }
    }

    /// Creates the tap on its thread; without Accessibility it retries.
    private func install() {
        let ready = DispatchSemaphore(value: 0)
        let thread = Thread { [self] in
            // Mouse presses too (not moves or scrolls: too many): they spoil
            // a modifier chord held for an ⌥-click or ⌥-drag, and always pass.
            let mask: CGEventMask = (1 << CGEventType.keyDown.rawValue) | (1 << CGEventType.keyUp.rawValue) |
                (1 << CGEventType.flagsChanged.rawValue) | (1 << CGEventType.leftMouseDown.rawValue) |
                (1 << CGEventType.rightMouseDown.rawValue) | (1 << CGEventType.otherMouseDown.rawValue)
            let callback: CGEventTapCallBack = { _, type, event, refcon in
                guard let refcon else { return Unmanaged.passUnretained(event) }
                let me = Unmanaged<HotkeyTap>.fromOpaque(refcon).takeUnretainedValue()
                return me.handle(type, event) ? nil : Unmanaged.passUnretained(event)
            }
            guard let port = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap, options: .defaultTap,
                                               eventsOfInterest: mask, callback: callback,
                                               userInfo: Unmanaged.passUnretained(self).toOpaque()) else {
                let trusted = AXIsProcessTrusted()
                lock.withLock {
                    lastError = trusted ? "CGEvent.tapCreate failed; retrying every 2 s"
                        : "CGEvent.tapCreate failed: VoxtNew may not use Accessibility; retrying every 2 s"
                }
                ready.signal()
                scheduleRetry()
                return
            }
            let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, port, 0)
            let loop = CFRunLoopGetCurrent()!
            CFRunLoopAddSource(loop, source, .commonModes)
            CGEvent.tapEnable(tap: port, enable: true)
            let wasRetrying = lock.withLock { () -> Bool in
                tap = port
                runLoop = loop
                lastError = nil
                defer { retry = nil }
                return retry != nil
            }
            ready.signal()
            if wasRetrying { out.notify(Method.hotkeyState, status()) }
            watchTrust()
            CFRunLoopRun()
        }
        thread.name = "voxtnew.hotkey"
        thread.qualityOfService = .userInteractive
        thread.start()
        _ = ready.wait(timeout: .now() + 1)
    }

    private func scheduleRetry() {
        let t = DispatchSource.makeTimerSource(queue: .global())
        t.schedule(deadline: .now() + 2)
        t.setEventHandler { [weak self] in
            guard let self else { return }
            let need = self.lock.withLock { self.tap == nil && !self.recognizer.config.bindings.isEmpty }
            if need {
                self.install()
            } else {
                self.lock.withLock { self.retry = nil }
            }
        }
        lock.withLock { retry = t }
        t.resume()
    }

    /// Every 2 s while installed: Accessibility taken away leaves a tap that
    /// sees nothing; remove it, say so, and wait for it to come back.
    private func watchTrust() {
        let t = DispatchSource.makeTimerSource(queue: .global())
        t.schedule(deadline: .now() + 2, repeating: 2)
        t.setEventHandler { [weak self, weak t] in
            guard let self else { t?.cancel(); return }
            guard self.lock.withLock({ self.tap != nil }) else { t?.cancel(); return }
            if !AXIsProcessTrusted() {
                t?.cancel()
                self.uninstall()
                self.lock.withLock { self.lastError = "Accessibility was turned off; retrying every 2 s" }
                self.out.notify(Method.hotkeyState, self.status())
                self.scheduleRetry()
            }
        }
        t.resume()
    }

    private func uninstall() {
        let (port, loop, r): (CFMachPort?, CFRunLoop?, DispatchSourceTimer?) = lock.withLock {
            defer {
                tap = nil
                runLoop = nil
                retry = nil
                recognizer.reset()
            }
            return (tap, runLoop, retry)
        }
        r?.cancel()
        if let port {
            CGEvent.tapEnable(tap: port, enable: false)
            CFMachPortInvalidate(port)
        }
        if let loop { CFRunLoopStop(loop) }
    }

    /// The tap callback: true swallows the event.
    private func handle(_ type: CGEventType, _ event: CGEvent) -> Bool {
        let start = monotonicMs()
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            lock.withLock {
                disabledByTimeout += 1
                recognizer.reset()
                if let tap { CGEvent.tapEnable(tap: tap, enable: true) }
            }
            log("event tap was disabled (\(type.rawValue)); enabled again")
            return false
        }
        let injected = InjectedEvent.isOurs(event.getIntegerValueField(.eventSourceUserData))
        let keyCode = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
        let flags = event.flags.rawValue
        let input: KeyInput
        switch type {
        case .flagsChanged: input = .flagsChanged(keyCode: keyCode, flags: flags, t: start)
        case .keyDown: input = .keyDown(keyCode: keyCode, flags: flags, isRepeat: event.getIntegerValueField(.keyboardEventAutorepeat) != 0, t: start)
        case .keyUp: input = .keyUp(keyCode: keyCode, flags: flags, t: start)
        case .leftMouseDown, .rightMouseDown, .otherMouseDown: input = .pointer(t: start)
        default: return false
        }
        let (fired, swallow, deadline): ([HotkeyEvent], Bool, Double?) = lock.withLock {
            events += 1
            if injected { return ([], false, nil) }
            let r = recognizer.handle(input)
            return (r.events, r.swallow, recognizer.nextDeadline)
        }
        send(fired)
        armTimer(deadline)
        let micros = (monotonicMs() - start) * 1000
        lock.withLock { maxCallbackMicros = max(maxCallbackMicros, micros) }
        return swallow
    }

    /// Sends what fired, with the app in front (as cached at this moment).
    private func send(_ fired: [HotkeyEvent]) {
        guard !fired.isEmpty else { return }
        let app = front.cached
        sendQueue.async { [out] in
            for e in fired { out.notify(Method.hotkeyEvent, HotkeyEventParams(e, front: app)) }
        }
    }

    /// Re-arms the one-shot timer for deferred taps and long presses (on
    /// the tap's run loop, so the recognizer only runs on this thread or
    /// under the lock).
    private func armTimer(_ deadline: Double?) {
        guard let loop = lock.withLock({ runLoop }) else { return }
        if let old = lock.withLock({ timer }) { CFRunLoopTimerInvalidate(old) }
        guard let deadline else { return }
        let delay = max(0, deadline - monotonicMs()) / 1000
        let t = CFRunLoopTimerCreateWithHandler(kCFAllocatorDefault, CFAbsoluteTimeGetCurrent() + delay, 0, 0, 0) { [weak self] _ in
            guard let self else { return }
            let (fired, next) = self.lock.withLock { () -> ([HotkeyEvent], Double?) in
                let f = self.recognizer.tick(now: monotonicMs())
                return (f, self.recognizer.nextDeadline)
            }
            self.send(fired)
            self.armTimer(next)
        }
        lock.withLock { timer = t }
        CFRunLoopAddTimer(loop, t, .commonModes)
    }
}

/// The system's double-click interval in ms, at most 800 (else 500): the
/// default double-tap window. (NSEvent.doubleClickInterval without
/// AppKit's main thread.)
func systemDoubleClickMs() -> Double {
    let v = UserDefaults.standard.double(forKey: "com.apple.mouse.doubleClickThreshold")
    return v > 0 ? min(v, 0.8) * 1000 : 500
}
