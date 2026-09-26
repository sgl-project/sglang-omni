// SPDX-License-Identifier: Apache-2.0
import CoreGraphics
import Carbon
import Combine
import Foundation

@MainActor
final class GlobalShortcut: ObservableObject {
    @Published private(set) var errorCode: String?
    private var conflictCode: String?
    private var tap: CFMachPort?
    private var source: CFRunLoopSource?
    private var monitorTask: Task<Void, Never>?
    private var keyCode: UInt16 = 49
    private var modifiers: UInt64 = 0
    private var hold = false
    private(set) var pressed = false
    private var capturedKey = false
    private var onStart: (() -> Void)?
    private var onStop: (() -> Void)?
    private var onCancel: (() -> Void)?
    private let modifierMask: UInt64 = CGEventFlags.maskCommand.rawValue | CGEventFlags.maskAlternate.rawValue
        | CGEventFlags.maskControl.rawValue | CGEventFlags.maskShift.rawValue | CGEventFlags.maskSecondaryFn.rawValue

    static func validate(keyCode: UInt16, modifiers: UInt64) throws {
        let flags = CGEventFlags(rawValue: modifiers)
        var carbon: UInt32 = 0
        if flags.contains(.maskCommand) { carbon |= UInt32(cmdKey) }
        if flags.contains(.maskAlternate) { carbon |= UInt32(optionKey) }
        if flags.contains(.maskControl) { carbon |= UInt32(controlKey) }
        if flags.contains(.maskShift) { carbon |= UInt32(shiftKey) }
        if ShortcutCapture.modifierFlag(for: keyCode) != nil { return }
        guard carbon != 0 || ShortcutCapture.functionKeys.contains(keyCode) else {
            throw Failure("shortcut.needsModifier")
        }
        var symbolic: Unmanaged<CFArray>?
        guard CopySymbolicHotKeys(&symbolic) == noErr,
              let shortcuts = symbolic?.takeRetainedValue() as? [[String: Any]] else {
            throw Failure("shortcut.checkFailed")
        }
        if shortcuts.contains(where: {
            ($0[kHISymbolicHotKeyEnabled] as? Bool) == true
                && ($0[kHISymbolicHotKeyCode] as? UInt16) == keyCode
                && ($0[kHISymbolicHotKeyModifiers] as? UInt32) == carbon
        }) { throw Failure("shortcut.systemConflict") }
        // ponytail: Carbon detects registered hotkeys; other event taps and app menu shortcuts remain invisible.
        var probe: EventHotKeyRef?
        let result = RegisterEventHotKey(UInt32(keyCode), carbon,
                                        EventHotKeyID(signature: 0x4F4D5459, id: 1), GetApplicationEventTarget(),
                                        OptionBits(kEventHotKeyExclusive), &probe)
        defer { if let probe { UnregisterEventHotKey(probe) } }
        guard result != eventHotKeyExistsErr else { throw Failure("shortcut.appConflict") }
        guard result == noErr else { throw Failure("shortcut.checkFailed") }
    }

    private var triggerModifier: UInt64 {
        switch keyCode {
        case 63: return CGEventFlags.maskSecondaryFn.rawValue
        case 54, 55: return CGEventFlags.maskCommand.rawValue
        case 56, 60: return CGEventFlags.maskShift.rawValue
        case 58, 61: return CGEventFlags.maskAlternate.rawValue
        case 59, 62: return CGEventFlags.maskControl.rawValue
        default: return 0
        }
    }

    private func matchingFlags(_ flags: CGEventFlags) -> UInt64 {
        // Note (Codex): Function keys carry secondaryFn even when Fn is not held.
        let fn = CGEventFlags.maskSecondaryFn.rawValue
        let mask = modifiers & fn != 0 || keyCode == 63 ? modifierMask : modifierMask & ~fn
        return flags.rawValue & mask
    }

    deinit {
        monitorTask?.cancel()
        if let source { CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes) }
        if let tap { CFMachPortInvalidate(tap) }
    }

    func start(keyCode: UInt16, modifiers: UInt64, hold: Bool,
               onStart: @escaping () -> Void, onStop: @escaping () -> Void,
               onCancel: @escaping () -> Void) {
        let modifiers = modifiers & modifierMask
        let unchanged = monitorTask != nil && self.keyCode == keyCode
            && self.modifiers == modifiers && self.hold == hold
        if !unchanged { stop() }
        self.keyCode = keyCode
        self.modifiers = modifiers
        self.hold = hold
        self.onStart = onStart
        self.onStop = onStop
        self.onCancel = onCancel
        guard !unchanged else { return }
        conflictCode = nil
        do { try Self.validate(keyCode: keyCode, modifiers: modifiers) }
        catch { conflictCode = (error as? Failure)?.code ?? "shortcut.checkFailed" }
        installTap()
        monitorTask = Task { [weak self] in
            var ticks = 0
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 100_000_000)
                guard !Task.isCancelled, let self else { break }
                ticks += 1
                if self.tap == nil && ticks % 10 == 0 { self.installTap() }
                self.pollForRelease()
            }
        }
    }

    func pollForRelease(keyState: (CGEventSourceStateID, CGKeyCode) -> Bool = CGEventSource.keyState,
                        flagsState: (CGEventSourceStateID) -> CGEventFlags = CGEventSource.flagsState) {
        guard pressed else { return }
        // Note (Codex): Consumed session events hide held keys; poll the physical state before our event tap.
        let flags = matchingFlags(flagsState(.hidSystemState))
        let down = triggerModifier == 0
            ? keyState(.hidSystemState, keyCode)
            : flags & triggerModifier != 0
        if !down || flags != modifiers | triggerModifier { release() }
    }

    func stop() {
        monitorTask?.cancel()
        monitorTask = nil
        release()
        if let tap { CGEvent.tapEnable(tap: tap, enable: false) }
        if let source { CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes) }
        if let tap { CFMachPortInvalidate(tap) }
        source = nil
        tap = nil
        onStart = nil
        onStop = nil
        onCancel = nil
        capturedKey = false
    }

    private func installTap() {
        guard tap == nil else { return }
        let mask = (CGEventMask(1) << CGEventType.keyDown.rawValue)
            | (CGEventMask(1) << CGEventType.keyUp.rawValue)
            | (CGEventMask(1) << CGEventType.flagsChanged.rawValue)
        guard let tap = CGEvent.tapCreate(tap: .cgSessionEventTap, place: .headInsertEventTap,
                                         options: .defaultTap, eventsOfInterest: mask,
                                         callback: { _, type, event, context in
            guard let context else { return Unmanaged.passUnretained(event) }
            let owner = Unmanaged<GlobalShortcut>.fromOpaque(context).takeUnretainedValue()
            let consumed = MainActor.assumeIsolated { owner.receive(type, event: event) }
            return consumed ? nil : Unmanaged.passUnretained(event)
        }, userInfo: Unmanaged.passUnretained(self).toOpaque()) else {
            if errorCode != "shortcut.listenFailed" { errorCode = "shortcut.listenFailed" }
            return
        }
        if errorCode != conflictCode { errorCode = conflictCode }
        self.tap = tap
        source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
    }

    func receive(_ type: CGEventType, event: CGEvent) -> Bool {
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            release()
            if let tap { CGEvent.tapEnable(tap: tap, enable: true) }
            return false
        }
        let code = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
        if type == .keyDown && code == 53 {
            // Note (Codex): Escape must not finalize a held recording.
            pressed = false
            deliver(onCancel)
            return false
        }
        if type == .keyUp && code == keyCode {
            let consumed = capturedKey
            capturedKey = false
            release()
            return consumed
        }
        let flags = matchingFlags(event.flags)
        if type == .flagsChanged && code == keyCode && triggerModifier != 0 {
            if flags == modifiers | triggerModifier, flags & triggerModifier != 0 {
                if !pressed { pressed = true; capturedKey = true; deliver(onStart) }
                return true
            }
            let consumed = capturedKey
            capturedKey = false
            release()
            return consumed
        }
        if type == .flagsChanged && pressed && flags != modifiers | triggerModifier {
            release()
        } else if type == .keyDown && code == keyCode && flags == modifiers {
            if event.getIntegerValueField(.keyboardEventAutorepeat) == 0 && !pressed {
                pressed = true
                capturedKey = true
                deliver(onStart)
            }
            return true
        }
        return false
    }

    private func release() {
        guard pressed else { return }
        pressed = false
        if hold { deliver(onStop) }
    }

    private func deliver(_ callback: (() -> Void)?) {
        guard let callback else { return }
        // Note (Codex): Defer UI and permission work to avoid blocking the event tap.
        DispatchQueue.main.async { callback() }
    }
}
