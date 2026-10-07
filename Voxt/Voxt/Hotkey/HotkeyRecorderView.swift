// HotkeyRecorderView.swift
// Provides Hotkey Recorder View for hotkey handling.

import SwiftUI
import AppKit
import Carbon
import ApplicationServices

struct HotkeyRecorderView: NSViewRepresentable {
    @Binding var isRecording: Bool
    let onCapture: (HotkeyPreference.Hotkey) -> Void
    let onCancelCapture: () -> Void
    let onRecorderMessageChange: (String?) -> Void

    func makeNSView(context: Context) -> KeyCaptureView {
        let view = KeyCaptureView()
        view.onHotkeyCaptured = { hotkey in
            self.onCapture(hotkey)
        }
        view.onCancel = {
            self.isRecording = false
            self.onCancelCapture()
        }
        view.onRecorderMessageChange = onRecorderMessageChange
        return view
    }

    func updateNSView(_ nsView: KeyCaptureView, context: Context) {
        nsView.isRecording = isRecording
        nsView.onRecorderMessageChange = onRecorderMessageChange
        HotkeyCaptureState.shared.setCaptureInProgress(isRecording)
        if isRecording {
            DispatchQueue.main.async {
                nsView.window?.makeFirstResponder(nsView)
            }
        } else {
            onRecorderMessageChange(nil)
        }
    }
}

final class KeyCaptureView: NSView {
    private struct PendingModifierCapture {
        let modifiers: NSEvent.ModifierFlags
        let sidedModifiers: SidedModifierFlags
        let count: Int
    }

    var onHotkeyCaptured: ((HotkeyPreference.Hotkey) -> Void)?
    var onCancel: (() -> Void)?
    var onRecorderMessageChange: ((String?) -> Void)?
    var isRecording: Bool = false {
        didSet {
            guard isRecording != oldValue else { return }
            if isRecording {
                startLocalEventMonitor()
            } else {
                stopLocalEventMonitor()
            }
        }
    }
    private var currentSidedModifiers: SidedModifierFlags = []
    private var currentModifierFlags: NSEvent.ModifierFlags = []
    private var pendingModifierCaptureTask: Task<Void, Never>?
    private var pendingModifierCapture: PendingModifierCapture?
    private var localEventMonitor: Any?
    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var hasCapturedChordDuringCurrentRecording = false
    private var lastModifierOnlyCaptureCount = 0

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
    }

    deinit {
        stopLocalEventMonitor()
        HotkeyCaptureState.shared.setCaptureInProgress(false)
    }

    override var acceptsFirstResponder: Bool { true }

    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func keyDown(with event: NSEvent) {
        guard isRecording else { return }
        if event.keyCode == UInt16(kVK_Escape) {
            pendingModifierCaptureTask?.cancel()
            onCancel?()
            return
        }
        VoxtLog.hotkey("Hotkey recorder keyDown. keyCode=\(event.keyCode), modifiers=\(HotkeyPreference.modifierSymbols(for: combinedModifiers(with: event.modifierFlags.intersection(.hotkeyRelevant))))", verbose: true)
        captureKeyDown(
            keyCode: event.keyCode,
            modifiers: event.modifierFlags.intersection(.hotkeyRelevant),
            sidedModifiers: currentSidedModifiers
        )
    }

    override func performKeyEquivalent(with event: NSEvent) -> Bool {
        guard isRecording else {
            return super.performKeyEquivalent(with: event)
        }
        keyDown(with: event)
        return true
    }

    override func flagsChanged(with event: NSEvent) {
        guard isRecording else { return }

        let modifiers = updateModifierFlags(
            keyCode: event.keyCode,
            rawModifiers: event.modifierFlags.intersection(.hotkeyRelevant)
        )
        updateSidedModifiers(for: event.keyCode)
        VoxtLog.hotkey("Hotkey recorder flagsChanged(local). keyCode=\(event.keyCode), modifiers=\(HotkeyPreference.modifierSymbols(for: modifiers))", verbose: true)
        scheduleModifierOnlyCaptureIfNeeded(keyCode: event.keyCode, modifiers: modifiers)
    }

    private func startLocalEventMonitor() {
        stopLocalEventMonitor()
        hasCapturedChordDuringCurrentRecording = false
        lastModifierOnlyCaptureCount = 0
        currentSidedModifiers = []
        currentModifierFlags = []
        pendingModifierCapture = nil
        startEventTap()
        localEventMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .flagsChanged, .otherMouseDown]) { [weak self] event in
            guard let self, self.isRecording else { return event }
            if self.eventTap != nil {
                return nil
            }
            switch event.type {
            case .keyDown:
                self.keyDown(with: event)
                return nil
            case .flagsChanged:
                self.flagsChanged(with: event)
                return nil
            case .otherMouseDown:
                self.captureMouseButtonDown(
                    buttonNumber: event.buttonNumber,
                    modifiers: event.modifierFlags,
                    sidedModifiers: self.currentSidedModifiers
                )
                return nil
            default:
                return event
            }
        }
    }

    private func stopLocalEventMonitor() {
        pendingModifierCaptureTask?.cancel()
        pendingModifierCaptureTask = nil
        hasCapturedChordDuringCurrentRecording = false
        lastModifierOnlyCaptureCount = 0
        currentSidedModifiers = []
        currentModifierFlags = []
        pendingModifierCapture = nil
        stopEventTap()
        onRecorderMessageChange?(nil)
        if let localEventMonitor {
            NSEvent.removeMonitor(localEventMonitor)
            self.localEventMonitor = nil
        }
    }

    private func startEventTap() {
        stopEventTap()
        // Local events remain available without global authorization.
        guard AccessibilityPermissionManager.isTrusted() else { return }

        let eventMask: CGEventMask =
            (1 << CGEventType.keyDown.rawValue) |
            (1 << CGEventType.flagsChanged.rawValue) |
            (1 << CGEventType.otherMouseDown.rawValue)

        guard let (tap, tapLocation) = createEventTap(eventMask: eventMask) else {
            onRecorderMessageChange?("System-reserved shortcuts such as fn+space may still be intercepted by macOS. If capture fails, disable or remap the Globe / input source shortcut in System Settings.")
            return
        }

        eventTap = tap
        let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        runLoopSource = source
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
        VoxtLog.hotkey("Hotkey recorder event tap started. location=\(tapLocation.debugName)")
        onRecorderMessageChange?(nil)
    }

    private func stopEventTap() {
        if let tap = eventTap {
            CGEvent.tapEnable(tap: tap, enable: false)
            CFMachPortInvalidate(tap)
        }
        if let source = runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes)
        }
        eventTap = nil
        runLoopSource = nil
    }

    private func createEventTap(eventMask: CGEventMask) -> (tap: CFMachPort, location: CGEventTapLocation)? {
        let callback: CGEventTapCallBack = { _, type, event, refcon in
            guard let refcon else { return Unmanaged.passUnretained(event) }
            let view = Unmanaged<KeyCaptureView>.fromOpaque(refcon).takeUnretainedValue()
            view.handleTapEvent(type: type, event: event)
            return Unmanaged.passUnretained(event)
        }

        // Prefer the Accessibility-backed session tap. The HID tap may be
        // creatable while still missing Fn events without Input Monitoring.
        for tapLocation in [CGEventTapLocation.cgSessionEventTap, .cghidEventTap] {
            if let tap = CGEvent.tapCreate(
                tap: tapLocation,
                place: .tailAppendEventTap,
                options: .defaultTap,
                eventsOfInterest: eventMask,
                callback: callback,
                userInfo: Unmanaged.passUnretained(self).toOpaque()
            ) {
                return (tap, tapLocation)
            }
        }
        return nil
    }

    private func handleTapEvent(type: CGEventType, event: CGEvent) {
        guard isRecording else { return }
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            if let eventTap { CGEvent.tapEnable(tap: eventTap, enable: true) }
            return
        }
        guard NSApp.isActive else { return }
        let keyCode = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
        if type == .keyDown, keyCode == UInt16(kVK_Escape) {
            pendingModifierCaptureTask?.cancel()
            onCancel?()
            return
        }
        switch type {
        case .keyDown:
            VoxtLog.hotkey("Hotkey recorder keyDown(tap). keyCode=\(keyCode), modifiers=\(HotkeyPreference.modifierSymbols(for: combinedModifiers(with: modifierFlags(from: event.flags).intersection(.hotkeyRelevant))))", verbose: true)
            captureKeyDown(
                keyCode: keyCode,
                modifiers: combinedModifiers(with: modifierFlags(from: event.flags).intersection(.hotkeyRelevant)),
                sidedModifiers: currentSidedModifiers
            )
        case .otherMouseDown:
            let buttonNumber = Int(event.getIntegerValueField(.mouseEventButtonNumber))
            captureMouseButtonDown(
                buttonNumber: buttonNumber,
                modifiers: combinedModifiers(with: modifierFlags(from: event.flags).intersection(.hotkeyRelevant)),
                sidedModifiers: currentSidedModifiers
            )
        case .flagsChanged:
            let modifiers = updateModifierFlags(
                keyCode: keyCode,
                rawModifiers: modifierFlags(from: event.flags).intersection(.hotkeyRelevant)
            )
            currentSidedModifiers = SidedModifierFlags.from(eventFlags: event.flags).filtered(by: modifiers)
            VoxtLog.hotkey("Hotkey recorder flagsChanged(tap). keyCode=\(keyCode), modifiers=\(HotkeyPreference.modifierSymbols(for: modifiers))", verbose: true)
            scheduleModifierOnlyCaptureIfNeeded(keyCode: keyCode, modifiers: modifiers)
        default:
            break
        }
    }

    private func modifierFlags(from cgFlags: CGEventFlags) -> NSEvent.ModifierFlags {
        var flags: NSEvent.ModifierFlags = []
        if cgFlags.contains(.maskCommand) { flags.insert(.command) }
        if cgFlags.contains(.maskAlternate) { flags.insert(.option) }
        if cgFlags.contains(.maskControl) { flags.insert(.control) }
        if cgFlags.contains(.maskShift) { flags.insert(.shift) }
        if cgFlags.contains(.maskSecondaryFn) { flags.insert(.function) }
        return flags
    }

    private func updateSidedModifiers(for keyCode: UInt16) {
        guard SidedModifierFlags.sidedFlag(for: keyCode) != nil else { return }
        currentSidedModifiers = SidedModifierFlags.snapshotFromCurrentKeyState(
            filteredBy: currentModifierFlags.intersection(.hotkeyRelevant)
        )
    }

    private func captureKeyDown(
        keyCode: UInt16,
        modifiers: NSEvent.ModifierFlags,
        sidedModifiers: SidedModifierFlags
    ) {
        guard HotkeyPreference.isRecordableKeyboardKeyCode(keyCode) else {
            VoxtLog.hotkey("Hotkey recorder ignored unsupported key. keyCode=\(keyCode)")
            return
        }
        let capturedModifiers = combinedModifiers(with: modifiers.intersection(.hotkeyRelevant))
        pendingModifierCaptureTask?.cancel()
        pendingModifierCapture = nil
        hasCapturedChordDuringCurrentRecording = true
        lastModifierOnlyCaptureCount = 0
        Task { @MainActor [weak self] in
            guard let self else { return }
            VoxtLog.hotkey("Hotkey recorder captured chord. keyCode=\(keyCode), modifiers=\(HotkeyPreference.modifierSymbols(for: capturedModifiers))")
            self.onHotkeyCaptured?(
                HotkeyPreference.Hotkey(
                    keyCode: keyCode,
                    modifiers: capturedModifiers,
                    sidedModifiers: sidedModifiers.filtered(by: capturedModifiers)
                )
            )
        }
    }

    private func captureMouseButtonDown(
        buttonNumber: Int,
        modifiers: NSEvent.ModifierFlags,
        sidedModifiers: SidedModifierFlags
    ) {
        guard buttonNumber >= HotkeyPreference.middleMouseButtonNumber else { return }
        let capturedModifiers = combinedModifiers(with: modifiers.intersection(.hotkeyRelevant))
        pendingModifierCaptureTask?.cancel()
        pendingModifierCapture = nil
        hasCapturedChordDuringCurrentRecording = true
        lastModifierOnlyCaptureCount = 0
        Task { @MainActor [weak self] in
            guard let self else { return }
            VoxtLog.hotkey("Hotkey recorder captured mouse button. button=\(buttonNumber), modifiers=\(HotkeyPreference.modifierSymbols(for: capturedModifiers))")
            self.onHotkeyCaptured?(
                HotkeyPreference.Hotkey(
                    mouseButtonNumber: buttonNumber,
                    modifiers: capturedModifiers,
                    sidedModifiers: sidedModifiers.filtered(by: capturedModifiers)
                )
            )
        }
    }

    private func updateModifierFlags(
        keyCode: UInt16,
        rawModifiers: NSEvent.ModifierFlags
    ) -> NSEvent.ModifierFlags {
        var modifiers = rawModifiers.intersection(.hotkeyRelevant)

        if currentModifierFlags.contains(.function) {
            modifiers.insert(.function)
        }

        if HotkeyModifierInterpreter.isFunctionKeyEvent(keyCode) {
            let functionIsDown = rawModifiers.contains(.function) || !currentModifierFlags.contains(.function)
            if functionIsDown {
                modifiers.insert(.function)
            } else {
                modifiers.remove(.function)
            }
        }

        currentModifierFlags = modifiers
        return modifiers
    }

    private func combinedModifiers(with rawModifiers: NSEvent.ModifierFlags) -> NSEvent.ModifierFlags {
        var modifiers = rawModifiers.intersection(.hotkeyRelevant)
        if currentModifierFlags.contains(.function) {
            modifiers.insert(.function)
        }
        return modifiers
    }

    private func scheduleModifierOnlyCaptureIfNeeded(
        keyCode: UInt16,
        modifiers: NSEvent.ModifierFlags
    ) {
        guard !modifiers.isEmpty else {
            finalizePendingModifierCaptureIfNeeded(reason: "modifier release")
            return
        }

        let modifierOnlyKeyCodes: Set<UInt16> = [
            UInt16(kVK_Shift), UInt16(kVK_RightShift),
            UInt16(kVK_Control), UInt16(kVK_RightControl),
            UInt16(kVK_Option), UInt16(kVK_RightOption),
            UInt16(kVK_Command), UInt16(kVK_RightCommand),
            UInt16(kVK_Function), UInt16(kVK_CapsLock)
        ]

        guard modifierOnlyKeyCodes.contains(keyCode) else { return }
        let capturedModifiers = modifiers.intersection(.hotkeyRelevant)
        let capturedSidedModifiers = currentSidedModifiers.filtered(by: capturedModifiers)
        let capturedModifierCount = modifierCount(for: capturedModifiers)
        if let pendingModifierCapture, capturedModifierCount < pendingModifierCapture.count {
            return
        }
        pendingModifierCapture = PendingModifierCapture(
            modifiers: capturedModifiers,
            sidedModifiers: capturedSidedModifiers,
            count: capturedModifierCount
        )
        pendingModifierCaptureTask?.cancel()
        pendingModifierCaptureTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(160))
            guard let self, !Task.isCancelled, self.isRecording else { return }
            guard !self.hasCapturedChordDuringCurrentRecording else {
                self.pendingModifierCapture = nil
                return
            }
            guard capturedModifierCount >= self.lastModifierOnlyCaptureCount else {
                self.pendingModifierCapture = nil
                return
            }
            self.lastModifierOnlyCaptureCount = capturedModifierCount
            self.pendingModifierCapture = nil
            VoxtLog.hotkey("Hotkey recorder captured modifier-only after hold. modifiers=\(HotkeyPreference.modifierSymbols(for: capturedModifiers))")
            self.onHotkeyCaptured?(
                HotkeyPreference.Hotkey(
                    keyCode: HotkeyPreference.modifierOnlyKeyCode,
                    modifiers: capturedModifiers,
                    sidedModifiers: capturedSidedModifiers
                )
            )
        }
    }

    private func finalizePendingModifierCaptureIfNeeded(reason: String) {
        pendingModifierCaptureTask?.cancel()
        pendingModifierCaptureTask = nil
        defer { pendingModifierCapture = nil }
        guard let pendingModifierCapture else { return }
        guard !hasCapturedChordDuringCurrentRecording else { return }
        guard pendingModifierCapture.count >= lastModifierOnlyCaptureCount else { return }
        lastModifierOnlyCaptureCount = pendingModifierCapture.count
        VoxtLog.hotkey("Hotkey recorder captured modifier-only on \(reason). modifiers=\(HotkeyPreference.modifierSymbols(for: pendingModifierCapture.modifiers))")
        onHotkeyCaptured?(
            HotkeyPreference.Hotkey(
                keyCode: HotkeyPreference.modifierOnlyKeyCode,
                modifiers: pendingModifierCapture.modifiers,
                sidedModifiers: pendingModifierCapture.sidedModifiers
            )
        )
    }

    private func modifierCount(for modifiers: NSEvent.ModifierFlags) -> Int {
        var count = 0
        if modifiers.contains(.command) { count += 1 }
        if modifiers.contains(.option) { count += 1 }
        if modifiers.contains(.control) { count += 1 }
        if modifiers.contains(.shift) { count += 1 }
        if modifiers.contains(.function) { count += 1 }
        return count
    }
}

#if DEBUG
extension KeyCaptureView {
    func debugCaptureKeyDownForTests(
        keyCode: UInt16,
        modifiers: NSEvent.ModifierFlags = [],
        sidedModifiers: SidedModifierFlags = []
    ) {
        captureKeyDown(
            keyCode: keyCode,
            modifiers: modifiers,
            sidedModifiers: sidedModifiers
        )
    }

    func debugCaptureMouseButtonDownForTests(
        buttonNumber: Int,
        modifiers: NSEvent.ModifierFlags,
        sidedModifiers: SidedModifierFlags
    ) {
        captureMouseButtonDown(
            buttonNumber: buttonNumber,
            modifiers: modifiers,
            sidedModifiers: sidedModifiers
        )
    }
}
#endif
