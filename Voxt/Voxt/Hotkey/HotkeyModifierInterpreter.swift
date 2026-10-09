// HotkeyModifierInterpreter.swift
// Provides Hotkey Modifier Interpreter for hotkey handling.

import Foundation
import Carbon
import ApplicationServices

nonisolated struct HotkeyModifierInterpreter {
    static func isModifierOnly(_ hotkey: HotkeyPreference.Hotkey) -> Bool {
        guard case .keyboard(let keyCode) = hotkey.input else { return false }
        return keyCode == HotkeyPreference.modifierOnlyKeyCode
    }

    static func isFunctionKeyEvent(_ keyCode: UInt16) -> Bool {
        keyCode == UInt16(kVK_Function)
    }

    static func translationTriggerDown(
        keyCode: UInt16,
        comboIsDown: Bool,
        eventFlags: CGEventFlags,
        translationFlags: CGEventFlags
    ) -> Bool {
        let isFnOnlyHotkey = translationFlags == .maskSecondaryFn
        let fnPressedForModifierHotkey =
            isFnOnlyHotkey &&
            isFunctionKeyEvent(keyCode) &&
            eventFlags.contains(.maskSecondaryFn)
        return comboIsDown || fnPressedForModifierHotkey
    }

    static func transcriptionTriggerDown(
        keyCode: UInt16,
        comboIsDown: Bool,
        eventFlags: CGEventFlags,
        transcriptionFlags: CGEventFlags
    ) -> Bool {
        let isFnOnlyHotkey = transcriptionFlags == .maskSecondaryFn
        // For fn-only hotkey, some keyboards report keyCode=Function with flags jitter.
        let fnPressedForModifierHotkey =
            isFnOnlyHotkey &&
            isFunctionKeyEvent(keyCode) &&
            eventFlags.contains(.maskSecondaryFn)
        return comboIsDown || fnPressedForModifierHotkey
    }

    static func shouldDelayTranscriptionTap(
        transcriptionHotkey: HotkeyPreference.Hotkey,
        prioritizedModifierHotkeys: [HotkeyPreference.Hotkey],
        transcriptionFlags: CGEventFlags,
        prioritizedFlags: [CGEventFlags]
    ) -> Bool {
        // Only needed for overlap such as transcription=fn and translation/rewrite=fn+shift/fn+control.
        guard isModifierOnly(transcriptionHotkey) else { return false }
        for (index, hotkey) in prioritizedModifierHotkeys.enumerated() {
            guard isModifierOnly(hotkey) else { continue }
            let flags = prioritizedFlags[index]
            guard transcriptionFlags != flags else { continue }
            if flags.contains(transcriptionFlags) {
                return true
            }
        }
        return false
    }
}
