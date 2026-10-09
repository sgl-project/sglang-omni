// HotkeyEventSupport.swift
// Provides Hotkey Event Support for hotkey handling.

import Foundation
import AppKit
import ApplicationServices
import Carbon

nonisolated enum HotkeyEventSupport {
    private static let voxtInjectedEventUserData: Int64 = 0x566F7874_496E6A

    static func markAsVoxtInjected(_ event: CGEvent?) {
        event?.setIntegerValueField(.eventSourceUserData, value: voxtInjectedEventUserData)
    }

    static func isVoxtInjected(_ event: CGEvent) -> Bool {
        event.getIntegerValueField(.eventSourceUserData) == voxtInjectedEventUserData
    }

    static func isVoxtInjected(eventSourceUserData: Int64) -> Bool {
        eventSourceUserData == voxtInjectedEventUserData
    }

    static func shouldLogFlagsChangedEvent(
        keyCode: UInt16,
        flags: CGEventFlags,
        triggerMode: HotkeyPreference.TriggerMode,
        transcriptionHotkey: HotkeyPreference.Hotkey,
        translationHotkey: HotkeyPreference.Hotkey,
        rewriteHotkey: HotkeyPreference.Hotkey?,
        isKeyDown: Bool,
        isTranslationKeyDown: Bool,
        isRewriteKeyDown: Bool,
        hasTranscriptionModifierTapCandidate: Bool,
        hasTranslationModifierTapCandidate: Bool,
        hasRewriteModifierTapCandidate: Bool,
        sawNonModifierKeyDuringFunctionChord: Bool
    ) -> Bool {
        guard typeRequiresTapFlagsLog(
            triggerMode: triggerMode,
            transcriptionHotkey: transcriptionHotkey,
            translationHotkey: translationHotkey,
            rewriteHotkey: rewriteHotkey
        ) else {
            return false
        }

        return HotkeyModifierInterpreter.isFunctionKeyEvent(keyCode) ||
            flags.contains(.maskSecondaryFn) ||
            isKeyDown ||
            isTranslationKeyDown ||
            isRewriteKeyDown ||
            hasTranscriptionModifierTapCandidate ||
            hasTranslationModifierTapCandidate ||
            hasRewriteModifierTapCandidate ||
            sawNonModifierKeyDuringFunctionChord
    }

    static func typeRequiresTapFlagsLog(
        triggerMode: HotkeyPreference.TriggerMode,
        transcriptionHotkey: HotkeyPreference.Hotkey,
        translationHotkey: HotkeyPreference.Hotkey,
        rewriteHotkey: HotkeyPreference.Hotkey?
    ) -> Bool {
        guard triggerMode == .tap else { return false }
        return HotkeyModifierInterpreter.isModifierOnly(transcriptionHotkey)
            || HotkeyModifierInterpreter.isModifierOnly(translationHotkey)
            || (rewriteHotkey.map { HotkeyModifierInterpreter.isModifierOnly($0) } ?? false)
    }

    static func isModifierKeyCode(_ keyCode: UInt16) -> Bool {
        switch Int(keyCode) {
        case kVK_Command,
             kVK_RightCommand,
             kVK_Shift,
             kVK_RightShift,
             kVK_Option,
             kVK_RightOption,
             kVK_Control,
             kVK_RightControl,
             kVK_Function,
             kVK_CapsLock:
            return true
        default:
            return false
        }
    }

    static func debugDescription(for flags: CGEventFlags) -> String {
        var values: [String] = []
        if flags.contains(.maskSecondaryFn) { values.append("fn") }
        if flags.contains(.maskShift) { values.append("shift") }
        if flags.contains(.maskControl) { values.append("ctrl") }
        if flags.contains(.maskAlternate) { values.append("opt") }
        if flags.contains(.maskCommand) { values.append("cmd") }
        return values.isEmpty ? "none" : values.joined(separator: "+")
    }

    static func modifierFlags(from cgFlags: CGEventFlags) -> NSEvent.ModifierFlags {
        var flags: NSEvent.ModifierFlags = []
        if cgFlags.contains(.maskCommand) { flags.insert(.command) }
        if cgFlags.contains(.maskAlternate) { flags.insert(.option) }
        if cgFlags.contains(.maskControl) { flags.insert(.control) }
        if cgFlags.contains(.maskShift) { flags.insert(.shift) }
        if cgFlags.contains(.maskSecondaryFn) { flags.insert(.function) }
        return flags
    }
}
