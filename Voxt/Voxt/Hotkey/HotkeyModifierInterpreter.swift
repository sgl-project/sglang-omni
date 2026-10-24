// HotkeyModifierInterpreter.swift
// Provides Hotkey Modifier Interpreter for hotkey handling.

import Foundation
import AppKit
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

    static func hotkeySpecificity(_ hotkey: HotkeyPreference.Hotkey) -> Int {
        var score = modifierSpecificity(hotkey.modifiers)
        if !hotkey.sidedModifiers.isEmpty {
            score += sidedModifierSpecificity(hotkey.sidedModifiers)
        }
        if !isModifierOnly(hotkey) {
            score += 100
        }
        return score
    }

    static func behaviorPriority(_ behavior: HotkeyPreference.TriggerBehavior) -> Int {
        switch behavior {
        case .doubleTap:
            return 0
        case .longPress:
            return 1
        case .tap:
            return 2
        }
    }

    static func modifierOnlyHotkey(
        _ candidate: HotkeyPreference.Hotkey,
        isMoreSpecificThan base: HotkeyPreference.Hotkey,
        distinguishModifierSides: Bool
    ) -> Bool {
        guard isModifierOnly(candidate),
              isModifierOnly(base),
              candidate.modifiers.intersection(base.modifiers) == base.modifiers
        else {
            return false
        }

        if distinguishModifierSides, !base.sidedModifiers.isEmpty {
            guard candidate.sidedModifiers.isSuperset(of: base.sidedModifiers) else { return false }
        }

        if candidate.modifiers != base.modifiers {
            return true
        }

        guard distinguishModifierSides else { return false }
        return candidate.sidedModifiers.isStrictSuperset(of: base.sidedModifiers)
    }

    private static func modifierSpecificity(_ modifiers: NSEvent.ModifierFlags) -> Int {
        var score = 0
        if modifiers.contains(.function) { score += 10 }
        if modifiers.contains(.shift) { score += 10 }
        if modifiers.contains(.control) { score += 10 }
        if modifiers.contains(.option) { score += 10 }
        if modifiers.contains(.command) { score += 10 }
        return score
    }

    private static func sidedModifierSpecificity(_ sidedModifiers: SidedModifierFlags) -> Int {
        var score = 0
        if sidedModifiers.contains(.leftShift) { score += 1 }
        if sidedModifiers.contains(.rightShift) { score += 1 }
        if sidedModifiers.contains(.leftControl) { score += 1 }
        if sidedModifiers.contains(.rightControl) { score += 1 }
        if sidedModifiers.contains(.leftOption) { score += 1 }
        if sidedModifiers.contains(.rightOption) { score += 1 }
        if sidedModifiers.contains(.leftCommand) { score += 1 }
        if sidedModifiers.contains(.rightCommand) { score += 1 }
        return score
    }
}
