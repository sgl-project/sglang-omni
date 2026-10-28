// FeatureShortcutSettings.swift
// Provides Feature Shortcut Settings for feature settings.

import AppKit
import Carbon

struct FeatureShortcutSettings: Codable, Hashable, Sendable {
    var keyCode: UInt16
    var modifiersRawValue: UInt
    var sidedModifiersRawValue: Int

    init(
        keyCode: UInt16 = UInt16(kVK_Space),
        modifiers: NSEvent.ModifierFlags = [],
        sidedModifiers: SidedModifierFlags = []
    ) {
        self.keyCode = keyCode
        self.modifiersRawValue = modifiers.intersection(.hotkeyRelevant).rawValue
        self.sidedModifiersRawValue = sidedModifiers.rawValue
    }

    var modifiers: NSEvent.ModifierFlags {
        NSEvent.ModifierFlags(rawValue: modifiersRawValue).intersection(.hotkeyRelevant)
    }

    var sidedModifiers: SidedModifierFlags {
        SidedModifierFlags(rawValue: sidedModifiersRawValue).filtered(by: modifiers)
    }

    var hotkey: HotkeyPreference.Hotkey {
        HotkeyPreference.Hotkey(
            keyCode: keyCode,
            modifiers: modifiers,
            sidedModifiers: sidedModifiers
        )
    }

    static let defaultShortcut = Self(keyCode: UInt16(kVK_Space))
}

typealias TranscriptionNoteTriggerSettings = FeatureShortcutSettings
typealias TranscriptionContinueShortcutSettings = FeatureShortcutSettings
