// HotkeyRuntimeConfiguration.swift
// Provides Hotkey Runtime Configuration for hotkey handling.

import AppKit
import ApplicationServices
import Foundation

nonisolated struct HotkeyRuntimeConfiguration {
    let transcriptionBindings: [HotkeyPreference.HotkeyBinding]
    let translationBindings: [HotkeyPreference.HotkeyBinding]
    let rewriteBindings: [HotkeyPreference.HotkeyBinding]
    let meetingBindings: [HotkeyPreference.HotkeyBinding]
    let noteBindings: [HotkeyPreference.HotkeyBinding]
    let transcriptionHotkey: HotkeyPreference.Hotkey
    let translationHotkey: HotkeyPreference.Hotkey
    let rewriteHotkey: HotkeyPreference.Hotkey
    let meetingHotkey: HotkeyPreference.Hotkey?
    let noteHotkey: HotkeyPreference.Hotkey
    let customPasteHotkey: HotkeyPreference.Hotkey?
    let distinguishModifierSides: Bool
    let triggerMode: HotkeyPreference.TriggerMode
    let rewriteActivationMode: HotkeyPreference.RewriteActivationMode

    static func load(defaults: UserDefaults = .standard) -> HotkeyRuntimeConfiguration {
        let customPasteEnabled = defaults.bool(forKey: AppPreferenceKey.customPasteHotkeyEnabled)
        HotkeyPreference.migrateHotkeyBindingsIfNeeded(defaults: defaults)
        let availability = FeatureSettingsStore.availability(defaults: defaults)
        let transcriptionBindings = HotkeyPreference.loadTranscriptionBindings(defaults: defaults)
        let translationBindings = availability.translationEnabled
            ? HotkeyPreference.loadTranslationBindings(defaults: defaults)
            : []
        let rewriteBindings = availability.rewriteEnabled
            ? HotkeyPreference.loadRewriteBindings(defaults: defaults)
            : []
        let meetingBindings = availability.meetingEnabled
            ? HotkeyPreference.loadMeetingBindings(defaults: defaults)
            : []
        let noteBindings = availability.notesEnabled
            ? HotkeyPreference.loadNoteBindings(defaults: defaults)
            : []
        let rewriteActivationMode = availability.rewriteEnabled
            ? HotkeyPreference.loadRewriteActivationMode(defaults: defaults)
            : .dedicatedHotkey

        return HotkeyRuntimeConfiguration(
            transcriptionBindings: transcriptionBindings,
            translationBindings: translationBindings,
            rewriteBindings: rewriteBindings,
            meetingBindings: meetingBindings,
            noteBindings: noteBindings,
            transcriptionHotkey: transcriptionBindings.first?.hotkey ?? HotkeyPreference.load(),
            translationHotkey: translationBindings.first?.hotkey ?? HotkeyPreference.loadTranslation(),
            rewriteHotkey: rewriteBindings.first?.hotkey ?? HotkeyPreference.loadRewrite(),
            meetingHotkey: meetingBindings.first?.hotkey,
            noteHotkey: noteBindings.first?.hotkey ?? HotkeyPreference.Hotkey(
                keyCode: HotkeyPreference.defaultNoteKeyCode,
                modifiers: HotkeyPreference.defaultNoteModifiers,
                sidedModifiers: HotkeyPreference.defaultNoteSidedModifiers
            ),
            customPasteHotkey: customPasteEnabled ? HotkeyPreference.loadCustomPaste() : nil,
            distinguishModifierSides: HotkeyPreference.loadDistinguishModifierSides(),
            triggerMode: transcriptionBindings.first?.behavior.legacyTriggerMode ?? HotkeyPreference.loadTriggerMode(defaults: defaults),
            rewriteActivationMode: rewriteActivationMode
        )
    }

    var transcriptionFlags: CGEventFlags {
        HotkeyPreference.cgFlags(from: transcriptionHotkey.modifiers)
    }

    var translationFlags: CGEventFlags {
        HotkeyPreference.cgFlags(from: translationHotkey.modifiers)
    }

    var rewriteFlags: CGEventFlags {
        HotkeyPreference.cgFlags(from: rewriteHotkey.modifiers)
    }

    var meetingFlags: CGEventFlags {
        meetingHotkey.map { HotkeyPreference.cgFlags(from: $0.modifiers) } ?? []
    }

    var customPasteFlags: CGEventFlags {
        customPasteHotkey.map { HotkeyPreference.cgFlags(from: $0.modifiers) } ?? []
    }

    var noteFlags: CGEventFlags {
        HotkeyPreference.cgFlags(from: noteHotkey.modifiers)
    }

    var debugBindingsDescription: String {
        let meetingDescription = meetingHotkey.map {
            HotkeyPreference.displayString(for: $0, distinguishModifierSides: distinguishModifierSides)
        } ?? "disabled"
        let customPasteDescription = customPasteHotkey.map {
            HotkeyPreference.displayString(for: $0, distinguishModifierSides: distinguishModifierSides)
        } ?? "disabled"

        return "Hotkey bindings. transcription=\(HotkeyPreference.displayString(for: transcriptionHotkey, distinguishModifierSides: distinguishModifierSides)), translation=\(HotkeyPreference.displayString(for: translationHotkey, distinguishModifierSides: distinguishModifierSides)), rewrite=\(HotkeyPreference.displayString(for: rewriteHotkey, distinguishModifierSides: distinguishModifierSides)), note=\(HotkeyPreference.displayString(for: noteHotkey, distinguishModifierSides: distinguishModifierSides)), rewriteActivation=\(rewriteActivationMode.rawValue), meeting=\(meetingDescription), customPaste=\(customPasteDescription), trigger=\(triggerMode.rawValue)"
    }
}
