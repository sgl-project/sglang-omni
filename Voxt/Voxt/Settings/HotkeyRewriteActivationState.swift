// HotkeyRewriteActivationState.swift
// Provides Hotkey Rewrite Activation State for settings screens.

import Foundation

struct HotkeyRewriteActivationState: Equatable {
    let mode: HotkeyPreference.RewriteActivationMode

    init(rawValue: String) {
        self.mode = HotkeyPreference.RewriteActivationMode(rawValue: rawValue)
            ?? HotkeyPreference.defaultRewriteActivationMode
    }

    var isDoubleTapWakeEnabled: Bool {
        mode == .doubleTapTranscriptionHotkey
    }

    var toggledMode: HotkeyPreference.RewriteActivationMode {
        isDoubleTapWakeEnabled ? .dedicatedHotkey : .doubleTapTranscriptionHotkey
    }

    func displayText(
        for transcriptionHotkey: HotkeyPreference.Hotkey,
        distinguishModifierSides: Bool
    ) -> String {
        AppLocalization.format(
            "Double-tap %@",
            HotkeyPreference.displayString(
                for: transcriptionHotkey,
                distinguishModifierSides: distinguishModifierSides
            )
        )
    }

    func enforcedTriggerMode(
        from requestedMode: HotkeyPreference.TriggerMode
    ) -> HotkeyPreference.TriggerMode {
        HotkeyPreference.enforcedTriggerMode(
            requestedMode,
            rewriteActivationMode: mode
        )
    }
}
