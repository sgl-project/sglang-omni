// HotkeySettingsValidation.swift
// Provides Hotkey Settings Validation for settings screens.

import AppKit
import Carbon
import Foundation

private struct HotkeyConflictRule {
    let keyCode: UInt16
    let modifiers: NSEvent.ModifierFlags
    let messageKey: String
}

private let hotkeySettingsConflictRules: [HotkeyConflictRule] = [
    HotkeyConflictRule(keyCode: UInt16(kVK_Space), modifiers: [.function], messageKey: "May conflict with Globe / input source switching (fn Space). Disable or remap the macOS shortcut if needed."),
    HotkeyConflictRule(keyCode: UInt16(kVK_Space), modifiers: [.command], messageKey: "Conflicts with Spotlight (⌘Space)."),
    HotkeyConflictRule(keyCode: UInt16(kVK_Space), modifiers: [.command, .option], messageKey: "Conflicts with Finder search (⌥⌘Space)."),
    HotkeyConflictRule(keyCode: UInt16(kVK_Tab), modifiers: [.command], messageKey: "Conflicts with App Switcher (⌘Tab)."),
    HotkeyConflictRule(keyCode: UInt16(kVK_ANSI_Grave), modifiers: [.command], messageKey: "Conflicts with window switcher (⌘`)."),
    HotkeyConflictRule(keyCode: UInt16(kVK_ANSI_Q), modifiers: [.command], messageKey: "Conflicts with Quit (⌘Q)."),
    HotkeyConflictRule(keyCode: UInt16(kVK_ANSI_H), modifiers: [.command], messageKey: "Conflicts with Hide (⌘H)."),
    HotkeyConflictRule(keyCode: UInt16(kVK_ANSI_M), modifiers: [.command], messageKey: "Conflicts with Minimise (⌘M)."),
    HotkeyConflictRule(keyCode: UInt16(kVK_ANSI_W), modifiers: [.command], messageKey: "Conflicts with Close (⌘W)."),
    HotkeyConflictRule(keyCode: UInt16(kVK_ANSI_V), modifiers: [.command], messageKey: "Conflicts with Paste (⌘V).")
]

struct HotkeySettingsValidation {
    struct State {
        let transcriptionBindings: [HotkeyPreference.HotkeyBinding]
        let translationBindings: [HotkeyPreference.HotkeyBinding]
        let meetingBindings: [HotkeyPreference.HotkeyBinding]
        let rewriteBindings: [HotkeyPreference.HotkeyBinding]
        let customPasteHotkey: HotkeyPreference.Hotkey?
        let noteBindings: [HotkeyPreference.HotkeyBinding]

        init(
            transcriptionHotkey: HotkeyPreference.Hotkey,
            translationHotkey: HotkeyPreference.Hotkey,
            meetingHotkey: HotkeyPreference.Hotkey? = nil,
            rewriteHotkey: HotkeyPreference.Hotkey,
            shouldValidateRewriteHotkey: Bool,
            customPasteHotkey: HotkeyPreference.Hotkey?
        ) {
            self.transcriptionBindings = [.init(hotkey: transcriptionHotkey, behavior: .tap)]
            self.translationBindings = [.init(hotkey: translationHotkey, behavior: .tap)]
            self.meetingBindings = meetingHotkey.map { [.init(hotkey: $0, behavior: .tap)] } ?? []
            self.rewriteBindings = shouldValidateRewriteHotkey ? [.init(hotkey: rewriteHotkey, behavior: .tap)] : []
            self.customPasteHotkey = customPasteHotkey
            self.noteBindings = []
        }

        init(
            transcriptionBindings: [HotkeyPreference.HotkeyBinding],
            translationBindings: [HotkeyPreference.HotkeyBinding],
            meetingBindings: [HotkeyPreference.HotkeyBinding],
            rewriteBindings: [HotkeyPreference.HotkeyBinding],
            customPasteHotkey: HotkeyPreference.Hotkey?,
            noteBindings: [HotkeyPreference.HotkeyBinding] = []
        ) {
            self.transcriptionBindings = transcriptionBindings
            self.translationBindings = translationBindings
            self.meetingBindings = meetingBindings
            self.rewriteBindings = rewriteBindings
            self.customPasteHotkey = customPasteHotkey
            self.noteBindings = noteBindings.map {
                .init(id: $0.id, hotkey: $0.hotkey, behavior: .tap)
            }
        }
    }

    struct Message: Identifiable, Equatable {
        let id: String
        let text: String
    }

    static func messages(for state: State) -> [Message] {
        var messages: [Message] = []

        let groups: [(key: String, title: String, bindings: [HotkeyPreference.HotkeyBinding])] = [
            ("transcription", "Transcription shortcut: %@", state.transcriptionBindings),
            ("translation", "Translation shortcut: %@", state.translationBindings),
            ("meeting", "Meeting shortcut: %@", state.meetingBindings),
            ("rewrite", "Content rewrite shortcut: %@", state.rewriteBindings),
            ("note", "Note shortcut: %@", state.noteBindings)
        ]

        for group in groups {
            for (index, binding) in group.bindings.enumerated() {
                appendInvalidShortcutMessage(
                    for: binding.hotkey,
                    formatKey: group.title,
                    id: "invalid.\(group.key).\(index)",
                    to: &messages
                )
                appendConflictMessage(
                    for: binding.hotkey,
                    formatKey: group.title,
                    id: "conflict.\(group.key).\(index)",
                    to: &messages
                )
            }
        }
        if let customPasteHotkey = state.customPasteHotkey {
            appendInvalidShortcutMessage(
                for: customPasteHotkey,
                formatKey: "Custom paste shortcut: %@",
                id: "invalid.customPaste",
                to: &messages
            )
            appendConflictMessage(
                for: customPasteHotkey,
                formatKey: "Custom paste shortcut: %@",
                id: "conflict.customPaste",
                to: &messages
            )
        }

        appendDuplicateBindingMessages(groups: groups, to: &messages)
        appendDuplicateBindingMessagesWithinGroups(groups: groups, to: &messages)

        if let customPasteHotkey = state.customPasteHotkey {
            appendCustomPasteDuplicateMessages(customPasteHotkey, groups: groups, to: &messages)
        }

        return messages
    }

    private static func appendDuplicateBindingMessages(
        groups: [(key: String, title: String, bindings: [HotkeyPreference.HotkeyBinding])],
        to messages: inout [Message]
    ) {
        var seen: [(key: String, binding: HotkeyPreference.HotkeyBinding)] = []
        for group in groups {
            for binding in group.bindings {
                if let duplicate = seen.first(where: {
                    $0.key != group.key &&
                    $0.binding.hotkey == binding.hotkey &&
                    $0.binding.behavior == binding.behavior
                }) {
                    messages.append(.init(
                        id: "duplicate.\(duplicate.key).\(group.key)",
                        text: duplicateText(first: duplicate.key, second: group.key)
                    ))
                } else {
                    seen.append((group.key, binding))
                }
            }
        }
    }

    private static func appendDuplicateBindingMessagesWithinGroups(
        groups: [(key: String, title: String, bindings: [HotkeyPreference.HotkeyBinding])],
        to messages: inout [Message]
    ) {
        for group in groups {
            var seen: [HotkeyPreference.Hotkey] = []
            for binding in group.bindings {
                if seen.contains(binding.hotkey) {
                    messages.append(.init(
                        id: "duplicate.\(group.key).\(group.key)",
                        text: AppLocalization.localizedString("A workflow cannot use the same shortcut more than once.")
                    ))
                    break
                }
                seen.append(binding.hotkey)
            }
        }
    }

    private static func duplicateText(first: String, second: String) -> String {
        let pair = Set([first, second])
        if pair == Set(["transcription", "translation"]) {
            return AppLocalization.localizedString("Transcription and translation shortcuts should be different.")
        }
        if pair == Set(["transcription", "meeting"]) {
            return AppLocalization.localizedString("Transcription and meeting shortcuts should be different.")
        }
        if pair == Set(["translation", "meeting"]) {
            return AppLocalization.localizedString("Translation and meeting shortcuts should be different.")
        }
        if pair == Set(["rewrite", "meeting"]) {
            return AppLocalization.localizedString("Content rewrite and meeting shortcuts should be different.")
        }
        if pair == Set(["transcription", "rewrite"]) {
            return AppLocalization.localizedString("Transcription and content rewrite shortcuts should be different.")
        }
        if pair == Set(["translation", "rewrite"]) {
            return AppLocalization.localizedString("Translation and content rewrite shortcuts should be different.")
        }
        return AppLocalization.localizedString("Shortcuts with the same trigger behavior should be different.")
    }

    private static func appendCustomPasteDuplicateMessages(
        _ customPasteHotkey: HotkeyPreference.Hotkey,
        groups: [(key: String, title: String, bindings: [HotkeyPreference.HotkeyBinding])],
        to messages: inout [Message]
    ) {
        for group in groups {
            for binding in group.bindings where binding.behavior == .tap && binding.hotkey == customPasteHotkey {
                messages.append(.init(
                    id: "duplicate.\(group.key).customPaste",
                    text: AppLocalization.localizedString("Custom paste shortcut should be different from tap shortcuts.")
                ))
            }
        }
    }

    private static func appendConflictMessage(
        for hotkey: HotkeyPreference.Hotkey,
        formatKey: String,
        id: String,
        to messages: inout [Message]
    ) {
        guard let conflictMessage = conflictMessage(for: hotkey) else { return }
        messages.append(.init(id: id, text: AppLocalization.format(formatKey, conflictMessage)))
    }

    private static func appendInvalidShortcutMessage(
        for hotkey: HotkeyPreference.Hotkey,
        formatKey: String,
        id: String,
        to messages: inout [Message]
    ) {
        guard !HotkeyPreference.isAllowedGlobalShortcut(hotkey) else { return }
        messages.append(.init(
            id: id,
            text: AppLocalization.format(
                formatKey,
                AppLocalization.localizedString("Keyboard shortcuts must include at least one modifier key.")
            )
        ))
    }

    private static func appendEqualityMessage(
        _ condition: Bool,
        id: String,
        textKey: String,
        to messages: inout [Message]
    ) {
        guard condition else { return }
        messages.append(.init(id: id, text: AppLocalization.localizedString(textKey)))
    }

    private static func conflictMessage(
        for hotkey: HotkeyPreference.Hotkey
    ) -> String? {
        hotkeySettingsConflictRules.first {
            hotkey.keyCode == $0.keyCode && hotkey.modifiers == $0.modifiers
        }.map { AppLocalization.localizedString($0.messageKey) }
    }
}
