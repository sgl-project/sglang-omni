// HotkeySupport.swift
// Provides Hotkey Support for hotkey handling.

import AppKit
import Carbon
import SwiftUI
import IOKit.hidsystem

nonisolated struct HotkeyPreference {
    enum TriggerBehavior: String, CaseIterable, Identifiable, Codable {
        case tap
        case longPress
        case doubleTap

        var id: String { rawValue }

        var title: String {
            switch self {
            case .tap:
                return AppLocalization.localizedString("Tap")
            case .longPress:
                return AppLocalization.localizedString("Long Press")
            case .doubleTap:
                return AppLocalization.localizedString("Double Tap")
            }
        }

        var legacyTriggerMode: TriggerMode {
            switch self {
            case .longPress:
                return .longPress
            case .tap, .doubleTap:
                return .tap
            }
        }

        init(_ triggerMode: TriggerMode) {
            switch triggerMode {
            case .longPress:
                self = .longPress
            case .tap:
                self = .tap
            }
        }
    }

    struct HotkeyBinding: Identifiable, Equatable, Codable {
        let id: UUID
        var hotkey: Hotkey
        var behavior: TriggerBehavior

        init(
            id: UUID = UUID(),
            hotkey: Hotkey,
            behavior: TriggerBehavior
        ) {
            self.id = id
            self.hotkey = hotkey
            self.behavior = behavior
        }

        private enum CodingKeys: String, CodingKey {
            case id
            case inputType
            case keyCode
            case mouseButtonNumber
            case modifiers
            case sidedModifiers
            case behavior
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            id = try container.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
            behavior = try container.decodeIfPresent(TriggerBehavior.self, forKey: .behavior) ?? .tap

            let inputType = try container.decodeIfPresent(String.self, forKey: .inputType)
            let keyCode = try container.decodeIfPresent(UInt16.self, forKey: .keyCode) ?? HotkeyPreference.modifierOnlyKeyCode
            let mouseButtonNumber = try container.decodeIfPresent(Int.self, forKey: .mouseButtonNumber)
            let input: Hotkey.Input
            if Hotkey.Input.Kind(rawValue: inputType ?? "") == .mouseButton,
               let mouseButtonNumber,
               mouseButtonNumber >= HotkeyPreference.middleMouseButtonNumber {
                input = .mouseButton(mouseButtonNumber)
            } else {
                input = .keyboard(keyCode)
            }

            let modifiersRaw = try container.decodeIfPresent(UInt.self, forKey: .modifiers) ?? 0
            let modifiers = NSEvent.ModifierFlags(rawValue: modifiersRaw).intersection(.hotkeyRelevant)
            let sidedRaw = try container.decodeIfPresent(Int.self, forKey: .sidedModifiers) ?? 0
            hotkey = HotkeyPreference.canonicalHotkey(
                input: input,
                modifiers: modifiers,
                sidedModifiers: SidedModifierFlags(rawValue: sidedRaw).filtered(by: modifiers)
            )
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(id, forKey: .id)
            try container.encode(hotkey.input.kind.rawValue, forKey: .inputType)
            switch hotkey.input {
            case .keyboard(let keyCode):
                try container.encode(keyCode, forKey: .keyCode)
            case .mouseButton(let buttonNumber):
                try container.encode(buttonNumber, forKey: .mouseButtonNumber)
            }
            try container.encode(hotkey.modifiers.rawValue, forKey: .modifiers)
            try container.encode(hotkey.sidedModifiers.filtered(by: hotkey.modifiers).rawValue, forKey: .sidedModifiers)
            try container.encode(behavior, forKey: .behavior)
        }
    }

    enum TriggerMode: String, CaseIterable, Identifiable {
        case longPress
        case tap

        var id: String { rawValue }

        var titleKey: LocalizedStringKey {
            switch self {
            case .longPress: return "Long Press (Release to End)"
            case .tap: return "Tap (Press to Toggle)"
            }
        }

        var title: String {
            switch self {
            case .longPress: return AppLocalization.localizedString("Long Press (Release to End)")
            case .tap: return AppLocalization.localizedString("Tap (Press to Toggle)")
            }
        }
    }

    enum RewriteActivationMode: String, CaseIterable, Identifiable {
        case dedicatedHotkey
        case doubleTapTranscriptionHotkey

        var id: String { rawValue }
    }

    enum Preset: String, CaseIterable, Identifiable {
        case fnCombo
        case commandCombo
        case mouseMiddleFnShift
        case custom

        var id: String { rawValue }

        var title: String {
            switch self {
            case .fnCombo:
                return AppLocalization.localizedString("fn Combo")
            case .commandCombo:
                return AppLocalization.localizedString("Command Combo")
            case .mouseMiddleFnShift:
                return AppLocalization.localizedString("Mouse Middle + fn Shift")
            case .custom:
                return AppLocalization.localizedString("Custom")
            }
        }
    }

    struct Hotkey: Equatable, Codable {
        enum Input: Equatable {
            case keyboard(UInt16)
            case mouseButton(Int)

            enum Kind: String {
                case keyboard
                case mouseButton
            }

            var kind: Kind {
                switch self {
                case .keyboard:
                    return .keyboard
                case .mouseButton:
                    return .mouseButton
                }
            }
        }

        let input: Input
        let modifiers: NSEvent.ModifierFlags
        let sidedModifiers: SidedModifierFlags

        private enum CodingKeys: String, CodingKey {
            case inputType
            case keyCode
            case mouseButtonNumber
            case modifiers
            case sidedModifiers
        }

        init(
            input: Input,
            modifiers: NSEvent.ModifierFlags,
            sidedModifiers: SidedModifierFlags
        ) {
            self.input = input
            self.modifiers = modifiers
            self.sidedModifiers = sidedModifiers
        }

        init(
            keyCode: UInt16,
            modifiers: NSEvent.ModifierFlags,
            sidedModifiers: SidedModifierFlags
        ) {
            self.init(input: .keyboard(keyCode), modifiers: modifiers, sidedModifiers: sidedModifiers)
        }

        init(
            mouseButtonNumber: Int,
            modifiers: NSEvent.ModifierFlags = [],
            sidedModifiers: SidedModifierFlags = []
        ) {
            self.init(input: .mouseButton(mouseButtonNumber), modifiers: modifiers, sidedModifiers: sidedModifiers)
        }

        var keyCode: UInt16 {
            switch input {
            case .keyboard(let keyCode):
                return keyCode
            case .mouseButton:
                return HotkeyPreference.modifierOnlyKeyCode
            }
        }

        var mouseButtonNumber: Int? {
            switch input {
            case .keyboard:
                return nil
            case .mouseButton(let buttonNumber):
                return buttonNumber
            }
        }

        var isMouseButton: Bool {
            mouseButtonNumber != nil
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let inputType = try container.decodeIfPresent(String.self, forKey: .inputType)
            let keyCode = try container.decodeIfPresent(UInt16.self, forKey: .keyCode) ?? HotkeyPreference.modifierOnlyKeyCode
            let mouseButtonNumber = try container.decodeIfPresent(Int.self, forKey: .mouseButtonNumber)
            if Input.Kind(rawValue: inputType ?? "") == .mouseButton,
               let mouseButtonNumber,
               mouseButtonNumber >= HotkeyPreference.middleMouseButtonNumber {
                input = .mouseButton(mouseButtonNumber)
            } else {
                input = .keyboard(keyCode)
            }
            let modifiersRaw = try container.decodeIfPresent(UInt.self, forKey: .modifiers) ?? 0
            modifiers = NSEvent.ModifierFlags(rawValue: modifiersRaw).intersection(.hotkeyRelevant)
            let sidedRaw = try container.decodeIfPresent(Int.self, forKey: .sidedModifiers) ?? 0
            sidedModifiers = SidedModifierFlags(rawValue: sidedRaw).filtered(by: modifiers)
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(input.kind.rawValue, forKey: .inputType)
            switch input {
            case .keyboard(let keyCode):
                try container.encode(keyCode, forKey: .keyCode)
            case .mouseButton(let buttonNumber):
                try container.encode(buttonNumber, forKey: .mouseButtonNumber)
            }
            try container.encode(modifiers.rawValue, forKey: .modifiers)
            try container.encode(sidedModifiers.filtered(by: modifiers).rawValue, forKey: .sidedModifiers)
        }
    }

    struct PresetHotkeys: Equatable {
        let distinguishSides: Bool
        let transcription: Hotkey
        let translation: Hotkey
        let rewrite: Hotkey
        let meeting: Hotkey
        let note: Hotkey
        let customPaste: Hotkey
        let triggerMode: TriggerMode
        let rewriteActivationMode: RewriteActivationMode

        var transcriptionBindings: [HotkeyBinding] {
            [.init(id: Self.transcriptionBindingID, hotkey: transcription, behavior: TriggerBehavior(triggerMode))]
        }

        var translationBindings: [HotkeyBinding] {
            [.init(id: Self.translationBindingID, hotkey: translation, behavior: TriggerBehavior(triggerMode))]
        }

        var rewriteBindings: [HotkeyBinding] {
            let behavior: TriggerBehavior = rewriteActivationMode == .doubleTapTranscriptionHotkey
                ? .doubleTap
                : TriggerBehavior(triggerMode)
            return [.init(id: Self.rewriteBindingID, hotkey: rewrite, behavior: behavior)]
        }

        var meetingBindings: [HotkeyBinding] {
            [.init(id: Self.meetingBindingID, hotkey: meeting, behavior: TriggerBehavior(triggerMode))]
        }

        var noteBindings: [HotkeyBinding] {
            [.init(id: Self.noteBindingID, hotkey: note, behavior: .tap)]
        }

        private static let transcriptionBindingID = UUID(uuidString: "00000000-0000-0000-0000-000000000101")!
        private static let translationBindingID = UUID(uuidString: "00000000-0000-0000-0000-000000000102")!
        private static let rewriteBindingID = UUID(uuidString: "00000000-0000-0000-0000-000000000103")!
        private static let meetingBindingID = UUID(uuidString: "00000000-0000-0000-0000-000000000104")!
        private static let noteBindingID = UUID(uuidString: "00000000-0000-0000-0000-000000000105")!

        init(
            distinguishSides: Bool,
            transcription: Hotkey,
            translation: Hotkey,
            rewrite: Hotkey,
            meeting: Hotkey,
            note: Hotkey,
            customPaste: Hotkey,
            triggerMode: TriggerMode = .tap,
            rewriteActivationMode: RewriteActivationMode = .dedicatedHotkey
        ) {
            self.distinguishSides = distinguishSides
            self.transcription = transcription
            self.translation = translation
            self.rewrite = rewrite
            self.meeting = meeting
            self.note = note
            self.customPaste = customPaste
            self.triggerMode = triggerMode
            self.rewriteActivationMode = rewriteActivationMode
        }
    }

    static let modifierOnlyKeyCode: UInt16 = 0xFFFF
    static let defaultKeyCode: UInt16 = modifierOnlyKeyCode
    static let defaultModifiers: NSEvent.ModifierFlags = [.function]
    static let defaultTranslationKeyCode: UInt16 = modifierOnlyKeyCode
    static let defaultTranslationModifiers: NSEvent.ModifierFlags = [.function, .shift]
    static let defaultRewriteKeyCode: UInt16 = modifierOnlyKeyCode
    static let defaultRewriteModifiers: NSEvent.ModifierFlags = [.function, .control]
    static let defaultMeetingKeyCode: UInt16 = modifierOnlyKeyCode
    static let defaultMeetingModifiers: NSEvent.ModifierFlags = [.function, .option]
    static let defaultNoteKeyCode: UInt16 = modifierOnlyKeyCode
    static let defaultNoteModifiers: NSEvent.ModifierFlags = [.command]
    static let defaultNoteSidedModifiers: SidedModifierFlags = [.rightCommand]
    static let defaultCustomPasteKeyCode: UInt16 = UInt16(kVK_ANSI_V)
    static let defaultCustomPasteModifiers: NSEvent.ModifierFlags = [.control, .command]
    static let defaultTriggerMode: TriggerMode = .tap
    static let defaultRewriteActivationMode: RewriteActivationMode = .dedicatedHotkey
    static let defaultDistinguishModifierSides = true
    static let defaultPreset: Preset = .fnCombo
    static let middleMouseButtonNumber = 2
    private static let maximumRecordableKeyboardKeyCode: UInt16 = 0x7F

    static func isRecordableKeyboardKeyCode(_ keyCode: UInt16) -> Bool {
        keyCode <= maximumRecordableKeyboardKeyCode
    }

    static func isAllowedGlobalShortcut(_ hotkey: Hotkey) -> Bool {
        switch hotkey.input {
        case .keyboard(let keyCode):
            if keyCode == modifierOnlyKeyCode {
                return !hotkey.modifiers.isEmpty
            }
            return isRecordableKeyboardKeyCode(keyCode)
        case .mouseButton(let buttonNumber):
            return buttonNumber >= middleMouseButtonNumber
        }
    }

    static func presetHotkeys(for preset: Preset) -> PresetHotkeys? {
        switch preset {
        case .fnCombo:
            return PresetHotkeys(
                distinguishSides: true,
                transcription: Hotkey(keyCode: defaultKeyCode, modifiers: defaultModifiers, sidedModifiers: []),
                translation: Hotkey(keyCode: defaultTranslationKeyCode, modifiers: defaultTranslationModifiers, sidedModifiers: []),
                rewrite: Hotkey(keyCode: defaultRewriteKeyCode, modifiers: defaultRewriteModifiers, sidedModifiers: []),
                meeting: Hotkey(keyCode: defaultMeetingKeyCode, modifiers: defaultMeetingModifiers, sidedModifiers: []),
                note: Hotkey(keyCode: defaultNoteKeyCode, modifiers: defaultNoteModifiers, sidedModifiers: defaultNoteSidedModifiers),
                customPaste: Hotkey(keyCode: defaultCustomPasteKeyCode, modifiers: defaultCustomPasteModifiers, sidedModifiers: [])
            )
        case .commandCombo:
            return PresetHotkeys(
                distinguishSides: true,
                transcription: Hotkey(keyCode: modifierOnlyKeyCode, modifiers: [.command], sidedModifiers: [.rightCommand]),
                translation: Hotkey(keyCode: modifierOnlyKeyCode, modifiers: [.command, .shift], sidedModifiers: [.rightCommand, .rightShift]),
                rewrite: Hotkey(keyCode: modifierOnlyKeyCode, modifiers: [.command, .option], sidedModifiers: [.rightCommand, .rightOption]),
                meeting: Hotkey(keyCode: UInt16(kVK_ANSI_L), modifiers: [.command], sidedModifiers: [.rightCommand]),
                note: Hotkey(keyCode: modifierOnlyKeyCode, modifiers: [.option], sidedModifiers: [.rightOption]),
                customPaste: Hotkey(keyCode: defaultCustomPasteKeyCode, modifiers: defaultCustomPasteModifiers, sidedModifiers: [])
            )
        case .mouseMiddleFnShift:
            return PresetHotkeys(
                distinguishSides: true,
                transcription: Hotkey(mouseButtonNumber: middleMouseButtonNumber),
                translation: Hotkey(keyCode: defaultTranslationKeyCode, modifiers: defaultTranslationModifiers, sidedModifiers: []),
                rewrite: Hotkey(mouseButtonNumber: middleMouseButtonNumber),
                meeting: Hotkey(keyCode: defaultMeetingKeyCode, modifiers: defaultMeetingModifiers, sidedModifiers: []),
                note: Hotkey(keyCode: modifierOnlyKeyCode, modifiers: [.command], sidedModifiers: [.rightCommand]),
                customPaste: Hotkey(keyCode: defaultCustomPasteKeyCode, modifiers: defaultCustomPasteModifiers, sidedModifiers: []),
                triggerMode: .tap,
                rewriteActivationMode: .doubleTapTranscriptionHotkey
            )
        case .custom:
            return nil
        }
    }

    static func hotkeyMatches(
        _ hotkey: Hotkey,
        eventFlags: CGEventFlags,
        sidedModifiers: SidedModifierFlags,
        distinguishModifierSides: Bool
    ) -> Bool {
        let requiredFlags = cgFlags(from: hotkey.modifiers)
        if case .keyboard(let keyCode) = hotkey.input,
           keyCode != modifierOnlyKeyCode {
            let relevantFlags = eventFlags.intersection([
                .maskCommand,
                .maskAlternate,
                .maskControl,
                .maskShift,
                .maskSecondaryFn
            ])
            guard relevantFlags == requiredFlags else { return false }
        } else {
            guard eventFlags.contains(requiredFlags) else { return false }
        }
        guard distinguishModifierSides, !hotkey.sidedModifiers.isEmpty else { return true }
        return sidedModifiers.isSuperset(of: hotkey.sidedModifiers) && hotkey.sidedModifiers.matches(requiredModifiers: hotkey.modifiers)
    }

    static func cgFlags(from modifiers: NSEvent.ModifierFlags) -> CGEventFlags {
        var flags: CGEventFlags = []
        if modifiers.contains(.command) { flags.insert(.maskCommand) }
        if modifiers.contains(.option) { flags.insert(.maskAlternate) }
        if modifiers.contains(.control) { flags.insert(.maskControl) }
        if modifiers.contains(.shift) { flags.insert(.maskShift) }
        if modifiers.contains(.function) { flags.insert(.maskSecondaryFn) }
        return flags
    }

    static func canonicalHotkey(
        input: Hotkey.Input,
        modifiers: NSEvent.ModifierFlags,
        sidedModifiers: SidedModifierFlags
    ) -> Hotkey {
        guard case .keyboard(let keyCode) = input else {
            return Hotkey(
                input: input,
                modifiers: modifiers,
                sidedModifiers: sidedModifiers.filtered(by: modifiers)
            )
        }
        guard let representedModifier = SidedModifierFlags.fromModifierKeyCode(keyCode),
              modifiers.contains(representedModifier.modifiers)
        else {
            return Hotkey(
                input: input,
                modifiers: modifiers,
                sidedModifiers: sidedModifiers.filtered(by: modifiers)
            )
        }

        return Hotkey(
            keyCode: modifierOnlyKeyCode,
            modifiers: modifiers,
            sidedModifiers: sidedModifiers
                .union(representedModifier.sided)
                .filtered(by: modifiers)
        )
    }

}
