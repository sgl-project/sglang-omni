// SPDX-License-Identifier: Apache-2.0
import AppKit
import Carbon

/// Decides what a shortcut-recording session does with each key event.
struct ShortcutCapture {
    enum Outcome: Equatable {
        case pending
        case cancel
        case reject
        case record(keyCode: UInt16, modifiers: UInt64)
    }

    private static let chordMask: NSEvent.ModifierFlags = [.command, .control, .option, .shift]
    private static let functionKeys: [UInt16: String] = [122: "F1", 120: "F2", 99: "F3", 118: "F4", 96: "F5",
        97: "F6", 98: "F7", 100: "F8", 101: "F9", 109: "F10", 103: "F11", 111: "F12", 105: "F13",
        107: "F14", 113: "F15", 106: "F16", 64: "F17", 79: "F18", 80: "F19", 90: "F20"]
    private var pendingModifier: UInt16?
    private var fnHeld = false

    static func modifierFlag(for keyCode: UInt16) -> NSEvent.ModifierFlags? {
        switch keyCode {
        case 63: return .function
        case 54, 55: return .command
        case 56, 60: return .shift
        case 58, 61: return .option
        case 59, 62: return .control
        default: return nil
        }
    }

    // Note (Yifei Leng): A lone modifier such as Fn never produces keyDown. Record it on release,
    // so holding modifiers to form a chord is not mistaken for a lone key.
    mutating func flagsChanged(keyCode: UInt16, flags: NSEvent.ModifierFlags) -> Outcome {
        if keyCode == 63 { fnHeld = flags.contains(.function) }
        let held = flags.intersection(Self.chordMask.union(.function))
        if let flag = Self.modifierFlag(for: keyCode), held == flag {
            pendingModifier = keyCode
        } else if held.isEmpty, let keyCode = pendingModifier {
            pendingModifier = nil
            return .record(keyCode: keyCode, modifiers: 0)
        } else {
            pendingModifier = nil
        }
        return .pending
    }

    mutating func keyDown(keyCode: UInt16, flags: NSEvent.ModifierFlags) -> Outcome {
        pendingModifier = nil
        if keyCode == 53 { return .cancel }
        var chord = flags.intersection(Self.chordMask)
        if fnHeld && flags.contains(.function) { chord.insert(.function) }
        guard !chord.isEmpty || Self.functionKeys[keyCode] != nil else { return .reject }
        return .record(keyCode: keyCode, modifiers: UInt64(chord.rawValue))
    }

    static func label(keyCode: UInt16, modifiers: UInt64) -> String {
        let flags = NSEvent.ModifierFlags(rawValue: UInt(modifiers))
        var label = ""
        if flags.contains(.function) && keyCode != 63 { label += "Fn " }
        if flags.contains(.control) { label += "⌃" }
        if flags.contains(.option) { label += "⌥" }
        if flags.contains(.shift) { label += "⇧" }
        if flags.contains(.command) { label += "⌘" }
        if let name = functionKeys[keyCode] { return label + name }
        let special: [UInt16: String] = [49: L("shortcut.space"), 63: "Fn", 36: "↩", 76: "⌤", 48: "⇥",
            51: "⌫", 117: "⌦", 53: "⎋", 57: "⇪", 71: "⌧", 114: "Help", 115: "↖", 119: "↘",
            116: "⇞", 121: "⇟", 123: "←", 124: "→", 125: "↓", 126: "↑",
            54: "⌘", 55: "⌘", 56: "⇧", 60: "⇧", 58: "⌥", 61: "⌥", 59: "⌃", 62: "⌃"]
        if let name = special[keyCode] { return label + name }
        for source in [TISCopyCurrentKeyboardLayoutInputSource().takeRetainedValue(),
                       TISCopyCurrentASCIICapableKeyboardLayoutInputSource().takeRetainedValue()] {
            guard let property = TISGetInputSourceProperty(source, kTISPropertyUnicodeKeyLayoutData) else { continue }
            let data = Unmanaged<CFData>.fromOpaque(property).takeUnretainedValue()
            guard let bytes = CFDataGetBytePtr(data) else { continue }
            let layout = UnsafeRawPointer(bytes).assumingMemoryBound(to: UCKeyboardLayout.self)
            var deadKeyState: UInt32 = 0
            var length = 0
            var characters = [UniChar](repeating: 0, count: 8)
            let status = UCKeyTranslate(layout, keyCode, UInt16(kUCKeyActionDisplay), 0, UInt32(LMGetKbdType()),
                                        OptionBits(kUCKeyTranslateNoDeadKeysMask), &deadKeyState,
                                        characters.count, &length, &characters)
            if status == noErr, length > 0 {
                let name = String(utf16CodeUnits: characters, count: length)
                    .trimmingCharacters(in: .whitespacesAndNewlines.union(.controlCharacters))
                if !name.isEmpty { return label + name.uppercased() }
            }
        }
        return label + L("shortcut.key", String(keyCode))
    }
}
