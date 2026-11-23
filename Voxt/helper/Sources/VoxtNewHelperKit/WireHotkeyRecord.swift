// Chord strings for hotkey.record: must match internal/hotkey's grammar.

import Foundation

public enum WireHotkeyRecord {
    /// macOS virtual key code → grammar key name (Carbon kVK_*).
    private static let keyNames: [UInt16: String] = [
        0: "A", 1: "S", 2: "D", 3: "F", 4: "H", 5: "G", 6: "Z", 7: "X", 8: "C", 9: "V", 11: "B", 12: "Q", 13: "W",
        14: "E", 15: "R", 16: "Y", 17: "T", 18: "1", 19: "2", 20: "3", 21: "4", 22: "6", 23: "5", 24: "=", 25: "9",
        26: "7", 27: "-", 28: "8", 29: "0", 30: "]", 31: "O", 32: "U", 33: "[", 34: "I", 35: "P", 36: "Return",
        37: "L", 38: "J", 39: "'", 40: "K", 41: ";", 42: "\\", 43: ",", 44: "/", 45: "N", 46: "M", 47: ".",
        48: "Tab", 49: "Space", 50: "`", 51: "Delete", 53: "Escape",
        64: "F17", 79: "F18", 80: "F19", 90: "F20", 96: "F5", 97: "F6", 98: "F7", 99: "F3", 100: "F8", 101: "F9",
        103: "F11", 105: "F13", 106: "F16", 107: "F14", 109: "F10", 111: "F12", 113: "F15",
        115: "Home", 116: "PageUp", 117: "ForwardDelete", 118: "F4", 119: "End", 120: "F2", 121: "PageDown",
        122: "F1", 123: "Left", 124: "Right", 125: "Down", 126: "Up",
    ]

    public static func keyName(keyCode: UInt16) -> String? { keyNames[keyCode] }

    private static let modOrder: [ModKey] = [
        .fn, .leftControl, .rightControl, .leftOption, .rightOption, .leftShift, .rightShift, .leftCommand, .rightCommand,
    ]

    private static func modGrammar(_ m: ModKey) -> String {
        switch m {
        case .fn: "Fn"
        case .leftControl: "LeftControl"
        case .rightControl: "RightControl"
        case .leftOption: "LeftOption"
        case .rightOption: "RightOption"
        case .leftShift: "LeftShift"
        case .rightShift: "RightShift"
        case .leftCommand: "LeftCommand"
        case .rightCommand: "RightCommand"
        }
    }

    /// Builds the grammar string for modifiers plus an optional key or mouse button (events are 0-based; Mouse4 is button 3).
    public static func chordString(mods: Set<ModKey>, keyCode: UInt16? = nil, mouseButton: Int? = nil) -> String? {
        var parts: [String] = []
        for m in modOrder where mods.contains(m) {
            parts.append(modGrammar(m))
        }
        if let b = mouseButton, b >= 2 {
            parts.append("Mouse\(b + 1)")
        } else if let k = keyCode, let name = keyName(keyCode: k), ModKey.forKeyCode(k) == nil {
            parts.append(name)
        } else if parts.isEmpty {
            return nil
        }
        return parts.joined(separator: "+")
    }

    public static func mods(from flags: UInt64) -> Set<ModKey> { RawFlags.pressed(flags) }
}
