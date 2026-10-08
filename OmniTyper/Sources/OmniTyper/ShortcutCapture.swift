// SPDX-License-Identifier: Apache-2.0
import AppKit

/// Decides what a shortcut-recording session does with each key event.
struct ShortcutCapture {
    enum Outcome: Equatable {
        case pending
        case cancel
        case reject
        case record(keyCode: UInt16, modifiers: UInt64)
    }

    private static let chordMask: NSEvent.ModifierFlags = [.command, .control, .option, .shift]
    private static let functionKeys: Set<UInt16> = [96, 97, 98, 99, 100, 101, 109, 111]
    private var pendingModifier: UInt16?

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
        let chord = flags.intersection(Self.chordMask)
        guard !chord.isEmpty || Self.functionKeys.contains(keyCode) else { return .reject }
        return .record(keyCode: keyCode, modifiers: UInt64(chord.rawValue))
    }
}
