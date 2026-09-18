// SPDX-License-Identifier: Apache-2.0
import AppKit
import Testing
@testable import OmniTyper

struct ShortcutCaptureTests {
    @Test func loneModifierIsRecordedOnRelease() {
        for (keyCode, flag): (UInt16, NSEvent.ModifierFlags) in [(63, .function), (61, .option), (54, .command)] {
            var capture = ShortcutCapture()
            #expect(capture.flagsChanged(keyCode: keyCode, flags: flag) == .pending)
            #expect(capture.flagsChanged(keyCode: keyCode, flags: []) == .record(keyCode: keyCode, modifiers: 0))
        }
    }

    @Test func chordIsNotMistakenForALoneModifier() {
        var capture = ShortcutCapture()
        #expect(capture.flagsChanged(keyCode: 59, flags: .control) == .pending)
        #expect(capture.flagsChanged(keyCode: 58, flags: [.control, .option]) == .pending)
        let chord = NSEvent.ModifierFlags([.control, .option])
        #expect(capture.keyDown(keyCode: 49, flags: chord) == .record(keyCode: 49, modifiers: UInt64(chord.rawValue)))

        // Note (Yifei Leng): Releasing the chord's modifiers afterwards must not record a lone key.
        var released = ShortcutCapture()
        _ = released.flagsChanged(keyCode: 59, flags: .control)
        _ = released.flagsChanged(keyCode: 58, flags: [.control, .option])
        #expect(released.flagsChanged(keyCode: 58, flags: .control) == .pending)
        #expect(released.flagsChanged(keyCode: 59, flags: []) == .pending)
    }

    @Test func keyDownClearsAPendingModifier() {
        var capture = ShortcutCapture()
        _ = capture.flagsChanged(keyCode: 63, flags: .function)
        #expect(capture.keyDown(keyCode: 0, flags: .function) == .record(keyCode: 0, modifiers: UInt64(NSEvent.ModifierFlags.function.rawValue)))
        #expect(capture.flagsChanged(keyCode: 63, flags: []) == .pending)
    }

    @Test func fnChordsPreserveOnlyAnExplicitlyHeldFnKey() {
        var capture = ShortcutCapture()
        _ = capture.flagsChanged(keyCode: 63, flags: .function)
        _ = capture.flagsChanged(keyCode: 59, flags: [.function, .control])
        #expect(capture.keyDown(keyCode: 1, flags: [.function, .control])
            == .record(keyCode: 1, modifiers: UInt64(NSEvent.ModifierFlags([.function, .control]).rawValue)))
        #expect(capture.flagsChanged(keyCode: 63, flags: .control) == .pending)
        #expect(capture.flagsChanged(keyCode: 59, flags: []) == .pending)
        #expect(capture.keyDown(keyCode: 123, flags: [.command, .function])
            == .record(keyCode: 123, modifiers: UInt64(NSEvent.ModifierFlags.command.rawValue)))
        for code in ShortcutCapture.functionKeys.keys {
            #expect(capture.keyDown(keyCode: code, flags: .function) == .record(keyCode: code, modifiers: 0))
        }
    }

    @Test @MainActor func shortcutLabelsNameKeysAndAllModifiers() {
        let flags = UInt64(NSEvent.ModifierFlags([.control, .option]).rawValue)
        #expect(ShortcutCapture.label(keyCode: 123, modifiers: flags) == "⌃⌥←")
        #expect(ShortcutCapture.label(keyCode: 63, modifiers: 0) == "Fn")
        #expect(ShortcutCapture.label(keyCode: 122, modifiers: UInt64(NSEvent.ModifierFlags.function.rawValue)) == "Fn F1")
        #expect(ShortcutCapture.keyName(111) == "F12")
        #expect(ShortcutCapture.keyName(90) == "F20")
        for code: UInt16 in [1, 7] {
            #expect(ShortcutCapture.keyName(code) != L("shortcut.key", String(code)))
            #expect(!ShortcutCapture.keyName(code).isEmpty)
        }
    }

    @Test func existingKeyDownRulesAreUnchanged() {
        var capture = ShortcutCapture()
        #expect(capture.keyDown(keyCode: 53, flags: []) == .cancel)
        #expect(capture.keyDown(keyCode: 0, flags: []) == .reject)
        #expect(capture.keyDown(keyCode: 96, flags: []) == .record(keyCode: 96, modifiers: 0))
        // Note (Yifei Leng): Fn accompanies arrow and function keys; it must not leak into stored modifiers.
        #expect(capture.keyDown(keyCode: 0, flags: [.command, .function])
            == .record(keyCode: 0, modifiers: UInt64(NSEvent.ModifierFlags.command.rawValue)))
    }
}
