// The hotkey tap's events (VoxtNewHelper's Hotkey.swift): which ones it asks
// macOS for, and what HotkeyRecognizer gets from each.

import CoreGraphics

public enum HotkeyTapEvents {
    /// Keys, modifiers and mouse presses (not moves or scrolls: too many):
    /// a press spoils a modifier chord held for an ⌥-click or ⌥-drag; the
    /// other buttons' presses and releases fire mouse bindings.
    public static let mask: CGEventMask = [CGEventType.keyDown, .keyUp, .flagsChanged, .leftMouseDown, .rightMouseDown,
                                           .otherMouseDown, .otherMouseUp].reduce(0) { $0 | (1 << $1.rawValue) }

    /// The recognizer's input for an event of the mask, at time t; nil for
    /// any other event.
    public static func input(_ type: CGEventType, _ event: CGEvent, t: Double) -> KeyInput? {
        let keyCode = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
        let button = Int(event.getIntegerValueField(.mouseEventButtonNumber))
        let flags = event.flags.rawValue
        switch type {
        case .flagsChanged: return .flagsChanged(keyCode: keyCode, flags: flags, t: t)
        case .keyDown: return .keyDown(keyCode: keyCode, flags: flags, isRepeat: event.getIntegerValueField(.keyboardEventAutorepeat) != 0, t: t)
        case .keyUp: return .keyUp(keyCode: keyCode, flags: flags, t: t)
        case .leftMouseDown, .rightMouseDown, .otherMouseDown: return .mouseDown(button: button, flags: flags, t: t)
        case .otherMouseUp: return .mouseUp(button: button, t: t)
        default: return nil
        }
    }
}
