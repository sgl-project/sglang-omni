import AppKit
import Carbon
import IOKit.hidsystem

nonisolated extension HotkeyPreference {
    static func displayString(for hotkey: Hotkey, distinguishModifierSides: Bool) -> String {
        let symbols = modifierSymbols(
            for: hotkey.modifiers,
            sidedModifiers: distinguishModifierSides ? hotkey.sidedModifiers : []
        )
        if case .keyboard(let keyCode) = hotkey.input, keyCode == modifierOnlyKeyCode {
            return symbols.isEmpty ? AppLocalization.localizedString("Unassigned") : symbols
        }
        let key = inputDisplayString(hotkey.input)
        return symbols.isEmpty ? key : "\(symbols) \(key)"
    }

    static func modifierSymbols(
        for modifiers: NSEvent.ModifierFlags,
        sidedModifiers: SidedModifierFlags = []
    ) -> String {
        let usesSides = !sidedModifiers.isEmpty
        var parts: [String] = []
        if modifiers.contains(.control) {
            parts.append(usesSides ? sidedModifierLabel(primary: .leftControl, secondary: .rightControl, sidedModifiers: sidedModifiers, fallback: "Control") : "⌃")
        }
        if modifiers.contains(.option) {
            parts.append(usesSides ? sidedModifierLabel(primary: .leftOption, secondary: .rightOption, sidedModifiers: sidedModifiers, fallback: "Option") : "⌥")
        }
        if modifiers.contains(.shift) {
            parts.append(usesSides ? sidedModifierLabel(primary: .leftShift, secondary: .rightShift, sidedModifiers: sidedModifiers, fallback: "Shift") : "⇧")
        }
        if modifiers.contains(.command) {
            parts.append(usesSides ? sidedModifierLabel(primary: .leftCommand, secondary: .rightCommand, sidedModifiers: sidedModifiers, fallback: "Command") : "⌘")
        }
        if modifiers.contains(.function) {
            parts.append("fn")
        }
        return parts.joined(separator: usesSides ? " + " : "")
    }

    static func keyCodeDisplayString(_ keyCode: UInt16) -> String {
        switch Int(keyCode) {
        case kVK_Space: return "Space"
        case kVK_Return: return "Return"
        case kVK_Escape: return "Esc"
        case kVK_Delete: return "Delete"
        case kVK_Tab: return "Tab"
        case kVK_LeftArrow: return "←"
        case kVK_RightArrow: return "→"
        case kVK_UpArrow: return "↑"
        case kVK_DownArrow: return "↓"
        default:
            break
        }

        guard Thread.isMainThread else {
            return AppLocalization.format("Key %d", Int(keyCode))
        }

        if let translated = translateKeyCode(keyCode), !translated.isEmpty {
            return translated.uppercased()
        }
        return AppLocalization.format("Key %d", Int(keyCode))
    }

    static func inputDisplayString(_ input: Hotkey.Input) -> String {
        switch input {
        case .keyboard(let keyCode):
            return keyCodeDisplayString(keyCode)
        case .mouseButton(let buttonNumber):
            return mouseButtonDisplayString(buttonNumber)
        }
    }

    static func mouseButtonDisplayString(_ buttonNumber: Int) -> String {
        switch buttonNumber {
        case middleMouseButtonNumber:
            return AppLocalization.localizedString("Mouse Middle Button")
        default:
            return AppLocalization.format("Mouse Button %d", buttonNumber)
        }
    }

    private static func sidedModifierLabel(
        primary: SidedModifierFlags,
        secondary: SidedModifierFlags,
        sidedModifiers: SidedModifierFlags,
        fallback: String
    ) -> String {
        if sidedModifiers.contains(primary), sidedModifiers.contains(secondary) {
            return localizedModifierName(fallback)
        }
        if sidedModifiers.contains(primary) { return AppLocalization.format("Left %@", localizedModifierName(fallback)) }
        if sidedModifiers.contains(secondary) { return AppLocalization.format("Right %@", localizedModifierName(fallback)) }
        return localizedModifierName(fallback)
    }

    private static func localizedModifierName(_ fallback: String) -> String {
        AppLocalization.localizedString(fallback)
    }

    private static func translateKeyCode(_ keyCode: UInt16) -> String? {
        guard let source = TISCopyCurrentKeyboardLayoutInputSource()?.takeRetainedValue(),
              let layoutData = TISGetInputSourceProperty(source, kTISPropertyUnicodeKeyLayoutData)
        else {
            return nil
        }

        let data = unsafeBitCast(layoutData, to: CFData.self)
        var deadKeyState: UInt32 = 0
        var length = 0
        var chars = [UniChar](repeating: 0, count: 4)

        let status: OSStatus = (data as Data).withUnsafeBytes { rawBuffer in
            guard let base = rawBuffer.bindMemory(to: UCKeyboardLayout.self).baseAddress else {
                return OSStatus(kUCKeyTranslateNoDeadKeysBit)
            }

            return UCKeyTranslate(
                base,
                keyCode,
                UInt16(kUCKeyActionDisplay),
                0,
                UInt32(LMGetKbdType()),
                OptionBits(kUCKeyTranslateNoDeadKeysBit),
                &deadKeyState,
                chars.count,
                &length,
                &chars
            )
        }

        guard status == noErr else { return nil }
        return String(utf16CodeUnits: chars, count: length)
    }
}
