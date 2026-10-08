import AppKit
import Carbon
import IOKit.hidsystem

nonisolated struct SidedModifierFlags: OptionSet, Equatable {
    let rawValue: Int

    static let leftShift = SidedModifierFlags(rawValue: 1 << 0)
    static let rightShift = SidedModifierFlags(rawValue: 1 << 1)
    static let leftControl = SidedModifierFlags(rawValue: 1 << 2)
    static let rightControl = SidedModifierFlags(rawValue: 1 << 3)
    static let leftOption = SidedModifierFlags(rawValue: 1 << 4)
    static let rightOption = SidedModifierFlags(rawValue: 1 << 5)
    static let leftCommand = SidedModifierFlags(rawValue: 1 << 6)
    static let rightCommand = SidedModifierFlags(rawValue: 1 << 7)

    static let allShift: SidedModifierFlags = [.leftShift, .rightShift]
    static let allControl: SidedModifierFlags = [.leftControl, .rightControl]
    static let allOption: SidedModifierFlags = [.leftOption, .rightOption]
    static let allCommand: SidedModifierFlags = [.leftCommand, .rightCommand]

    static func toggled(from current: SidedModifierFlags, keyCode: UInt16) -> SidedModifierFlags {
        guard let flag = sidedFlag(for: keyCode) else { return current }
        if current.contains(flag) {
            return current.subtracting(flag)
        }
        return current.union(flag)
    }

    static func updating(from current: SidedModifierFlags, keyCode: UInt16, isPressed: Bool) -> SidedModifierFlags {
        guard let flag = sidedFlag(for: keyCode) else { return current }
        if isPressed {
            return current.union(flag)
        }
        return current.subtracting(flag)
    }

    func filtered(by modifiers: NSEvent.ModifierFlags) -> SidedModifierFlags {
        var filtered: SidedModifierFlags = []
        if modifiers.contains(.shift) {
            filtered.formUnion(intersection(.allShift))
        }
        if modifiers.contains(.control) {
            filtered.formUnion(intersection(.allControl))
        }
        if modifiers.contains(.option) {
            filtered.formUnion(intersection(.allOption))
        }
        if modifiers.contains(.command) {
            filtered.formUnion(intersection(.allCommand))
        }
        return filtered
    }

    func matches(requiredModifiers modifiers: NSEvent.ModifierFlags) -> Bool {
        if modifiers.contains(.shift), isDisjoint(with: .allShift) { return false }
        if modifiers.contains(.control), isDisjoint(with: .allControl) { return false }
        if modifiers.contains(.option), isDisjoint(with: .allOption) { return false }
        if modifiers.contains(.command), isDisjoint(with: .allCommand) { return false }
        return true
    }

    static func sidedFlag(for keyCode: UInt16) -> SidedModifierFlags? {
        switch Int(keyCode) {
        case kVK_Shift:
            return .leftShift
        case kVK_RightShift:
            return .rightShift
        case kVK_Control:
            return .leftControl
        case kVK_RightControl:
            return .rightControl
        case kVK_Option:
            return .leftOption
        case kVK_RightOption:
            return .rightOption
        case kVK_Command:
            return .leftCommand
        case kVK_RightCommand:
            return .rightCommand
        default:
            return nil
        }
    }

    static func fromModifierKeyCode(_ keyCode: UInt16) -> (modifiers: NSEvent.ModifierFlags, sided: SidedModifierFlags)? {
        switch Int(keyCode) {
        case kVK_Shift:
            return ([.shift], .leftShift)
        case kVK_RightShift:
            return ([.shift], .rightShift)
        case kVK_Control:
            return ([.control], .leftControl)
        case kVK_RightControl:
            return ([.control], .rightControl)
        case kVK_Option:
            return ([.option], .leftOption)
        case kVK_RightOption:
            return ([.option], .rightOption)
        case kVK_Command:
            return ([.command], .leftCommand)
        case kVK_RightCommand:
            return ([.command], .rightCommand)
        case kVK_Function:
            return ([.function], [])
        default:
            return nil
        }
    }

    static func from(eventFlags: CGEventFlags) -> SidedModifierFlags {
        let raw = eventFlags.rawValue
        var sided: SidedModifierFlags = []

        if raw & UInt64(NX_DEVICELSHIFTKEYMASK) != 0 { sided.insert(.leftShift) }
        if raw & UInt64(NX_DEVICERSHIFTKEYMASK) != 0 { sided.insert(.rightShift) }
        if raw & UInt64(NX_DEVICELCTLKEYMASK) != 0 { sided.insert(.leftControl) }
        if raw & UInt64(NX_DEVICERCTLKEYMASK) != 0 { sided.insert(.rightControl) }
        if raw & UInt64(NX_DEVICELALTKEYMASK) != 0 { sided.insert(.leftOption) }
        if raw & UInt64(NX_DEVICERALTKEYMASK) != 0 { sided.insert(.rightOption) }
        if raw & UInt64(NX_DEVICELCMDKEYMASK) != 0 { sided.insert(.leftCommand) }
        if raw & UInt64(NX_DEVICERCMDKEYMASK) != 0 { sided.insert(.rightCommand) }

        return sided
    }

    static func snapshotFromCurrentKeyState(filteredBy modifiers: NSEvent.ModifierFlags) -> SidedModifierFlags {
        var sided: SidedModifierFlags = []

        let keyCodes: [(UInt16, SidedModifierFlags)] = [
            (UInt16(kVK_Shift), .leftShift),
            (UInt16(kVK_RightShift), .rightShift),
            (UInt16(kVK_Control), .leftControl),
            (UInt16(kVK_RightControl), .rightControl),
            (UInt16(kVK_Option), .leftOption),
            (UInt16(kVK_RightOption), .rightOption),
            (UInt16(kVK_Command), .leftCommand),
            (UInt16(kVK_RightCommand), .rightCommand)
        ]

        for (keyCode, flag) in keyCodes {
            if CGEventSource.keyState(.hidSystemState, key: CGKeyCode(keyCode)) {
                sided.insert(flag)
            }
        }

        return sided.filtered(by: modifiers)
    }
}

nonisolated extension NSEvent.ModifierFlags {
    static let hotkeyRelevant: NSEvent.ModifierFlags = [.command, .option, .control, .shift, .function]
}
