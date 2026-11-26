import CoreGraphics
import Foundation

struct KeySpec: Equatable {
    let name: String
    let code: CGKeyCode
    /// note (luojiaxuan): The flag a modifier key sets (with its left/right device bit); nil for other keys.
    let modifierFlags: UInt64?
}

enum KeyNames {
    // note (luojiaxuan): Generic modifier masks (CGEventFlags) plus the NX_DEVICE* bit that says which side was pressed.
    private static let modifiers: [String: (CGKeyCode, UInt64)] = [
        "fn": (63, CGEventFlags.maskSecondaryFn.rawValue),
        "left_command": (55, CGEventFlags.maskCommand.rawValue | 0x08),
        "right_command": (54, CGEventFlags.maskCommand.rawValue | 0x10),
        "left_shift": (56, CGEventFlags.maskShift.rawValue | 0x02),
        "right_shift": (60, CGEventFlags.maskShift.rawValue | 0x04),
        "left_option": (58, CGEventFlags.maskAlternate.rawValue | 0x20),
        "right_option": (61, CGEventFlags.maskAlternate.rawValue | 0x40),
        "left_control": (59, CGEventFlags.maskControl.rawValue | 0x01),
        "right_control": (62, CGEventFlags.maskControl.rawValue | 0x2000),
    ]
    private static let aliases = ["command": "left_command", "cmd": "left_command", "shift": "left_shift",
                                  "option": "left_option", "alt": "left_option", "control": "left_control",
                                  "ctrl": "left_control"]
    private static let keys: [String: CGKeyCode] = [
        "space": 49, "return": 36, "tab": 48, "escape": 53,
        "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100,
        "f9": 101, "f10": 109, "f11": 103, "f12": 111, "f13": 105, "f14": 107, "f15": 113,
        "f16": 106, "f17": 64, "f18": 79, "f19": 80, "f20": 90,
    ]

    static var names: [String] { (Array(modifiers.keys) + Array(aliases.keys) + Array(keys.keys)).sorted() }

    /// note (luojiaxuan): A key by name (see `names`) or by virtual key code.
    static func parse(_ text: String) throws -> KeySpec {
        let name = aliases[text.lowercased()] ?? text.lowercased()
        if let (code, flags) = modifiers[name] { return KeySpec(name: name, code: code, modifierFlags: flags) }
        if let code = keys[name] { return KeySpec(name: name, code: code, modifierFlags: nil) }
        if let code = CGKeyCode(text) { return KeySpec(name: "keycode \(code)", code: code, modifierFlags: nil) }
        throw ReplayError("unknown key \"\(text)\"; use a virtual key code or one of: \(names.joined(separator: ", "))")
    }
}

/// note (luojiaxuan): The dictation hotkey: `modifiers` are held, in order, around `key`.
struct Hotkey {
    let key: KeySpec
    let modifiers: [KeySpec]

    init(key: String, modifiers: String?) throws {
        self.key = try KeyNames.parse(key)
        self.modifiers = try (modifiers ?? "").split(separator: ",").map { try KeyNames.parse(String($0)) }
        for modifier in self.modifiers where modifier.modifierFlags == nil {
            throw ReplayError("--modifiers takes modifier keys only, not \(modifier.name)")
        }
    }

    var description: String { (modifiers.map(\.name) + [key.name]).joined(separator: "+") }
}

/// note (luojiaxuan): Posts the hotkey as HID-level keyboard events. Modifier keys go out as
/// flagsChanged events, other keys as keyDown/keyUp; every event carries the flags
/// of the modifiers held at that moment.
final class HotkeyPoster {
    private let hotkey: Hotkey
    private let source: CGEventSource
    private let lock = NSLock()
    private var held: [KeySpec] = []

    init(hotkey: Hotkey) throws {
        guard let source = CGEventSource(stateID: .hidSystemState) else { throw ReplayError("cannot create a CGEventSource") }
        self.hotkey = hotkey
        self.source = source
    }

    /// note (luojiaxuan): Presses the modifiers, then the key; returns the post time of the key's down event.
    func down() throws -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        for modifier in hotkey.modifiers { _ = try post(modifier, down: true) }
        return try post(hotkey.key, down: true)
    }

    /// note (luojiaxuan): Releases the key, then the modifiers; returns the post time of the key's up event.
    func up() throws -> UInt64 {
        lock.lock()
        defer { lock.unlock() }
        let t = try post(hotkey.key, down: false)
        for modifier in hotkey.modifiers.reversed() { _ = try post(modifier, down: false) }
        return t
    }

    /// note (luojiaxuan): `presses` taps of the hotkey, each held `holdMs`, `gapMs` apart. Returns the
    /// post times of the first down event, the last down event and the last up event.
    func tap(presses: Int, holdMs: Int, gapMs: Int) throws -> (firstDown: UInt64, lastDown: UInt64, lastUp: UInt64) {
        var firstDown: UInt64 = 0, lastDown: UInt64 = 0, lastUp: UInt64 = 0
        for index in 0..<presses {
            if index > 0 { usleep(useconds_t(gapMs * 1000)) }
            lastDown = try down()
            if index == 0 { firstDown = lastDown }
            usleep(useconds_t(holdMs * 1000))
            lastUp = try up()
        }
        return (firstDown, lastDown, lastUp)
    }

    /// note (luojiaxuan): Releases every key still held. Called when the driver is interrupted.
    func releaseAll() {
        lock.lock()
        defer { lock.unlock() }
        for key in held.reversed() { _ = try? post(key, down: false) }
    }

    private func post(_ key: KeySpec, down: Bool) throws -> UInt64 {
        guard let event = CGEvent(keyboardEventSource: source, virtualKey: key.code, keyDown: down) else {
            throw ReplayError("cannot create a keyboard event for \(key.name)")
        }
        if down {
            held.append(key)
        } else {
            held.removeAll { $0 == key }
        }
        if key.modifierFlags != nil { event.type = .flagsChanged }
        let modifierFlags = held.compactMap(\.modifierFlags).reduce(0, |)
        event.flags = CGEventFlags(rawValue: modifierFlags | CGEventFlags.maskNonCoalesced.rawValue)
        event.post(tap: .cghidEventTap)
        return MachClock.now()
    }
}
