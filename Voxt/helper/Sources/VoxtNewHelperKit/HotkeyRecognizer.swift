import Foundation

// The hotkey recognizer: pure logic, no CoreGraphics. The event tap feeds it
// every keyboard event (flags, key down/up) and mouse button press (and the
// releases of the buttons besides the left and right ones) with a
// timestamp and asks, at once, whether to swallow the event; timers call tick(now:) at
// nextDeadline. Go configures the bindings (hotkey.configure) and gets
// hotkey.event notifications; what a trigger means (start, stop, cancel) is
// Go's business.
//
// Modifier-only chords (fn, fn+⇧, fn+⌃, right ⌘ …) are judged per "chord
// session": from the first modifier pressed to the last one released. The
// chord is every modifier seen during the session, so fn then ⇧ is fn+⇧, never
// fn. A non-modifier key during the session (fn+→, ⌘C, right ⌥+E typing é)
// spoils it, and so does a mouse button (⌥-click, ⌥-drag): the
// modifier was used for something else. Modifier-only chords are never
// swallowed (fn's own events aside, with swallowFn), so a modifier the
// user holds to type or click works as usual: the shortcut just does not
// fire. Sides are told apart by the device bits of the flags: a binding
// for right ⌥ ignores left ⌥, so ⌥-characters typed with the left key are
// untouched. A mouse button binding (the middle button, a side button)
// is judged like a key binding, on the button's press.

/// One modifier key, with its side. fn has none.
public enum ModKey: String, CaseIterable, Codable, Sendable {
    case fn
    case leftShift, rightShift
    case leftControl, rightControl
    case leftOption, rightOption
    case leftCommand, rightCommand

    public var family: ModFamily {
        switch self {
        case .fn: .fn
        case .leftShift, .rightShift: .shift
        case .leftControl, .rightControl: .control
        case .leftOption, .rightOption: .option
        case .leftCommand, .rightCommand: .command
        }
    }

    public var isRight: Bool { [.rightShift, .rightControl, .rightOption, .rightCommand].contains(self) }

    /// The key a flagsChanged event's keycode names (kVK_*).
    public static func forKeyCode(_ k: UInt16) -> ModKey? {
        switch k {
        case 63: .fn
        case 56: .leftShift
        case 60: .rightShift
        case 59: .leftControl
        case 62: .rightControl
        case 58: .leftOption
        case 61: .rightOption
        case 55: .leftCommand
        case 54: .rightCommand
        default: nil
        }
    }
}

public enum ModFamily: String, CaseIterable, Codable, CodingKeyRepresentable, Sendable {
    case fn, shift, control, option, command
}

/// Raw CGEventFlags bits the recognizer reads (IOKit's NX_* device masks).
public enum RawFlags {
    public static let secondaryFn: UInt64 = 0x80_0000
    public static let shift: UInt64 = 0x2_0000
    public static let control: UInt64 = 0x4_0000
    public static let option: UInt64 = 0x8_0000
    public static let command: UInt64 = 0x10_0000
    static let deviceLeftControl: UInt64 = 0x0001
    static let deviceLeftShift: UInt64 = 0x0002
    static let deviceRightShift: UInt64 = 0x0004
    static let deviceLeftCommand: UInt64 = 0x0008
    static let deviceRightCommand: UInt64 = 0x0010
    static let deviceLeftOption: UInt64 = 0x0020
    static let deviceRightOption: UInt64 = 0x0040
    static let deviceRightControl: UInt64 = 0x2000

    /// The modifier keys down according to flags. When a family's flag is set
    /// but no device bit says which side (synthetic events), the left key.
    public static func pressed(_ flags: UInt64) -> Set<ModKey> {
        var s = Set<ModKey>()
        if flags & secondaryFn != 0 { s.insert(.fn) }
        func side(_ family: UInt64, _ l: UInt64, _ r: UInt64, _ lk: ModKey, _ rk: ModKey) {
            let hasL = flags & l != 0, hasR = flags & r != 0
            if hasL { s.insert(lk) }
            if hasR { s.insert(rk) }
            if flags & family != 0, !hasL, !hasR { s.insert(lk) }
        }
        side(shift, deviceLeftShift, deviceRightShift, .leftShift, .rightShift)
        side(control, deviceLeftControl, deviceRightControl, .leftControl, .rightControl)
        side(option, deviceLeftOption, deviceRightOption, .leftOption, .rightOption)
        side(command, deviceLeftCommand, deviceRightCommand, .leftCommand, .rightCommand)
        return s
    }
}

/// Virtual key codes the recognizer knows (Carbon kVK_*).
public enum KeyCode {
    public static let function: UInt16 = 63
    public static let escape: UInt16 = 53
    public static let capsLock: UInt16 = 57
    /// The 🌐 key's key down/up after fn (seen on macOS 26 with an Apple keyboard).
    public static let globe: UInt16 = 179
    public static let modifiers: Set<UInt16> = [54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
    /// Keys whose events macOS marks with fn's flag whether fn is held or
    /// not (a laptop reaches them through fn): F1–F20, Help, Home, Page Up,
    /// Forward Delete, End, Page Down and the arrows.
    public static let fnFlagged: Set<UInt16> = [
        122, 120, 99, 118, 96, 97, 98, 100, 101, 109, 103, 111, 105, 107, 113, 106, 64, 79, 80, 90,
        114, 115, 116, 117, 119, 121, 123, 124, 125, 126,
    ]
}

public enum Behavior: String, Codable, Sendable {
    case tap, doubleTap, longPress
}

/// What one modifier family must be in a chord.
public enum SideRule: String, Codable, Sendable {
    case any, left, right
}

/// What a key or mouse binding presses besides its modifiers.
enum Press: Hashable {
    case key(UInt16)
    case button(Int)
}

/// A binding: a modifier-only chord (key and button nil), a key with
/// modifiers, or a mouse button with modifiers.
public struct Binding: Codable, Equatable, Sendable {
    public var id: String
    /// Modifier families required, each with a side rule; others must be up.
    public var modifiers: [ModFamily: SideRule]
    /// The non-modifier key (kVK), or nil.
    public var key: UInt16?
    /// The mouse button (the events' button number: 2 is the middle one,
    /// 3 and up the side buttons), or nil.
    public var button: Int?
    public var behavior: Behavior
    /// Swallow the key's or button's events when it matches (key and mouse
    /// bindings only; the default is true). Modifier-only chords are never
    /// swallowed except fn with Config.swallowFn.
    public var swallow: Bool?

    public init(id: String, modifiers: [ModFamily: SideRule], key: UInt16? = nil, button: Int? = nil, behavior: Behavior, swallow: Bool? = nil) {
        self.id = id
        self.modifiers = modifiers
        self.key = key
        self.button = button
        self.behavior = behavior
        self.swallow = swallow
    }

    var press: Press? { key.map(Press.key) ?? button.map(Press.button) }

    /// Whether a chord (modifier keys seen) is exactly this binding's.
    public func matches(chord: Set<ModKey>) -> Bool {
        for family in ModFamily.allCases {
            let keys = chord.filter { $0.family == family }
            guard let rule = modifiers[family] else {
                if !keys.isEmpty { return false }
                continue
            }
            if keys.isEmpty { return false }
            switch rule {
            case .any: break
            case .left: if keys.contains(where: { $0.isRight }) { return false }
            case .right: if keys.contains(where: { !$0.isRight }) { return false }
            }
        }
        return true
    }

    /// Whether a key or button press is this binding's: its key or button,
    /// with exactly its modifier keys (RawFlags.pressed), sides and fn
    /// included. For the keys in KeyCode.fnFlagged fn's flag says nothing,
    /// so fn is not judged there: F5 and Fn+F5 are one shortcut.
    func matches(_ p: Press, pressed: Set<ModKey>) -> Bool {
        guard press == p else { return false }
        guard case .key(let keyCode) = p, KeyCode.fnFlagged.contains(keyCode) else { return matches(chord: pressed) }
        var withoutFn = self
        withoutFn.modifiers[.fn] = nil
        return withoutFn.matches(chord: pressed.subtracting([.fn]))
    }

    /// More modifiers is more specific (fn+⇧ before fn).
    var specificity: Int { modifiers.count * 2 + (modifiers.values.contains { $0 != .any } ? 1 : 0) }
}

public struct HotkeyConfig: Codable, Equatable, Sendable {
    public var bindings: [Binding]
    /// A second press starting within this long of the first release makes
    /// a double tap (judged at its release; real double taps on mac-air had
    /// 196–450 ms between releases, so the release alone is too strict).
    public var doubleTapMs: Double
    /// Held this long, a longPress binding fires down.
    public var longPressMs: Double
    /// A modifier-only press held longer than this is no tap.
    public var tapMaxMs: Double
    /// Swallow fn's own flagsChanged events while an fn chord is bound, so
    /// the 🌐 key's system action (emoji, input source) does not run.
    public var swallowFn: Bool
    /// While a session records or the answer card shows, a tap fires at
    /// once (no wait for a double tap) and the cancel shortcut is reported
    /// (and swallowed).
    public var active: Bool
    /// The shortcut that cancels while active (id "cancel", a tap). Nil:
    /// nothing cancels. Never judged while inactive.
    public var cancel: Binding?

    public init(bindings: [Binding], doubleTapMs: Double = 500, longPressMs: Double = 300, tapMaxMs: Double = 1500,
                swallowFn: Bool = false, active: Bool = false, cancel: Binding? = nil) {
        self.bindings = bindings
        self.doubleTapMs = doubleTapMs
        self.longPressMs = longPressMs
        self.tapMaxMs = tapMaxMs
        self.swallowFn = swallowFn
        self.active = active
        self.cancel = cancel
    }
}

public enum Phase: String, Codable, Sendable {
    /// tap or doubleTap fired; a key binding's down for tap.
    case trigger
    /// longPress: held long enough.
    case down
    /// longPress: released after down.
    case up
}

public struct HotkeyEvent: Codable, Equatable, Sendable {
    public var binding: String
    public var phase: Phase
    public var behavior: Behavior
    /// Milliseconds the chord was held (release − press), when known.
    public var heldMs: Double?
    /// Timestamp of the event that decided it (ms, monotonic).
    public var t: Double
}

/// A keyboard event, as the tap sees it.
public enum KeyInput: Sendable {
    case flagsChanged(keyCode: UInt16, flags: UInt64, t: Double)
    case keyDown(keyCode: UInt16, flags: UInt64, isRepeat: Bool, t: Double)
    case keyUp(keyCode: UInt16, flags: UInt64, t: Double)
    /// A mouse button pressed (0 left, 1 right, 2 middle, 3… side
    /// buttons): it spoils a chord, and may be a mouse binding's.
    case mouseDown(button: Int, flags: UInt64, t: Double)
    case mouseUp(button: Int, t: Double)

    var t: Double {
        switch self {
        case .flagsChanged(_, _, let t), .keyDown(_, _, _, let t), .keyUp(_, _, let t), .mouseDown(_, _, let t), .mouseUp(_, let t): t
        }
    }
}

public struct HotkeyRecognizer: Sendable {
    public private(set) var config: HotkeyConfig
    public init(config: HotkeyConfig) { self.config = config }

    /// Modifier events found missing so far (diagnostics).
    public private(set) var missedEvents = 0

    // Chord session.
    private var pressed = Set<ModKey>()
    private var chordStart: Double?
    private var chordMax = Set<ModKey>()
    private var spoiled = false
    private var longPressFired: Binding?
    private var longPressDeadline: (binding: Binding, at: Double)?
    // Double tap.
    private var doubleArm: (chord: Set<ModKey>, until: Double)?
    private var pendingTap: (binding: Binding, at: Double, heldMs: Double)?
    // Key and mouse bindings.
    private var swallowedKeys = Set<Press>()
    private var keyDownAt: [Press: Double] = [:]
    private var keyLongFired: [Press: Binding] = [:]
    private var keyDoubleArm: (key: Press, until: Double)?

    /// Sets the config. Changing only the timings or "active" keeps a
    /// chord in progress (Go flips active as a session starts, while the
    /// key that started it may still be down).
    public mutating func configure(_ c: HotkeyConfig) {
        let same = c.bindings == config.bindings
        config = c
        if !same { reset() }
    }

    /// Forgets transient state (the tap was disabled, the config changed).
    public mutating func reset() {
        chordStart = nil
        chordMax = []
        spoiled = false
        longPressFired = nil
        longPressDeadline = nil
        doubleArm = nil
        pendingTap = nil
        keyDoubleArm = nil
        keyLongFired = [:]
        keyDownAt = [:]
    }

    /// When tick(now:) must run next, if at all.
    public var nextDeadline: Double? {
        // A deferred tap waits while a chord is down: it may be the second tap.
        [chordStart == nil ? pendingTap?.at : nil, longPressDeadline?.at].compactMap { $0 }.min()
    }

    /// Feeds one event; returns what fired and whether to swallow it.
    public mutating func handle(_ input: KeyInput) -> (events: [HotkeyEvent], swallow: Bool) {
        var out = tick(now: input.t)
        var swallow = false
        switch input {
        case .flagsChanged(let keyCode, let flags, let t):
            if keyCode == KeyCode.capsLock { break }
            swallow = handleFlags(keyCode: keyCode, flags: flags, t: t, &out)
        case .keyDown(let keyCode, _, _, _) where keyCode == KeyCode.globe,
             .keyUp(let keyCode, _, _) where keyCode == KeyCode.globe:
            // The 🌐 key's own key events, sent as fn is released; macOS runs
            // the 🌐 action (emoji, input source) on them. Swallowed with
            // swallowFn; they never spoil a chord.
            swallow = config.swallowFn && chordBindings.contains { $0.modifiers[.fn] != nil }
        case .keyDown(let keyCode, let flags, let isRepeat, let t):
            if !pressed.isEmpty { spoiled = true; longPressDeadline = nil }
            swallow = handleKeyDown(.key(keyCode), flags: flags, isRepeat: isRepeat, t: t, &out)
        case .keyUp(let keyCode, _, let t):
            swallow = handleKeyUp(.key(keyCode), t: t, &out)
        case .mouseDown(let button, let flags, let t):
            if !pressed.isEmpty { spoiled = true; longPressDeadline = nil }
            swallow = handleKeyDown(.button(button), flags: flags, isRepeat: false, t: t, &out)
        case .mouseUp(let button, let t):
            swallow = handleKeyUp(.button(button), t: t, &out)
        }
        return (out, swallow)
    }

    /// Fires what is due at now (deferred taps, long presses).
    public mutating func tick(now: Double) -> [HotkeyEvent] {
        var out: [HotkeyEvent] = []
        if let p = pendingTap, now >= p.at, chordStart == nil {
            out.append(HotkeyEvent(binding: p.binding.id, phase: .trigger, behavior: .tap, heldMs: p.heldMs, t: p.at))
            pendingTap = nil
            doubleArm = nil
        }
        if let d = longPressDeadline, now >= d.at {
            longPressDeadline = nil
            if !spoiled, pressed == chordMax, d.binding.matches(chord: pressed) {
                longPressFired = d.binding
                out.append(HotkeyEvent(binding: d.binding.id, phase: .down, behavior: .longPress, heldMs: d.at - (chordStart ?? d.at), t: d.at))
            }
        }
        return out
    }

    private var chordBindings: [Binding] {
        config.bindings.filter { $0.press == nil }.sorted { $0.specificity > $1.specificity }
    }

    private mutating func handleFlags(keyCode: UInt16, flags: UInt64, t: Double, _ out: inout [HotkeyEvent]) -> Bool {
        var now = RawFlags.pressed(flags)
        // fn's own event decides fn (some keyboards jitter the flag on others).
        if keyCode == KeyCode.function {
            if flags & RawFlags.secondaryFn != 0 { now.insert(.fn) } else { now.remove(.fn) }
        }
        let swallowFn = keyCode == KeyCode.function && config.swallowFn &&
            chordBindings.contains { $0.modifiers[.fn] != nil }
        var was = pressed
        // A modifier's flagsChanged always toggles that key. If its state
        // did not change, we missed an event (seen: the system eats fn's
        // release after fn+⌃): the chord in progress is unreliable, drop it
        // without firing and start over from this event.
        if let k = ModKey.forKeyCode(keyCode), was.contains(k) == now.contains(k) {
            missedEvents += 1
            chordStart = nil
            chordMax = []
            spoiled = false
            longPressFired = nil
            longPressDeadline = nil
            doubleArm = nil
            pendingTap = nil
            was = now.contains(k) ? now.subtracting([k]) : now
        }
        pressed = now
        if was.isEmpty, !now.isEmpty {
            chordStart = t
            chordMax = now
            spoiled = false
            longPressFired = nil
            if let arm = doubleArm, t > arm.until { doubleArm = nil }
        } else {
            chordMax.formUnion(now)
        }

        if now.isEmpty, !was.isEmpty {
            release(t: t, &out)
        } else if !now.isEmpty {
            // A long press waits for the chord to settle.
            longPressDeadline = nil
            if !spoiled, now == chordMax, longPressFired == nil,
               let b = chordBindings.first(where: { $0.behavior == .longPress && $0.matches(chord: now) }) {
                longPressDeadline = (b, t + config.longPressMs)
            }
        }
        return swallowFn
    }

    private mutating func release(t: Double, _ out: inout [HotkeyEvent]) {
        defer {
            chordStart = nil
            chordMax = []
            longPressDeadline = nil
            longPressFired = nil
            spoiled = false
        }
        let held = t - (chordStart ?? t)
        if let fired = longPressFired {
            out.append(HotkeyEvent(binding: fired.id, phase: .up, behavior: .longPress, heldMs: held, t: t))
            doubleArm = nil
            return
        }
        guard !spoiled, held <= config.tapMaxMs else {
            doubleArm = nil
            return
        }
        let chord = chordMax
        // A modifier-only cancel chord fires while active, before the
        // features' taps; like every modifier-only chord it is never
        // swallowed.
        if config.active, let c = config.cancel, c.press == nil, c.matches(chord: chord) {
            pendingTap = nil
            doubleArm = nil
            out.append(HotkeyEvent(binding: c.id, phase: .trigger, behavior: .tap, heldMs: held, t: t))
            return
        }
        let matching = chordBindings.filter { $0.matches(chord: chord) }
        let tapB = matching.first { $0.behavior == .tap }
        let dblB = matching.first { $0.behavior == .doubleTap }
        if config.active, let b = tapB ?? dblB {
            pendingTap = nil
            doubleArm = nil
            out.append(HotkeyEvent(binding: b.id, phase: .trigger, behavior: b.behavior, heldMs: held, t: t))
            return
        }
        if let arm = doubleArm, arm.chord == chord, (chordStart ?? t) <= arm.until, let dblB {
            doubleArm = nil
            pendingTap = nil
            out.append(HotkeyEvent(binding: dblB.id, phase: .trigger, behavior: .doubleTap, heldMs: held, t: t))
            return
        }
        // A different chord ends a pending double tap: its tap fires now.
        if let p = pendingTap {
            out.append(HotkeyEvent(binding: p.binding.id, phase: .trigger, behavior: .tap, heldMs: p.heldMs, t: t))
            pendingTap = nil
        }
        doubleArm = nil
        if dblB != nil {
            doubleArm = (chord, t + config.doubleTapMs)
            if let tapB { pendingTap = (tapB, t + config.doubleTapMs, held) }
        } else if let tapB {
            out.append(HotkeyEvent(binding: tapB.id, phase: .trigger, behavior: .tap, heldMs: held, t: t))
        }
    }

    private var keyBindings: [Binding] { config.bindings.filter { $0.press != nil }.sorted { $0.specificity > $1.specificity } }

    /// A key or a mouse button pressed.
    private mutating func handleKeyDown(_ press: Press, flags: UInt64, isRepeat: Bool, t: Double, _ out: inout [HotkeyEvent]) -> Bool {
        let pressed = RawFlags.pressed(flags)
        // The cancel shortcut, only while active: its keys are swallowed
        // then, and reach the app in front at any other time.
        if config.active, let c = config.cancel, c.matches(press, pressed: pressed) {
            if !isRepeat { out.append(HotkeyEvent(binding: c.id, phase: .trigger, behavior: .tap, heldMs: nil, t: t)) }
            swallowedKeys.insert(press)
            return true
        }
        guard let b = keyBindings.first(where: { $0.matches(press, pressed: pressed) }) else {
            if swallowedKeys.contains(press) { return true } // repeats of a swallowed key
            return false
        }
        let swallow = b.swallow ?? true
        if swallow { swallowedKeys.insert(press) }
        if isRepeat { return swallow }
        keyDownAt[press] = t
        switch b.behavior {
        case .tap:
            out.append(HotkeyEvent(binding: b.id, phase: .trigger, behavior: .tap, heldMs: nil, t: t))
        case .longPress:
            keyLongFired[press] = b
            out.append(HotkeyEvent(binding: b.id, phase: .down, behavior: .longPress, heldMs: nil, t: t))
        case .doubleTap:
            if let arm = keyDoubleArm, arm.key == press, t <= arm.until {
                keyDoubleArm = nil
                out.append(HotkeyEvent(binding: b.id, phase: .trigger, behavior: .doubleTap, heldMs: nil, t: t))
            } else {
                keyDoubleArm = (press, t + config.doubleTapMs)
            }
        }
        return swallow
    }

    /// A key or a mouse button released.
    private mutating func handleKeyUp(_ press: Press, t: Double, _ out: inout [HotkeyEvent]) -> Bool {
        if let b = keyLongFired.removeValue(forKey: press) {
            out.append(HotkeyEvent(binding: b.id, phase: .up, behavior: .longPress, heldMs: t - (keyDownAt[press] ?? t), t: t))
        }
        keyDownAt[press] = nil
        // A key up is swallowed only when its key down was.
        return swallowedKeys.remove(press) != nil
    }
}
