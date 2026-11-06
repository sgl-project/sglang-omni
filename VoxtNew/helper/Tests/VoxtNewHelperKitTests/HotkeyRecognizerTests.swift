@testable import VoxtNewHelperKit
import XCTest

// Table-style tests of the recognizer with synthetic flag sequences.
final class HotkeyRecognizerTests: XCTestCase {
    let fn = RawFlags.secondaryFn
    let shift = RawFlags.shift | 0x2 // left shift device bit
    let ctrl = RawFlags.control | 0x1
    let rcmd = RawFlags.command | 0x10
    let lcmd = RawFlags.command | 0x8

    func defaults(_ extra: [Binding] = []) -> HotkeyConfig {
        HotkeyConfig(bindings: [
            Binding(id: "transcribe", modifiers: [.fn: .any], behavior: .tap),
            Binding(id: "translate", modifiers: [.fn: .any, .shift: .any], behavior: .tap),
            Binding(id: "rewrite", modifiers: [.fn: .any, .control: .any], behavior: .tap),
        ] + extra, doubleTapMs: 400, longPressMs: 300, tapMaxMs: 1000)
    }

    /// Feeds (t, keyCode, flags) flag events and ticks to `until`.
    func run(_ c: HotkeyConfig, _ seq: [(Double, UInt16, UInt64)], until: Double? = nil) -> [String] {
        var r = HotkeyRecognizer(config: c)
        var fired: [String] = []
        for (t, k, f) in seq {
            fired += r.handle(.flagsChanged(keyCode: k, flags: f, t: t)).events.map { "\($0.binding):\($0.phase.rawValue)" }
        }
        if let until { fired += r.tick(now: until).map { "\($0.binding):\($0.phase.rawValue)" } }
        return fired
    }

    func testFnTap() {
        XCTAssertEqual(run(defaults(), [(0, 63, fn), (80, 63, 0)]), ["transcribe:trigger"])
    }

    func testFnShiftIsTranslateNotTranscribe() {
        // fn down, shift down, fn up, shift up.
        XCTAssertEqual(run(defaults(), [(0, 63, fn), (40, 56, fn | shift), (120, 63, shift), (150, 56, 0)]), ["translate:trigger"])
    }

    func testShiftFirstThenFn() {
        XCTAssertEqual(run(defaults(), [(0, 56, shift), (30, 63, fn | shift), (100, 63, shift), (110, 56, 0)]), ["translate:trigger"])
    }

    func testFnCtrl() {
        XCTAssertEqual(run(defaults(), [(0, 63, fn), (40, 59, fn | ctrl), (90, 59, fn), (100, 63, 0)]), ["rewrite:trigger"])
    }

    func testFnWithOtherKeyIsSpoiled() {
        var r = HotkeyRecognizer(config: defaults())
        _ = r.handle(.flagsChanged(keyCode: 63, flags: fn, t: 0))
        _ = r.handle(.keyDown(keyCode: 124, flags: fn, isRepeat: false, t: 50)) // fn+→
        _ = r.handle(.keyUp(keyCode: 124, flags: fn, t: 80))
        XCTAssertTrue(r.handle(.flagsChanged(keyCode: 63, flags: 0, t: 100)).events.isEmpty)
    }

    func testLongHoldIsNoTap() {
        XCTAssertEqual(run(defaults(), [(0, 63, fn), (1500, 63, 0)]), [])
    }

    func testDoubleTapDefersTap() {
        let c = defaults([Binding(id: "dbl", modifiers: [.fn: .any], behavior: .doubleTap)])
        // single tap: fires after the window
        XCTAssertEqual(run(c, [(0, 63, fn), (60, 63, 0)], until: 459), [])
        XCTAssertEqual(run(c, [(0, 63, fn), (60, 63, 0)], until: 461), ["transcribe:trigger"])
        // double tap
        XCTAssertEqual(run(c, [(0, 63, fn), (60, 63, 0), (200, 63, fn), (260, 63, 0)], until: 2000), ["dbl:trigger"])
        // second tap too late: two taps
        XCTAssertEqual(run(c, [(0, 63, fn), (60, 63, 0), (600, 63, fn), (660, 63, 0)], until: 2000),
                       ["transcribe:trigger", "transcribe:trigger"])
    }

    /// Real double tap on mac-air (10:45:28): releases 450 ms apart, but the
    /// second press began 303 ms after the first release. It is a double tap,
    /// and the deferred single tap must not fire while the second is down.
    func testDoubleTapJudgedBySecondPress() {
        let c = defaults([Binding(id: "dbl", modifiers: [.fn: .any], behavior: .doubleTap)])
        var r = HotkeyRecognizer(config: c)
        var fired: [String] = []
        _ = r.handle(.flagsChanged(keyCode: 63, flags: fn, t: 0))
        _ = r.handle(.flagsChanged(keyCode: 63, flags: 0, t: 155))
        _ = r.handle(.flagsChanged(keyCode: 63, flags: fn, t: 458))
        XCTAssertNil(r.nextDeadline) // no timer while the chord is down
        fired += r.tick(now: 560).map(\.binding) // past 155+400, second tap still down
        fired += r.handle(.flagsChanged(keyCode: 63, flags: 0, t: 605)).events.map(\.binding)
        fired += r.tick(now: 3000).map(\.binding)
        XCTAssertEqual(fired, ["dbl"])
    }

    func testActiveFiresTapAtOnce() {
        var c = defaults([Binding(id: "dbl", modifiers: [.fn: .any], behavior: .doubleTap)])
        c.active = true
        XCTAssertEqual(run(c, [(0, 63, fn), (60, 63, 0)]), ["transcribe:trigger"])
    }

    func testRightCommandLongPress() {
        let c = HotkeyConfig(bindings: [Binding(id: "ptt", modifiers: [.command: .right], behavior: .longPress)], longPressMs: 300)
        var r = HotkeyRecognizer(config: c)
        XCTAssertTrue(r.handle(.flagsChanged(keyCode: 54, flags: rcmd, t: 0)).events.isEmpty)
        XCTAssertEqual(r.nextDeadline, 300)
        XCTAssertEqual(r.tick(now: 300).map(\.phase), [.down])
        XCTAssertEqual(r.handle(.flagsChanged(keyCode: 54, flags: 0, t: 2000)).events.map(\.phase), [.up])
    }

    func testLeftCommandIsNotRightCommand() {
        let c = HotkeyConfig(bindings: [Binding(id: "rc", modifiers: [.command: .right], behavior: .tap)])
        XCTAssertEqual(run(c, [(0, 55, lcmd), (60, 55, 0)]), [])
        XCTAssertEqual(run(c, [(0, 54, rcmd), (60, 54, 0)]), ["rc:trigger"])
    }

    func testLongPressWaitsForMoreSpecific() {
        let c = HotkeyConfig(bindings: [
            Binding(id: "t", modifiers: [.fn: .any], behavior: .longPress),
            Binding(id: "tr", modifiers: [.fn: .any, .shift: .any], behavior: .tap),
        ], longPressMs: 300)
        var r = HotkeyRecognizer(config: c)
        _ = r.handle(.flagsChanged(keyCode: 63, flags: fn, t: 0))
        _ = r.handle(.flagsChanged(keyCode: 56, flags: fn | shift, t: 100))
        XCTAssertTrue(r.tick(now: 400).isEmpty)
        _ = r.handle(.flagsChanged(keyCode: 56, flags: fn, t: 450))
        XCTAssertEqual(r.handle(.flagsChanged(keyCode: 63, flags: 0, t: 500)).events.map(\.binding), ["tr"])
    }

    func testKeyBindingSwallowsDownAndUp() {
        let c = HotkeyConfig(bindings: [Binding(id: "paste", modifiers: [.control: .any, .command: .any], key: 9, behavior: .tap)])
        var r = HotkeyRecognizer(config: c)
        let d = r.handle(.keyDown(keyCode: 9, flags: RawFlags.control | RawFlags.command, isRepeat: false, t: 0))
        XCTAssertTrue(d.swallow)
        XCTAssertEqual(d.events.map(\.binding), ["paste"])
        XCTAssertTrue(r.handle(.keyUp(keyCode: 9, flags: RawFlags.control | RawFlags.command, t: 50)).swallow)
        // plain ⌘V passes
        XCTAssertFalse(r.handle(.keyDown(keyCode: 9, flags: RawFlags.command, isRepeat: false, t: 100)).swallow)
        XCTAssertFalse(r.handle(.keyUp(keyCode: 9, flags: RawFlags.command, t: 120)).swallow)
    }

    /// A key shortcut fires only on its own modifier keys, fn and the side
    /// included. It used to compare families without fn and without sides,
    /// so an Fn+Space shortcut fired on a bare Space (and swallowed it) and
    /// a right ⌥+Space one fired on left ⌥+Space. macOS sets fn's flag on
    /// every F-key, arrow and navigation key event, fn held or not: for
    /// those keys fn is not judged, so F5 and Fn+F5 both still fire.
    func testKeyShortcutMatchesFnAndSide() {
        let space: UInt16 = 49, f5: UInt16 = 96, left: UInt16 = 123
        let rOpt = RawFlags.option | 0x40, lOpt = RawFlags.option | 0x20
        let cases: [(name: String, binding: Binding, key: UInt16, flags: UInt64, fires: Bool)] = [
            ("fn+space on space", Binding(id: "b", modifiers: [.fn: .any], key: space, behavior: .tap), space, 0, false),
            ("fn+space on fn+space", Binding(id: "b", modifiers: [.fn: .any], key: space, behavior: .tap), space, fn, true),
            ("space on fn+space", Binding(id: "b", modifiers: [:], key: space, behavior: .tap), space, fn, false),
            ("right⌥+space on left⌥+space", Binding(id: "b", modifiers: [.option: .right], key: space, behavior: .tap), space, lOpt, false),
            ("right⌥+space on right⌥+space", Binding(id: "b", modifiers: [.option: .right], key: space, behavior: .tap), space, rOpt, true),
            ("⌥+space on left⌥+space", Binding(id: "b", modifiers: [.option: .any], key: space, behavior: .tap), space, lOpt, true),
            ("f5 with fn's flag", Binding(id: "b", modifiers: [:], key: f5, behavior: .tap), f5, fn, true),
            ("f5 without fn's flag", Binding(id: "b", modifiers: [:], key: f5, behavior: .tap), f5, 0, true),
            ("fn+f5 with fn's flag", Binding(id: "b", modifiers: [.fn: .any], key: f5, behavior: .tap), f5, fn, true),
            ("⌃+← with fn's flag", Binding(id: "b", modifiers: [.control: .any], key: left, behavior: .tap), left, fn | ctrl, true),
            ("⌃+← without ⌃", Binding(id: "b", modifiers: [.control: .any], key: left, behavior: .tap), left, fn, false),
        ]
        for c in cases {
            var r = HotkeyRecognizer(config: HotkeyConfig(bindings: [c.binding]))
            let down = r.handle(.keyDown(keyCode: c.key, flags: c.flags, isRepeat: false, t: 0))
            XCTAssertEqual(down.events.map(\.binding), c.fires ? ["b"] : [], c.name)
            XCTAssertEqual(down.swallow, c.fires, "\(c.name): swallowed")
            XCTAssertEqual(r.handle(.keyUp(keyCode: c.key, flags: c.flags, t: 10)).swallow, c.fires, "\(c.name): key up swallowed")
        }
    }

    /// The cancel shortcut matches the same way: a right ⌥+Space cancel
    /// does not take left ⌥+Space, and an F5 cancel takes F5 with fn's flag.
    func testCancelShortcutMatchesFnAndSide() {
        let space: UInt16 = 49, f5: UInt16 = 96
        for (name, cancel, key, flags, fires) in [
            ("right⌥+space on left⌥+space", Binding(id: "cancel", modifiers: [.option: .right], key: space, behavior: .tap), space, RawFlags.option | 0x20, false),
            ("right⌥+space on right⌥+space", Binding(id: "cancel", modifiers: [.option: .right], key: space, behavior: .tap), space, RawFlags.option | 0x40, true),
            ("fn+space on space", Binding(id: "cancel", modifiers: [.fn: .any], key: space, behavior: .tap), space, 0, false),
            ("f5 with fn's flag", Binding(id: "cancel", modifiers: [:], key: f5, behavior: .tap), f5, fn, true),
        ] as [(String, Binding, UInt16, UInt64, Bool)] {
            var c = defaults()
            c.active = true
            c.cancel = cancel
            var r = HotkeyRecognizer(config: c)
            let down = r.handle(.keyDown(keyCode: key, flags: flags, isRepeat: false, t: 0))
            XCTAssertEqual(down.events.map(\.binding), fires ? ["cancel"] : [], name)
            XCTAssertEqual(down.swallow, fires, "\(name): swallowed")
        }
    }

    /// The cancel shortcut is taken only while active: its key down and up
    /// are swallowed then and reported as "cancel"; while inactive the same
    /// keys pass through untouched. Escape is no longer special: with no
    /// cancel configured nothing is taken even while active, and with
    /// another key configured Escape passes.
    func testCancelShortcutOnlyWhenActive() {
        var c = defaults()
        c.cancel = Binding(id: "cancel", modifiers: [:], key: 53, behavior: .tap)
        var r = HotkeyRecognizer(config: c)
        let idle = r.handle(.keyDown(keyCode: 53, flags: 0, isRepeat: false, t: 0))
        XCTAssertFalse(idle.swallow)
        XCTAssertTrue(idle.events.isEmpty)
        XCTAssertFalse(r.handle(.keyUp(keyCode: 53, flags: 0, t: 10)).swallow)
        c.active = true
        r.configure(c)
        let down = r.handle(.keyDown(keyCode: 53, flags: 0, isRepeat: false, t: 20))
        XCTAssertTrue(down.swallow)
        XCTAssertEqual(down.events.map(\.binding), ["cancel"])
        let held = r.handle(.keyDown(keyCode: 53, flags: 0, isRepeat: true, t: 25))
        XCTAssertTrue(held.swallow)
        XCTAssertTrue(held.events.isEmpty)
        XCTAssertTrue(r.handle(.keyUp(keyCode: 53, flags: 0, t: 30)).swallow)
        c.active = false
        r.configure(c)
        XCTAssertFalse(r.handle(.keyDown(keyCode: 53, flags: 0, isRepeat: false, t: 40)).swallow)
    }

    func testCancelKeyIsConfigurable() {
        var c = defaults()
        c.active = true
        c.cancel = Binding(id: "cancel", modifiers: [.control: .any], key: 47, behavior: .tap) // ⌃.
        var r = HotkeyRecognizer(config: c)
        let down = r.handle(.keyDown(keyCode: 47, flags: RawFlags.control, isRepeat: false, t: 0))
        XCTAssertTrue(down.swallow)
        XCTAssertEqual(down.events.map(\.binding), ["cancel"])
        XCTAssertTrue(r.handle(.keyUp(keyCode: 47, flags: RawFlags.control, t: 10)).swallow)
        // Escape and the bare key are not the cancel shortcut: they pass.
        let esc = r.handle(.keyDown(keyCode: 53, flags: 0, isRepeat: false, t: 20))
        XCTAssertFalse(esc.swallow)
        XCTAssertTrue(esc.events.isEmpty)
        let bare = r.handle(.keyDown(keyCode: 47, flags: 0, isRepeat: false, t: 30))
        XCTAssertFalse(bare.swallow)
        XCTAssertTrue(bare.events.isEmpty)
    }

    func testNoCancelConfiguredTakesNothing() {
        var c = defaults()
        c.active = true
        var r = HotkeyRecognizer(config: c)
        let esc = r.handle(.keyDown(keyCode: 53, flags: 0, isRepeat: false, t: 0))
        XCTAssertFalse(esc.swallow)
        XCTAssertTrue(esc.events.isEmpty)
    }

    func testModifierOnlyCancel() {
        var c = defaults()
        c.active = true
        c.cancel = Binding(id: "cancel", modifiers: [.command: .right], behavior: .tap)
        var r = HotkeyRecognizer(config: c)
        // A modifier-only chord is never swallowed, and fires on release.
        XCTAssertFalse(r.handle(.flagsChanged(keyCode: 54, flags: rcmd, t: 0)).swallow)
        let up = r.handle(.flagsChanged(keyCode: 54, flags: 0, t: 60))
        XCTAssertFalse(up.swallow)
        XCTAssertEqual(up.events.map(\.binding), ["cancel"])
        // The left command is not the right one.
        XCTAssertTrue(run(c, [(0, 55, lcmd), (60, 55, 0)]).isEmpty)
        // Inactive: nothing fires.
        c.active = false
        XCTAssertTrue(run(c, [(0, 54, rcmd), (60, 54, 0)]).isEmpty)
    }

    /// Seen on macOS 26 (synthetic events): fn's release after fn+⌃ never
    /// arrives. The next fn press must not be read as a continuation.
    func testMissedFnReleaseRecovers() {
        let c = defaults([Binding(id: "dbl", modifiers: [.fn: .any], behavior: .doubleTap)])
        var r = HotkeyRecognizer(config: c)
        var fired: [String] = []
        for (t, k, f) in [(0.0, UInt16(63), fn), (40, 59, fn | ctrl), (120, 59, fn), /* fn up lost */
                          (1500, 63, fn), (1570, 63, 0), (1690, 63, fn), (1760, 63, 0)] as [(Double, UInt16, UInt64)] {
            fired += r.handle(.flagsChanged(keyCode: k, flags: f, t: t)).events.map(\.binding)
        }
        fired += r.tick(now: 3000).map(\.binding)
        XCTAssertEqual(fired, ["dbl"])
        XCTAssertEqual(r.missedEvents, 1)
    }

    func testGlobeKeyEventsDoNotSpoilAndAreSwallowedWithSwallowFn() {
        var c = defaults()
        var r = HotkeyRecognizer(config: c)
        _ = r.handle(.flagsChanged(keyCode: 63, flags: fn, t: 0))
        _ = r.handle(.flagsChanged(keyCode: 63, flags: 0, t: 80))
        XCTAssertFalse(r.handle(.keyDown(keyCode: 179, flags: 0, isRepeat: false, t: 81)).swallow)
        c.swallowFn = true
        r = HotkeyRecognizer(config: c)
        XCTAssertTrue(r.handle(.keyDown(keyCode: 179, flags: 0, isRepeat: false, t: 81)).swallow)
        XCTAssertTrue(r.handle(.keyUp(keyCode: 179, flags: 0, t: 82)).swallow)
    }

    func testSwallowFn() {
        var c = defaults()
        c.swallowFn = true
        var r = HotkeyRecognizer(config: c)
        XCTAssertTrue(r.handle(.flagsChanged(keyCode: 63, flags: fn, t: 0)).swallow)
        XCTAssertFalse(r.handle(.flagsChanged(keyCode: 56, flags: fn | shift, t: 10)).swallow)
        XCTAssertTrue(r.handle(.flagsChanged(keyCode: 63, flags: shift, t: 20)).swallow)
    }
}

final class PCMFramerTests: XCTestCase {
    func testFramesAndLevels() {
        var f = PCMFramer(sampleRate: 16000, frameMs: 100, levelHz: 20)
        let r = f.push([Float](repeating: 0.5, count: 3300))
        XCTAssertEqual(r.frames.map(\.seq), [0, 1])
        XCTAssertEqual(r.frames.map(\.offset), [0, 1600])
        XCTAssertEqual(r.frames[0].pcm.count, 3200)
        XCTAssertEqual(r.levels.count, 4)
        XCTAssertEqual(r.levels[0].rms, -6.0, accuracy: 0.1)
        let last = f.flush()
        XCTAssertEqual(last?.pcm.count, 200)
        XCTAssertEqual(last?.offset, 3200)
        // 0.5 → 16383 little-endian
        XCTAssertEqual(r.frames[0].pcm[0], 0xFF)
        XCTAssertEqual(r.frames[0].pcm[1], 0x3F)
    }
}
