@testable import VoxtNewHelperKit
import XCTest

// VoxtNew's default shortcuts, the ActNow family's (spec §10.2): right ⌥
// tapped alone transcribes, with ⇧ translates, with ⌃ rewrites, ⌃⌥V pastes
// the last result. Auralink has fn, so the two apps can run together.
//
// The flags are the ones a real keyboard sends (CGEventFlags raw values,
// as an event tap reads them on mac-air): the family bit, the device bit of
// the side, and NX_NONCOALSESCEDMASK (0x100).
final class RightOptionTests: XCTestCase {
    static let real: UInt64 = 0x100
    let rOpt = RawFlags.option | 0x40 | real // right ⌥: 0x80140
    let lOpt = RawFlags.option | 0x20 | real // left ⌥: 0x80120
    let lShift = RawFlags.shift | 0x2 | real // 0x20102
    let lCtrl = RawFlags.control | 0x1 | real // 0x40101
    let up = real
    let fn = RawFlags.secondaryFn | real

    /// What hotkey.configure brings with the defaults (internal/hotkey Build).
    func family() -> HotkeyConfig {
        HotkeyConfig(bindings: [
            Binding(id: "transcribe/0", modifiers: [.option: .right], behavior: .tap),
            Binding(id: "translate/0", modifiers: [.option: .right, .shift: .any], behavior: .tap),
            Binding(id: "rewrite/0", modifiers: [.control: .any, .option: .right], behavior: .tap),
            Binding(id: "paste-last", modifiers: [.control: .any, .option: .any], key: 9, behavior: .tap),
        ], doubleTapMs: 500, longPressMs: 300, tapMaxMs: 1500)
    }

    /// Feeds the events; returns what fired and whether any was swallowed.
    func run(_ inputs: [KeyInput], until: Double? = nil) -> (fired: [String], swallowed: [Int]) {
        var r = HotkeyRecognizer(config: family())
        var fired: [String] = [], swallowed: [Int] = []
        for (i, input) in inputs.enumerated() {
            let res = r.handle(input)
            fired += res.events.map { "\($0.binding):\($0.phase.rawValue)" }
            if res.swallow { swallowed.append(i) }
        }
        if let until { fired += r.tick(now: until).map { "\($0.binding):\($0.phase.rawValue)" } }
        return (fired, swallowed)
    }

    func flags(_ k: UInt16, _ f: UInt64, _ t: Double) -> KeyInput { .flagsChanged(keyCode: k, flags: f, t: t) }
    func down(_ k: UInt16, _ f: UInt64, _ t: Double) -> KeyInput { .keyDown(keyCode: k, flags: f, isRepeat: false, t: t) }
    func keyUp(_ k: UInt16, _ f: UInt64, _ t: Double) -> KeyInput { .keyUp(keyCode: k, flags: f, t: t) }

    func testRightOptionTapTranscribesAndPasses() {
        let r = run([flags(61, rOpt, 0), flags(61, up, 95)])
        XCTAssertEqual(r.fired, ["transcribe/0:trigger"])
        XCTAssertEqual(r.swallowed, [], "a modifier is never taken from the apps")
    }

    func testLeftOptionTapDoesNothing() {
        XCTAssertEqual(run([flags(58, lOpt, 0), flags(58, up, 95)]).fired, [])
    }

    /// A synthetic event without a device bit reads as the left key.
    func testOptionWithoutSideIsLeft() {
        XCTAssertEqual(run([flags(61, RawFlags.option, 0), flags(61, 0, 95)]).fired, [])
    }

    func testRightOptionShiftTranslatesInEitherOrder() {
        XCTAssertEqual(run([flags(61, rOpt, 0), flags(56, rOpt | lShift, 40), flags(61, lShift, 120), flags(56, up, 150)]).fired,
                       ["translate/0:trigger"])
        XCTAssertEqual(run([flags(56, lShift, 0), flags(61, rOpt | lShift, 30), flags(56, rOpt, 100), flags(61, up, 110)]).fired,
                       ["translate/0:trigger"])
    }

    func testControlRightOptionRewrites() {
        XCTAssertEqual(run([flags(59, lCtrl, 0), flags(61, lCtrl | rOpt, 40), flags(61, lCtrl, 90), flags(59, up, 100)]).fired,
                       ["rewrite/0:trigger"])
    }

    func testBothOptionsDoNothing() {
        XCTAssertEqual(run([flags(58, lOpt, 0), flags(61, lOpt | rOpt, 30), flags(61, lOpt, 80), flags(58, up, 100)]).fired, [])
    }

    /// Left ⌥ typing (⌥E E = é, ⌥A = å) is not touched at all.
    func testLeftOptionCharactersPass() {
        let r = run([flags(58, lOpt, 0), down(14, lOpt, 30), keyUp(14, lOpt, 60), flags(58, up, 80),
                     down(14, up, 100), keyUp(14, up, 130), flags(58, lOpt, 200), down(0, lOpt, 220), keyUp(0, lOpt, 250), flags(58, up, 270)])
        XCTAssertEqual(r.fired, [])
        XCTAssertEqual(r.swallowed, [])
    }

    /// Right ⌥ held to type (é, AltGr-style layouts): the keys pass and no
    /// shortcut fires on its release; nor with ⇧ (⌥⇧ characters).
    func testRightOptionHeldForTypingPassesAndDoesNotFire() {
        var r = run([flags(61, rOpt, 0), down(14, rOpt, 40), keyUp(14, rOpt, 70), down(14, rOpt, 120), keyUp(14, rOpt, 150), flags(61, up, 200)])
        XCTAssertEqual(r.fired, [])
        XCTAssertEqual(r.swallowed, [])
        r = run([flags(61, rOpt, 0), flags(56, rOpt | lShift, 20), down(2, rOpt | lShift, 60), keyUp(2, rOpt | lShift, 90),
                 flags(56, rOpt, 120), flags(61, up, 140)])
        XCTAssertEqual(r.fired, [])
        XCTAssertEqual(r.swallowed, [])
        // Typing goes on right after: the next tap works again.
        XCTAssertEqual(run([flags(61, rOpt, 0), down(14, rOpt, 40), keyUp(14, rOpt, 70), flags(61, up, 90),
                            flags(61, rOpt, 400), flags(61, up, 480)]).fired, ["transcribe/0:trigger"])
    }

    /// Right ⌥ held for an ⌥-click or ⌥-drag: no shortcut.
    func testRightOptionClickDoesNotFire() {
        let r = run([flags(61, rOpt, 0), .pointer(t: 50), flags(61, up, 300)])
        XCTAssertEqual(r.fired, [])
        XCTAssertEqual(r.swallowed, [])
    }

    func testRightOptionHeldLongIsNoTap() {
        XCTAssertEqual(run([flags(61, rOpt, 0), flags(61, up, 1600)]).fired, [])
    }

    /// ⌃⌥V with either ⌥ pastes the last result and is taken from the app;
    /// ⌥V (√) and ⌃V pass.
    func testControlOptionVWithEitherOption() {
        for opt in [lOpt, rOpt] {
            let r = run([down(9, lCtrl | opt, 0), keyUp(9, lCtrl | opt, 40)])
            XCTAssertEqual(r.fired, ["paste-last:trigger"])
            XCTAssertEqual(r.swallowed, [0, 1])
        }
        XCTAssertEqual(run([down(9, rOpt, 0), keyUp(9, rOpt, 40)]).swallowed, [])
        XCTAssertEqual(run([down(9, lCtrl, 0), keyUp(9, lCtrl, 40)]).swallowed, [])
        // ⌃ + right ⌥ held, then V: the paste, not Rewrite.
        let r = run([flags(59, lCtrl, 0), flags(61, lCtrl | rOpt, 30), down(9, lCtrl | rOpt, 60), keyUp(9, lCtrl | rOpt, 90),
                     flags(61, lCtrl, 120), flags(59, up, 130)])
        XCTAssertEqual(r.fired, ["paste-last:trigger"])
    }

    /// fn is Auralink's: VoxtNew's defaults neither fire on it nor take it.
    func testFnIsLeftAlone() {
        let r = run([flags(63, fn, 0), flags(63, up, 80), down(179, up, 81), keyUp(179, up, 90),
                     flags(63, fn, 200), flags(56, fn | lShift, 220), flags(56, fn, 260), flags(63, up, 280)])
        XCTAssertEqual(r.fired, [])
        XCTAssertEqual(r.swallowed, [])
    }

    func testInjectedMarkerIsVoxtNews() {
        let ascii = withUnsafeBytes(of: InjectedEvent.marker.bigEndian) { Array($0.suffix(4)) }
        XCTAssertEqual(String(decoding: ascii, as: UTF8.self), "VOXT")
        XCTAssertNotEqual(InjectedEvent.marker, InjectedEvent.auralinkMarker)
        XCTAssertTrue(InjectedEvent.isOurs(0x564F_5854))
        XCTAssertFalse(InjectedEvent.isOurs(InjectedEvent.auralinkMarker))
        XCTAssertFalse(InjectedEvent.isOurs(0))
    }
}
