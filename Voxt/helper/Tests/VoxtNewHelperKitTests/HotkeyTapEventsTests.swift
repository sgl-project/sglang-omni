@testable import VoxtNewHelperKit
import CoreGraphics
import XCTest

final class HotkeyTapEventsTests: XCTestCase {
    /// The tap asks for keys, modifiers, every button's press and the other
    /// buttons' releases (a bound side button's release is swallowed too);
    /// not moves, drags, scrolls or the left and right releases.
    func testMask() {
        for type in [CGEventType.keyDown, .keyUp, .flagsChanged, .leftMouseDown, .rightMouseDown, .otherMouseDown, .otherMouseUp] {
            XCTAssertNotEqual(HotkeyTapEvents.mask & (1 << type.rawValue), 0, "type \(type.rawValue)")
        }
        for type in [CGEventType.mouseMoved, .leftMouseUp, .rightMouseUp, .otherMouseDragged, .scrollWheel] {
            XCTAssertEqual(HotkeyTapEvents.mask & (1 << type.rawValue), 0, "type \(type.rawValue)")
        }
    }

    /// A button's press and release reach the recognizer with the number
    /// macOS gives the button (3: the first side button, Mouse4), the press
    /// with its modifier flags.
    func testMouseInput() {
        func input(_ type: CGEventType, _ button: UInt32) -> KeyInput? {
            let e = CGEvent(mouseEventSource: nil, mouseType: type, mouseCursorPosition: .zero, mouseButton: CGMouseButton(rawValue: button)!)!
            e.flags = .maskAlternate
            return HotkeyTapEvents.input(type, e, t: 7)
        }
        guard case .mouseDown(button: 3, let flags, t: 7) = input(.otherMouseDown, 3) else { return XCTFail("otherMouseDown") }
        XCTAssertNotEqual(flags & CGEventFlags.maskAlternate.rawValue, 0)
        guard case .mouseUp(button: 3, t: 7) = input(.otherMouseUp, 3) else { return XCTFail("otherMouseUp") }
        guard case .mouseDown(button: 0, _, _) = input(.leftMouseDown, 0) else { return XCTFail("leftMouseDown") }
        XCTAssertNil(input(.otherMouseDragged, 3))
    }
}
