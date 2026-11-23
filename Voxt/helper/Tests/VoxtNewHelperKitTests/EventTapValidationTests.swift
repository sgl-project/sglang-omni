@testable import VoxtNewHelperKit
import CoreGraphics
import XCTest

// The tap validation Voxt's #2996 added: macOS may create a tap while
// stripping unauthorized keyboard events, so a nonnil tap must not be
// reported as ready unless its granted mask kept every requested event.
final class EventTapValidationTests: XCTestCase {
    /// The mask HotkeyTap installs: keys, flags and the mouse presses.
    let requestedMask: CGEventMask = (1 << CGEventType.keyDown.rawValue) | (1 << CGEventType.keyUp.rawValue) |
        (1 << CGEventType.flagsChanged.rawValue) | (1 << CGEventType.leftMouseDown.rawValue) |
        (1 << CGEventType.rightMouseDown.rawValue) | (1 << CGEventType.otherMouseDown.rawValue)

    func testTapWithFilteredKeyboardEventsIsRejected() {
        let keyboardMask: CGEventMask = (1 << CGEventType.keyDown.rawValue) | (1 << CGEventType.keyUp.rawValue)
        let cases: [(grantedMask: CGEventMask, expected: Bool)] = [
            (requestedMask, true),
            (requestedMask | (1 << CGEventType.mouseMoved.rawValue), true),
            (requestedMask & ~keyboardMask, false),
            (requestedMask & ~(1 << CGEventType.keyDown.rawValue), false),
            (requestedMask & ~(1 << CGEventType.keyUp.rawValue), false),
        ]
        for testCase in cases {
            XCTAssertEqual(
                EventTapValidation.newTapHasRequiredEvents(
                    eventMask: requestedMask,
                    location: .cgSessionEventTap,
                    processIdentifier: 100,
                    previousTapIDs: [],
                    installedTaps: [tapInformation(identifier: 1, eventMask: testCase.grantedMask)]
                ),
                testCase.expected,
                "grantedMask=\(String(testCase.grantedMask, radix: 16))"
            )
        }
    }

    func testValidationDoesNotBorrowKeyboardAccessFromAnotherTap() {
        var foreignTap = tapInformation(identifier: 2, eventMask: requestedMask)
        foreignTap.tappingProcess = 200
        var otherLocation = tapInformation(identifier: 3, eventMask: requestedMask)
        otherLocation.tapPoint = .cghidEventTap
        var passiveTap = tapInformation(identifier: 4, eventMask: requestedMask)
        passiveTap.options = .listenOnly
        let unrelatedTaps = [
            tapInformation(identifier: 1, eventMask: requestedMask),
            foreignTap,
            otherLocation,
            passiveTap,
        ]
        for grantedMask in [CGEventMask(0), requestedMask] {
            XCTAssertEqual(
                EventTapValidation.newTapHasRequiredEvents(
                    eventMask: requestedMask,
                    location: .cgSessionEventTap,
                    processIdentifier: 100,
                    previousTapIDs: [1],
                    installedTaps: unrelatedTaps + [tapInformation(identifier: 5, eventMask: grantedMask)]
                ),
                grantedMask == requestedMask
            )
        }
    }

    func testMissingOrAmbiguousTapCannotBeReportedAsReady() {
        for installedTaps in [
            [],
            [tapInformation(identifier: 1, eventMask: requestedMask)],
            [
                tapInformation(identifier: 2, eventMask: requestedMask),
                tapInformation(identifier: 3, eventMask: requestedMask),
            ],
        ] {
            XCTAssertFalse(
                EventTapValidation.newTapHasRequiredEvents(
                    eventMask: requestedMask,
                    location: .cgSessionEventTap,
                    processIdentifier: 100,
                    previousTapIDs: [1],
                    installedTaps: installedTaps
                )
            )
        }
    }

    func testEventTapMetadataCanBeReadWithoutInstallingKeyboardMonitor() {
        XCTAssertNotNil(EventTapValidation.currentTaps())
    }

    private func tapInformation(identifier: UInt32, eventMask: CGEventMask) -> CGEventTapInformation {
        var information = CGEventTapInformation()
        information.eventTapID = identifier
        information.tappingProcess = 100
        information.tapPoint = .cgSessionEventTap
        information.options = .defaultTap
        information.eventsOfInterest = eventMask
        return information
    }
}
