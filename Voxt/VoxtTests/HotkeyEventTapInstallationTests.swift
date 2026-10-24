import XCTest
import ApplicationServices
@testable import Voxt

final class HotkeyEventTapInstallationTests: XCTestCase {
    func testTapWithFilteredKeyboardEventsIsRejected() {
        let requestedMask: CGEventMask =
            (1 << CGEventType.keyDown.rawValue) |
            (1 << CGEventType.keyUp.rawValue) |
            (1 << CGEventType.flagsChanged.rawValue) |
            (1 << CGEventType.otherMouseDown.rawValue) |
            (1 << CGEventType.otherMouseUp.rawValue)
        let keyboardMask: CGEventMask =
            (1 << CGEventType.keyDown.rawValue) | (1 << CGEventType.keyUp.rawValue)
        let cases: [(grantedMask: CGEventMask, expected: Bool)] = [
            (requestedMask, true),
            (requestedMask | (1 << CGEventType.mouseMoved.rawValue), true),
            (requestedMask & ~keyboardMask, false),
            (requestedMask & ~(1 << CGEventType.keyDown.rawValue), false),
            (requestedMask & ~(1 << CGEventType.keyUp.rawValue), false)
        ]
        for testCase in cases {
            XCTAssertEqual(
                HotkeyEventTapInstallation.newTapHasRequiredEvents(
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
        let requestedMask: CGEventMask = 1 << CGEventType.keyDown.rawValue
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
            passiveTap
        ]
        for grantedMask in [CGEventMask(0), requestedMask] {
            XCTAssertEqual(
                HotkeyEventTapInstallation.newTapHasRequiredEvents(
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
        let requestedMask: CGEventMask = 1 << CGEventType.keyDown.rawValue
        for installedTaps in [
            [],
            [tapInformation(identifier: 1, eventMask: requestedMask)],
            [
                tapInformation(identifier: 2, eventMask: requestedMask),
                tapInformation(identifier: 3, eventMask: requestedMask)
            ]
        ] {
            XCTAssertFalse(
                HotkeyEventTapInstallation.newTapHasRequiredEvents(
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
        XCTAssertNotNil(HotkeyEventTapInstallation.currentEventTaps())
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
