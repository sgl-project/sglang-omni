import ApplicationServices
import Foundation
@testable import VoxtNewHelper
import XCTest

final class BrowserActivationTests: XCTestCase {
    /// Chrome rejects Electron's attribute: the request must reach its enhanced setter.
    func testChromeActivationFallsBackFromElectronAttribute() {
        let front = FrontApps(out: Writer(handle: FileHandle.nullDevice))
        front.manual.configure(["com.google.Chrome"])
        var attributes: [String] = []
        front.setManualAccessibility(pid: 42, bundleId: "com.google.Chrome", why: "no web area", set: { _, name, value in
            attributes.append(name as String)
            XCTAssertEqual(value as? Bool, true)
            return name as String == "AXManualAccessibility" ? .attributeUnsupported : .notImplemented
        })
        XCTAssertEqual(attributes, ["AXManualAccessibility", "AXEnhancedUserInterface"])
    }

    /// Electron already enabled its tree; requesting enhanced mode can slow its animations.
    func testElectronActivationKeepsItsManualAttribute() {
        let front = FrontApps(out: Writer(handle: FileHandle.nullDevice))
        var attributes: [String] = []
        front.setManualAccessibility(pid: 42, bundleId: "com.tinyspeck.slackmacgap", why: "Electron", set: { _, name, _ in
            attributes.append(name as String)
            return .success
        })
        XCTAssertEqual(attributes, ["AXManualAccessibility"])
    }

    func testEnhancedTreeIsReleasedOnlyAfterItsLastUse() {
        let front = FrontApps(out: Writer(handle: FileHandle.nullDevice))
        front.manual.configure(["com.google.Chrome"])
        XCTAssertTrue(front.manual.claim(pid: 42, .activation))
        front.setManualAccessibility(pid: 42, bundleId: "com.google.Chrome", why: "meeting", set: { _, name, _ in
            name as String == "AXManualAccessibility" ? .attributeUnsupported : .notImplemented
        }, read: { _, _, value in
            value.pointee = kCFBooleanFalse
            return .success
        })
        var released = 0
        let setter: (AXUIElement, CFString, CFTypeRef) -> AXError = { _, name, value in
            XCTAssertEqual(name as String, "AXEnhancedUserInterface")
            XCTAssertEqual(value as? Bool, false)
            released += 1
            return .success
        }
        front.releaseUnusedAccessibility(set: setter)
        XCTAssertEqual(released, 0, "app enhancement or a recording browser still needs the tree")
        front.manual.configure([])
        front.releaseUnusedAccessibility(set: setter)
        front.releaseUnusedAccessibility(set: setter)
        XCTAssertEqual(released, 1)
        XCTAssertTrue(front.manual.claim(pid: 42, .activation), "the next use must be able to activate it again")
    }

    func testEnhancedTreeAlreadyOnOrUnreadableIsNotDisabled() {
        for prior: Bool? in [true, nil] {
            let front = FrontApps(out: Writer(handle: FileHandle.nullDevice))
            front.manual.configure(["com.google.Chrome"])
            front.setManualAccessibility(pid: 42, bundleId: "com.google.Chrome", why: "meeting", set: { _, name, _ in
                name as String == "AXManualAccessibility" ? .attributeUnsupported : .success
            }, read: { _, _, value in
                if let prior {
                    value.pointee = prior ? kCFBooleanTrue : kCFBooleanFalse
                    return .success
                }
                return .attributeUnsupported
            })
            front.manual.configure([])
            front.releaseUnusedAccessibility(set: { _, _, _ in
                XCTFail("an existing or unknown accessibility setting is not ours to disable")
                return .success
            })
        }
    }

    func testStaleBrowserReadCannotEnableAnUnlistedTree() {
        let front = FrontApps(out: Writer(handle: FileHandle.nullDevice))
        XCTAssertTrue(front.manual.claim(pid: 42, .retry))
        var attributes: [String] = []
        front.setManualAccessibility(pid: 42, bundleId: "com.google.Chrome", why: "late URL read", set: { _, name, _ in
            attributes.append(name as String)
            return .attributeUnsupported
        })
        XCTAssertEqual(attributes, ["AXManualAccessibility"])
        XCTAssertTrue(front.manual.claim(pid: 42, .retry), "a later configured use may try again")
    }
}
