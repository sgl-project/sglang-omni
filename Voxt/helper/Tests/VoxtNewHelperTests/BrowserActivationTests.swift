import ApplicationServices
import Foundation
@testable import VoxtNewHelper
import XCTest

final class BrowserActivationTests: XCTestCase {
    /// Chrome rejects Electron's attribute: the request must reach its enhanced setter.
    func testChromeActivationFallsBackFromElectronAttribute() {
        let front = FrontApps(out: Writer(handle: FileHandle.nullDevice))
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
}
