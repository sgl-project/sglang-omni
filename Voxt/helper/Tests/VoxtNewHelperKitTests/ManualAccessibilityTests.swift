import Foundation
import XCTest
@testable import VoxtNewHelperKit

/// Which apps get AXManualAccessibility (Electron apps, app.configure's
/// bundle ids) and that each process gets it once per reason.
final class ManualAccessibilityTests: XCTestCase {
    func testWants() {
        let m = ManualAccessibility()
        XCTAssertFalse(m.wants(bundleId: "company.thebrowser.Browser", electron: false), "nothing listed yet")
        XCTAssertTrue(m.wants(bundleId: "com.tinyspeck.slackmacgap", electron: true), "Electron apps always")
        m.configure(["com.google.Chrome", " company.thebrowser.Browser ", "", "com.example.MyBrowser"])
        XCTAssertTrue(m.wants(bundleId: "company.thebrowser.Browser", electron: false))
        XCTAssertTrue(m.wants(bundleId: "com.google.chrome", electron: false), "case does not matter")
        XCTAssertTrue(m.listed("COM.EXAMPLE.MYBROWSER"))
        XCTAssertFalse(m.wants(bundleId: "com.apple.Safari", electron: false))
        XCTAssertFalse(m.wants(bundleId: nil, electron: false))
        XCTAssertFalse(m.listed(""))
        m.configure([])
        XCTAssertFalse(m.listed("com.google.Chrome"), "a new list replaces the old one")
    }

    func testOncePerProcess() {
        let m = ManualAccessibility()
        XCTAssertTrue(m.claim(pid: 905, .activation))
        XCTAssertFalse(m.claim(pid: 905, .activation), "once per process")
        XCTAssertTrue(m.claim(pid: 905, .retry), "browser.url's retry is its own")
        XCTAssertFalse(m.claim(pid: 905, .retry))
        XCTAssertTrue(m.claim(pid: 906, .activation))
        XCTAssertFalse(m.claim(pid: 0, .activation), "no process")
        m.forget(pid: 905)
        XCTAssertTrue(m.claim(pid: 905, .activation), "a pid that came back is a new process")
        XCTAssertTrue(m.claim(pid: 905, .retry))
        XCTAssertFalse(m.claim(pid: 906, .activation))
    }
}
