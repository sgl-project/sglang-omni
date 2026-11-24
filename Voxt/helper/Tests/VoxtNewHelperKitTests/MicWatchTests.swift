import XCTest
@testable import VoxtNewHelperKit

final class MicWatchTests: XCTestCase {
    func testUsersAreTheProcessesRecordingWithoutOurOwn() {
        let users = micUsers([
            AudioProcess(pid: 704, bundleId: "us.zoom.xos", input: true),
            AudioProcess(pid: 690, bundleId: "com.apple.Music", input: false),
            AudioProcess(pid: 50, bundleId: "dev.hehehai.voxtnew.helper", input: true),
            AudioProcess(pid: 49, bundleId: "dev.hehehai.voxtnew", input: true),
            AudioProcess(pid: 388, bundleId: "", input: true),
            AudioProcess(pid: 704, bundleId: "us.zoom.xos", input: true),
        ], ownPIDs: [50, 49])
        XCTAssertEqual(users.apps, [App(pid: 388, bundleId: nil, name: nil), App(pid: 704, bundleId: "us.zoom.xos", name: nil)])
    }

    func testWatchSendsChangesOnce() {
        let w = MicWatch()
        let none = MicUsers()
        let zoom = MicUsers(apps: [App(pid: 704, bundleId: "us.zoom.xos", name: nil)])
        XCTAssertNil(w.changed(zoom), "off")
        w.set(true, at: none)
        XCTAssertNil(w.changed(none))
        XCTAssertEqual(w.changed(zoom), zoom)
        XCTAssertNil(w.changed(zoom))
        XCTAssertEqual(w.changed(none), none)
        w.set(false, at: none)
        XCTAssertNil(w.changed(zoom))
    }
}
