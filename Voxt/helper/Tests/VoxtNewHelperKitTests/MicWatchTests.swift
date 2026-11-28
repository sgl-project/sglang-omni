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

final class MicHostTests: XCTestCase {
    /// A KERN_PROCARGS2 buffer as Zoom's meeting process has it.
    func testServedPIDFromTheArguments() {
        var raw = withUnsafeBytes(of: Int32(5)) { Array($0) }
        raw += Array("/Applications/zoom.us.app/Contents/Frameworks/aomhost.app/Contents/MacOS/aomhost".utf8) + [0, 0, 0]
        for arg in ["aomhost", "-pid", "758", "-evtname", "aomhost758"] {
            raw += Array(arg.utf8) + [0]
        }
        raw += Array("HOME=/Users/someone".utf8) + [0]
        let args = procArgs(raw)
        XCTAssertEqual(args, ["aomhost", "-pid", "758", "-evtname", "aomhost758"])
        XCTAssertEqual(servedPID(args), 758)
        XCTAssertNil(servedPID(["caphost", "-evtname", "x"]))
        XCTAssertEqual(procArgs([1, 0]), [])
    }

    /// A meeting process that starts or stops is a change to send.
    func testHostsAreAChange() {
        let w = MicWatch()
        let zoom = [App(pid: 758, bundleId: "us.zoom.xos", name: nil)]
        w.set(true, at: MicUsers(apps: zoom, hosts: []))
        let meeting = MicUsers(apps: zoom, hosts: [MicHost(pid: 37502, name: "aomhost", app: 758)])
        XCTAssertEqual(w.changed(meeting), meeting)
        XCTAssertNil(w.changed(meeting))
    }
}
