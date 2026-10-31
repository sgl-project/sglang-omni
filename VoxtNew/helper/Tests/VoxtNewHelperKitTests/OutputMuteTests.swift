import Foundation
import XCTest
@testable import VoxtNewHelperKit

/// MuteClaims' rules on a pretend Core Audio: the same cases as the Go
/// fake's (internal/helper/helpertest/output.go) and internal/outmute's.
final class OutputMuteTests: XCTestCase {
    final class Devices: OutputHardware, @unchecked Sendable {
        struct Dev { var device: OutputDevice; var muted: Bool?; var settable: Bool }
        var list: [Dev] = []
        var current: String?
        var writes: [String] = []
        var refuse = false

        func defaultOutput() -> OutputDevice? { list.first { $0.device.uid == current }?.device }
        func device(uid: String) -> OutputDevice? { list.first { $0.device.uid == uid }?.device }
        func muted(_ d: OutputDevice) -> Bool? { list.first { $0.device.uid == d.uid }?.muted }
        func settable(_ d: OutputDevice) -> Bool { list.first { $0.device.uid == d.uid }?.settable ?? false }
        func setMuted(_ m: Bool, _ d: OutputDevice) -> Bool {
            guard !refuse, let i = list.firstIndex(where: { $0.device.uid == d.uid }), list[i].settable else { return false }
            list[i].muted = m
            writes.append("\(d.uid):\(m)")
            return true
        }
        func isMuted(_ uid: String) -> Bool? { list.first { $0.device.uid == uid }?.muted }
    }

    let speakers = OutputDevice(id: 52, uid: "BuiltInSpeakerDevice", name: "MacBook Air Speakers")
    let airpods = OutputDevice(id: 91, uid: "AirPods:output", name: "AirPods Pro")

    func devices(muted: Bool = false) -> Devices {
        let d = Devices()
        d.list = [.init(device: speakers, muted: muted, settable: true), .init(device: airpods, muted: false, settable: true)]
        d.current = speakers.uid
        return d
    }

    func journalDir() -> String {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("voxtnew-output-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir.path
    }

    func files(_ dir: String) -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []).filter { $0.hasSuffix(".json") }.sorted()
    }

    func testMuteAndRelease() throws {
        let hw = devices(), dir = journalDir()
        let c = MuteClaims(hardware: hw, pid: 100, alive: { _ in true })
        XCTAssertEqual(c.state(), OutputState(device: speakers, muted: false, settable: true))
        let r = try c.mute(OutputMuteParams(device: speakers, journal: dir))
        XCTAssertTrue(r.took)
        XCTAssertEqual(r.state.muted, true)
        XCTAssertEqual(files(dir), ["100.json"])
        XCTAssertEqual(c.release(OutputReleaseParams(device: speakers, unmute: true)), OutputReleaseResult(unmuted: true))
        XCTAssertEqual(files(dir), [])
        // Twice: not ours any more.
        XCTAssertEqual(c.release(OutputReleaseParams(device: speakers, unmute: true)), OutputReleaseResult(unmuted: false))
        XCTAssertEqual(hw.writes, ["BuiltInSpeakerDevice:true", "BuiltInSpeakerDevice:false"])
    }

    func testUsersMuteIsNeverClaimed() throws {
        let hw = devices(muted: true)
        let c = MuteClaims(hardware: hw, pid: 100, alive: { _ in true })
        XCTAssertFalse(try c.mute(OutputMuteParams(device: speakers)).took)
        XCTAssertEqual(c.release(OutputReleaseParams(device: speakers, unmute: true)).unmuted, false)
        XCTAssertEqual(c.releaseAll(), [])
        XCTAssertEqual(hw.isMuted(speakers.uid), true)
        XCTAssertEqual(hw.writes, [])
    }

    func testUserUnmutedThenMuted() throws {
        let hw = devices()
        let c = MuteClaims(hardware: hw, pid: 100, alive: { _ in true })
        XCTAssertTrue(try c.mute(OutputMuteParams(device: speakers)).took)
        // The app saw the user unmute it: the claim goes, the mute stays theirs.
        XCTAssertEqual(c.release(OutputReleaseParams(device: speakers)).unmuted, false)
        hw.list[0].muted = true // the user mutes it again
        XCTAssertEqual(c.releaseAll(), [])
        XCTAssertEqual(hw.isMuted(speakers.uid), true)
    }

    func testCannotMute() {
        let hw = devices()
        hw.list[0].settable = false
        let c = MuteClaims(hardware: hw, pid: 100, alive: { _ in true })
        XCTAssertThrowsError(try c.mute(OutputMuteParams(device: speakers))) { XCTAssertEqual(($0 as? RPCError)?.code, Code.unsupported) }
        hw.list[0].muted = nil // no mute at all
        XCTAssertEqual(c.state(), OutputState(device: speakers))
        XCTAssertThrowsError(try c.mute(OutputMuteParams(device: speakers))) { XCTAssertEqual(($0 as? RPCError)?.code, Code.unsupported) }
        XCTAssertThrowsError(try c.mute(OutputMuteParams(device: OutputDevice(id: 7, uid: "gone")))) {
            XCTAssertEqual(($0 as? RPCError)?.code, Code.deviceUnavailable)
        }
        let refusing = devices()
        refusing.refuse = true
        let dir = journalDir()
        let c2 = MuteClaims(hardware: refusing, pid: 100, alive: { _ in true })
        XCTAssertThrowsError(try c2.mute(OutputMuteParams(device: speakers, journal: dir))) { XCTAssertEqual(($0 as? RPCError)?.code, Code.unsupported) }
        XCTAssertEqual(files(dir), [], "a refused mute leaves no claim")
        XCTAssertEqual(c2.claimed, [])
    }

    func testJournalUnwritableMutesNothing() throws {
        let hw = devices()
        let file = journalDir()
        try Data("a file".utf8).write(to: URL(fileURLWithPath: file)) // not a directory
        let c = MuteClaims(hardware: hw, pid: 100, alive: { _ in true })
        XCTAssertThrowsError(try c.mute(OutputMuteParams(device: speakers, journal: file))) { XCTAssertEqual(($0 as? RPCError)?.code, Code.internal) }
        XCTAssertEqual(hw.writes, [])
    }

    func testReleaseAllAndRecover() throws {
        let hw = devices(), dir = journalDir()
        final class Dead: @unchecked Sendable { var pids = Set<Int32>() }
        let dead = Dead()
        let first = MuteClaims(hardware: hw, pid: 100, alive: { !dead.pids.contains($0) })
        XCTAssertTrue(try first.mute(OutputMuteParams(device: speakers, journal: dir)).took)
        hw.current = airpods.uid
        XCTAssertTrue(try first.mute(OutputMuteParams(device: airpods, journal: dir)).took)
        // A second helper while the first runs: not its business.
        let second = MuteClaims(hardware: hw, pid: 200, alive: { !dead.pids.contains($0) })
        XCTAssertNil(second.recover(OutputRecoverParams(journal: dir)).restored)
        XCTAssertEqual(files(dir), ["100.json"])
        // The first one dies (killed): the next recovers its claims.
        dead.pids.insert(100)
        hw.list[1].muted = false // the user unmuted AirPods meanwhile: nothing to do there
        let restored = second.recover(OutputRecoverParams(journal: dir)).restored
        XCTAssertEqual(restored, [speakers])
        XCTAssertEqual(files(dir), [])
        XCTAssertEqual(hw.isMuted(speakers.uid), false)
        // releaseAll: everything back, the file gone.
        hw.current = speakers.uid
        XCTAssertTrue(try second.mute(OutputMuteParams(device: speakers, journal: dir)).took)
        XCTAssertEqual(second.releaseAll(), [speakers])
        XCTAssertEqual(hw.isMuted(speakers.uid), false)
        XCTAssertEqual(files(dir), [])
        // An unreadable journal is dropped.
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        try Data("{".utf8).write(to: URL(fileURLWithPath: dir).appendingPathComponent("7.json"))
        XCTAssertNil(second.recover(OutputRecoverParams(journal: dir)).restored)
        XCTAssertEqual(files(dir), [])
    }

    func testJournalFormatIsGos() throws {
        // internal/helper/helpertest/output.go reads and writes the same file.
        let hw = devices(), dir = journalDir()
        let c = MuteClaims(hardware: hw, pid: 100, alive: { _ in true })
        _ = try c.mute(OutputMuteParams(device: speakers, journal: dir))
        let data = try Data(contentsOf: URL(fileURLWithPath: dir).appendingPathComponent("100.json"))
        let obj = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(obj["pid"] as? Int, 100)
        XCTAssertEqual((obj["devices"] as? [[String: Any]])?.first?["uid"] as? String, speakers.uid)
    }

    func testWatchSendsChangesOnce() {
        let w = OutputWatch()
        let a = OutputState(device: speakers, muted: false, settable: true)
        var b = a
        b.muted = true
        XCTAssertNil(w.changed(b), "off")
        w.set(true, at: a)
        XCTAssertNil(w.changed(a))
        XCTAssertEqual(w.changed(b), b)
        XCTAssertNil(w.changed(b))
        w.set(false, at: b)
        XCTAssertNil(w.changed(a))
    }

    func testNoJournalIsNotTheWorkingDirectory() throws {
        let dir = journalDir()
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        try Data(#"{"name":"x"}"#.utf8).write(to: URL(fileURLWithPath: dir).appendingPathComponent("mygo.json"))
        let c = MuteClaims(hardware: devices(), pid: 100, alive: { _ in false })
        XCTAssertNil(c.recover(OutputRecoverParams(journal: "")).restored)
        XCTAssertNil(c.recover(OutputRecoverParams(journal: dir)).restored)
        XCTAssertTrue(FileManager.default.fileExists(atPath: dir + "/mygo.json"), "only <pid>.json files are journals")
    }

    func testProcessRuns() {
        XCTAssertTrue(processRuns(getpid()))
        XCTAssertFalse(processRuns(99_999_999))
    }
}
