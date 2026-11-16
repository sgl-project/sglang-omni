@testable import VoxtNewHelperKit
import XCTest

/// capture.system's pure parts: the tap it asks Core Audio for, the slots
/// that let it run beside one capture.start, and its test-only file input.
final class SystemCaptureTests: XCTestCase {
    final class Run {}

    func testTapPlanLeavesOutOwnProcesses() {
        let p = systemTapPlan(ownPIDs: [812, 0, 1, 811, 812], outputUID: "BuiltInSpeakerDevice")
        XCTAssertEqual(p.excludedPIDs, [812, 811])
        XCTAssertEqual(p.outputUID, "BuiltInSpeakerDevice")
        XCTAssertTrue(p.mono && p.mixdown && p.isPrivate && p.exclusive)
        XCTAssertFalse(p.muted, "the user must still hear the call")
    }

    func testOneOfEachKind() throws {
        var slots = CaptureSlots<Run>()
        let mic = Run(), sys = Run()
        try slots.claim(Method.captureStart, job: "m", mic)
        try slots.claim(Method.captureSystem, job: "s", sys)
        XCTAssertThrowsError(try slots.claim(Method.captureSystem, job: "s2", Run())) { e in
            XCTAssertEqual((e as? RPCError)?.code, Code.busy)
            XCTAssertEqual((e as? RPCError)?.message, "A recording is running already (s)")
        }
        XCTAssertThrowsError(try slots.claim(Method.captureStart, job: "m2", Run()))
        XCTAssertTrue(slots.run(job: "s") === sys)
        XCTAssertTrue(slots.run(job: "m") === mic)
        XCTAssertNil(slots.run(job: "x"))
        XCTAssertEqual(slots.all.count, 2)

        slots.release(sys)
        XCTAssertNil(slots.run(job: "s"))
        slots.release(sys)
        XCTAssertTrue(slots.run(job: "m") === mic)
        try slots.claim(Method.captureSystem, job: "s3", Run())
    }

    func testSystemFileFromEnvironment() {
        XCTAssertNil(CaptureFileConfig.system(from: [:]))
        XCTAssertNil(CaptureFileConfig.system(from: ["VOXTNEW_HELPER_CAPTURE_FILE": "/tmp/mic.wav"]))
        XCTAssertNil(CaptureFileConfig.system(from: ["VOXTNEW_HELPER_SYSTEM_FILE": ""]))
        let c = CaptureFileConfig.system(from: ["VOXTNEW_HELPER_SYSTEM_FILE": "/tmp/call.wav",
                                                "VOXTNEW_HELPER_CAPTURE_FILE": "/tmp/mic.wav",
                                                "VOXTNEW_HELPER_CAPTURE_SPEED": "4"])
        XCTAssertEqual(c, CaptureFileConfig(path: "/tmp/call.wav", speed: 4))
    }

    func testSystemParamsCarryOver() {
        let c = SystemParams(sampleRate: 16000, frameMs: 100, levelHz: 20, maxSeconds: 7200).capture
        XCTAssertNil(c.devices)
        XCTAssertEqual(c.frameMs, 100)
        XCTAssertEqual(c.levelHz, 20)
        XCTAssertEqual(c.maxSeconds, 7200)
    }
}
