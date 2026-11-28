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

    func testIdleOutputKeepsItsPlaceBeforeFirstBuffer() {
        var clock = SystemCaptureClock(sampleRate: 16000, startedAt: 1000)
        XCTAssertEqual(clock.silenceWhileIdle(at: 1150, written: 0), 0)
        XCTAssertEqual(clock.silenceWhileIdle(at: 3100, written: 0), 30400)
        // A 20 ms buffer arrives 2.2 seconds into the recording. Account
        // for both its duration and the silence already emitted.
        XCTAssertEqual(clock.silenceBeforeBuffer(at: 3200, samples: 320, written: 30400), 4480)
        XCTAssertEqual(clock.silenceBeforeBuffer(at: 3220, samples: 320, written: 35200), 0)
        XCTAssertEqual(clock.silenceWhileIdle(at: 3350, written: 35520), 0)
        XCTAssertEqual(clock.silenceWhileIdle(at: 3720, written: 35520), 4800)
    }

    func testShortGapIsNotPaddedLater() {
        var clock = SystemCaptureClock(sampleRate: 16000, startedAt: 0)
        XCTAssertEqual(clock.silenceBeforeBuffer(at: 100, samples: 1600, written: 0), 0)
        // 150 ms later: under the grace, left alone (the archive places it by its time stamp).
        XCTAssertEqual(clock.silenceBeforeBuffer(at: 250, samples: 1600, written: 1600), 0)
        // After a 300 ms gap only that gap is padded (less the buffer), not
        // the 50 ms left alone before it as well.
        XCTAssertEqual(clock.silenceBeforeBuffer(at: 550, samples: 1600, written: 3200), 3200)
        XCTAssertEqual(clock.silenceWhileIdle(at: 1050, written: 8000), 4800)
    }

    func testCappedPaddingIsNotPaddedLater() {
        var clock = SystemCaptureClock(sampleRate: 16000, startedAt: 0)
        XCTAssertEqual(clock.silenceBeforeBuffer(at: 100, samples: 1600, written: 0), 0)
        // 400 ms of gap, but the buffer's time stamp leaves room for 100 ms
        // (the output's latency fell, AirPods to the speakers).
        XCTAssertEqual(clock.silenceBeforeBuffer(at: 500, samples: 1600, written: 1600, limit: 1600), 1600)
        // Idle after it: only what the idle time adds, not the 200 ms held back.
        XCTAssertEqual(clock.silenceWhileIdle(at: 1000, written: 4800), 4800)
    }

    func testHostAnchorCarriesStampsAndFindsBreaks() {
        var host = HostAnchor(sampleRate: 16000)
        XCTAssertNil(host.ms(at: 0))
        XCTAssertFalse(host.breaks(1000, at: 0))
        host.set(1000, at: 0)
        XCTAssertEqual(host.ms(at: 1600), 1100)
        XCTAssertFalse(host.breaks(1115, at: 1600)) // jitter
        XCTAssertTrue(host.breaks(1600, at: 1600)) // a restart's gap
        XCTAssertTrue(host.breaks(1050, at: 1600)) // an overlap
        // Padding up to a buffer stamped 1600 ms fits 500 ms, none past it.
        XCTAssertEqual(host.room(before: 1600, at: 1600), 8000)
        XCTAssertEqual(host.room(before: 1050, at: 1600), 0)
    }

    func testIdleOutputFillsToStopWithoutOverwritingAudio() {
        let clock = SystemCaptureClock(sampleRate: 16000, startedAt: 500)
        XCTAssertEqual(clock.silenceWhileIdle(at: 1500, written: 9600, stopping: true), 6400)
        XCTAssertEqual(clock.silenceWhileIdle(at: 1500, written: 16200, stopping: true), 0)
    }
}
