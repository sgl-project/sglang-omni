import Foundation
@testable import VoxtNewHelper
import VoxtNewHelperKit
import XCTest

final class CaptureSilenceTests: XCTestCase {
    func testIdleSystemCaptureEmitsSilenceLevels() throws {
        let output = try capture(CaptureParams(sampleRate: 16000, frameMs: 100, levelHz: 20)) { run in
            let clock = SystemCaptureClock(sampleRate: 16000, startedAt: 0)
            run.padSystemSilence(clock.silenceWhileIdle(at: 31000, written: 0))
        }
        XCTAssertEqual(output.levels.count, 616)
        XCTAssertEqual(output.levels.first?.seconds, 0.05)
        XCTAssertEqual(output.levels.last?.seconds, 30.8)
        XCTAssertTrue(output.levels.allSatisfy { $0.job == "system-idle" && $0.rms == -120 && $0.peak == -120 })
        XCTAssertEqual(output.frames.count, 308)
        for (index, frame) in output.frames.enumerated() {
            XCTAssertEqual(frame.job, "system-idle")
            XCTAssertEqual(frame.seq, index)
            XCTAssertEqual(frame.offset, index * 1600)
            XCTAssertEqual(frame.data, Data(repeating: 0, count: 3200))
        }
    }

    func testPaddingAcrossRatesAndFrameBoundaries() throws {
        for rate in [8000, 16000, 44100, 48000] {
            for frameMs in [10, 100, 137] {
                for levelHz in [1, 20, 60] {
                    let samples = rate + 37
                    let output = try capture(CaptureParams(sampleRate: rate, frameMs: frameMs, levelHz: levelHz)) { run in
                        for count in [0, -1, 1, 1023, 1024, 1025, samples - 3073] {
                            run.padSystemSilence(count)
                        }
                        XCTAssertTrue(run.finish(reason: CaptureReason.stopped))
                        XCTAssertFalse(run.finish(reason: CaptureReason.stopped))
                        XCTAssertFalse(run.cancel())
                    }
                    let frameSamples = rate * frameMs / 1000
                    XCTAssertEqual(output.frames.count, (samples + frameSamples - 1) / frameSamples)
                    for (index, frame) in output.frames.enumerated() {
                        XCTAssertEqual(frame.seq, index)
                        XCTAssertEqual(frame.offset, index * frameSamples)
                        XCTAssertEqual(frame.data, Data(repeating: 0, count: min(frameSamples, samples - frame.offset) * 2))
                    }
                    let levelSamples = rate / levelHz
                    XCTAssertEqual(output.levels.count, samples / levelSamples)
                    for (index, level) in output.levels.enumerated() {
                        XCTAssertEqual(level.seconds, Double((index + 1) * levelSamples) / Double(rate))
                        XCTAssertEqual(level.rms, -120)
                        XCTAssertEqual(level.peak, -120)
                    }
                    XCTAssertEqual(output.result?.samples, samples)
                    XCTAssertEqual(output.result?.seconds, Double(samples) / Double(rate))
                    XCTAssertEqual(output.result?.reason, CaptureReason.stopped)
                    XCTAssertNil(output.error)
                }
            }
        }
    }

    func testPaddingHonorsDurationLimit() throws {
        let output = try capture(CaptureParams(sampleRate: 16000, frameMs: 100, levelHz: 20, maxSeconds: 0.137)) { run in
            run.padSystemSilence(16000)
            run.padSystemSilence(16000)
            XCTAssertTrue(run.finish(reason: CaptureReason.maxSeconds))
        }
        XCTAssertEqual(output.result?.samples, 2192)
        XCTAssertEqual(output.result?.seconds, 0.137)
        XCTAssertEqual(output.result?.reason, CaptureReason.maxSeconds)
        XCTAssertEqual(output.frames.map { $0.data.count }, [3200, 1184])
        XCTAssertEqual(output.levels.map { $0.seconds }, [0.05, 0.1])
    }

    func testCancelEndsPaddedCaptureOnce() throws {
        let output = try capture(CaptureParams(sampleRate: 16000)) { run in
            run.padSystemSilence(8000)
            XCTAssertTrue(run.cancel())
            XCTAssertFalse(run.cancel())
            XCTAssertFalse(run.finish(reason: CaptureReason.stopped))
        }
        XCTAssertEqual(output.frames.count, 5)
        XCTAssertEqual(output.levels.count, 10)
        XCTAssertNil(output.result)
        XCTAssertEqual(output.error?.code, Code.cancelled)
    }

    private struct Output {
        var levels: [LevelParams] = []
        var frames: [PCMParams] = []
        var result: CaptureResult?
        var error: RPCError?
    }

    private func capture(_ params: CaptureParams, _ action: (CaptureRun) -> Void) throws -> Output {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        XCTAssertTrue(FileManager.default.createFile(atPath: url.path, contents: nil))
        defer { try? FileManager.default.removeItem(at: url) }
        let handle = try FileHandle(forUpdating: url)
        defer { try? handle.close() }
        let run = CaptureRun(out: Writer(handle: handle), job: "system-idle",
                             params: params,
                             system: true, file: nil, t0: 0, onEnd: { _ in })
        action(run)

        try handle.seek(toOffset: 0)
        let data = try XCTUnwrap(handle.readToEnd())
        var output = Output()
        for line in data.split(separator: 0x0A) {
            let message = try Incoming.decode(Data(line))
            if message.method == Method.captureLevel {
                output.levels.append(try JSONDecoder().decode(LevelParams.self, from: XCTUnwrap(message.params)))
            } else if message.method == Method.capturePCM {
                output.frames.append(try JSONDecoder().decode(PCMParams.self, from: XCTUnwrap(message.params)))
            } else if let result = message.result {
                XCTAssertNil(output.result)
                output.result = try JSONDecoder().decode(CaptureResult.self, from: result)
            } else if let error = message.error {
                XCTAssertNil(output.error)
                output.error = error
            } else {
                XCTFail("unexpected notification: \(message.method ?? "nil")")
            }
        }
        return output
    }
}
