import Foundation
@testable import VoxtNewHelper
import VoxtNewHelperKit
import XCTest

final class CaptureSilenceTests: XCTestCase {
    func testIdleSystemCaptureEmitsSilenceLevels() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        XCTAssertTrue(FileManager.default.createFile(atPath: url.path, contents: nil))
        defer { try? FileManager.default.removeItem(at: url) }
        let handle = try FileHandle(forUpdating: url)
        defer { try? handle.close() }
        let run = CaptureRun(out: Writer(handle: handle), job: "system-idle",
                             params: CaptureParams(sampleRate: 16000, frameMs: 100, levelHz: 20),
                             system: true, file: nil, t0: 0, onEnd: { _ in })
        let clock = SystemCaptureClock(sampleRate: 16000, startedAt: 0)
        run.padSystemSilence(clock.silenceWhileIdle(at: 31000, written: 0))

        try handle.seek(toOffset: 0)
        let data = try XCTUnwrap(handle.readToEnd())
        var levels: [LevelParams] = []
        var frames: [PCMParams] = []
        for line in data.split(separator: 0x0A) {
            let message = try Incoming.decode(Data(line))
            if message.method == Method.captureLevel {
                levels.append(try JSONDecoder().decode(LevelParams.self, from: XCTUnwrap(message.params)))
            } else if message.method == Method.capturePCM {
                frames.append(try JSONDecoder().decode(PCMParams.self, from: XCTUnwrap(message.params)))
            } else {
                XCTFail("unexpected notification: \(message.method ?? "nil")")
            }
        }
        XCTAssertEqual(levels.count, 616)
        XCTAssertEqual(levels.first?.seconds, 0.05)
        XCTAssertEqual(levels.last?.seconds, 30.8)
        XCTAssertTrue(levels.allSatisfy { $0.job == "system-idle" && $0.rms == -120 && $0.peak == -120 })
        XCTAssertEqual(frames.count, 308)
        for (index, frame) in frames.enumerated() {
            XCTAssertEqual(frame.job, "system-idle")
            XCTAssertEqual(frame.seq, index)
            XCTAssertEqual(frame.offset, index * 1600)
            XCTAssertEqual(frame.data, Data(repeating: 0, count: 3200))
        }
    }
}
