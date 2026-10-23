@preconcurrency import AVFoundation
@testable import VoxtNewHelperKit
import XCTest

/// The test-only file input (VOXTNEW_HELPER_CAPTURE_FILE) and the capture
/// path it shares with the microphone: a 22.05 kHz stereo fixture played
/// at a microphone's pace, downmixed, resampled to 16 kHz and framed.
final class CaptureFileTests: XCTestCase {
    /// 0.4 s of 440 Hz, 22050 Hz, 16-bit stereo: the left channel at 0.5,
    /// the right at 0.25 (mono: 0.375, -11.5 dBFS RMS, -8.5 dBFS peak).
    static let tone = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .appendingPathComponent("Fixtures/audio/tone-22050-stereo.wav").path

    func testConfigFromEnvironment() {
        XCTAssertNil(CaptureFileConfig.from(environment: [:]))
        XCTAssertNil(CaptureFileConfig.from(environment: ["VOXTNEW_HELPER_CAPTURE_FILE": ""]))
        XCTAssertEqual(CaptureFileConfig.from(environment: ["VOXTNEW_HELPER_CAPTURE_FILE": "/tmp/a.wav"]),
                       CaptureFileConfig(path: "/tmp/a.wav", speed: 1))
        let fast = CaptureFileConfig.from(environment: ["VOXTNEW_HELPER_CAPTURE_FILE": "/tmp/a.wav", "VOXTNEW_HELPER_CAPTURE_SPEED": "4"])
        XCTAssertEqual(fast?.speed, 4)
        XCTAssertEqual(fast?.name, "a.wav")
        XCTAssertEqual(CaptureFileConfig.from(environment: ["VOXTNEW_HELPER_CAPTURE_FILE": "/a", "VOXTNEW_HELPER_CAPTURE_SPEED": "-1"])?.speed, 1)
    }

    func testPlayerPacesLikeAMicrophone() throws {
        let p = try CaptureFilePlayer(path: Self.tone, speed: 1, chunkMs: 20)
        XCTAssertEqual(p.format.sampleRate, 22050)
        XCTAssertEqual(p.format.channelCount, 2)
        XCTAssertEqual(p.length, 8820)
        XCTAssertEqual(p.chunk, 441)
        XCTAssertTrue(p.due(now: 1000).isEmpty, "nothing is heard at the start")
        XCTAssertEqual(p.due(now: 1019).count, 0)
        XCTAssertEqual(p.due(now: 1020).count, 1)
        XCTAssertEqual(p.due(now: 1100).count, 4) // 100 ms: 5 chunks in all
        XCTAssertFalse(p.atEnd)
        XCTAssertEqual(p.due(now: 1400).count, 15)
        XCTAssertTrue(p.atEnd)
        // After the file: silence, at the same pace.
        let tail = p.due(now: 1500)
        XCTAssertEqual(tail.count, 5)
        XCTAssertEqual(tail[0].frameLength, 441)
        XCTAssertEqual(tail[0].floatChannelData![0][100], 0)
        XCTAssertEqual(p.fileFrames, 8820)
    }

    func testPlayerSpeed() throws {
        let p = try CaptureFilePlayer(path: Self.tone, speed: 4, chunkMs: 20)
        XCTAssertTrue(p.due(now: 0).isEmpty)
        XCTAssertEqual(p.due(now: 100).count, 20) // 400 ms of audio in 100 ms
        XCTAssertTrue(p.atEnd)
    }

    /// The whole path: the file's chunks, downmixed and resampled to
    /// 16 kHz, framed into 100 ms of PCM16 with levels.
    func testFileThroughTheCapturePath() throws {
        let p = try CaptureFilePlayer(path: Self.tone, speed: 1, chunkMs: 20)
        let res = try XCTUnwrap(MonoResampler(inputRate: p.format.sampleRate, rate: 16000))
        var framer = PCMFramer(sampleRate: 16000, frameMs: 100, levelHz: 20)
        var frames: [PCMFramer.Frame] = []
        var levels: [PCMFramer.Level] = []
        for b in p.due(now: 0) + p.due(now: 400) {
            let mono = try XCTUnwrap(MonoResampler.downmix(b, into: res.mono))
            XCTAssertEqual(mono.format.channelCount, 1)
            guard let out = res.convert(mono) else { continue }
            XCTAssertEqual(out.format.sampleRate, 16000)
            let (f, l) = framer.push(UnsafeBufferPointer(start: out.floatChannelData![0], count: Int(out.frameLength)))
            frames += f
            levels += l
        }
        if let last = framer.flush() { frames.append(last) }
        // 0.4 s at 16 kHz, less what the resampler holds back.
        XCTAssertEqual(Double(framer.samplesSeen), 6400, accuracy: 200)
        XCTAssertEqual(frames.map(\.seq), Array(0 ..< frames.count))
        var offset = 0
        for f in frames {
            XCTAssertEqual(f.offset, offset)
            offset += f.pcm.count / 2
        }
        XCTAssertEqual(offset, framer.samplesSeen)
        XCTAssertEqual(frames[0].pcm.count, 3200)
        // Levels in the steady middle: the mono mix of 0.5 and 0.25.
        let mid = levels[2 ..< levels.count - 2]
        XCTAssertFalse(mid.isEmpty)
        for l in mid {
            XCTAssertEqual(l.rms, -11.5, accuracy: 0.5)
            XCTAssertEqual(l.peak, -8.5, accuracy: 0.5)
        }
    }

    func testDownmixInterleaved() throws {
        let fmt = try XCTUnwrap(AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48000, channels: 2, interleaved: true))
        let b = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: 4))
        b.frameLength = 4
        let d = b.floatChannelData![0]
        for i in 0 ..< 4 {
            d[2 * i] = 1
            d[2 * i + 1] = 0
        }
        let res = try XCTUnwrap(MonoResampler(inputRate: 48000, rate: 16000))
        let mono = try XCTUnwrap(MonoResampler.downmix(b, into: res.mono))
        XCTAssertEqual(mono.frameLength, 4)
        XCTAssertEqual(mono.floatChannelData![0][3], 0.5)
    }

    func testMissingFile() {
        XCTAssertThrowsError(try CaptureFilePlayer(path: "/nonexistent/voxtnew.wav"))
    }
}
