@preconcurrency import AVFoundation
import Foundation

// The audio path of a recording before PCMFramer: an input's buffers
// downmixed to mono at the input's rate, then resampled to the capture
// rate. The microphone (AVAudioEngine's tap) and the test-only file input
// share it, so a file recording goes through the same conversion, framing
// and notifications as a real one.

/// Mono at the input's rate → mono float at the capture rate: one
/// AVAudioConverter per input, so the resampler's state carries over from
/// one buffer to the next.
public final class MonoResampler: @unchecked Sendable {
    /// Mono at the input's rate (what downmix makes).
    public let mono: AVAudioFormat
    /// Mono at the capture rate.
    public let out: AVAudioFormat
    private let converter: AVAudioConverter

    public init?(inputRate: Double, rate: Int) {
        guard inputRate > 0, rate > 0,
              let mono = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: inputRate, channels: 1, interleaved: false),
              let out = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(rate), channels: 1, interleaved: false),
              let conv = AVAudioConverter(from: mono, to: out) else { return nil }
        conv.sampleRateConverterQuality = AVAudioQuality.medium.rawValue
        self.mono = mono
        self.out = out
        converter = conv
    }

    /// buffer (float, any channels, interleaved or not) averaged into a
    /// new mono buffer of format mono.
    public static func downmix(_ buffer: AVAudioPCMBuffer, into mono: AVAudioFormat) -> AVAudioPCMBuffer? {
        guard let data = buffer.floatChannelData else { return nil }
        let n = Int(buffer.frameLength)
        let channels = Int(buffer.format.channelCount)
        guard n > 0, channels > 0, let copy = AVAudioPCMBuffer(pcmFormat: mono, frameCapacity: AVAudioFrameCount(n)),
              let dst = copy.floatChannelData?[0] else { return nil }
        copy.frameLength = AVAudioFrameCount(n)
        if buffer.format.isInterleaved {
            for i in 0 ..< n {
                var v: Float = 0
                for c in 0 ..< channels { v += data[0][i * channels + c] }
                dst[i] = v / Float(channels)
            }
        } else if channels == 1 {
            dst.update(from: data[0], count: n)
        } else {
            for i in 0 ..< n {
                var v: Float = 0
                for c in 0 ..< channels { v += data[c][i] }
                dst[i] = v / Float(channels)
            }
        }
        return copy
    }

    /// Resamples one mono buffer (format mono); the samples come out as
    /// the converter has them, a few held back for the next buffer.
    public func convert(_ buffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer? {
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * out.sampleRate / buffer.format.sampleRate + 64)
        guard let converted = AVAudioPCMBuffer(pcmFormat: out, frameCapacity: capacity) else { return nil }
        let given = OnceFlag()
        var error: NSError?
        converter.convert(to: converted, error: &error) { _, status in
            if given.take() {
                status.pointee = .haveData
                return buffer
            }
            status.pointee = .noDataNow
            return nil
        }
        return converted.frameLength > 0 ? converted : nil
    }
}

/// True the first time take() is called (the converter's input block).
private final class OnceFlag: @unchecked Sendable {
    private var done = false
    func take() -> Bool {
        if done { return false }
        done = true
        return true
    }
}

// MARK: The test-only file input

/// VOXTNEW_HELPER_CAPTURE_FILE (tests only): capture.start records this
/// audio file instead of the microphone, at the pace of a microphone
/// (VOXTNEW_HELPER_CAPTURE_SPEED times as fast, default 1), then
/// silence until it is stopped, without asking for the microphone.
public struct CaptureFileConfig: Sendable, Equatable {
    public static let fileVariable = "VOXTNEW_HELPER_CAPTURE_FILE"
    public static let speedVariable = "VOXTNEW_HELPER_CAPTURE_SPEED"

    public var path: String
    public var speed: Double

    public init(path: String, speed: Double = 1) {
        self.path = path
        self.speed = speed
    }

    /// The configuration env asks for, or nil (the microphone).
    public static func from(environment env: [String: String]) -> CaptureFileConfig? {
        guard let path = env[fileVariable], !path.isEmpty else { return nil }
        var speed = 1.0
        if let s = env[speedVariable], let v = Double(s), v > 0, v.isFinite { speed = min(v, 100) }
        return CaptureFileConfig(path: path, speed: speed)
    }

    /// The file's name, which the recording reports as its device.
    public var name: String { (path as NSString).lastPathComponent }
}

/// An audio file played as a microphone: AVAudioFile's buffers (float,
/// the file's rate and channels, which the capture path downmixes and
/// resamples), in chunks of chunkMs, as many as are due at the time
/// given, then silence. Not thread-safe: one queue drives it.
public final class CaptureFilePlayer: @unchecked Sendable {
    public let format: AVAudioFormat
    /// Frames in the file.
    public let length: AVAudioFramePosition
    public let speed: Double
    public let chunk: AVAudioFrameCount
    private let file: AVAudioFile
    private var start: Double?
    private var sent: Int64 = 0
    /// Frames of the file read so far (silence after its end not counted).
    public private(set) var fileFrames: Int64 = 0

    public init(path: String, speed: Double = 1, chunkMs: Int = 20) throws {
        file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
        format = file.processingFormat
        length = file.length
        self.speed = speed > 0 ? speed : 1
        chunk = AVAudioFrameCount(max(1, Int(format.sampleRate * Double(max(1, chunkMs)) / 1000)))
    }

    /// True once the whole file was played.
    public var atEnd: Bool { fileFrames >= length }

    /// The chunks whose time has passed at now (monotonic ms), as a
    /// microphone delivers a buffer once it was heard: the first call
    /// starts the clock.
    public func due(now: Double) -> [AVAudioPCMBuffer] {
        let t0 = start ?? now
        start = t0
        let dueFrames = Int64(((now - t0) / 1000 * format.sampleRate * speed).rounded(.down))
        var out: [AVAudioPCMBuffer] = []
        while sent + Int64(chunk) <= dueFrames, let b = next() {
            sent += Int64(b.frameLength)
            out.append(b)
        }
        return out
    }

    /// The next chunk: the file's frames, padded with silence at its end.
    private func next() -> AVAudioPCMBuffer? {
        guard let b = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: chunk) else { return nil }
        var got: AVAudioFrameCount = 0
        if !atEnd {
            do {
                try file.read(into: b, frameCount: chunk)
                got = b.frameLength
            } catch {
                got = 0
            }
            fileFrames += Int64(got)
            if got == 0 { fileFrames = length } // unreadable rest: treat as the end
        }
        if got < chunk, let data = b.floatChannelData {
            let channels = format.isInterleaved ? 1 : Int(format.channelCount)
            let stride = format.isInterleaved ? Int(format.channelCount) : 1
            for c in 0 ..< channels {
                (data[c] + Int(got) * stride).update(repeating: 0, count: Int(chunk - got) * stride)
            }
            b.frameLength = chunk
        }
        return b
    }
}
