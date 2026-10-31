import Foundation

/// Cuts 16 kHz mono float audio into fixed frames of little-endian PCM16
/// (capture.pcm) and meters it (capture.level). Pure, so tests cover it.
public struct PCMFramer: Sendable {
    public let sampleRate: Int
    public let frameSamples: Int
    public let levelSamples: Int

    private var pending: [Int16] = []
    private var seq = 0
    private var total = 0
    // Meter.
    private var meterN = 0
    private var meterSum: Double = 0
    private var meterPeak: Float = 0

    public struct Frame: Sendable, Equatable {
        public var seq: Int
        /// Sample offset of the frame's first sample since the start.
        public var offset: Int
        public var pcm: Data
    }

    public struct Level: Sendable, Equatable {
        /// Seconds since the start, at the end of the window.
        public var seconds: Double
        /// RMS and peak in dBFS (-120 for silence).
        public var rms: Double
        public var peak: Double
    }

    public init(sampleRate: Int = 16000, frameMs: Int = 100, levelHz: Int = 20) {
        self.sampleRate = sampleRate
        frameSamples = max(1, sampleRate * frameMs / 1000)
        levelSamples = max(1, sampleRate / max(1, levelHz))
        pending.reserveCapacity(frameSamples * 2)
    }

    /// Adds samples; returns the frames and levels they complete.
    public mutating func push(_ samples: UnsafeBufferPointer<Float>) -> (frames: [Frame], levels: [Level]) {
        var frames: [Frame] = []
        var levels: [Level] = []
        for v in samples {
            let c = max(-1, min(1, v))
            pending.append(Int16(c * 32767))
            let a = abs(c)
            meterSum += Double(c * c)
            if a > meterPeak { meterPeak = a }
            meterN += 1
            total += 1
            if meterN >= levelSamples {
                levels.append(Level(seconds: Double(total) / Double(sampleRate),
                                    rms: Self.db(sqrt(meterSum / Double(meterN))), peak: Self.db(Double(meterPeak))))
                meterN = 0
                meterSum = 0
                meterPeak = 0
            }
            if pending.count >= frameSamples {
                frames.append(makeFrame(count: frameSamples))
            }
        }
        return (frames, levels)
    }

    public mutating func push(_ samples: [Float]) -> (frames: [Frame], levels: [Level]) {
        samples.withUnsafeBufferPointer { push($0) }
    }

    /// The partial last frame, if any (at stop).
    public mutating func flush() -> Frame? {
        pending.isEmpty ? nil : makeFrame(count: pending.count)
    }

    public var samplesSeen: Int { total }

    private mutating func makeFrame(count: Int) -> Frame {
        let chunk = pending.prefix(count)
        var data = Data(capacity: count * 2)
        for s in chunk {
            let le = s.littleEndian
            withUnsafeBytes(of: le) { data.append(contentsOf: $0) }
        }
        let f = Frame(seq: seq, offset: total - pending.count, pcm: data)
        pending.removeFirst(count)
        seq += 1
        return f
    }

    static func db(_ x: Double) -> Double {
        x <= 1e-6 ? -120 : max(-120, (20 * log10(x) * 10).rounded() / 10)
    }
}
