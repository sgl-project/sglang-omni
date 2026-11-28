/// When a converted stream's samples were in the air, in host ms: the
/// latest buffer's time stamp (already shifted by the device's latency)
/// for the sample it starts at, carried forward at the stream's rate.
public struct HostAnchor: Sendable {
    /// A stamp further than this from where the stream's earlier stamps
    /// put it is a break (a restart, a dropout).
    public static let breakMs = 20.0

    private let sampleRate: Int
    private var anchor: (sample: Int, ms: Double)?

    public init(sampleRate: Int) {
        self.sampleRate = sampleRate
    }

    /// The host time of sample, nil before the first stamp.
    public func ms(at sample: Int) -> Double? {
        anchor.map { $0.ms + Double(sample - $0.sample) * 1000 / Double(sampleRate) }
    }

    /// Whether a stamp of ms for sample breaks with the earlier ones: the
    /// samples not framed yet should go out first, on the earlier times.
    public func breaks(_ ms: Double, at sample: Int) -> Bool {
        guard let expected = self.ms(at: sample) else { return false }
        return abs(ms - expected) > Self.breakMs
    }

    public mutating func set(_ ms: Double, at sample: Int) {
        anchor = (sample, ms)
    }

    /// How many samples of silence fit between sample and a buffer stamped
    /// ms: padding never runs past the audio that follows it.
    public func room(before ms: Double, at sample: Int) -> Int? {
        self.ms(at: sample).map { max(0, Int((ms - $0) * Double(sampleRate) / 1000)) }
    }
}
