/// Keeps a system track on the recording's clock when Core Audio has no
/// output buffers to deliver. The clock uses monotonic milliseconds.
public struct SystemCaptureClock: Sendable {
    private static let graceMs = 200.0

    private let sampleRate: Int
    private let startedAt: Double
    private var lastBufferAt: Double

    public init(sampleRate: Int, startedAt: Double) {
        self.sampleRate = sampleRate
        self.startedAt = startedAt
        lastBufferAt = startedAt
    }

    /// Insert silence up to the beginning of the next buffer after a gap.
    /// A continuous callback stream is framed at its own sample rate.
    public mutating func silenceBeforeBuffer(at time: Double, samples: Int, written: Int) -> Int {
        let gap = time - lastBufferAt
        lastBufferAt = max(lastBufferAt, time)
        guard gap >= Self.graceMs else { return 0 }
        return max(0, sampleOffset(at: time) - samples - written)
    }

    /// Stay behind wall time so a buffer arriving after an idle period
    /// has room on the timeline. At stop there are no more buffers.
    public func silenceWhileIdle(at time: Double, written: Int, stopping: Bool = false) -> Int {
        guard time - lastBufferAt >= Self.graceMs else { return 0 }
        let target = stopping ? time : time - Self.graceMs
        return max(0, sampleOffset(at: target) - written)
    }

    private func sampleOffset(at time: Double) -> Int {
        Int(max(0, time - startedAt) * Double(sampleRate) / 1000)
    }
}
