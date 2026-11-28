/// Keeps a system track on the recording's clock when Core Audio has no
/// output buffers to deliver. The clock uses monotonic milliseconds.
/// Each buffer sets it again (it ends where the buffer arrived), so a
/// short gap left alone or an output clock slower than the Mac's is not
/// padded later, past the audio that came after it.
public struct SystemCaptureClock: Sendable {
    private static let graceMs = 200.0

    private let sampleRate: Int
    /// The time of sample base on the track.
    private var baseAt: Double
    private var base = 0
    private var lastBufferAt: Double

    public init(sampleRate: Int, startedAt: Double) {
        self.sampleRate = sampleRate
        baseAt = startedAt
        lastBufferAt = startedAt
    }

    /// Insert silence up to the beginning of the next buffer after a gap.
    /// A continuous callback stream is framed at its own sample rate. limit
    /// caps it (the room the buffer's time stamp leaves, HostAnchor.room).
    public mutating func silenceBeforeBuffer(at time: Double, samples: Int, written: Int, limit: Int? = nil) -> Int {
        let gap = time - lastBufferAt
        lastBufferAt = max(lastBufferAt, time)
        var missing = gap >= Self.graceMs ? max(0, sampleOffset(at: time) - samples - written) : 0
        if let limit { missing = min(missing, max(0, limit)) }
        baseAt = lastBufferAt
        base = written + missing + samples
        return missing
    }

    /// Stay behind wall time so a buffer arriving after an idle period
    /// has room on the timeline. At stop there are no more buffers.
    public func silenceWhileIdle(at time: Double, written: Int, stopping: Bool = false) -> Int {
        guard time - lastBufferAt >= Self.graceMs else { return 0 }
        let target = stopping ? time : time - Self.graceMs
        return max(0, sampleOffset(at: target) - written)
    }

    private func sampleOffset(at time: Double) -> Int {
        base + Int(max(0, time - baseAt) * Double(sampleRate) / 1000)
    }
}
