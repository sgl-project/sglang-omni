import Foundation

nonisolated enum ModelDownloadProgress {
    /// Shared display-only estimate, never used to validate downloaded bytes.
    /// Clamp before integer conversion so a corrupt size or clock jump cannot trap.
    static func inFlightBytes(
        progress: Progress,
        expectedFileBytes: Int64,
        startTime: Date,
        now: Date = Date()
    ) -> Int64 {
        let reported = max(progress.completedUnitCount, 0)
        guard reported == 0 else { return reported }
        let expected = Double(max(expectedFileBytes, 0))
        let elapsed = now.timeIntervalSince(startTime)
        guard elapsed.isFinite else { return 0 }
        let rate = max(expected / (10 * 60), 256 * 1024)
        let cap = Int64(expected * 0.95)
        return Int64(min(max(elapsed * rate, 0), Double(cap)))
    }
}
