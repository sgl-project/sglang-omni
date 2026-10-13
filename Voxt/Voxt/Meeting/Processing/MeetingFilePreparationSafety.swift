import Foundation

/// Limits application-owned buffers and decoded output, not AVFoundation's
/// internal allocations. Decoder RSS still requires on-device stress testing.
nonisolated struct MeetingFilePreparationLimits: Sendable {
    // Keep the existing source admission ceiling until large-container decoder
    // stress tests justify raising it. Output limits are enforced independently.
    var maximumSourceBytes: Int64 = 4 * 1024 * 1024 * 1024
    var maximumDurationSeconds: TimeInterval = MeetingFileImportSupport.maximumAnalysisDurationSeconds
    var maximumOutputBytes: Int64 = Int64(MeetingFileImportSupport.maximumAnalysisDurationSeconds) * 16_000 * 2 + 44
    var minimumFreeDiskBytes: Int64 = 512 * 1024 * 1024
    static let maximumDecoderBufferBytes = 4 * 1024 * 1024
    static let conversionBufferBytes = 64 * 1024
    static let copyBufferBytes = 1024 * 1024

    var maximumSampleCount: Int {
        let duration = max(0, min(
            maximumDurationSeconds.isFinite ? maximumDurationSeconds : 0,
            MeetingFileImportSupport.maximumAnalysisDurationSeconds
        ))
        let byteLimit = max(0, min(maximumOutputBytes, MeetingImportedWAVFormat.maximumDataByteCount))
        return min(Int(duration * 16_000), Int(max(0, byteLimit - 44) / 2))
    }

    func checkDiskSpace(at url: URL, additionalBytes: Int64 = 0) throws {
        let values = try url.resourceValues(forKeys: [
            .volumeAvailableCapacityForImportantUsageKey, .volumeAvailableCapacityKey
        ])
        let available = values.volumeAvailableCapacityForImportantUsage
            ?? values.volumeAvailableCapacity.map(Int64.init)
        let (required, overflow) = max(0, additionalBytes).addingReportingOverflow(max(0, minimumFreeDiskBytes))
        guard !overflow, let available, available >= required else {
            throw MeetingFileTaskStagingError.insufficientDiskSpace
        }
    }
}

/// A paused reader retains at most its current bounded buffer. No producer runs
/// ahead of the consumer while memory/thermal pressure is elevated.
nonisolated final class MeetingFilePreparationResources: @unchecked Sendable {
    private let lock = NSLock()
    private var memoryConstrained = false
    private let monitor = MeetingMemoryPressureMonitor()

    init() {
        monitor.start(criticalOnly: true) { [weak self] constrained in
            guard let self else { return }
            self.lock.withLock { self.memoryConstrained = constrained }
        }
    }

    func waitUntilAvailable() async throws {
        var loggedWait = false
        while true {
            try Task.checkCancellation()
            let thermal = ProcessInfo.processInfo.thermalState
            let memoryPressure = MeetingMemoryPressureMonitor.currentFileConstraint()
                ?? lock.withLock { memoryConstrained }
            if !memoryPressure, thermal != .serious, thermal != .critical {
                return
            }
            if !loggedWait {
                VoxtLog.meeting("File preparation is waiting for system resources.")
                loggedWait = true
            }
            try await Task.sleep(for: .milliseconds(250))
        }
    }
}
