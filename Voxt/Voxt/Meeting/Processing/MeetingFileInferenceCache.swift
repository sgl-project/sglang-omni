import Foundation
@preconcurrency import MLX

/// File inference keeps a modest reusable allocator cache between work units.
/// This is a boundary policy, NOT a cap on model weights or total process memory.
/// Only a work-unit-scoped unused-cache limit is changed; live-memory limits and
/// speaker state are untouched.
nonisolated enum MeetingFileInferenceCache {
    static let retainedCacheThresholdBytes = 128 * 1_024 * 1_024

    /// Bound unused allocator storage while a file work unit is executing, not
    /// just after it returns. Preserve a stricter existing setting and restore
    /// the previous global value on every exit; never alter the live-memory limit.
    static func beginWorkUnit() -> @Sendable () -> Void {
        let previous = Memory.cacheLimit
        let applied = min(previous, retainedCacheThresholdBytes)
        if applied != previous { Memory.cacheLimit = applied }
        if Memory.cacheMemory > applied { Memory.clearCache() }
        return {
            // All file units share one permit. Do not overwrite an explicit
            // setting made by another caller while this unit was running.
            if applied != previous, Memory.cacheLimit == applied {
                Memory.cacheLimit = previous
            }
        }
    }

    static func shouldTrim(cacheBytes: Int, underPressure: Bool) -> Bool {
        cacheBytes > 0 && (underPressure || cacheBytes > retainedCacheThresholdBytes)
    }

    static func trimIfNeeded(underPressure: Bool) {
        let before = Memory.snapshot()
        guard shouldTrim(cacheBytes: before.cacheMemory, underPressure: underPressure) else { return }
        // MLX releases only allocator-owned unused buffers. Live weights, the
        // Sortformer FIFO and in-flight allocations stay owned by their users.
        // Called after file native work returns, before its permit is released,
        // or before admission while this coordinator's lane is idle.
        Memory.clearCache()
    }
}
