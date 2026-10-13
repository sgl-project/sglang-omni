// MeetingLocalInferenceCoordinator.swift
// Provides one bounded, priority-aware lane for heavyweight local meeting inference.

import Foundation

nonisolated enum MeetingLocalInferenceWorkClass: String, Sendable {
    case liveASRFeed
    case liveASRFinal
    case liveASRPartial
    case realtimeTranslation
    case fileASR
    case fileSpeakerAnalysis
    case finalASR
    case speakerAnalysis
    case detailTranslation
    case summary

    var priority: Int {
        switch self {
        case .liveASRFeed: return 100
        case .liveASRFinal: return 95
        case .liveASRPartial: return 85
        case .realtimeTranslation: return 70
        case .fileASR, .fileSpeakerAnalysis: return 10
        case .finalASR: return 60
        case .speakerAnalysis: return 50
        case .detailTranslation: return 40
        case .summary: return 20
        }
    }

    var waitsWhileRecording: Bool {
        switch self {
        case .fileASR, .fileSpeakerAnalysis, .finalASR, .speakerAnalysis, .detailTranslation, .summary:
            return true
        case .liveASRFeed, .liveASRFinal, .liveASRPartial, .realtimeTranslation:
            return false
        }
    }

    var isThermallyDeferrable: Bool {
        switch self {
        case .liveASRFeed, .liveASRFinal, .finalASR:
            return false
        case .liveASRPartial, .realtimeTranslation, .fileASR, .fileSpeakerAnalysis, .speakerAnalysis, .detailTranslation, .summary:
            return true
        }
    }

    var isMemoryDeferrable: Bool {
        switch self {
        case .liveASRFeed, .liveASRFinal, .finalASR:
            return false
        case .liveASRPartial, .realtimeTranslation, .fileASR, .fileSpeakerAnalysis, .speakerAnalysis, .detailTranslation, .summary:
            return true
        }
    }
}

nonisolated struct MeetingLocalInferenceStatistics: Equatable, Sendable {
    var submittedCount = 0
    var completedCount = 0
    var cancelledCount = 0
    var overloadedCount = 0
    var thermalDeferralCount = 0
    var memoryDeferralCount = 0
    var peakQueuedCount = 0
    var totalWaitMilliseconds: Int64 = 0
}

nonisolated enum MeetingLocalInferenceCoordinatorError: LocalizedError, Sendable {
    case overloaded
    case thermallyConstrained
    case memoryConstrained

    var errorDescription: String? {
        switch self {
        case .overloaded:
            return "The local meeting inference queue reached its safety limit."
        case .thermallyConstrained:
            return "The Mac is too warm for this nonessential local inference task right now."
        case .memoryConstrained:
            return "The Mac is under memory pressure, so this nonessential local inference task was deferred."
        }
    }
}

actor MeetingLocalInferenceCoordinator {
    static let shared = MeetingLocalInferenceCoordinator(
        maintainFileCache: { MeetingFileInferenceCache.trimIfNeeded(underPressure: $0) },
        readMemoryPressure: { MeetingMemoryPressureMonitor.currentFileConstraint() },
        beginFileWorkUnit: { MeetingFileInferenceCache.beginWorkUnit() }
    )

    private struct Waiter {
        let id: UUID
        let workClass: MeetingLocalInferenceWorkClass
        let sequence: Int64
        let submittedAt: ContinuousClock.Instant
        let continuation: CheckedContinuation<UUID, Error>
    }

    private static let maximumQueuedWork = 32

    private var activeToken: UUID?
    private var waiters: [Waiter] = []
    private var cancelledWaiterIDs = Set<UUID>()
    private var admittingWaiterIDs = Set<UUID>()
    private var sequence: Int64 = 0
    private var recordingActive = false
    private var memoryPressureConstrained = false
    private var memoryPressureSampleAvailable = false
    private var statistics = MeetingLocalInferenceStatistics()
    private let clock = ContinuousClock()
    private let maintainFileCache: @Sendable (Bool) -> Void
    private let readMemoryPressure: @Sendable () -> Bool?
    private let beginFileWorkUnit: @Sendable () -> (@Sendable () -> Void)

    // Isolated coordinators in tests must not initialize or mutate the MLX GPU
    // allocator. The shared production instance installs the file cache hooks.
    init(
        maintainFileCache: @escaping @Sendable (Bool) -> Void = { _ in },
        readMemoryPressure: @escaping @Sendable () -> Bool? = { nil },
        beginFileWorkUnit: @escaping @Sendable () -> (@Sendable () -> Void) = { {} }
    ) {
        self.maintainFileCache = maintainFileCache
        self.readMemoryPressure = readMemoryPressure
        self.beginFileWorkUnit = beginFileWorkUnit
    }

    func setRecordingActive(_ isActive: Bool) {
        recordingActive = isActive
        scheduleNextIfPossible()
    }

    func setMemoryPressureConstrained(_ isConstrained: Bool) {
        if memoryPressureConstrained != isConstrained {
            VoxtLog.meeting("Meeting memory pressure changed. source=notification, constrained=\(isConstrained)")
        }
        memoryPressureConstrained = isConstrained
        scheduleNextIfPossible()
    }

    private func fileMemoryBlocked() -> Bool {
        let sample = readMemoryPressure()
        memoryPressureSampleAvailable = sample != nil
        // Do not change the notification-based policy for unrelated live/LLM
        // callers. Only bounded file work treats a known warning as nonblocking.
        return sample ?? memoryPressureConstrained
    }

    func withPermit<T: Sendable>(
        _ workClass: MeetingLocalInferenceWorkClass,
        operation: @escaping @Sendable () async throws -> T
    ) async throws -> T {
        do {
            let token = try await acquire(workClass)
            let isFile = workClass == .fileASR || workClass == .fileSpeakerAnalysis
            let restoreCacheLimit: (@Sendable () -> Void)? = isFile ? beginFileWorkUnit() : nil
            defer {
                if isFile {
                    // Work has actually returned (including errors/cancellation).
                    // Trim before admitting another operation, not during native feed.
                    maintainFileCache(fileMemoryBlocked())
                }
                restoreCacheLimit?()
                release(token)
            }
            try Task.checkCancellation()
            let value = try await operation()
            statistics.completedCount += 1
            return value
        } catch {
            throw error
        }
    }

    func currentStatistics() -> MeetingLocalInferenceStatistics {
        statistics
    }

    func resetStatistics() {
        statistics = MeetingLocalInferenceStatistics()
    }

    private func acquire(_ workClass: MeetingLocalInferenceWorkClass) async throws -> UUID {
        try Task.checkCancellation()
        statistics.submittedCount += 1

        if workClass == .fileASR || workClass == .fileSpeakerAnalysis {
            return try await acquireFilePermit(workClass)
        }

        if workClass.isThermallyDeferrable {
            switch ProcessInfo.processInfo.thermalState {
            case .serious, .critical:
                statistics.thermalDeferralCount += 1
                VoxtLog.meetingWarning("Local inference admission rejected. workClass=\(workClass.rawValue), reason=thermal-pressure, thermalState=\(ProcessInfo.processInfo.thermalState.rawValue)")
                throw MeetingLocalInferenceCoordinatorError.thermallyConstrained
            case .nominal, .fair:
                break
            @unknown default:
                break
            }
        }

        if memoryPressureConstrained, workClass.isMemoryDeferrable {
            statistics.memoryDeferralCount += 1
            VoxtLog.meetingWarning("Local inference admission rejected. workClass=\(workClass.rawValue), reason=memory-pressure")
            throw MeetingLocalInferenceCoordinatorError.memoryConstrained
        }

        if activeToken == nil, canRun(workClass) {
            let token = UUID()
            activeToken = token
            return token
        }
        guard waiters.count < Self.maximumQueuedWork else {
            statistics.overloadedCount += 1
            VoxtLog.meetingWarning("Local inference admission rejected. workClass=\(workClass.rawValue), reason=inference-queue-full, queuedCount=\(waiters.count)")
            throw MeetingLocalInferenceCoordinatorError.overloaded
        }

        let waiterID = UUID()
        admittingWaiterIDs.insert(waiterID)
        let submittedAt = clock.now
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                admittingWaiterIDs.remove(waiterID)
                let cancelledBeforeAdmission = cancelledWaiterIDs.remove(waiterID) != nil
                if Task.isCancelled || cancelledBeforeAdmission {
                    statistics.cancelledCount += 1
                    continuation.resume(throwing: CancellationError())
                    return
                }
                sequence += 1
                waiters.append(
                    Waiter(
                        id: waiterID,
                        workClass: workClass,
                        sequence: sequence,
                        submittedAt: submittedAt,
                        continuation: continuation
                    )
                )
                waiters.sort {
                    if $0.workClass.priority == $1.workClass.priority {
                        return $0.sequence < $1.sequence
                    }
                    return $0.workClass.priority > $1.workClass.priority
                }
                statistics.peakQueuedCount = max(statistics.peakQueuedCount, waiters.count)
            }
        } onCancel: {
            Task { await self.cancelWaiter(waiterID) }
        }
    }

    private func cancelWaiter(_ id: UUID) {
        if let index = waiters.firstIndex(where: { $0.id == id }) {
            let waiter = waiters.remove(at: index)
            statistics.cancelledCount += 1
            waiter.continuation.resume(throwing: CancellationError())
        } else if admittingWaiterIDs.contains(id) {
            // Only retain a cancellation that can still race admission. A delayed
            // cancellation for an already granted/completed waiter needs no tombstone.
            cancelledWaiterIDs.insert(id)
        }
    }

    private func release(_ token: UUID) {
        guard activeToken == token else { return }
        activeToken = nil
        scheduleNextIfPossible()
    }

    private func scheduleNextIfPossible() {
        guard activeToken == nil,
              let index = waiters.firstIndex(where: { canRun($0.workClass) })
        else {
            return
        }
        let waiter = waiters.remove(at: index)
        let token = UUID()
        activeToken = token
        let waited = waiter.submittedAt.duration(to: clock.now)
        let components = waited.components
        let milliseconds = (components.seconds * 1_000) + (components.attoseconds / 1_000_000_000_000_000)
        statistics.totalWaitMilliseconds += Int64(max(0, milliseconds))
        waiter.continuation.resume(returning: token)
    }

    private func acquireFilePermit(_ workClass: MeetingLocalInferenceWorkClass) async throws -> UUID {
        // The file queue is serial. Recheck pressure AND lane availability on
        // every admission, including recovery after waiting for another caller.
        // No native work or permit is left running while sleeping.
        var didWait = false
        defer {
            if didWait {
                postFileResourceWait(isWaiting: false)
            }
        }
        var retryDelayMilliseconds = 250
        var reclaimedForPressure = false
        var loggedWait = false
        while true {
            try Task.checkCancellation()
            var fileMemoryConstrained = fileMemoryBlocked()
            if !fileMemoryConstrained { reclaimedForPressure = false }
            if activeToken == nil {
                // An inherited multi-GB cache can prevent the very recovery we
                // are waiting for. Reclaim once per pressure episode; subsequent
                // polls only trim if unused buffers again exceed the threshold.
                maintainFileCache(fileMemoryConstrained && !reclaimedForPressure)
                reclaimedForPressure = fileMemoryConstrained
                fileMemoryConstrained = fileMemoryBlocked()
            }
            try Task.checkCancellation()
            if activeToken == nil,
               canRun(workClass, memoryBlocked: fileMemoryConstrained),
               !waiters.contains(where: { canRun($0.workClass) }) {
                let token = UUID()
                activeToken = token
                return token
            }
            if !didWait {
                didWait = true
                postFileResourceWait(isWaiting: true)
            }
            if !loggedWait {
                VoxtLog.meeting("File analysis is waiting for system resources.")
                loggedWait = true
            }
            try await Task.sleep(for: .milliseconds(retryDelayMilliseconds))
            retryDelayMilliseconds = min(retryDelayMilliseconds * 2, 5_000)
            scheduleNextIfPossible()
        }
    }

    private func postFileResourceWait(isWaiting: Bool) {
        guard let taskID = MeetingFileTaskContext.taskID else { return }
        NotificationCenter.default.post(
            name: .voxtMeetingFileResourceWaitDidChange,
            object: nil,
            userInfo: [
                "taskID": taskID.uuidString,
                "isWaiting": isWaiting
            ]
        )
    }

    private func canRun(_ workClass: MeetingLocalInferenceWorkClass, memoryBlocked: Bool? = nil) -> Bool {
        let thermalState = ProcessInfo.processInfo.thermalState
        let thermalBlocked = workClass.isThermallyDeferrable
            && (thermalState == .serious || thermalState == .critical)
        return !(recordingActive && workClass.waitsWhileRecording)
            && !((memoryBlocked ?? memoryPressureConstrained) && workClass.isMemoryDeferrable)
            && !thermalBlocked
    }
}
