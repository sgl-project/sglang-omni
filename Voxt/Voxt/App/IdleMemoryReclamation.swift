// IdleMemoryReclamation.swift
// Coordinates delayed MLX runtime reclamation after model unload.

import Darwin
import Foundation
import MLX

nonisolated enum ModelUnloadReclamationNotificationPolicy {
    static func shouldNotify(
        wasLoaded: Bool,
        isLoaded: Bool,
        isApplicationTerminating: Bool
    ) -> Bool {
        wasLoaded && !isLoaded && !isApplicationTerminating
    }
}

nonisolated struct DeepIdleMemoryReclamationState: Equatable, Sendable {
    let isApplicationTerminating: Bool
    let isRecordingSessionActive: Bool
    let isMeetingActive: Bool
    let hasPendingRecordingWork: Bool
    let hasPendingLLMWork: Bool
    let hasLoadedASRModel: Bool
    let hasActiveASRUse: Bool
    let hasLoadedLLMModel: Bool
    let hasActiveLLMInference: Bool
    let isTranscriberRecording: Bool
    let isTranscriberFinalizing: Bool
    var hasPendingASRLoad = false
    var hasPendingLLMLoad = false

    var disposition: DeepIdleMemoryReclamationDisposition {
        if isApplicationTerminating {
            return .stop
        }
        if hasLoadedASRModel || hasLoadedLLMModel {
            return .waitForModelUnload
        }
        if isRecordingSessionActive
            || isMeetingActive
            || hasPendingRecordingWork
            || hasPendingLLMWork
            || hasActiveASRUse
            || hasActiveLLMInference
            || isTranscriberRecording
            || isTranscriberFinalizing
            || hasPendingASRLoad
            || hasPendingLLMLoad
        {
            return .retryAfterTransientWork
        }
        return .reclaim
    }

    var canReclaim: Bool {
        disposition == .reclaim
    }

    var blockerSummary: String {
        var blockers: [String] = []
        if isApplicationTerminating { blockers.append("application-terminating") }
        if isRecordingSessionActive { blockers.append("recording-session") }
        if isMeetingActive { blockers.append("meeting-session") }
        if hasPendingRecordingWork { blockers.append("pending-recording-work") }
        if hasPendingLLMWork { blockers.append("pending-llm-work") }
        if hasLoadedASRModel { blockers.append("asr-model-loaded") }
        if hasActiveASRUse { blockers.append("asr-active-use") }
        if hasLoadedLLMModel { blockers.append("llm-model-loaded") }
        if hasActiveLLMInference { blockers.append("llm-active-inference") }
        if isTranscriberRecording { blockers.append("transcriber-recording") }
        if isTranscriberFinalizing { blockers.append("transcriber-finalizing") }
        if hasPendingASRLoad { blockers.append("asr-loading") }
        if hasPendingLLMLoad { blockers.append("llm-loading") }
        return blockers.isEmpty ? "none" : blockers.joined(separator: ",")
    }
}

nonisolated enum DeepIdleMemoryReclamationDisposition: Equatable, Sendable {
    case reclaim
    case retryAfterTransientWork
    case waitForModelUnload
    case stop
}

enum IdleMemoryReclamationSupport {
    static let deepReclamationDelay: Duration = .seconds(2)
    static let transientRetryDelay: Duration = .seconds(5)

    @discardableResult
    nonisolated static func releaseAllocatorCaches() async -> Int {
        // Reclaim only allocator-owned UNUSED buffers. Do not synchronize global
        // streams: that can block UI and accidentally wait for a newly started
        // session. MLX keeps in-flight allocations alive; later unload notifications
        // can reclaim buffers that are not yet idle. No live model state is mutated.
        let task = Task.detached(priority: .utility) {
            guard !Task.isCancelled else { return 0 }
            Memory.clearCache()
            return malloc_zone_pressure_relief(nil, 0)
        }
        return await withTaskCancellationHandler {
            await task.value
        } onCancel: {
            task.cancel()
        }
    }
}

@MainActor
extension AppDelegate {
    func cancelDeepIdleMemoryReclamation() {
        deepIdleMemoryReclamationID = UUID()
        pendingDeepIdleMemoryReclamationTask?.cancel()
        pendingDeepIdleMemoryReclamationTask = nil
    }

    func scheduleDeepIdleMemoryReclamation() {
        guard !isApplicationTerminating else { return }
        cancelDeepIdleMemoryReclamation()
        let reclamationID = UUID()
        deepIdleMemoryReclamationID = reclamationID
        pendingDeepIdleMemoryReclamationTask = Task { @MainActor [weak self] in
            var delay = IdleMemoryReclamationSupport.deepReclamationDelay
            var didLogTransientDeferral = false
            while !Task.isCancelled {
                do {
                    try await Task.sleep(for: delay)
                } catch {
                    return
                }
                guard let self,
                      !Task.isCancelled,
                      self.deepIdleMemoryReclamationID == reclamationID
                else { return }

                let disposition = await self.performDeepIdleMemoryReclamationIfPossible(
                    reclamationID: reclamationID
                )
                switch disposition {
                case .retryAfterTransientWork:
                    let state = self.deepIdleMemoryReclamationState
                    VoxtLog.modelInfo(
                        "Deep idle memory reclamation deferred. id=\(reclamationID.uuidString), blockers=\(state.blockerSummary)",
                        verbose: didLogTransientDeferral
                    )
                    didLogTransientDeferral = true
                    delay = IdleMemoryReclamationSupport.transientRetryDelay
                case .waitForModelUnload:
                    let state = self.deepIdleMemoryReclamationState
                    VoxtLog.modelInfo(
                        "Deep idle memory reclamation waiting for model unload. id=\(reclamationID.uuidString), blockers=\(state.blockerSummary)"
                    )
                    if self.deepIdleMemoryReclamationID == reclamationID {
                        self.pendingDeepIdleMemoryReclamationTask = nil
                    }
                    return
                case .reclaim, .stop:
                    if self.deepIdleMemoryReclamationID == reclamationID {
                        self.pendingDeepIdleMemoryReclamationTask = nil
                    }
                    return
                }
            }
        }
    }

    private var deepIdleMemoryReclamationState: DeepIdleMemoryReclamationState {
        DeepIdleMemoryReclamationState(
            isApplicationTerminating: isApplicationTerminating,
            isRecordingSessionActive: isSessionActive,
            isMeetingActive: meetingSessionCoordinator.isActive,
            hasPendingRecordingWork: pendingTranscriptionStartTask != nil
                || pendingMeetingStartupTask != nil
                || !recordingCaptureStartTasks.isEmpty,
            hasPendingLLMWork: llmRequests.hasPendingWork
                || !llmWarmupTasksByRepo.isEmpty
                || !remoteLLMWarmupTasksByKey.isEmpty
                || pauseLLMTask != nil
                || pendingDictionaryHistoryScanTask != nil
                || pendingAutomaticDictionaryLearningTask != nil,
            hasLoadedASRModel: mlxModelManager.hasLoadedModel,
            hasActiveASRUse: mlxModelManager.hasActiveUse,
            hasLoadedLLMModel: customLLMManager.hasLoadedInferenceModel,
            hasActiveLLMInference: customLLMManager.hasActiveInference,
            isTranscriberRecording: mlxTranscriber?.isRecording == true,
            isTranscriberFinalizing: mlxTranscriber?.isFinalizingTranscription == true,
            hasPendingASRLoad: mlxModelManager.hasOutstandingModelLoad,
            hasPendingLLMLoad: customLLMManager.hasOutstandingModelLoad
        )
    }

    private func performDeepIdleMemoryReclamationIfPossible(
        reclamationID: UUID
    ) async -> DeepIdleMemoryReclamationDisposition {
        guard !Task.isCancelled else { return .stop }
        let initialDisposition = deepIdleMemoryReclamationState.disposition
        guard initialDisposition == .reclaim else { return initialDisposition }

        let recordingVAD = recordingVoiceActivityFrameDecider
        let hadRecordingVAD = recordingVAD != nil
        recordingVoiceActivityFrameDecider = nil
        recordingVoiceActivitySegmenter = nil
        recordingVoiceActivityMode = nil
        recordingVoiceActivityUseCase = nil

        await recordingVAD?.releaseResources()
        await meetingSessionCoordinator.releaseIdleVADResources()

        // Actor calls above can yield to a new recording/meeting request. Recheck before
        // touching the shared MLX allocator or the reusable dictation transcriber.
        guard !Task.isCancelled else { return .stop }
        let finalDisposition = deepIdleMemoryReclamationState.disposition
        guard finalDisposition == .reclaim else {
            VoxtLog.modelInfo("Deep idle memory reclamation cancelled after activity resumed.", verbose: true)
            return finalDisposition
        }

        let memoryBefore = Memory.snapshot()
        let releasedTranscriber = mlxTranscriber?.releaseIdleResources() == true
        if releasedTranscriber {
            mlxTranscriber = nil
        }
        let reclaimedBytes = await IdleMemoryReclamationSupport.releaseAllocatorCaches()
        let memoryAfter = Memory.snapshot()
        VoxtLog.modelInfo(
            "Deep idle memory reclamation completed. id=\(reclamationID.uuidString), "
                + "recordingVADReleased=\(hadRecordingVAD), transcriberReleased=\(releasedTranscriber), "
                + "mlxActiveBeforeBytes=\(memoryBefore.activeMemory), mlxCacheBeforeBytes=\(memoryBefore.cacheMemory), "
                + "mlxActiveAfterBytes=\(memoryAfter.activeMemory), mlxCacheAfterBytes=\(memoryAfter.cacheMemory), "
                + "mallocPressureReliefBytes=\(reclaimedBytes)"
        )
        return .reclaim
    }
}
