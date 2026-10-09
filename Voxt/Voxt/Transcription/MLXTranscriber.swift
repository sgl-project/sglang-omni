// MLXTranscriber.swift
// Provides MLXTranscriber for transcription engines.

import Foundation
import AVFoundation
import Combine
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioSTT
import MLXAudioVAD
import AudioToolbox

private struct MLXUnsafeSendableBox<Value>: @unchecked Sendable {
    nonisolated(unsafe) let value: Value
}

private enum MLXCaptureStartError: LocalizedError {
    case engineStartTimedOut(Double)

    var errorDescription: String? {
        switch self {
        case .engineStartTimedOut(let seconds):
            return "Audio engine failed to start within \(String(format: "%.0f", seconds))s."
        }
    }
}

/// Lets the non-`Sendable` `AVAudioEngine` be handed to a detached task so the blocking
/// `start()` call can run off the main actor. Only one start is admitted by the
/// recording-start barrier. Timeout/cancellation may request stop from another
/// thread; completion still waits for the native start to return.
private struct MLXAudioEngineBox: @unchecked Sendable {
    // CoreAudio's start/stop cancellation behavior requires device acceptance;
    // this box does not turn cancellation into an awaitable native-exit API.
    nonisolated(unsafe) let engine: AVAudioEngine
}

@MainActor
class MLXTranscriber: ObservableObject, TranscriberProtocol {
    @Published var isRecording = false
    @Published var isModelInitializing = false
    @Published var audioLevel: Float = 0.0
    @Published var transcribedText = ""
    @Published var isEnhancing = false
    @Published var isFinalizingTranscription = false

    var onTranscriptionFinished: ((String) -> Void)?
    var onPartialTranscription: ((String) -> Void)?
    var dictionaryEntryProvider: (() -> [DictionaryEntry])?

    private let audioEngine = AVAudioEngine()
    private let inferenceTaskPriority: TaskPriority
    private let audioLevelDelivery = MLXAudioLevelDelivery()
    private let sampleStore = AudioSampleStore()
    private let voiceActivityFrameStore = VoiceActivityFrameStore()
    private let voiceActivityFilteredSampleStore = AudioSampleStore()
    private var inputSampleRate: Double = 16000
    private var completedAudioArchiveURL: URL?
    private let modelManager: MLXModelManager
    private let transcriptionPurpose: MLXTranscriptionPurpose
    private var preferredInputDeviceID: AudioDeviceID?
    private let targetSampleRate = 16000

    /// Upper bound for the off-main `AVAudioEngine.start()`. A wedged coreaudiod can block
    /// `kAUStartIO` for ~10s; failing fast surfaces an overlay error instead of stalling.
    private static let captureStartTimeoutSeconds: Double = 6

    private let correctionPollInterval: Duration = .milliseconds(600)
    private let quickPassMinimumDurationSeconds: Double = 14.0
    private let qwenLiveFeedPollInterval: Duration = .milliseconds(100)
    private let senseVoiceDirectPassMaximumDurationSeconds: Double = 30.0
    private let senseVoiceChunkMaximumDurationSeconds: Double = 24.0
    private let senseVoiceChunkOverlapSeconds: Double = 0.35
    private let senseVoiceVADThreshold: Float = 0.5
    private let senseVoiceVADMinSpeechDurationMs = 220
    private let senseVoiceVADMinSilenceDurationMs = 420
    private let senseVoiceVADSpeechPadMs = 180

    private var sessionRevision = 0
    private var correctionLoopTask: Task<Void, Never>?
    private var finalizationTask: Task<Void, Never>?
    private var preloadTask: Task<Void, Never>?
    /// Survive `cancelActiveTasks()` at session start so hotkey-time load is not aborted.
    private var earlyPrewarmTask: Task<Void, Never>?
    private var captureWatchdogTask: Task<Void, Never>?
    private let liveSessionSetupTasks = TrackedTaskStore()
    private let sessionTasks = TrackedTaskStore()
    private let prewarmTasks = TrackedTaskStore()
    private let correctionPasses = MLXCorrectionPassCoordinator()
    private var activeLiveMode = MLXModelManager.liveMode(for: MLXModelManager.defaultModelRepo)
    private let nativeLiveRuntime = MLXNativeLiveRuntime()
    /// Keeps the ASR model resident for the whole recording + Final window so idle
    /// unload cannot race live release → postStopFinal.
    private var sessionModelPinned = false
    private var qwenFeedCursor = 0
    private var qwenVoiceActivityFeedCursor = 0
    private var voiceActivityFinalizationFilteringEnabled = false
    private var latestNativeLiveConfirmedText = ""
    private var latestNativeLivePreviewText = ""
    private var latestNativeLiveEndedText = ""
    private var latestNativeLiveEndedSegments: [MLXStructuredTranscriptSegment] = []
    private var nativeQwenLiveUsesAutomaticLanguageProtocol = false
    var sessionAllowsRealtimeTextDisplay = true
    private var didRetryCaptureStartup = false
    private var activeCaptureUsesPreferredInputDevice = false
    private var loggedSampleExtractionFailure = false
    private var activeSessionBehavior = MLXModelManager.transcriptionBehavior(
        for: MLXModelManager.defaultModelRepo
    )

    private var stableCommittedText = ""
    private var lastCandidateText = ""
    private var internalTranscribedText = ""
    private var nextCorrectionAtSeconds: Double = 6.0
    private(set) var lastCaptureMetrics: TranscriptionCaptureMetrics?
    @Published private(set) var latestSenseVoiceMetadata: SenseVoiceTranscriptMetadata?
    private var senseVoiceVADModel: SileroVAD?
    private var pendingRuntimeFailureMessage: String?

    init(
        modelManager: MLXModelManager,
        transcriptionPurpose: MLXTranscriptionPurpose = .dictation,
        inferenceTaskPriority: TaskPriority = .userInitiated
    ) {
        self.modelManager = modelManager
        self.transcriptionPurpose = transcriptionPurpose
        self.inferenceTaskPriority = inferenceTaskPriority
    }

    func setPreferredInputDevice(_ deviceID: AudioDeviceID?) {
        preferredInputDeviceID = deviceID
    }

    func requestPermissions() async -> Bool {
        await RecordingPermissionRequest.microphoneAccess()
    }

    func consumeCompletedAudioArchiveURL() -> URL? {
        let url = completedAudioArchiveURL
        completedAudioArchiveURL = nil
        return url
    }

    func discardCompletedAudioArchive() {
        removeCompletedAudioArchiveIfNeeded()
    }

    func consumePendingRuntimeFailureMessage() -> String? {
        let message = pendingRuntimeFailureMessage
        pendingRuntimeFailureMessage = nil
        return message
    }

    func consumeVoiceActivityFrames() -> [ASRVoiceActivityAudioFrame] {
        voiceActivityFrameStore.drain()
    }

    func configureVoiceActivityFinalizationFiltering(enabled: Bool) {
        voiceActivityFinalizationFilteringEnabled = enabled
        voiceActivityFilteredSampleStore.configureVoiceActivityFiltering(enabled: enabled)
    }

    func appendVoiceActivityFinalizationFrame(_ frame: ASRVoiceActivityAudioFrame, isSpeech: Bool) {
        voiceActivityFilteredSampleStore.appendVoiceActivityFrame(frame, isSpeech: isSpeech)
    }

    func finishVoiceActivityFinalizationFiltering() {
        voiceActivityFilteredSampleStore.finishVoiceActivityFiltering()
    }

    /// Starts ASR model load as early as possible (hotkey / session prepare), overlapping
    /// mic startup. Does not change UI or session lifecycle — load is shared with later
    /// `loadModel()` calls via the model manager coordinator.
    func prewarmModelForUpcomingSession() {
        guard modelManager.currentTranscriptionBehavior.preloadsOnRecordingStart else {
            return
        }
        pinModelForSessionIfNeeded()
        isModelInitializing = !modelManager.isCurrentModelLoaded
        startEarlyModelPrewarmIfNeeded()
    }

    /// Drops a prepare-time model pin when capture never entered recording/finalization
    /// (start failure or cancel-before-start). Idempotent with normal session teardown.
    func discardPreparedSessionModelUse() {
        guard !isRecording, !isFinalizingTranscription else { return }
        preloadTask?.cancel()
        preloadTask = nil
        earlyPrewarmTask?.cancel()
        earlyPrewarmTask = nil
        unpinModelForSessionIfNeeded()
        isModelInitializing = false
    }

    func startRecording() {
        Task { [weak self] in
            _ = await self?.startRecordingSession()
        }
    }

    /// Starts capture and returns a user-facing failure message, or `nil` on success.
    ///
    /// The blocking `AVAudioEngine.start()` runs off the main actor with a timeout
    /// (`startAudioCaptureGraphWithTimeout()`), so a wedged CoreAudio device can no longer
    /// freeze the hotkey/UI thread — it surfaces an overlay error instead.
    @discardableResult
    func startRecordingSession() async -> String? {
        guard !isRecording else { return nil }

        cancelActiveTasks()
        removeCompletedAudioArchiveIfNeeded()
        resetTransientState()
        sessionRevision += 1
        let revision = sessionRevision
        activeSessionBehavior = modelManager.currentTranscriptionBehavior
        activeLiveMode = resolvedSessionLiveMode()
        activeCaptureUsesPreferredInputDevice = preferredInputDeviceID != nil
        pinModelForSessionIfNeeded()
        isModelInitializing = !modelManager.isCurrentModelLoaded
        // Overlap model load with mic graph startup; do not wait for capture first.
        startModelPreloadIfNeeded(revision: revision)
        VoxtLog.asr(
            "MLX transcription session started. repo=\(modelManager.currentModelRepo), correctionMode=\(activeSessionBehavior.correctionMode), realtimeDisplay=\(sessionAllowsRealtimeTextDisplay), liveMode=\(String(describing: activeLiveMode)), modelState=\(String(describing: modelManager.state))",
            verbose: true
        )

        do {
            try await startAudioCaptureGraphWithTimeout()
            isRecording = true
            scheduleCaptureStartupWatchdog(revision: revision)

            if MLXTranscriptionPlanning.isNativeLiveMode(activeLiveMode) {
                startNativeLiveSession(revision: revision)
            } else if activeSessionBehavior.runsIntermediateCorrections {
                correctionLoopTask = sessionTasks.start { [weak self] in
                    await self?.runIntermediateCorrectionLoop(revision: revision)
                }
            } else {
                VoxtLog.asr(
                    "MLX transcription intermediate corrections disabled for repo=\(modelManager.currentModelRepo); finalization-only mode enabled.",
                    verbose: true
                )
            }
            return nil
        } catch {
            VoxtLog.asrError("MLXTranscriber start recording failed: \(error)")
            stopAudioEngine()
            audioEngine.inputNode.removeTap(onBus: 0)
            discardPreparedSessionModelUse()
            return AppLocalization.localizedString("Failed to start the microphone. Please try again.")
        }
    }

    func stopRecording() {
        guard isRecording else {
            discardPreparedSessionModelUse()
            return
        }

        stopAudioEngine()
        audioEngine.inputNode.removeTap(onBus: 0)
        isRecording = false

        correctionLoopTask?.cancel()
        correctionLoopTask = nil
        liveSessionSetupTasks.cancelAll()
        nativeLiveRuntime.stopFeeding()
        drainPendingSamplesIntoQwenLiveSession()
        nativeLiveRuntime.session?.stop()

        let revision = sessionRevision
        let sampleRate = inputSampleRate
        let callbackCount = sampleStore.callbacksReceived()
        let sampleCount = sampleStore.count()
        lastCaptureMetrics = TranscriptionCaptureMetrics(
            callbackCount: callbackCount,
            sampleCount: sampleCount,
            sampleRate: sampleRate
        )
        let capturedAudioSec = String(format: "%.2f", lastCaptureMetrics?.capturedAudioSeconds ?? 0)
        VoxtLog.asr(
            "MLX recording stop captured. callbacks=\(callbackCount), samples=\(sampleCount), sampleRate=\(Int(sampleRate)), capturedAudioSec=\(capturedAudioSec)",
            verbose: true
        )

        guard sampleCount > 0 else {
            isFinalizingTranscription = false
            if callbackCount > 0 {
                VoxtLog.asrWarning(
                    "MLX recording stopped with audio callbacks but no extracted samples. sampleRate=\(Int(sampleRate))"
                )
            }
            releaseNativeLiveSession(cancelSession: false)
            onTranscriptionFinished?("")
            releaseCompletedSessionResources(revision: revision)
            return
        }

        isFinalizingTranscription = true
        finalizationTask?.cancel()
        finalizationTask = sessionTasks.start { [weak self] in
            await self?.runFinalizationPipeline(revision: revision, sampleRate: sampleRate)
        }
    }

    func shutdownForApplicationTermination() async {
        let correctionTask = correctionLoopTask
        let finalizationTask = finalizationTask
        let preloadTask = preloadTask
        let earlyPrewarmTask = earlyPrewarmTask
        let watchdogTask = captureWatchdogTask
        let setupTasks = liveSessionSetupTasks.cancelAll()
        let correctionPassTask = correctionPasses.currentTask

        sessionRevision += 1
        stopAudioEngine()
        audioEngine.inputNode.removeTap(onBus: 0)
        isRecording = false
        isFinalizingTranscription = false
        cancelActiveTasks()
        prewarmTasks.cancelAll()
        self.earlyPrewarmTask = nil
        unpinModelForSessionIfNeeded()

        await correctionTask?.value
        await finalizationTask?.value
        await preloadTask?.value
        await earlyPrewarmTask?.value
        await watchdogTask?.value
        for task in setupTasks { await task.value }
        _ = await correctionPassTask?.value
        await nativeLiveRuntime.waitForRetirement()
        await sessionTasks.waitForAll()
        await prewarmTasks.waitForAll()

        sampleStore.clear()
        voiceActivityFrameStore.clear()
        voiceActivityFilteredSampleStore.clear()
        removeCompletedAudioArchiveIfNeeded()
        audioLevel = 0
        isEnhancing = false
        isFinalizingTranscription = false
        onTranscriptionFinished = nil
        onPartialTranscription = nil
    }

    /// Releases the reusable capture runtime after the model-level idle timeout has elapsed.
    /// The model manager remains responsible for model lifetime; this only tears down the
    /// dictation transcriber and its lightweight VAD/audio state.
    @discardableResult
    func releaseIdleResources() -> Bool {
        guard !isRecording, !isFinalizingTranscription, !correctionPasses.hasPendingWork,
              liveSessionSetupTasks.isEmpty, sessionTasks.isEmpty, prewarmTasks.isEmpty,
              !nativeLiveRuntime.hasPendingWork else { return false }

        sessionRevision += 1
        stopAudioEngine()
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.reset()
        cancelActiveTasks()
        earlyPrewarmTask?.cancel()
        earlyPrewarmTask = nil
        unpinModelForSessionIfNeeded()
        sampleStore.clear()
        voiceActivityFrameStore.clear()
        voiceActivityFilteredSampleStore.clear()
        removeCompletedAudioArchiveIfNeeded()
        senseVoiceVADModel = nil
        audioLevelDelivery.clear()
        audioLevel = 0
        isModelInitializing = false
        isEnhancing = false
        onTranscriptionFinished = nil
        onPartialTranscription = nil
        dictionaryEntryProvider = nil
        return true
    }

    /// Triggers an intermediate transcription pass while recording.
    /// Used to improve responsiveness during short pauses in speech.
    func forceIntermediateTranscription() {
        guard isRecording,
              activeSessionBehavior.runsIntermediateCorrections,
              !MLXTranscriptionPlanning.isNativeLiveMode(activeLiveMode)
        else { return }
        let revision = sessionRevision
        let sampleRate = inputSampleRate
        sessionTasks.start { [weak self] in
            _ = await self?.runManagedCorrectionPass(
                stage: .intermediate,
                revision: revision,
                explicitSamples: nil,
                sampleRate: sampleRate
            )
        }
    }

    func restartCaptureForPreferredInputDevice() throws {
        guard isRecording else { return }
        activeCaptureUsesPreferredInputDevice = preferredInputDeviceID != nil
        try startAudioCaptureGraph(usePreferredInputDevice: activeCaptureUsesPreferredInputDevice)
    }

    private func runIntermediateCorrectionLoop(revision: Int) async {
        while !Task.isCancelled, revision == sessionRevision, isRecording {
            do {
                try await Task.sleep(for: correctionPollInterval)
            } catch {
                return
            }

            guard revision == sessionRevision, isRecording else { return }
            let sampleCount = sampleStore.count()
            guard let decision = MLXTranscriptionPlanning.intermediateCorrectionDecision(
                sampleCount: sampleCount,
                sampleRate: inputSampleRate,
                nextCorrectionAtSeconds: nextCorrectionAtSeconds,
                behavior: activeSessionBehavior,
                firstCorrectionMinimumSeconds: currentFirstCorrectionMinimumSeconds,
                contextWindowSeconds: currentIntermediateContextWindowSeconds
            ) else { continue }
            let intermediateSamples = sampleStore.tail(sampleCount: decision.contextSampleCount)

            _ = await runManagedCorrectionPass(
                stage: .intermediate,
                revision: revision,
                explicitSamples: intermediateSamples,
                sampleRate: inputSampleRate
            )

            nextCorrectionAtSeconds = decision.elapsedSeconds + currentCorrectionIntervalSeconds
        }
    }

    private func runFinalizationPipeline(revision: Int, sampleRate: Double) async {
        guard !Task.isCancelled, revision == sessionRevision else { return }
        defer {
            if revision == sessionRevision {
                isFinalizingTranscription = false
                releaseCompletedSessionResources(revision: revision)
            }
        }
        pendingRuntimeFailureMessage = nil

        let fullSnapshot = sampleStore.snapshot()
        guard !fullSnapshot.isEmpty else {
            releaseNativeLiveSession(cancelSession: false)
            onTranscriptionFinished?("")
            sampleStore.clear()
            voiceActivityFilteredSampleStore.clear()
            return
        }
        voiceActivityFilteredSampleStore.finishVoiceActivityFiltering()
        let voiceActivityState = voiceActivityFilteredSampleStore.voiceActivityState()
        let capability = MLXModelCatalog.capability(for: modelManager.currentModelRepo)
        let vadPolicy = capability.vadPolicy
        let selection = MLXTranscriptionPlanning.finalizationSamples(
            fullSamples: fullSnapshot,
            voiceActivityFilteredSamples: voiceActivityFilteredSampleStore.snapshot(),
            localVADGateActive: voiceActivityState.enabled,
            observedVoiceActivityFrames: voiceActivityState.observedFrames,
            observedSpeech: voiceActivityState.observedSpeech,
            vadPolicy: vadPolicy,
            family: capability.family
        )
        let snapshot = selection.samples
        guard !snapshot.isEmpty else {
            VoxtLog.asr(
                "MLX finalization skipped because local VAD observed no speech. repo=\(modelManager.currentModelRepo), family=\(capability.family), fullAudioSec=\(String(format: "%.2f", Double(fullSnapshot.count) / safeSampleRate(sampleRate)))",
                verbose: true
            )
            stageCompletedAudioArchive(samples: fullSnapshot, sampleRate: sampleRate)
            transcribedText = ""
            publishPartial("")
            onTranscriptionFinished?("")
            sampleStore.clear()
            voiceActivityFilteredSampleStore.clear()
            releaseNativeLiveSession(cancelSession: false)
            return
        }

        let plan = MLXTranscriptionPlanning.finalizationPlan(
            sampleCount: snapshot.count,
            sampleRate: sampleRate,
            behavior: activeSessionBehavior,
            quickPassMinimumDurationSeconds: quickPassMinimumDurationSeconds,
            quickPassContextWindowSeconds: currentQuickPassContextWindowSeconds
        )
        // Offline Final is authoritative for quality. Do not wait on live `.ended` —
        // that only delayed postStopFinal while canceling the streaming session anyway.
        if MLXTranscriptionPlanning.isNativeLiveMode(activeLiveMode) {
            releaseNativeLiveSession(cancelSession: true)
        }
        let shouldRunQuickPass = MLXTranscriptionPlanning.shouldRunQuickStopPass(
            plan: plan,
            sessionAllowsRealtimeTextDisplay: sessionAllowsRealtimeTextDisplay,
            liveMode: activeLiveMode
        )
        let finalizationStartedAt = Date()
        VoxtLog.asr(
            "MLX finalization started. repo=\(modelManager.currentModelRepo), family=\(capability.family), audioSec=\(String(format: "%.2f", plan.durationSeconds)), source=\(selection.source.telemetryName), vadPolicy=\(String(describing: vadPolicy)), externalTrim=\(MLXTranscriptionPlanning.allowsExternalFinalSpeechTrim(vadPolicy: vadPolicy, family: capability.family)), fullAudioSec=\(String(format: "%.2f", Double(fullSnapshot.count) / safeSampleRate(sampleRate))), quickPass=\(shouldRunQuickPass), recognitionPreset=\(capability.configurationCapabilities.contains(.recognitionPreset))",
            verbose: true
        )
        // History WAV export is independent of Final inference — overlap I/O with ASR.
        let archiveTask = Task.detached(priority: .utility) {
            Self.exportCompletedAudioArchiveURL(samples: fullSnapshot, sampleRate: sampleRate)
        }
        let quickSource: [Float]?
        if shouldRunQuickPass, let quickPassSampleCount = plan.quickPassSampleCount {
            quickSource = latestWindow(from: snapshot, maxCount: quickPassSampleCount)
        } else {
            quickSource = nil
        }

        let quickResult: MLXCorrectionPassResult
        if let quickSource {
            quickResult = await runManagedCorrectionPass(
                stage: .postStopQuick,
                revision: revision,
                explicitSamples: quickSource,
                sampleRate: sampleRate
            )
        } else {
            quickResult = .success(nil)
        }

        let finalResult = await runManagedCorrectionPass(
            stage: .postStopFinal,
            revision: revision,
            explicitSamples: snapshot,
            sampleRate: sampleRate
        )

        guard !Task.isCancelled, revision == sessionRevision else {
            archiveTask.cancel()
            if let archiveURL = await archiveTask.value {
                try? FileManager.default.removeItem(at: archiveURL)
            }
            return
        }
        let archiveURL = await archiveTask.value
        guard !Task.isCancelled, revision == sessionRevision else {
            if let archiveURL { try? FileManager.default.removeItem(at: archiveURL) }
            return
        }
        if let archiveURL {
            removeCompletedAudioArchiveIfNeeded()
            completedAudioArchiveURL = archiveURL
        } else {
            // Fallback keeps history available if the overlapped export failed.
            stageCompletedAudioArchive(samples: fullSnapshot, sampleRate: sampleRate)
        }
        let fallbackText: String
        if !latestNativeLiveEndedSegments.isEmpty {
            // Prefer reliable timed segments captured at live `.ended` when post-stop
            // correction is empty or unavailable (for example Nemotron sentence timing).
            fallbackText = latestNativeLiveEndedSegments
                .map(\.text)
                .joined(separator: " ")
        } else if !latestNativeLiveEndedText.isEmpty {
            fallbackText = latestNativeLiveEndedText
        } else if !latestNativeLivePreviewText.isEmpty {
            fallbackText = latestNativeLivePreviewText
        } else {
            fallbackText = transcribedText
        }
        let resolved = normalizeText(finalResult.text ?? quickResult.text ?? fallbackText)
        if resolved.isEmpty, let error = finalResult.error ?? quickResult.error {
            let failureMessage = runtimeFailureMessage(for: error)
            pendingRuntimeFailureMessage = failureMessage
            VoxtLog.asrError(
                "MLX finalization produced no transcript because inference failed. repo=\(modelManager.currentModelRepo), error=\(failureMessage)"
            )
        }
        transcribedText = resolved
        publishPartial(resolved)
        onTranscriptionFinished?(resolved)
        let finalizationElapsedMs = Int(Date().timeIntervalSince(finalizationStartedAt) * 1000)
        VoxtLog.asr(
            "MLX finalization completed. repo=\(modelManager.currentModelRepo), audioSec=\(String(format: "%.2f", plan.durationSeconds)), textChars=\(resolved.count), finalizationMs=\(finalizationElapsedMs), source=\(selection.source.telemetryName)",
            verbose: true
        )
        sampleStore.clear()
        voiceActivityFilteredSampleStore.clear()
        releaseNativeLiveSession(cancelSession: false)
    }

    /// Drops per-session buffers and completed task/callback references while keeping the
    /// audio engine object and loaded model reusable for the next recording.
    private func releaseCompletedSessionResources(revision: Int) {
        guard revision == sessionRevision, !isRecording else { return }

        sampleStore.clear()
        voiceActivityFrameStore.clear()
        voiceActivityFilteredSampleStore.clear()
        preloadTask?.cancel()
        preloadTask = nil
        earlyPrewarmTask?.cancel()
        earlyPrewarmTask = nil
        captureWatchdogTask?.cancel()
        captureWatchdogTask = nil
        finalizationTask = nil
        audioLevelDelivery.clear()
        audioLevel = 0
        isModelInitializing = false
        onTranscriptionFinished = nil
        onPartialTranscription = nil
        dictionaryEntryProvider = nil
        unpinModelForSessionIfNeeded()
    }

    private func pinModelForSessionIfNeeded() {
        guard !sessionModelPinned else { return }
        modelManager.beginActiveUse()
        sessionModelPinned = true
    }

    private func unpinModelForSessionIfNeeded() {
        guard sessionModelPinned else { return }
        sessionModelPinned = false
        modelManager.endActiveUse()
    }

    private func runManagedCorrectionPass(
        stage: MLXCorrectionPassKind,
        revision: Int,
        explicitSamples: [Float]?,
        sampleRate: Double
    ) async -> MLXCorrectionPassResult {
        await correctionPasses.run(
            kind: stage,
            isCurrent: { [weak self] in
                guard let self, revision == self.sessionRevision else { return false }
                return stage != .intermediate || self.isRecording
            },
            operation: { [weak self] in
                guard let self else { return .success(nil) }
                return await self.executeCorrectionPass(
                    stage: stage,
                    revision: revision,
                    explicitSamples: explicitSamples,
                    sampleRate: sampleRate
                )
            }
        )
    }

    private func executeCorrectionPass(
        stage: MLXCorrectionPassKind,
        revision: Int,
        explicitSamples: [Float]?,
        sampleRate: Double
    ) async -> MLXCorrectionPassResult {
        guard revision == sessionRevision else { return .success(nil) }
        let rawSamples = explicitSamples ?? sampleStore.snapshot()
        guard !rawSamples.isEmpty else { return .success(nil) }
        let audioSeconds = Double(rawSamples.count) / safeSampleRate(sampleRate)
        let repo = modelManager.currentModelRepo
        let passStartedAt = Date()

        do {
            try Task.checkCancellation()
            let prepareStartedAt = Date()
            let targetRate = targetSampleRate
            // Resample/copy off the main actor and overlap with model pin/load.
            let prepareTask = Task.detached(priority: .userInitiated) {
                try Self.prepareInputSamplesDetached(
                    rawSamples,
                    sampleRate: sampleRate,
                    targetSampleRate: targetRate
                )
            }
            modelManager.beginActiveUse()
            defer { modelManager.endActiveUse() }
            let model = try await modelManager.loadModel()
            try Task.checkCancellation()
            guard revision == sessionRevision else { return .success(nil) }
            isModelInitializing = false
            let audioSamples = try await prepareTask.value
            try Task.checkCancellation()
            guard revision == sessionRevision else { return .success(nil) }
            let inferenceConfiguration = resolvedInferenceConfiguration(
                for: stage,
                audioDurationSeconds: audioSeconds
            )
            let prepareElapsedMs = Int(Date().timeIntervalSince(prepareStartedAt) * 1000)
            let inferenceStartedAt = Date()
            let inferenceResult = try await runStreamingInference(
                model: model,
                audioSamples: audioSamples,
                inferenceConfiguration: inferenceConfiguration
            )
            try Task.checkCancellation()
            guard revision == sessionRevision else { return .success(nil) }
            let inferenceElapsedMs = Int(Date().timeIntervalSince(inferenceStartedAt) * 1000)

            let rawCandidate = normalizeText(inferenceResult.rawText)
            let candidate = normalizeText(MLXTranscriptionPlanning.removingKnownASRContextLeakage(from: rawCandidate))
            if candidate != rawCandidate {
                VoxtLog.asrWarning(
                    "MLX ASR context leakage removed. repo=\(repo), stage=\(stageLabel(for: stage)), rawChars=\(rawCandidate.count), outputChars=\(candidate.count)"
                )
            }
            guard !candidate.isEmpty else { return .success(nil) }
            latestSenseVoiceMetadata = inferenceResult.senseVoiceMetadata
            applyCandidate(candidate, stage: stage)
            let elapsedMs = Int(Date().timeIntervalSince(passStartedAt) * 1000)
            VoxtLog.asr(
                "MLX correction pass completed. repo=\(repo), stage=\(stageLabel(for: stage)), audioSec=\(String(format: "%.2f", audioSeconds)), elapsedMs=\(elapsedMs), prepareMs=\(prepareElapsedMs), inferenceMs=\(inferenceElapsedMs), maxTokens=\(inferenceConfiguration.generationParameters.maxTokens), textChars=\(candidate.count)",
                verbose: true
            )
            return .success(candidate)
        } catch is CancellationError {
            let elapsedMs = Int(Date().timeIntervalSince(passStartedAt) * 1000)
            VoxtLog.asr(
                "MLX correction pass cancelled. repo=\(repo), stage=\(stageLabel(for: stage)), audioSec=\(String(format: "%.2f", audioSeconds)), elapsedMs=\(elapsedMs)",
                verbose: true
            )
            return .success(nil)
        } catch {
            guard revision == sessionRevision else { return .success(nil) }
            isModelInitializing = false
            let elapsedMs = Int(Date().timeIntervalSince(passStartedAt) * 1000)
            VoxtLog.asrError(
                "MLXTranscriber \(stageLabel(for: stage)) pass failed. repo=\(repo), audioSec=\(String(format: "%.2f", audioSeconds)), elapsedMs=\(elapsedMs), error=\(error.localizedDescription)"
            )
            return .failure(error)
        }
    }

    private func resetTransientState() {
        sampleStore.clear()
        voiceActivityFrameStore.clear()
        voiceActivityFilteredSampleStore.configureVoiceActivityFiltering(
            enabled: voiceActivityFinalizationFilteringEnabled
        )
        qwenFeedCursor = 0
        qwenVoiceActivityFeedCursor = 0
        transcribedText = ""
        internalTranscribedText = ""
        audioLevelDelivery.clear()
        audioLevel = 0
        isModelInitializing = false
        isFinalizingTranscription = false
        didRetryCaptureStartup = false
        activeCaptureUsesPreferredInputDevice = preferredInputDeviceID != nil
        stableCommittedText = ""
        lastCandidateText = ""
        nextCorrectionAtSeconds = currentCorrectionIntervalSeconds
        loggedSampleExtractionFailure = false
        lastCaptureMetrics = nil
        latestSenseVoiceMetadata = nil
        pendingRuntimeFailureMessage = nil
        qwenFeedCursor = 0
        latestNativeLiveConfirmedText = ""
        latestNativeLivePreviewText = ""
        latestNativeLiveEndedText = ""
        latestNativeLiveEndedSegments = []
        nativeQwenLiveUsesAutomaticLanguageProtocol = false
    }

    private var currentCorrectionIntervalSeconds: Double {
        currentCorrectionCadence.correctionIntervalSeconds
    }

    private var currentFirstCorrectionMinimumSeconds: Double {
        currentCorrectionCadence.firstCorrectionMinimumSeconds
    }

    private var currentIntermediateContextWindowSeconds: Double {
        currentCorrectionCadence.intermediateContextWindowSeconds
    }

    private var currentQuickPassContextWindowSeconds: Double {
        currentCorrectionCadence.quickPassContextWindowSeconds
    }

    private var currentCorrectionCadence: MLXCorrectionCadence {
        MLXTranscriptionPlanning.correctionCadence(
            for: modelManager.currentModelRepo,
            sessionAllowsRealtimeTextDisplay: sessionAllowsRealtimeTextDisplay
        )
    }

    private func stopAudioEngine() {
        if audioEngine.isRunning {
            audioEngine.stop()
        }
    }

    /// Configures the engine, input device, tap and `prepare()` — everything except the
    /// blocking `start()`. Returns the data needed to log once the engine is running.
    private func configureAudioCaptureGraph(usePreferredInputDevice: Bool? = nil) -> (format: AVAudioFormat, usedPreferredDevice: Bool) {
        if audioEngine.isRunning {
            audioEngine.stop()
        }
        audioEngine.reset()

        let inputNode = audioEngine.inputNode
        inputNode.removeTap(onBus: 0)

        let shouldUsePreferredInputDevice = usePreferredInputDevice ?? activeCaptureUsesPreferredInputDevice
        activeCaptureUsesPreferredInputDevice = shouldUsePreferredInputDevice
        let didApplyPreferredInputDevice = shouldUsePreferredInputDevice
            ? applyPreferredInputDeviceIfNeeded(inputNode: inputNode)
            : false
        let activeInputDeviceID = didApplyPreferredInputDevice ? preferredInputDeviceID : AudioInputDeviceManager.defaultInputDeviceID()
        let nodeOutputFormat = inputNode.outputFormat(forBus: 0)
        let hardwareSampleRate = AudioInputDeviceManager.nominalSampleRate(for: activeInputDeviceID)
        let recordingFormat = AudioInputDeviceManager.captureTapFormat(
            nodeOutputFormat: nodeOutputFormat,
            hardwareSampleRate: hardwareSampleRate
        )
        inputSampleRate = recordingFormat.sampleRate

        if abs(recordingFormat.sampleRate - nodeOutputFormat.sampleRate) > 1 {
            VoxtLog.warning(
                "MLX transcriber adjusted input tap format. deviceID=\(activeInputDeviceID.map(String.init(describing:)) ?? "default"), hardwareSampleRate=\(hardwareSampleRate.map { String(Int($0.rounded())) } ?? "unknown"), nodeSampleRate=\(Int(nodeOutputFormat.sampleRate.rounded())), tapSampleRate=\(Int(recordingFormat.sampleRate.rounded()))"
            )
        }

        let sampleStore = self.sampleStore
        let voiceActivityFrameStore = self.voiceActivityFrameStore

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            guard let self else { return }
            sampleStore.noteCallback()

            guard let samples = AudioLevelMeter.monoSamples(from: buffer), !samples.isEmpty else {
                if !self.loggedSampleExtractionFailure {
                    self.loggedSampleExtractionFailure = true
                    VoxtLog.asrWarning(
                        """
                        MLX audio sample extraction failed. sampleRate=\(Int(buffer.format.sampleRate)), channels=\(buffer.format.channelCount), format=\(buffer.format.commonFormat.rawValue), interleaved=\(buffer.format.isInterleaved)
                        """
                    )
                }
                return
            }

            sampleStore.append(samples)
            let normalized = AudioLevelMeter.normalizedLevel(fromSamples: samples)
            voiceActivityFrameStore.append(
                samples: samples,
                sampleRate: buffer.format.sampleRate,
                level: normalized
            )
            self.audioLevelDelivery.submit(normalized) { [weak self] latestLevel in
                self?.audioLevel = latestLevel
            }
        }

        audioEngine.prepare()
        return (recordingFormat, didApplyPreferredInputDevice)
    }

    private func logCaptureStarted(format: AVAudioFormat, usedPreferredDevice: Bool) {
        VoxtLog.asr(
            "MLX audio capture started. sampleRate=\(Int(format.sampleRate)), channels=\(format.channelCount), format=\(format.commonFormat.rawValue), interleaved=\(format.isInterleaved), routing=\(usedPreferredDevice ? "preferred" : "system-default"), deviceID=\(usedPreferredDevice ? (preferredInputDeviceID.map(String.init(describing:)) ?? "default") : "system-default")",
            verbose: true
        )
    }

    /// Synchronous start. Used only by mid-session recovery/device-switch paths, which are
    /// already off the hotkey thread. The hotkey start path uses the async timeout variant.
    private func startAudioCaptureGraph(usePreferredInputDevice: Bool? = nil) throws {
        let context = configureAudioCaptureGraph(usePreferredInputDevice: usePreferredInputDevice)
        try audioEngine.start()
        logCaptureStarted(format: context.format, usedPreferredDevice: context.usedPreferredDevice)
    }

    /// Same as `startAudioCaptureGraph`, but runs the blocking `AVAudioEngine.start()` off the
    /// main actor and gives up after `captureStartTimeoutSeconds`, so a wedged coreaudiod can
    /// never freeze the hotkey/UI thread.
    private func startAudioCaptureGraphWithTimeout(usePreferredInputDevice: Bool? = nil) async throws {
        let context = configureAudioCaptureGraph(usePreferredInputDevice: usePreferredInputDevice)
        try await startConfiguredEngineWithTimeout(timeoutSeconds: Self.captureStartTimeoutSeconds)
        logCaptureStarted(format: context.format, usedPreferredDevice: context.usedPreferredDevice)
    }

    /// Runs the already-configured engine's blocking `start()` on a detached task, racing it
    /// against a timeout. On timeout the engine is stopped so the start call unwinds promptly.
    private func startConfiguredEngineWithTimeout(timeoutSeconds: Double) async throws {
        let engineBox = MLXAudioEngineBox(engine: audioEngine)
        try await withTaskCancellationHandler {
            try Task.checkCancellation()
            try await withThrowingTaskGroup(of: Void.self) { group in
                group.addTask {
                    // Detached so the blocking start can never run on the main actor.
                    try await Task.detached(priority: .userInitiated) {
                        try engineBox.engine.start()
                    }.value
                    if Task.isCancelled { engineBox.engine.stop() }
                    try Task.checkCancellation()
                }
                group.addTask {
                    try await Task.sleep(for: .seconds(timeoutSeconds))
                    engineBox.engine.stop()
                    throw MLXCaptureStartError.engineStartTimedOut(timeoutSeconds)
                }
                defer { group.cancelAll() }
                _ = try await group.next()
            }
        } onCancel: {
            engineBox.engine.stop()
        }
    }

    private func cancelActiveTasks() {
        sessionTasks.cancelAll()
        correctionLoopTask?.cancel()
        correctionLoopTask = nil
        finalizationTask?.cancel()
        finalizationTask = nil
        liveSessionSetupTasks.cancelAll()
        correctionPasses.cancel()
        isFinalizingTranscription = false
        preloadTask?.cancel()
        preloadTask = nil
        captureWatchdogTask?.cancel()
        captureWatchdogTask = nil
        releaseNativeLiveSession(cancelSession: true)
    }

    private func stageCompletedAudioArchive(samples: [Float], sampleRate: Double) {
        removeCompletedAudioArchiveIfNeeded()
        guard let tempURL = Self.exportCompletedAudioArchiveURL(samples: samples, sampleRate: sampleRate) else {
            return
        }
        completedAudioArchiveURL = tempURL
    }

    private nonisolated static func exportCompletedAudioArchiveURL(
        samples: [Float],
        sampleRate: Double
    ) -> URL? {
        guard !samples.isEmpty else { return nil }
        let tempURL = HistoryAudioArchiveSupport.temporaryArchiveURL(prefix: "voxt-mlx-history")
        do {
            if try HistoryAudioArchiveSupport.exportWAV(samples: samples, sampleRate: sampleRate, to: tempURL) {
                return tempURL
            }
        } catch {
            try? FileManager.default.removeItem(at: tempURL)
            VoxtLog.asrWarning("MLX completed audio archive export failed: \(error.localizedDescription)")
        }
        return nil
    }

    private func removeCompletedAudioArchiveIfNeeded() {
        guard let completedAudioArchiveURL else { return }
        try? FileManager.default.removeItem(at: completedAudioArchiveURL)
        self.completedAudioArchiveURL = nil
    }

    private func safeSampleRate(_ value: Double) -> Double {
        max(value, 1)
    }

    private func latestWindow(from samples: [Float], maxCount: Int) -> [Float] {
        guard maxCount > 0, samples.count > maxCount else { return samples }
        return Array(samples.suffix(maxCount))
    }

    private func publishPartial(_ text: String) {
        guard sessionAllowsRealtimeTextDisplay else { return }
        onPartialTranscription?(text)
    }

    private func resolvedSessionLiveMode() -> MLXLiveMode {
        guard sessionAllowsRealtimeTextDisplay else { return .batchPreview }
        return MLXModelManager.liveMode(for: modelManager.currentModelRepo)
    }

    func makeMeetingNativeStreamingConfiguration() async throws -> MLXMeetingNativeStreamingConfiguration {
        let liveMode = MLXModelManager.liveMode(for: modelManager.currentModelRepo)
        let loaded = try await modelManager.loadModel()
        if let runtime = loaded.omniRuntime {
            return try await makeOmniMeetingStreamingConfiguration(runtime: runtime, liveMode: liveMode)
        }
        guard let loadedModel = loaded.mlxModel else { throw CancellationError() }

        switch liveMode {
        case .nativeQwenLive:
            guard let model = loadedModel as? Qwen3ASRModel else {
                throw NSError(
                    domain: "Voxt.Meeting.NativeMLX",
                    code: -1,
                    userInfo: [NSLocalizedDescriptionKey: "The selected Qwen ASR model could not create a streaming session."]
                )
            }
            let language = resolvedNativeQwenLiveLanguage()
            let kvCachePolicy = MLXModelCatalog.capability(
                for: modelManager.currentModelRepo
            ).kvCachePolicy
            return MLXMeetingNativeStreamingConfiguration(
                session: StreamingInferenceSession(
                    model: model,
                    config: StreamingConfig(
                        language: language,
                        temperature: 0,
                        maxTokensPerPass: 1024,
                        kvBits: kvCachePolicy?.bits,
                        kvGroupSize: kvCachePolicy?.groupSize ?? 64,
                        quantizedKVStart: kvCachePolicy?.quantizedStart ?? 0
                    )
                ),
                liveMode: liveMode,
                qwenUsesAutomaticLanguageProtocol: language == nil,
                mossVisibleOutputMode: nil
            )
        case .nativeStreamingLive:
            guard loadedModel is CohereTranscribeModel || loadedModel is MossTranscribeDiarizeModel else {
                throw NSError(
                    domain: "Voxt.Meeting.NativeMLX",
                    code: -2,
                    userInfo: [NSLocalizedDescriptionKey: "The selected MLX model does not support a native streaming meeting session."]
                )
            }
            let inferenceConfiguration = resolvedInferenceConfiguration(for: .intermediate)
            let isMoss = loadedModel is MossTranscribeDiarizeModel
            return MLXMeetingNativeStreamingConfiguration(
                session: StreamingInferenceSession(
                    model: loadedModel,
                    config: StreamingConfig(
                        language: inferenceConfiguration.languageHint,
                        temperature: inferenceConfiguration.generationParameters.temperature,
                        maxTokensPerPass: inferenceConfiguration.generationParameters.maxTokens,
                        prompt: isMoss ? inferenceConfiguration.mossPrompt : nil,
                        usePunctuation: inferenceConfiguration.generationParameters.usePunctuation
                    )
                ),
                liveMode: liveMode,
                qwenUsesAutomaticLanguageProtocol: false,
                // The final offline MOSS pass preserves structured speaker/timestamp output.
                // The live overlay strips protocol tags to keep incremental text readable.
                mossVisibleOutputMode: isMoss ? .plainText : nil
            )
        case .nativeNemotronLive:
            guard let model = loadedModel as? NemotronASRModel else {
                throw NSError(
                    domain: "Voxt.Meeting.NativeMLX",
                    code: -3,
                    userInfo: [NSLocalizedDescriptionKey: "The selected Nemotron model could not create a streaming session."]
                )
            }
            let tuningSettings = resolvedLocalTuningSettings()
            let chunkMilliseconds = tuningSettings.nemotronStreamLatency.rawValue
            let language = MLXTranscriptionPlanning.nativeNemotronLanguage(
                requested: resolvedNativeNemotronLiveLanguage(),
                availableLanguages: Array(model.promptDictionary.keys),
                defaultLanguage: model.defaultLanguage
            )
            return MLXMeetingNativeStreamingConfiguration(
                session: NemotronASRStreamingSession(
                    model: model,
                    config: StreamingConfig(
                        decodeIntervalSeconds: Double(chunkMilliseconds) / 1000,
                        boundaryDecodeIntervalSeconds: 0.2,
                        boundaryBoostSeconds: 1.0,
                        delayPreset: .custom(ms: chunkMilliseconds),
                        language: language,
                        temperature: 0,
                        maxTokensPerPass: 1024
                    )
                ),
                liveMode: liveMode,
                qwenUsesAutomaticLanguageProtocol: false,
                mossVisibleOutputMode: nil
            )
        case .batchPreview:
            throw NSError(
                domain: "Voxt.Meeting.NativeMLX",
                code: -4,
                userInfo: [NSLocalizedDescriptionKey: "The selected model is not eligible for the visible local meeting streaming path."]
            )
        }
    }

    private func makeOmniMeetingStreamingConfiguration(
        runtime: OmniASRRuntime,
        liveMode: MLXLiveMode
    ) async throws -> MLXMeetingNativeStreamingConfiguration {
        switch liveMode {
        case .nativeQwenLive:
            let language = resolvedNativeQwenLiveLanguage()
            return MLXMeetingNativeStreamingConfiguration(
                session: try await OmniNativeStreamingSession.qwen(runtime: runtime, language: language),
                liveMode: liveMode,
                qwenUsesAutomaticLanguageProtocol: language == nil,
                mossVisibleOutputMode: nil
            )
        default:
            throw NSError(
                domain: "Voxt.Meeting.NativeMLX",
                code: -5,
                userInfo: [NSLocalizedDescriptionKey: "The selected model is not eligible for the visible local meeting streaming path."]
            )
        }
    }

    private func startNativeLiveSession(revision: Int) {
        liveSessionSetupTasks.cancelAll()
        let expectedMode = activeLiveMode
        liveSessionSetupTasks.start { [weak self] in
            guard let self else { return }
            let startedAt = Date()
            let manager = self.modelManager
            manager.beginActiveUse()
            var transferredModelUse = false
            defer { if !transferredModelUse { manager.endActiveUse() } }
            do {
                let loaded = try await manager.loadModel()
                guard !Task.isCancelled, revision == self.sessionRevision,
                      self.isRecording, self.activeLiveMode == expectedMode else { return }
                let releaseModel: @MainActor () -> Void = { manager.endActiveUse() }
                if let runtime = loaded.omniRuntime {
                    guard try await self.installOmniLiveSession(
                        runtime: runtime,
                        mode: expectedMode,
                        revision: revision,
                        releaseModel: releaseModel
                    ) else { return }
                    transferredModelUse = true
                    self.isModelInitializing = false
                    VoxtLog.asr("Omni live session ready. repo=\(manager.currentModelRepo), mode=\(String(describing: expectedMode)), elapsedMs=\(Int(Date().timeIntervalSince(startedAt) * 1000))", verbose: true)
                    return
                }
                guard let model = loaded.mlxModel else { return }
                defer {
                    if !transferredModelUse {
                        VoxtLog.asrWarning("MLX native live requested for incompatible model. repo=\(manager.currentModelRepo), mode=\(String(describing: expectedMode))")
                    }
                }
                switch expectedMode {
                case .nativeQwenLive:
                    guard let model = model as? Qwen3ASRModel else { return }
                    self.installNativeQwenLiveSession(model, revision: revision, releaseModel: releaseModel)
                case .nativeStreamingLive:
                    guard model is CohereTranscribeModel || model is MossTranscribeDiarizeModel else { return }
                    self.installNativeStreamingLiveSession(model, revision: revision, releaseModel: releaseModel)
                case .nativeNemotronLive:
                    guard let model = model as? NemotronASRModel else { return }
                    self.installNativeNemotronLiveSession(model, revision: revision, releaseModel: releaseModel)
                case .batchPreview:
                    return
                }
                transferredModelUse = true
                self.isModelInitializing = false
                let elapsedMs = Int(Date().timeIntervalSince(startedAt) * 1000)
                VoxtLog.asr("MLX native live session ready. repo=\(manager.currentModelRepo), mode=\(String(describing: expectedMode)), elapsedMs=\(elapsedMs)", verbose: true)
            } catch {
                guard !Task.isCancelled, revision == self.sessionRevision else { return }
                self.isModelInitializing = false
                VoxtLog.asrWarning("MLX native live setup failed. repo=\(manager.currentModelRepo), mode=\(String(describing: expectedMode)), error=\(error.localizedDescription)")
            }
        }
    }

    /// Returns false when the mode has no Omni live session for this checkpoint.
    private func installOmniLiveSession(
        runtime: OmniASRRuntime,
        mode: MLXLiveMode,
        revision: Int,
        releaseModel: @escaping @MainActor () -> Void
    ) async throws -> Bool {
        let session: OmniNativeStreamingSession
        switch mode {
        case .nativeQwenLive:
            let language = resolvedNativeQwenLiveLanguage()
            session = try await OmniNativeStreamingSession.qwen(runtime: runtime, language: language)
            guard revision == sessionRevision, isRecording, activeLiveMode == mode else {
                session.cancel()
                return false
            }
            releaseNativeLiveSession(cancelSession: true)
            nativeQwenLiveUsesAutomaticLanguageProtocol = language == nil
        default:
            return false
        }
        installNativeLiveSession(session, revision: revision, releaseModel: releaseModel)
        return true
    }

    private func installNativeQwenLiveSession(_ model: Qwen3ASRModel, revision: Int, releaseModel: @escaping @MainActor () -> Void) {
        releaseNativeLiveSession(cancelSession: true)
        let language = resolvedNativeQwenLiveLanguage()
        let kvCachePolicy = MLXModelCatalog.capability(
            for: modelManager.currentModelRepo
        ).kvCachePolicy
        nativeQwenLiveUsesAutomaticLanguageProtocol = language == nil
        let session = StreamingInferenceSession(
            model: model,
            config: StreamingConfig(
                language: language,
                temperature: 0.0,
                maxTokensPerPass: 1024,
                kvBits: kvCachePolicy?.bits,
                kvGroupSize: kvCachePolicy?.groupSize ?? 64,
                quantizedKVStart: kvCachePolicy?.quantizedStart ?? 0
            )
        )
        installNativeLiveSession(session, revision: revision, releaseModel: releaseModel)
    }

    private func installNativeStreamingLiveSession(_ model: any STTGenerationModel, revision: Int, releaseModel: @escaping @MainActor () -> Void) {
        releaseNativeLiveSession(cancelSession: true)
        let inferenceConfiguration = resolvedInferenceConfiguration(for: .intermediate)
        let session = StreamingInferenceSession(
            model: model,
            config: StreamingConfig(
                language: inferenceConfiguration.languageHint,
                temperature: inferenceConfiguration.generationParameters.temperature,
                maxTokensPerPass: inferenceConfiguration.generationParameters.maxTokens,
                prompt: model is MossTranscribeDiarizeModel ? inferenceConfiguration.mossPrompt : nil,
                usePunctuation: inferenceConfiguration.generationParameters.usePunctuation
            )
        )
        installNativeLiveSession(session, revision: revision, releaseModel: releaseModel)
    }

    private func installNativeNemotronLiveSession(_ model: NemotronASRModel, revision: Int, releaseModel: @escaping @MainActor () -> Void) {
        releaseNativeLiveSession(cancelSession: true)
        let tuningSettings = resolvedLocalTuningSettings()
        let chunkMilliseconds = tuningSettings.nemotronStreamLatency.rawValue
        let language = MLXTranscriptionPlanning.nativeNemotronLanguage(
            requested: resolvedNativeNemotronLiveLanguage(),
            availableLanguages: Array(model.promptDictionary.keys),
            defaultLanguage: model.defaultLanguage
        )
        let session = NemotronASRStreamingSession(
            model: model,
            config: StreamingConfig(
                decodeIntervalSeconds: Double(chunkMilliseconds) / 1000,
                boundaryDecodeIntervalSeconds: 0.2,
                boundaryBoostSeconds: 1.0,
                delayPreset: .custom(ms: chunkMilliseconds),
                language: language,
                temperature: 0.0,
                maxTokensPerPass: 1024
            )
        )
        installNativeLiveSession(session, revision: revision, releaseModel: releaseModel)
    }

    private func installNativeLiveSession(
        _ session: any MLXNativeStreamingSession,
        revision: Int,
        releaseModel: @escaping @MainActor () -> Void
    ) {
        qwenFeedCursor = 0
        qwenVoiceActivityFeedCursor = 0
        latestNativeLiveConfirmedText = ""
        latestNativeLivePreviewText = ""
        latestNativeLiveEndedText = ""
        latestNativeLiveEndedSegments = []
        if activeLiveMode != .nativeQwenLive {
            nativeQwenLiveUsesAutomaticLanguageProtocol = false
        }
        nativeLiveRuntime.install(
            session,
            pollInterval: qwenLiveFeedPollInterval,
            nextSamples: { [weak self] in self?.drainPendingSamplesForQwenLiveFeed(revision: revision) ?? [] },
            shouldContinue: { [weak self] in self?.shouldContinueQwenLiveFeed(revision: revision) ?? false },
            onEvent: { [weak self] in self?.handleNativeLiveEvent($0, revision: revision) },
            releaseModel: releaseModel
        )
    }

    private func releaseNativeLiveSession(cancelSession: Bool) {
        nativeLiveRuntime.release(cancelSession: cancelSession)
    }

    private func resolvedNativeQwenLiveLanguage() -> String? {
        MLXTranscriptionPlanning.nativeLiveLanguage(from: resolvedHintPayload().language)
    }

    private func resolvedNativeNemotronLiveLanguage() -> String {
        let hintPayload = resolvedHintPayload()
        if let language = hintPayload.language?.trimmingCharacters(in: .whitespacesAndNewlines),
           !language.isEmpty {
            return language
        }
        return "auto"
    }

    private func drainPendingSamplesForQwenLiveFeed(revision: Int) -> [Float] {
        guard revision == sessionRevision,
              isRecording,
              MLXTranscriptionPlanning.isNativeLiveMode(activeLiveMode)
        else { return [] }
        let pendingSamples = pendingSamplesForNativeLiveFeed()
        guard !pendingSamples.isEmpty else { return [] }

        do {
            return try prepareInputSamples(pendingSamples, sampleRate: inputSampleRate)
        } catch {
            VoxtLog.asrWarning("MLX native Qwen live sample prepare failed: \(error.localizedDescription)")
            return []
        }
    }

    private func shouldContinueQwenLiveFeed(revision: Int) -> Bool {
        revision == sessionRevision
            && isRecording
            && MLXTranscriptionPlanning.isNativeLiveMode(activeLiveMode)
            && nativeLiveRuntime.session != nil
    }

    private func drainPendingSamplesIntoQwenLiveSession() {
        guard let session = nativeLiveRuntime.session else { return }
        let pendingSamples = pendingSamplesForNativeLiveFeed()
        guard !pendingSamples.isEmpty else { return }

        do {
            let prepared = try prepareInputSamples(pendingSamples, sampleRate: inputSampleRate)
            if !prepared.isEmpty {
                session.feedAudio(samples: prepared)
            }
        } catch {
            VoxtLog.asrWarning("MLX native Qwen live stop-drain prepare failed: \(error.localizedDescription)")
        }
    }

    private func pendingSamplesForNativeLiveFeed() -> [Float] {
        let voiceActivityState = voiceActivityFilteredSampleStore.voiceActivityState()
        if voiceActivityState.enabled {
            guard voiceActivityState.observedFrames,
                  voiceActivityState.observedSpeech
            else { return [] }

            let pending = voiceActivityFilteredSampleStore.samples(from: qwenVoiceActivityFeedCursor)
            qwenVoiceActivityFeedCursor = pending.nextIndex
            return pending.samples
        }

        let pending = sampleStore.samples(from: qwenFeedCursor)
        qwenFeedCursor = pending.nextIndex
        return pending.samples
    }

    private func handleNativeLiveEvent(_ event: TranscriptionEvent, revision: Int) {
        guard revision == sessionRevision else { return }

        switch event {
        case .displayUpdate(let confirmedText, let provisionalText):
            let visibleParts: (confirmedText: String, provisionalText: String)
            if nativeQwenLiveUsesAutomaticLanguageProtocol {
                visibleParts = MLXTranscriptionPlanning.qwenStreamingVisibleTextParts(
                    confirmedText: confirmedText,
                    provisionalText: provisionalText
                )
            } else {
                visibleParts = (
                    confirmedText: renderedMossTextIfNeeded(confirmedText),
                    provisionalText: renderedMossTextIfNeeded(provisionalText)
                )
            }
            guard let combined = MLXTranscriptionPlanning.resolvedNativeLiveVisiblePreview(
                previousPreview: latestNativeLivePreviewText,
                previousConfirmedText: latestNativeLiveConfirmedText,
                confirmedText: visibleParts.confirmedText,
                provisionalText: visibleParts.provisionalText
            ) else { return }
            latestNativeLiveConfirmedText = normalizeText(visibleParts.confirmedText)
            latestNativeLivePreviewText = combined
            internalTranscribedText = combined
            transcribedText = combined
            publishPartial(combined)
        case .ended(let output):
            let visibleText = nativeQwenLiveUsesAutomaticLanguageProtocol
                ? MLXTranscriptionPlanning.qwenStreamingVisibleText(
                    output.text,
                    suppressIncompleteWindowHeader: false
                )
                : renderedMossTextIfNeeded(output.text)
            let normalized = normalizeText(visibleText)
            latestNativeLiveConfirmedText = normalized
            latestNativeLiveEndedText = normalized
            let capability = MLXModelCatalog.capability(for: modelManager.currentModelRepo)
            latestNativeLiveEndedSegments = Self.structuredSegmentsForLiveEnded(
                output: output,
                modelFamily: capability.family,
                timingGranularity: capability.timingGranularity
            )
            if !latestNativeLiveEndedSegments.isEmpty {
                VoxtLog.asr(
                    "MLX native live ended with structured segments. repo=\(modelManager.currentModelRepo), segmentCount=\(latestNativeLiveEndedSegments.count), timing=\(String(describing: capability.timingGranularity)), language=\(output.language ?? "nil")",
                    verbose: true
                )
            }
            if !normalized.isEmpty {
                latestNativeLivePreviewText = normalized
            }
            if !normalized.isEmpty {
                internalTranscribedText = normalized
                transcribedText = normalized
                publishPartial(normalized)
            }
        case .failed(let failure):
            pendingRuntimeFailureMessage = failure.localizedDescription
            VoxtLog.asrError(
                "MLX native streaming session failed. repo=\(modelManager.currentModelRepo), error=\(failure.localizedDescription)"
            )
            releaseNativeLiveSession(cancelSession: false)
        case .confirmed, .provisional, .stats:
            break
        }
    }

    private func startEarlyModelPrewarmIfNeeded() {
        guard !modelManager.isCurrentModelLoaded else {
            isModelInitializing = false
            return
        }
        guard earlyPrewarmTask == nil else { return }

        earlyPrewarmTask = prewarmTasks.start { [weak self] in
            guard let self else { return }
            let startedAt = Date()
            do {
                // Session pin (if held) already blocks idle unload; avoid nested begin/end
                // so a cancelled prepare path cannot briefly schedule unload.
                _ = try await self.modelManager.loadModel()
                guard !Task.isCancelled else { return }
                await MainActor.run {
                    self.isModelInitializing = false
                }
                let elapsedMs = Int(Date().timeIntervalSince(startedAt) * 1000)
                VoxtLog.asr(
                    "MLX transcription early prewarm completed. repo=\(self.modelManager.currentModelRepo), elapsedMs=\(elapsedMs)",
                    verbose: true
                )
            } catch is CancellationError {
                return
            } catch {
                guard !Task.isCancelled else { return }
                let elapsedMs = Int(Date().timeIntervalSince(startedAt) * 1000)
                VoxtLog.asrWarning(
                    "MLX transcription early prewarm failed. repo=\(self.modelManager.currentModelRepo), elapsedMs=\(elapsedMs), error=\(error.localizedDescription)"
                )
            }
        }
    }

    private func startModelPreloadIfNeeded(revision: Int) {
        guard activeSessionBehavior.preloadsOnRecordingStart else {
            isModelInitializing = false
            return
        }
        guard !modelManager.isCurrentModelLoaded else {
            isModelInitializing = false
            return
        }

        preloadTask?.cancel()
        preloadTask = sessionTasks.start { [weak self] in
            guard let self else { return }
            let startedAt = Date()
            do {
                // Prefer session pin; only add a nested pin when session pin is absent.
                let nestedPin = !self.sessionModelPinned
                if nestedPin {
                    self.modelManager.beginActiveUse()
                }
                defer {
                    if nestedPin {
                        self.modelManager.endActiveUse()
                    }
                }
                _ = try await self.modelManager.loadModel()
                guard !Task.isCancelled, revision == self.sessionRevision else { return }
                await MainActor.run {
                    self.isModelInitializing = false
                }
                let elapsedMs = Int(Date().timeIntervalSince(startedAt) * 1000)
                VoxtLog.asr(
                    "MLX transcription preload completed. repo=\(self.modelManager.currentModelRepo), elapsedMs=\(elapsedMs)",
                    verbose: true
                )
            } catch {
                guard !Task.isCancelled else { return }
                let elapsedMs = Int(Date().timeIntervalSince(startedAt) * 1000)
                VoxtLog.asrWarning(
                    "MLX transcription preload failed. repo=\(self.modelManager.currentModelRepo), elapsedMs=\(elapsedMs), error=\(error.localizedDescription)"
                )
            }
        }
    }

    private func scheduleCaptureStartupWatchdog(revision: Int) {
        captureWatchdogTask?.cancel()
        captureWatchdogTask = sessionTasks.start { [weak self] in
            do {
                try await Task.sleep(for: .seconds(1.2))
            } catch {
                return
            }
            await self?.recoverAudioCaptureIfNeeded(revision: revision)
        }
    }

    private func recoverAudioCaptureIfNeeded(revision: Int) async {
        guard revision == sessionRevision, isRecording else { return }
        guard sampleStore.callbacksReceived() == 0 else { return }
        guard !didRetryCaptureStartup else { return }

        didRetryCaptureStartup = true
        let shouldFallbackToSystemDefault = preferredInputDeviceID != nil && activeCaptureUsesPreferredInputDevice
        if shouldFallbackToSystemDefault {
            VoxtLog.asrWarning(
                "MLX audio capture produced no initial callbacks. Retrying once with system default input instead of the preferred device."
            )
        } else {
            VoxtLog.asrWarning("MLX audio capture produced no initial callbacks. Restarting input graph once.")
        }

        do {
            try startAudioCaptureGraph(usePreferredInputDevice: shouldFallbackToSystemDefault ? false : activeCaptureUsesPreferredInputDevice)
            scheduleCaptureStartupWatchdog(revision: revision)
        } catch {
            VoxtLog.asrError("MLX audio capture recovery failed: \(error)")
        }
    }

    private func applyCandidate(_ candidate: String, stage: MLXCorrectionPassKind) {
        if !sessionAllowsRealtimeTextDisplay {
            switch stage {
            case .postStopFinal:
                internalTranscribedText = candidate
                transcribedText = candidate
                stableCommittedText = candidate
                lastCandidateText = candidate
                return
            case .postStopQuick:
                let trustedHiddenBaseline = resolvedTrustedHiddenPreviewBaseline(
                    base: internalTranscribedText,
                    candidate: candidate
                )
                let merged = MLXTranscriptionPlanning.mergedHiddenPostStopPreview(
                    base: trustedHiddenBaseline,
                    candidate: candidate
                )
                internalTranscribedText = merged
                transcribedText = merged
                lastCandidateText = merged
                stableCommittedText = merged
                return
            case .intermediate:
                // Keep hidden intermediate candidates off the UI, but preserve the most
                // recent full-context hypothesis as a baseline for stop-time quick-pass
                // merging. This lets final-only mode use a true tail-window quick pass
                // without losing earlier transcript context.
                let merged = MLXTranscriptionPlanning.mergedHiddenPostStopPreview(
                    base: internalTranscribedText,
                    candidate: candidate
                )
                internalTranscribedText = merged
                lastCandidateText = merged
                return
            }
        }

        internalTranscribedText = candidate
        switch stage {
        case .postStopFinal:
            transcribedText = candidate
            stableCommittedText = candidate
            lastCandidateText = candidate
            publishPartial(candidate)
        case .intermediate, .postStopQuick:
            if lastCandidateText.isEmpty {
                lastCandidateText = candidate
                transcribedText = candidate
                publishPartial(candidate)
                return
            }

            let stablePrefix = MLXTranscriptionPlanning.longestCommonPrefix(lastCandidateText, candidate)
            if stablePrefix.count > stableCommittedText.count {
                stableCommittedText = stablePrefix
            }

            lastCandidateText = candidate
            let merged = MLXTranscriptionPlanning.mergeStablePrefix(stableCommittedText, candidate: candidate)
            transcribedText = merged
            publishPartial(merged)
        }
    }

    private func normalizeText(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func renderedMossTextIfNeeded(_ text: String) -> String {
        guard MLXModelFamily.family(for: modelManager.currentModelRepo) == .mossTranscribeDiarize else {
            return text
        }
        return MossASRTranscriptRendering.renderedText(
            text,
            outputMode: resolvedLocalTuningSettings()
                .mossSettings(for: transcriptionPurpose.mossUsageScope)
                .outputMode
        )
    }

    private func resolvedTrustedHiddenPreviewBaseline(base: String, candidate: String) -> String {
        let stableBase = normalizeText(base)
        let stableCandidate = normalizeText(candidate)
        guard !stableBase.isEmpty, !stableCandidate.isEmpty else { return stableBase }

        let maxTrustedBaseCount = stableCandidate.count + max(48, stableCandidate.count / 2)
        if stableBase.count > maxTrustedBaseCount,
           !stableBase.contains(stableCandidate) {
            return ""
        }

        return stableBase
    }

    private func resolvedInferenceConfiguration(
        for stage: MLXCorrectionPassKind,
        audioDurationSeconds: Double? = nil
    ) -> ResolvedInferenceConfiguration {
        MLXInferenceConfiguration.resolve(
            for: stage,
            audioDurationSeconds: audioDurationSeconds,
            hintPayload: resolvedHintPayload(),
            tuningSettings: resolvedLocalTuningSettings(),
            transcriptionPurpose: transcriptionPurpose,
            userLanguageCodes: UserMainLanguageOption.storedSelection(
                from: UserDefaults.standard.string(forKey: AppPreferenceKey.userMainLanguageCodes)
            ),
            capability: MLXModelCatalog.capability(for: modelManager.currentModelRepo),
            dictionaryTerms: resolvedDictionaryTermsTemplateValue(),
            sessionAllowsRealtimeTextDisplay: sessionAllowsRealtimeTextDisplay
        )
    }

    private func resolvedHintPayload() -> ResolvedASRHintPayload {
        let defaults = UserDefaults.standard
        let settings = ASRHintSettingsStore.resolvedSettings(
            for: .mlxAudio,
            rawValue: defaults.string(forKey: AppPreferenceKey.asrHintSettings)
        )
        let userLanguageCodes = UserMainLanguageOption.storedSelection(
            from: defaults.string(forKey: AppPreferenceKey.userMainLanguageCodes)
        )
        return ASRHintResolver.resolve(
            target: .mlxAudio,
            settings: settings,
            userLanguageCodes: userLanguageCodes,
            mlxModelRepo: modelManager.currentModelRepo
        )
    }

    private func resolvedLocalTuningSettings() -> MLXLocalTuningSettings {
        MLXLocalTuningSettingsStore.resolvedSettings(
            for: modelManager.currentModelRepo,
            rawValue: UserDefaults.standard.string(forKey: AppPreferenceKey.mlxLocalASRTuningSettings)
        )
    }

    private func stageLabel(for stage: MLXCorrectionPassKind) -> String {
        switch stage {
        case .intermediate: return "intermediate"
        case .postStopQuick: return "post-stop quick"
        case .postStopFinal: return "post-stop final"
        }
    }

    private func resolvedDictionaryTermsTemplateValue() -> String {
        DictionaryEntryCollection.asrPromptTermsText(from: dictionaryEntryProvider?() ?? [])
    }

    private func prepareInputSamples(_ samples: [Float], sampleRate: Double) throws -> [Float] {
        try Self.prepareInputSamplesDetached(
            samples,
            sampleRate: sampleRate,
            targetSampleRate: targetSampleRate
        )
    }

    private func runStreamingInference(
        model loadedModel: LoadedASRModel,
        audioSamples: [Float],
        inferenceConfiguration: ResolvedInferenceConfiguration
    ) async throws -> MLXDetachedInferenceResult {
        try Task.checkCancellation()
        let model: any STTGenerationModel
        switch loadedModel {
        case .omni(let runtime):
            // The Omni branch starts before any MLXArray exists on the client.
            let targetSampleRate = targetSampleRate
            let inferenceTask = Task.detached(priority: inferenceTaskPriority) {
                try await Self.runOmniInferenceDetached(
                    runtime: runtime,
                    audioSamples: audioSamples,
                    inferenceConfiguration: inferenceConfiguration,
                    targetSampleRate: targetSampleRate
                )
            }
            return try await withTaskCancellationHandler {
                try await inferenceTask.value
            } onCancel: {
                inferenceTask.cancel()
            }
        case .mlx(let mlxModel):
            model = mlxModel
        }
        let longFormVADModel = try await resolvedLongFormVADModelIfNeeded(
            model: model,
            audioSamples: audioSamples,
            inferenceConfiguration: inferenceConfiguration
        )
        let modelBox = MLXUnsafeSendableBox(value: model)
        let vadBox = MLXUnsafeSendableBox(value: longFormVADModel)
        let targetSampleRate = targetSampleRate
        let directPassMaximumDurationSeconds = senseVoiceDirectPassMaximumDurationSeconds
        let chunkMaximumDurationSeconds = senseVoiceChunkMaximumDurationSeconds
        let chunkOverlapSeconds = senseVoiceChunkOverlapSeconds
        let vadThreshold = senseVoiceVADThreshold
        let vadMinSpeechDurationMs = senseVoiceVADMinSpeechDurationMs
        let vadMinSilenceDurationMs = senseVoiceVADMinSilenceDurationMs
        let vadSpeechPadMs = senseVoiceVADSpeechPadMs

        let inferenceTask = Task.detached(priority: inferenceTaskPriority) {
            try Task.checkCancellation()
            return try await Self.runStreamingInferenceDetached(
                model: modelBox.value,
                audioSamples: audioSamples,
                inferenceConfiguration: inferenceConfiguration,
                longFormVADModel: vadBox.value,
                targetSampleRate: targetSampleRate,
                directPassMaximumDurationSeconds: directPassMaximumDurationSeconds,
                chunkMaximumDurationSeconds: chunkMaximumDurationSeconds,
                chunkOverlapSeconds: chunkOverlapSeconds,
                vadThreshold: vadThreshold,
                vadMinSpeechDurationMs: vadMinSpeechDurationMs,
                vadMinSilenceDurationMs: vadMinSilenceDurationMs,
                vadSpeechPadMs: vadSpeechPadMs
            )
        }
        return try await withTaskCancellationHandler {
            try await inferenceTask.value
        } onCancel: {
            inferenceTask.cancel()
        }
    }

    private func resolvedLongFormVADModelIfNeeded(
        model: any STTGenerationModel,
        audioSamples: [Float],
        inferenceConfiguration: ResolvedInferenceConfiguration
    ) async throws -> SileroVAD? {
        guard MLXTranscriptionPlanning.shouldUseSenseVoiceVAD(
            sampleCount: audioSamples.count,
            sampleRate: targetSampleRate,
            directPassMaximumDurationSeconds: senseVoiceDirectPassMaximumDurationSeconds
        ) else {
            return nil
        }
        if model is CohereTranscribeModel {
            guard inferenceConfiguration.cohereLongFormStrategy == .voiceActivity else { return nil }
        } else if !(model is SenseVoiceModel) {
            return nil
        }
        if let senseVoiceVADModel {
            return senseVoiceVADModel
        }

        let modelDirectory = try await SileroVADModelProvisioner.shared.ensureModelDirectory()
        try Task.checkCancellation()
        let loadedModel = try SileroVADModelSupport.loadModel(from: modelDirectory)
        senseVoiceVADModel = loadedModel
        return loadedModel
    }

    private func runtimeFailureMessage(for error: Error) -> String {
        if let structuredError = error as? MLXStructuredTranscriptionError,
           let description = structuredError.errorDescription {
            return description
        }
        return error.localizedDescription
    }

    func transcribeBufferedChunk(samples: [Float], sampleRate: Double) async throws -> String? {
        try await transcribeBufferedResult(samples: samples, sampleRate: sampleRate)?.text
    }

    func transcribeBufferedResult(
        samples: [Float],
        sampleRate: Double
    ) async throws -> MLXBufferedTranscriptionResult? {
        guard !samples.isEmpty else { return nil }

        latestSenseVoiceMetadata = nil
        modelManager.beginActiveUse()
        defer { modelManager.endActiveUse() }
        defer { isModelInitializing = false }
        let model = try await modelManager.loadModel()
        let audioSamples = try prepareInputSamples(samples, sampleRate: sampleRate)
        let audioDurationSeconds = Double(samples.count) / safeSampleRate(sampleRate)
        let inferenceConfiguration = resolvedInferenceConfiguration(
            for: .postStopFinal,
            audioDurationSeconds: audioDurationSeconds
        )
        let inferenceResult = try await runStreamingInference(
            model: model,
            audioSamples: audioSamples,
            inferenceConfiguration: inferenceConfiguration
        )
        latestSenseVoiceMetadata = inferenceResult.senseVoiceMetadata
        let rawCandidate = normalizeText(inferenceResult.rawText)
        let candidate = normalizeText(MLXTranscriptionPlanning.removingKnownASRContextLeakage(from: rawCandidate))
        if candidate != rawCandidate {
            VoxtLog.asrWarning(
                "MLX ASR context leakage removed. repo=\(modelManager.currentModelRepo), stage=structured, rawChars=\(rawCandidate.count), outputChars=\(candidate.count)"
            )
        }
        guard !candidate.isEmpty else {
            latestSenseVoiceMetadata = nil
            return nil
        }
        return MLXBufferedTranscriptionResult(
            text: candidate,
            structuredSegments: inferenceResult.structuredSegments
        )
    }

    func transcribeAudioFile(_ fileURL: URL) async throws -> String {
        let loaded = try DebugAudioClipIO.loadMonoSamples(from: fileURL)
        return try await transcribeBufferedChunk(
            samples: loaded.samples,
            sampleRate: loaded.sampleRate
        ) ?? ""
    }

    func debugReplayAudioFileWithTrace(
        _ fileURL: URL,
        stepSeconds: Double = 4.0,
        allowsRealtimeTextDisplay: Bool
    ) async throws -> MLXRealtimeReplayDiagnostics {
        let loaded = try DebugAudioClipIO.loadMonoSamples(from: fileURL)
        let safeSampleRate = safeSampleRate(loaded.sampleRate)
        let stepSampleCount = max(Int(stepSeconds * safeSampleRate), 1)
        let revision = sessionRevision + 1

        resetTransientState()
        sessionRevision = revision
        activeSessionBehavior = modelManager.currentTranscriptionBehavior
        sessionAllowsRealtimeTextDisplay = allowsRealtimeTextDisplay
        let previousIsRecording = isRecording
        let previousInputSampleRate = inputSampleRate
        isRecording = true
        inputSampleRate = loaded.sampleRate
        defer {
            isRecording = previousIsRecording
            inputSampleRate = previousInputSampleRate
        }

        var events: [MLXRealtimeReplayEvent] = []
        var trace: [String] = []
        var endSample = stepSampleCount

        while endSample <= loaded.samples.count {
            let prefix = Array(loaded.samples.prefix(endSample))
            if let decision = MLXTranscriptionPlanning.intermediateCorrectionDecision(
                sampleCount: prefix.count,
                sampleRate: loaded.sampleRate,
                nextCorrectionAtSeconds: nextCorrectionAtSeconds,
                behavior: activeSessionBehavior,
                firstCorrectionMinimumSeconds: currentFirstCorrectionMinimumSeconds,
                contextWindowSeconds: currentIntermediateContextWindowSeconds
            ) {
                let intermediateSamples = latestWindow(from: prefix, maxCount: decision.contextSampleCount)
                let publishedBefore = transcribedText
                let candidate = await runManagedCorrectionPass(
                    stage: .intermediate,
                    revision: revision,
                    explicitSamples: intermediateSamples,
                    sampleRate: loaded.sampleRate
                )
                nextCorrectionAtSeconds = decision.elapsedSeconds + currentCorrectionIntervalSeconds
                let publishedAfter = normalizeText(transcribedText)
                trace.append(
                    String(
                        format: "[%.1fs] intermediate candidate=%@ published=%@",
                        Double(endSample) / safeSampleRate,
                        Self.traceQuoted(normalizeText(candidate.text ?? "")),
                        Self.traceQuoted(publishedAfter)
                    )
                )
                if !publishedAfter.isEmpty, publishedAfter != normalizeText(publishedBefore) {
                    events.append(
                        MLXRealtimeReplayEvent(
                            elapsedSeconds: Double(endSample) / safeSampleRate,
                            text: publishedAfter,
                            isFinal: false,
                            source: "intermediate"
                        )
                    )
                }
            }
            endSample += stepSampleCount
        }
        isRecording = false

        let snapshot = loaded.samples
        let plan = MLXTranscriptionPlanning.finalizationPlan(
            sampleCount: snapshot.count,
            sampleRate: loaded.sampleRate,
            behavior: activeSessionBehavior,
            quickPassMinimumDurationSeconds: quickPassMinimumDurationSeconds,
            quickPassContextWindowSeconds: currentQuickPassContextWindowSeconds
        )
        let shouldRunQuickPass = allowsRealtimeTextDisplay && plan.shouldRunQuickPass
        if shouldRunQuickPass, let quickPassSampleCount = plan.quickPassSampleCount {
            let quickSource = latestWindow(from: snapshot, maxCount: quickPassSampleCount)
            let publishedBefore = transcribedText
                let candidate = await runManagedCorrectionPass(
                    stage: .postStopQuick,
                    revision: revision,
                    explicitSamples: quickSource,
                    sampleRate: loaded.sampleRate
                )
            let publishedAfter = normalizeText(transcribedText)
            trace.append(
                    String(
                        format: "[%.1fs] post-stop-quick candidate=%@ published=%@",
                        plan.durationSeconds,
                        Self.traceQuoted(normalizeText(candidate.text ?? "")),
                        Self.traceQuoted(publishedAfter)
                    )
                )
            if !publishedAfter.isEmpty, publishedAfter != normalizeText(publishedBefore) {
                events.append(
                    MLXRealtimeReplayEvent(
                        elapsedSeconds: plan.durationSeconds,
                        text: publishedAfter,
                        isFinal: false,
                        source: "post-stop-quick"
                    )
                )
            }
        }

        let finalText = await runManagedCorrectionPass(
            stage: .postStopFinal,
            revision: revision,
            explicitSamples: snapshot,
            sampleRate: loaded.sampleRate
        )
        let resolvedFinal = normalizeText(finalText.text ?? transcribedText)
        trace.append(
            String(
                format: "[%.1fs] final text=%@",
                plan.durationSeconds,
                Self.traceQuoted(resolvedFinal)
            )
        )
        if !resolvedFinal.isEmpty {
            events.append(
                MLXRealtimeReplayEvent(
                    elapsedSeconds: plan.durationSeconds,
                    text: resolvedFinal,
                    isFinal: true,
                    source: "final"
                )
            )
        }
        return MLXRealtimeReplayDiagnostics(events: events, trace: trace)
    }

    func debugReplayRealtimeAudioFileWithTrace(
        _ fileURL: URL,
        stepSeconds: Double = 4.0
    ) async throws -> MLXRealtimeReplayDiagnostics {
        try await debugReplayAudioFileWithTrace(
            fileURL,
            stepSeconds: stepSeconds,
            allowsRealtimeTextDisplay: true
        )
    }

    func debugReplayFinalOnlyAudioFileWithTrace(
        _ fileURL: URL,
        stepSeconds: Double = 4.0
    ) async throws -> MLXRealtimeReplayDiagnostics {
        try await debugReplayAudioFileWithTrace(
            fileURL,
            stepSeconds: stepSeconds,
            allowsRealtimeTextDisplay: false
        )
    }

    var currentWorkingTranscriptText: String {
        let internalText = internalTranscribedText.trimmingCharacters(in: .whitespacesAndNewlines)
        if !internalText.isEmpty {
            return internalText
        }
        return transcribedText.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func traceQuoted(_ value: String) -> String {
        value.isEmpty ? "\"\"" : "\"\(value)\""
    }

    @discardableResult
    private func applyPreferredInputDeviceIfNeeded(inputNode: AVAudioInputNode) -> Bool {
        guard let preferredInputDeviceID,
              preferredInputDeviceID != AudioDeviceID(kAudioObjectUnknown),
              AudioInputDeviceManager.isAvailableInputDevice(preferredInputDeviceID)
        else {
            return false
        }
        guard let audioUnit = inputNode.audioUnit else { return false }
        var deviceID = preferredInputDeviceID
        let status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &deviceID,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        if status != noErr {
            VoxtLog.asrWarning("Unable to switch input device. status=\(status)")
            return false
        }
        return true
    }
}
