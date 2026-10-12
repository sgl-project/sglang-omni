// RecordingCaptureFlow.swift
// Provides Recording Capture Flow for recording session routing.

import Foundation
import AVFoundation
import Speech

extension AppDelegate {
    func startTrackedRecordingCaptureTask(
        _ operation: @escaping @MainActor () async -> Void
    ) {
        let retiringStarts = recordingCaptureStartTasks.cancelAll()
        // An old start can still be unwinding CoreAudio. It must finish its cleanup
        // before a replacement touches the same transcriber/engine.
        recordingCaptureStartTasks.start(after: retiringStarts) { [weak self] in
            guard let self, !self.isApplicationTerminating else { return }
            await operation()
        }
    }

    @discardableResult
    func cancelRecordingCaptureStartTask() -> [Task<Void, Never>] {
        recordingCaptureStartTasks.cancelAll()
    }

    private var isMLXReady: Bool {
        switch mlxModelManager.state {
        case .downloaded, .ready, .loading:
            return true
        default:
            return false
        }
    }

    func startMLXRecordingSession() {
        let mlx = mlxTranscriber ?? MLXTranscriber(modelManager: mlxModelManager)
        mlxTranscriber = mlx
        mlx.dictionaryEntryProvider = { [weak self] in
            guard let self else { return [] }
            return self.dictionaryStore.activeEntriesForRemoteRequest(
                activeGroupID: self.activeDictionaryGroupID(),
                limit: DictionaryEntryCollection.asrPromptTermLimit
            )
        }
        let sessionID = activeRecordingSessionID
        overlayState.statusMessage = ""
        mlx.transcribedText = ""
        mlx.sessionAllowsRealtimeTextDisplay = transcriptionCapturePipeline.usesLiveDisplay
        let localVADMode = LocalVADMode.stored()
        let localVADGatePolicy = ASRVoiceActivityRuntimePolicy.localGatePolicy(
            transcriptionEngine: .mlxAudio,
            mode: localVADMode
        )
        mlx.configureVoiceActivityFinalizationFiltering(enabled: localVADGatePolicy.isEnabled)
        // Hotkey-time warm: ASR load + Silero provision overlap overlay/mic setup.
        // No settings or interaction changes — same session path, earlier work.
        SileroVADModelProvisioner.prefetchIfNeeded(for: localVADMode)
        mlx.prewarmModelForUpcomingSession()
        mlx.setPreferredInputDevice(selectedInputDeviceID)
        mlx.onPartialTranscription = { [weak self] text in
            self?.handleLiveASRPartialTranscription(text, sessionID: sessionID)
        }
        mlx.onTranscriptionFinished = { [weak self] text in
            self?.stashPendingCompletedHistoryAudioArchive(self?.mlxTranscriber?.consumeCompletedAudioArchiveURL())
            self?.processTranscription(text, sessionID: sessionID)
        }
        overlayState.bind(to: mlx)
        overlayWindow.show(
            state: overlayState,
            position: overlayPosition
        )
        startTrackedRecordingCaptureTask { [weak self] in
            guard let self else { return }
            if let startFailureMessage = await mlx.startRecordingSession() {
                mlx.discardPreparedSessionModelUse()
                guard !Task.isCancelled,
                      !self.isApplicationTerminating,
                      self.shouldHandleCallbacks(for: sessionID),
                      self.isSessionActive
                else { return }
                VoxtLog.asrWarning("MLX recording session did not enter recording state. reason=\(startFailureMessage)")
                self.handleRecordingStartFailure(startFailureMessage)
                return
            }
            // Recording started, but the user may have released/cancelled the hotkey while the
            // engine was starting. If the session is no longer current, stop the stray capture.
            guard self.shouldHandleCallbacks(for: sessionID),
                  self.isSessionActive,
                  !self.isApplicationTerminating,
                  !Task.isCancelled,
                  !self.isSessionCancellationRequested,
                  self.recordingStoppedAt == nil
            else {
                mlx.stopRecording()
                self.systemAudioMuteController.restoreSystemAudioIfNeeded()
                return
            }
        }
    }

    func startSpeechRecordingSession() {
        startTrackedRecordingCaptureTask { [weak self] in
            guard let self else { return }
            let sessionID = self.activeRecordingSessionID
            let granted = await self.speechTranscriber.requestPermissions()
            guard !Task.isCancelled,
                  !self.isApplicationTerminating,
                  self.shouldHandleCallbacks(for: sessionID),
                  self.isSessionActive
            else { return }
            guard granted else {
                self.handleRecordingPermissionDenied()
                return
            }

            self.overlayState.statusMessage = ""
            self.speechTranscriber.transcribedText = ""
            self.speechTranscriber.sessionReportsPartialResultsOverride = self.transcriptionCapturePipeline.usesLiveDisplay
            self.speechTranscriber.onTranscriptionFinished = { [weak self] text in
                self?.stashPendingCompletedHistoryAudioArchive(self?.speechTranscriber.consumeCompletedAudioArchiveURL())
                self?.processTranscription(text, sessionID: sessionID)
            }
            self.speechTranscriber.startRecording()
            guard self.speechTranscriber.isRecording else {
                let failureMessage = self.speechTranscriber.lastStartFailureMessage
                    ?? AppLocalization.localizedString("Direct Dictation failed to start recording.")
                VoxtLog.asrWarning("Speech recording session did not enter recording state. reason=\(failureMessage)")
                self.handleRecordingStartFailure(failureMessage)
                return
            }

            self.overlayState.bind(to: self.speechTranscriber)
            self.overlayWindow.show(
                state: self.overlayState,
                position: self.overlayPosition
            )
        }
    }

    func startRemoteRecordingSession() {
        startTrackedRecordingCaptureTask { [weak self] in
            guard let self else { return }
            let sessionID = self.activeRecordingSessionID
            let granted = await self.remoteASRTranscriber.requestPermissions()
            guard !Task.isCancelled,
                  !self.isApplicationTerminating,
                  self.shouldHandleCallbacks(for: sessionID),
                  self.isSessionActive
            else { return }
            guard granted else {
                self.handleRecordingPermissionDenied()
                return
            }

            self.overlayState.statusMessage = ""
            self.remoteASRTranscriber.dictionaryEntryProvider = { [weak self] in
                guard let self else { return [] }
                return self.dictionaryStore.activeEntriesForRemoteRequest(
                    activeGroupID: self.activeDictionaryGroupID(),
                    limit: DictionaryEntryCollection.asrPromptTermLimit
                )
            }
            self.remoteASRTranscriber.transcribedText = ""
            self.remoteASRTranscriber.sessionAllowsRealtimeTextDisplay = self.transcriptionCapturePipeline.usesLiveDisplay
            self.remoteASRTranscriber.voiceActivityUseCase = self.voiceActivityUseCase
            self.remoteASRTranscriber.onTranscriptionFinished = { [weak self] text in
                self?.stashPendingCompletedHistoryAudioArchive(self?.remoteASRTranscriber.consumeCompletedAudioArchiveURL())
                self?.processTranscription(text, sessionID: sessionID)
            }
            self.remoteASRTranscriber.onStartFailure = { [weak self] message in
                guard let self, self.shouldHandleCallbacks(for: sessionID) else { return }
                self.handleRecordingStartFailure(message, autoHideAfter: 3.6)
            }
            self.remoteASRTranscriber.onRuntimeFailure = { [weak self] message in
                guard let self, self.shouldHandleCallbacks(for: sessionID), self.isSessionActive else { return }
                self.showOverlayStatus(message, clearAfter: 4.8)
            }
            self.overlayState.bind(to: self.remoteASRTranscriber)
            self.overlayWindow.show(
                state: self.overlayState,
                position: self.overlayPosition
            )
            self.remoteASRTranscriber.startRecording()
        }
    }

    func startRecordingCapture(using engine: TranscriptionEngine) {
        switch engine {
        case .mlxAudio:
            startMLXRecordingSession()
        case .remote:
            startRemoteRecordingSession()
        case .dictation:
            startSpeechRecordingSession()
        }

        startSilenceMonitoringIfNeeded()
    }

    func resetSessionAfterFailedStart() {
        OnboardingSessionEvent.ended(id: activeRecordingSessionID, message: overlayState.statusMessage).post()
        cancelSessionControlTasks()
        systemAudioMuteController.restoreSystemAudioIfNeeded()
        if transcriptionEngine == .remote {
            remoteASRTranscriber.discardPendingSessionOutput()
        }
        discardPendingCompletedHistoryAudio()
        isSessionActive = false
        recordingLifecycle.begin()
        invalidateActiveLLMRequest()
        sessionOutputMode = .transcription
        recordingRequestedAt = nil
        recordingStartedAt = nil
        recordingStoppedAt = nil
        transcriptionProcessingStartedAt = nil
        transcriptionResultReceivedAt = nil
        firstLiveASRPartialReceivedAt = nil
        sessionFinalOutputDeliveredAt = nil
        sessionLLMExecutionTimings = []
        transcriptionCapturePipeline = .liveDisplay
        isSelectedTextTranslationFlow = false
        sessionTargetApplicationPID = nil
        sessionTargetApplicationBundleID = nil
        enhancementContextSnapshot = nil
        lastEnhancementPromptContext = nil
        sessionOutputDestinationContext = nil
        selectedTextTranslationHadWritableFocusedInput = false
        rewriteSessionHasSelectedSourceText = false
        rewriteSessionSelectedSourceText = ""
        rewriteSessionHadWritableFocusedInput = false
        resetVoiceEndCommandState()
        resetSessionTranslationState()
        resetVoxtNoteSessionRuntimeState()
        overlayState.reset()
        overlayWindow.hide()
        scheduleDeepIdleMemoryReclamation()
    }

    func preflightPermissionsForRecording(engine: TranscriptionEngine) -> Bool {
        if AVCaptureDevice.authorizationStatus(for: .audio) != .authorized {
            VoxtLog.asrWarning("Recording blocked: microphone permission not granted.")
            showOverlayReminder(
                AppLocalization.localizedString("Microphone permission is required. Enable it in Settings > Permissions.")
            )
            return false
        }

        if engine == .dictation && SFSpeechRecognizer.authorizationStatus() != .authorized {
            VoxtLog.asrWarning("Recording blocked: speech recognition permission not granted for Direct Dictation.")
            showOverlayReminder(
                AppLocalization.localizedString("Speech Recognition permission is required for Direct Dictation. Enable it in Settings > Permissions.")
            )
            return false
        }

        if !AccessibilityPermissionManager.isTrusted() {
            VoxtLog.asrWarning("Recording start proceeding without accessibility trust. Injection may be unavailable.")
            showOverlayStatus(
                AppLocalization.localizedString("Please enable required permissions in Settings > Permissions."),
                clearAfter: 2.2
            )
        }

        return true
    }

    func applyPreferredInputDevice() {
        speechTranscriber.setPreferredInputDevice(selectedInputDeviceID)
        mlxTranscriber?.setPreferredInputDevice(selectedInputDeviceID)
        remoteASRTranscriber.setPreferredInputDevice(selectedInputDeviceID)
    }

    func handlePreferredInputDeviceChange(
        previousUID: String?,
        newUID: String?,
        reason: String
    ) {
        applyPreferredInputDevice()

        guard previousUID != newUID else { return }

        guard let currentDevice = microphoneResolvedState.activeDevice else {
            if isSessionActive {
                showOverlayReminder(AppLocalization.localizedString("No available microphone devices."))
                finishSession(after: 0)
            }
            return
        }

        guard isSessionActive else { return }

        let sessionKind = RecordingSessionSupport.outputLabel(for: sessionOutputMode)
        let remoteDebugState = remoteASRTranscriber.activeRealtimeDebugSummary() ?? "none"
        VoxtLog.asrWarning(
            """
            Preferred input device changed during recording. reason=\(reason), previousUID=\(previousUID ?? "none"), newUID=\(newUID ?? "none"), engine=\(transcriptionEngine.rawValue), output=\(sessionKind), remoteState=\(remoteDebugState)
            """
        )

        do {
            try restartCurrentRecordingCaptureForPreferredInputDevice()
            showOverlayStatus(
                AppLocalization.format("Switched microphone to %@.", currentDevice.name),
                clearAfter: 1.8
            )
            VoxtLog.asrWarning(
                "Preferred input device change applied during recording. reason=\(reason), newUID=\(newUID ?? "none"), engine=\(transcriptionEngine.rawValue), output=\(sessionKind)"
            )
        } catch {
            VoxtLog.asrError("Recording microphone switch failed: \(error.localizedDescription). reason=\(reason)")
            showOverlayReminder(
                AppLocalization.format("Failed to switch microphone to %@.", currentDevice.name)
            )
            finishSession(after: 0)
        }
    }

    func stopActiveRecordingTranscriber() {
        if transcriptionEngine == .mlxAudio {
            mlxTranscriber?.stopRecording()
        } else if transcriptionEngine == .remote {
            remoteASRTranscriber.stopRecording()
        } else {
            speechTranscriber.stopRecording()
        }
    }

    /// Stops microphone capture and immediately releases the system-audio mute.
    ///
    /// Final ASR and enhancement continue asynchronously after the transcriber
    /// stops capturing, so system audio no longer needs to remain muted here.
    func stopActiveRecordingTranscriberAndRestoreSystemAudio() {
        stopActiveRecordingTranscriber()
        systemAudioMuteController.restoreSystemAudioIfNeeded()
    }

    func stopActiveRecordingTranscriberAfterPendingVADFlush(sessionID: UUID) {
        Task { [weak self] in
            guard let self else { return }
            guard self.activeRecordingSessionID == sessionID else { return }

            let monitorTask = self.silenceMonitorTask
            self.silenceMonitorTask?.cancel()
            self.silenceMonitorTask = nil
            self.pauseLLMTask?.cancel()
            self.pauseLLMTask = nil
            await monitorTask?.value

            guard self.activeRecordingSessionID == sessionID else { return }
            await self.flushPendingRecordingVoiceActivityFramesBeforeStop()
            self.stopActiveRecordingTranscriberAndRestoreSystemAudio()
        }
    }

    func updateActiveRecordingTranscriberTranscribedText(_ text: String) {
        switch transcriptionEngine {
        case .remote:
            remoteASRTranscriber.transcribedText = text
        case .mlxAudio:
            mlxTranscriber?.transcribedText = text
        case .dictation:
            speechTranscriber.transcribedText = text
        }
    }

    func consumeActiveRecordingRuntimeFailureMessage() -> String? {
        switch transcriptionEngine {
        case .mlxAudio:
            return mlxTranscriber?.consumePendingRuntimeFailureMessage()
        case .remote, .dictation:
            return nil
        }
    }

    func setActiveRecordingTranscriberEnhancingState(_ isEnhancing: Bool) {
        switch transcriptionEngine {
        case .mlxAudio:
            mlxTranscriber?.isEnhancing = isEnhancing
        case .remote:
            remoteASRTranscriber.isEnhancing = isEnhancing
        case .dictation:
            speechTranscriber.isEnhancing = isEnhancing
        }
    }

    func cancelPendingFinishTasks() {
        pendingSessionFinishTask?.cancel()
        pendingSessionFinishTask = nil
    }

    func cancelActiveRecordingTasks() {
        silenceMonitorTask?.cancel()
        silenceMonitorTask = nil
        pauseLLMTask?.cancel()
        pauseLLMTask = nil
    }

    func cancelSessionControlTasks() {
        cancelPendingFinishTasks()
        cancelActiveRecordingTasks()
    }

    private func handleRecordingPermissionDenied() {
        handleRecordingStartFailure(
            AppLocalization.localizedString("Please enable required permissions in Settings > Permissions.")
        )
    }

    private func handleRecordingStartFailure(
        _ message: String,
        autoHideAfter seconds: TimeInterval = 2.4
    ) {
        releaseResidualRecordingResources(reason: "recording-start-failure")
        showOverlayReminder(message, autoHideAfter: seconds)
        resetSessionAfterFailedStart()
    }

    private func restartCurrentRecordingCaptureForPreferredInputDevice() throws {
        if transcriptionEngine == .mlxAudio {
            try mlxTranscriber?.restartCaptureForPreferredInputDevice()
            return
        }

        if transcriptionEngine == .remote {
            try remoteASRTranscriber.restartCaptureForPreferredInputDevice()
            return
        }


        try speechTranscriber.restartCaptureForPreferredInputDevice()
    }

    func startSilenceMonitoringIfNeeded() {
        cancelActiveRecordingTasks()

        resetSilenceMonitoringState()
        let localVADMode = LocalVADMode.stored()
        let localVADGatePolicy = ASRVoiceActivityRuntimePolicy.localGatePolicy(
            transcriptionEngine: transcriptionEngine,
            mode: localVADMode
        )
        let localVADGateActive = localVADGatePolicy.isEnabled
        VoxtLog.asr(
            "Recording local VAD gate \(localVADGateActive ? "enabled" : "disabled"). mode=\(localVADMode.rawValue), output=\(RecordingSessionSupport.outputLabel(for: sessionOutputMode))",
            verbose: true
        )

        prepareRecordingVoiceActivityProcessing(
            mode: localVADMode,
            useCase: voiceActivityUseCase
        )

        silenceMonitorTask = Task { [weak self] in
            guard let self else { return }
            var observedSpeechEnd = false
            while !Task.isCancelled, self.isSessionActive {
                guard self.overlayState.isRecording else {
                    self.recordingVoiceActivitySegmenter?.reset()
                    await self.recordingVoiceActivityFrameDecider?.reset()
                    observedSpeechEnd = false
                    do {
                        try await Task.sleep(for: .milliseconds(200))
                    } catch {
                        return
                    }
                    continue
                }

                let level = self.overlayState.audioLevel
                let voiceActivityResult = await self.processPendingRecordingVoiceActivityFrames(
                    localVADGateActive: localVADGateActive,
                    level: level,
                    logEvents: true
                )
                let vadEvents = voiceActivityResult.events
                let sawSpeechFrame = voiceActivityResult.sawSpeechFrame
                let sawVoiceActivityFrame = voiceActivityResult.sawVoiceActivityFrame
                let shouldUseLevelTiming = ASRVoiceActivityRuntimePolicy.shouldUseLevelTiming(
                    localVADGateActive: localVADGateActive,
                    hasVoiceActivityFrames: sawVoiceActivityFrame
                )
                if !localVADGateActive {
                    self.recordingVoiceActivitySegmenter?.reset()
                    observedSpeechEnd = false
                }
                if vadEvents.contains(where: { event in
                    if case .speechEnded = event { return true }
                    return false
                }) {
                    observedSpeechEnd = true
                }
                if sawVoiceActivityFrame {
                    self.localVADObservedFramesInCurrentSession = true
                }
                if sawSpeechFrame {
                    self.localVADObservedSpeechInCurrentSession = true
                }
                if sawSpeechFrame || (shouldUseLevelTiming && level > self.silenceAudioLevelThreshold) {
                    observedSpeechEnd = false
                    self.lastSignificantAudioAt = Date()
                    self.didTriggerPauseTranscription = false
                    self.didTriggerPauseLLM = false
                    self.pauseLLMTask?.cancel()
                    self.pauseLLMTask = nil
                    self.setEnhancingState(false)
                } else if sawVoiceActivityFrame || shouldUseLevelTiming {
                    let silentDuration = Date().timeIntervalSince(self.lastSignificantAudioAt)

                    if ASRLocalIntermediateGatePolicy.shouldTriggerIntermediateTranscription(
                        transcriptionEngine: self.transcriptionEngine,
                        localVADGateActive: localVADGateActive,
                        silentDuration: silentDuration,
                        didTriggerPauseTranscription: self.didTriggerPauseTranscription,
                        observedSpeechEnd: observedSpeechEnd
                    ) {
                        self.didTriggerPauseTranscription = true
                        observedSpeechEnd = false
                        VoxtLog.asr(
                            "Recording VAD triggered intermediate transcription after trailing silence. silentDurationSec=\(String(format: "%.3f", silentDuration))",
                            verbose: true
                        )
                        self.mlxTranscriber?.forceIntermediateTranscription()
                    }

                }

                if self.shouldStopRecordingForVoiceEndCommand() {
                    self.triggerVoiceEndCommandStop()
                    return
                }

                do {
                    try await Task.sleep(for: .milliseconds(200))
                } catch {
                    return
                }
            }
        }
    }

    private func resetSilenceMonitoringState() {
        lastSignificantAudioAt = Date()
        didTriggerPauseTranscription = false
        didTriggerPauseLLM = false
        localVADObservedFramesInCurrentSession = false
        localVADObservedSpeechInCurrentSession = false
        recordingVoiceActivityFrameDecider = nil
        recordingVoiceActivitySegmenter = nil
        recordingVoiceActivityMode = nil
        recordingVoiceActivityUseCase = nil
        recordingVoiceActivityDebugStats = RecordingVoiceActivityDebugStats()
        voiceEndCommandState.lastDetectedCommand = false
    }

    private var voiceActivityUseCase: ASRVoiceActivityUseCase {
        switch sessionOutputMode {
        case .transcription:
            return .transcription
        case .translation:
            return .translation
        case .rewrite:
            return .rewrite
        }
    }

    private func triggerVoiceEndCommandStop() {
        voiceEndCommandState.didAutoStop = true
        voiceEndCommandState.lastDetectedCommand = false
        VoxtLog.hotkey("Voice end command triggered stop after trailing silence.")
        endRecording()
    }

    private func prepareRecordingVoiceActivityProcessing(
        mode: LocalVADMode,
        useCase: ASRVoiceActivityUseCase
    ) {
        recordingVoiceActivityMode = mode
        recordingVoiceActivityUseCase = useCase
        recordingVoiceActivityFrameDecider = RecordingVoiceActivityFrameDecider(
            mode: mode,
            useCase: useCase,
            energyThreshold: silenceAudioLevelThreshold
        )
        recordingVoiceActivitySegmenter = ASRVoiceActivitySegmenter(
            configuration: ASRVoiceActivityConfiguration.profile(for: useCase)
        )
        recordingVoiceActivityDebugStats = RecordingVoiceActivityDebugStats(
            mode: mode,
            useCase: useCase,
            backend: ASRVoiceActivityRuntimePolicy.effectiveBackend(mode: mode, useCase: useCase)
        )
        VoxtLog.vad(
            "Recording VAD debug start. mode=\(mode.rawValue), backend=\(recordingVoiceActivityDebugStats.backend.rawValue), useCase=\(useCase.rawValue)"
        )
    }

    func flushPendingRecordingVoiceActivityFramesBeforeStop() async {
        guard transcriptionEngine == .mlxAudio else { return }
        let localVADMode = LocalVADMode.stored()
        let localVADGatePolicy = ASRVoiceActivityRuntimePolicy.localGatePolicy(
            transcriptionEngine: transcriptionEngine,
            mode: localVADMode
        )
        guard localVADGatePolicy.isEnabled else { return }

        if recordingVoiceActivityFrameDecider == nil || recordingVoiceActivitySegmenter == nil {
            prepareRecordingVoiceActivityProcessing(
                mode: localVADMode,
                useCase: voiceActivityUseCase
            )
        }

        let result = await processPendingRecordingVoiceActivityFrames(
            localVADGateActive: true,
            level: overlayState.audioLevel,
            logEvents: true
        )
        if result.sawVoiceActivityFrame {
            VoxtLog.asr(
                "Recording VAD stop flush processed pending frames. speech=\(result.sawSpeechFrame), events=\(result.events.count)",
                verbose: true
            )
        }
        logRecordingVoiceActivityDebugSummary(reason: "stop-flush")
        mlxTranscriber?.finishVoiceActivityFinalizationFiltering()
    }

    private func processPendingRecordingVoiceActivityFrames(
        localVADGateActive: Bool,
        level: Float,
        logEvents: Bool
    ) async -> RecordingVoiceActivityDrainResult {
        guard localVADGateActive,
              let mlxTranscriber,
              let frameDecider = recordingVoiceActivityFrameDecider
        else {
            recordingVoiceActivitySegmenter?.reset()
            return .empty
        }

        let frames = mlxTranscriber.consumeVoiceActivityFrames()
        guard !frames.isEmpty else { return .empty }

        if recordingVoiceActivitySegmenter == nil {
            recordingVoiceActivitySegmenter = ASRVoiceActivitySegmenter(
                configuration: ASRVoiceActivityConfiguration.profile(for: voiceActivityUseCase)
            )
        }
        var segmenter = recordingVoiceActivitySegmenter ?? ASRVoiceActivitySegmenter(
            configuration: ASRVoiceActivityConfiguration.profile(for: voiceActivityUseCase)
        )
        var events: [ASRVoiceActivityEvent] = []
        var sawSpeechFrame = false
        var sawVoiceActivityFrame = false

        for frame in frames {
            guard let result = await frameDecider.decision(for: frame) else {
                continue
            }
            sawVoiceActivityFrame = true
            let segmenterResult = segmenter.appendWithResolvedSpeechState(result.decision)
            if segmenterResult.isSpeech {
                sawSpeechFrame = true
            }
            recordingVoiceActivityDebugStats.record(
                frame: frame,
                result: result,
                resolvedSpeech: segmenterResult.isSpeech,
                events: segmenterResult.events,
                fallbackSource: .fallbackEnergy
            )
            mlxTranscriber.appendVoiceActivityFinalizationFrame(
                frame,
                isSpeech: segmenterResult.isSpeech
            )
            if logEvents {
                for event in segmenterResult.events {
                    VoxtLog.asr(
                        "Recording VAD event. \(event.telemetrySummary), source=\(result.source.telemetryName), level=\(String(format: "%.3f", frame.level ?? level)), probability=\(result.probabilityText), threshold=\(String(format: "%.3f", silenceAudioLevelThreshold))",
                        verbose: true
                    )
                }
            }
            events.append(contentsOf: segmenterResult.events)
        }

        recordingVoiceActivitySegmenter = segmenter
        if sawVoiceActivityFrame {
            localVADObservedFramesInCurrentSession = true
        }
        if sawSpeechFrame {
            localVADObservedSpeechInCurrentSession = true
        }
        return RecordingVoiceActivityDrainResult(
            events: events,
            sawSpeechFrame: sawSpeechFrame,
            sawVoiceActivityFrame: sawVoiceActivityFrame
        )
    }

    private func logRecordingVoiceActivityDebugSummary(reason: String) {
        VoxtLog.vad(
            "Recording VAD debug summary. reason=\(reason), \(recordingVoiceActivityDebugStats.telemetrySummary)"
        )
    }

}

private struct RecordingVoiceActivityDrainResult: Sendable {
    let events: [ASRVoiceActivityEvent]
    let sawSpeechFrame: Bool
    let sawVoiceActivityFrame: Bool

    static let empty = RecordingVoiceActivityDrainResult(
        events: [],
        sawSpeechFrame: false,
        sawVoiceActivityFrame: false
    )
}

struct RecordingVoiceActivityDecisionResult: Sendable {
    let decision: ASRVoiceActivityFrameDecision
    let source: Source

    enum Source: Sendable {
        case energy
        case silero
        case omni
        case fallbackEnergy

        var telemetryName: String {
            switch self {
            case .energy:
                return "energy"
            case .silero:
                return "silero"
            case .omni:
                return "omnivad"
            case .fallbackEnergy:
                return "fallback-energy"
            }
        }
    }

    var probabilityText: String {
        decision.probability.map { String(format: "%.3f", $0) } ?? "nil"
    }
}

struct RecordingVoiceActivityDebugStats: Sendable {
    var mode: LocalVADMode = .automatic
    var useCase: ASRVoiceActivityUseCase = .transcription
    var backend: ASRVoiceActivityBackendKind = .off
    private(set) var totalFrames = 0
    private(set) var speechFrames = 0
    private(set) var silenceFrames = 0
    private(set) var fallbackFrames = 0
    private(set) var events = 0
    private(set) var speechStartedEvents = 0
    private(set) var speechEndedEvents = 0
    private(set) var speechForcedEvents = 0
    private(set) var speechRejectedEvents = 0
    private(set) var probabilityFrames = 0
    private(set) var probabilitySum: Float = 0
    private(set) var minProbability: Float?
    private(set) var maxProbability: Float?
    private(set) var levelFrames = 0
    private(set) var levelSum: Float = 0
    private(set) var minLevel: Float?
    private(set) var maxLevel: Float?
    private(set) var firstFrameStartSeconds: TimeInterval?
    private(set) var lastFrameEndSeconds: TimeInterval?
    private(set) var sourceCounts: [String: Int] = [:]

    mutating func record(
        frame: ASRVoiceActivityAudioFrame,
        result: RecordingVoiceActivityDecisionResult,
        resolvedSpeech: Bool,
        events newEvents: [ASRVoiceActivityEvent],
        fallbackSource: RecordingVoiceActivityDecisionResult.Source
    ) {
        totalFrames += 1
        if resolvedSpeech {
            speechFrames += 1
        } else {
            silenceFrames += 1
        }
        if result.source.telemetryName == fallbackSource.telemetryName {
            fallbackFrames += 1
        }
        sourceCounts[result.source.telemetryName, default: 0] += 1

        if let probability = result.decision.probability {
            probabilityFrames += 1
            probabilitySum += probability
            minProbability = min(minProbability ?? probability, probability)
            maxProbability = max(maxProbability ?? probability, probability)
        }

        if let level = frame.level {
            levelFrames += 1
            levelSum += level
            minLevel = min(minLevel ?? level, level)
            maxLevel = max(maxLevel ?? level, level)
        }

        firstFrameStartSeconds = min(firstFrameStartSeconds ?? frame.startSeconds, frame.startSeconds)
        lastFrameEndSeconds = max(lastFrameEndSeconds ?? frame.endSeconds, frame.endSeconds)

        events += newEvents.count
        for event in newEvents {
            switch event {
            case .speechStarted:
                speechStartedEvents += 1
            case .speechEnded:
                speechEndedEvents += 1
            case .speechForced:
                speechForcedEvents += 1
            case .speechRejected:
                speechRejectedEvents += 1
            }
        }
    }

    var telemetrySummary: String {
        let speechRatio = totalFrames > 0 ? Double(speechFrames) / Double(totalFrames) : 0
        let probabilityAverage = probabilityFrames > 0 ? probabilitySum / Float(probabilityFrames) : nil
        let levelAverage = levelFrames > 0 ? levelSum / Float(levelFrames) : nil
        let audioSpanMs = firstFrameStartSeconds.flatMap { first in
            lastFrameEndSeconds.map { max(0, Int((($0 - first) * 1_000).rounded())) }
        }
        return [
            "mode=\(mode.rawValue)",
            "backend=\(backend.rawValue)",
            "useCase=\(useCase.rawValue)",
            "frames=\(totalFrames)",
            "speechFrames=\(speechFrames)",
            "silenceFrames=\(silenceFrames)",
            "speechRatio=\(Self.percentText(speechRatio))",
            "sources=\(Self.sourceCountsText(sourceCounts))",
            "fallbackFrames=\(fallbackFrames)",
            "events=\(events)",
            "speechStarted=\(speechStartedEvents)",
            "speechEnded=\(speechEndedEvents)",
            "speechForced=\(speechForcedEvents)",
            "speechRejected=\(speechRejectedEvents)",
            "probabilityMin=\(Self.floatText(minProbability))",
            "probabilityAvg=\(Self.floatText(probabilityAverage))",
            "probabilityMax=\(Self.floatText(maxProbability))",
            "levelMin=\(Self.floatText(minLevel))",
            "levelAvg=\(Self.floatText(levelAverage))",
            "levelMax=\(Self.floatText(maxLevel))",
            "audioSpanMs=\(audioSpanMs.map(String.init) ?? "nil")"
        ].joined(separator: ", ")
    }

    private static func sourceCountsText(_ counts: [String: Int]) -> String {
        guard !counts.isEmpty else { return "none" }
        return counts
            .sorted { $0.key < $1.key }
            .map { "\($0.key):\($0.value)" }
            .joined(separator: "|")
    }

    private static func floatText(_ value: Float?) -> String {
        value.map { String(format: "%.3f", $0) } ?? "nil"
    }

    private static func percentText(_ value: Double) -> String {
        String(format: "%.1f%%", value * 100)
    }
}

actor RecordingVoiceActivityFrameDecider {
    private let mode: LocalVADMode
    private let useCase: ASRVoiceActivityUseCase
    private let energyBackend: ASREnergyVoiceActivityBackend
    private let sileroThreshold: Float
    private let sileroDetector = ASRSileroStreamingVoiceActivityDetector()
    private let omniDetector: OmniStreamVoiceActivityBackend
    private var sileroFallbackWarningLogged = false
    private var omniDegradedWarningLogged = false

    init(
        mode: LocalVADMode,
        useCase: ASRVoiceActivityUseCase,
        energyThreshold: Float
    ) {
        self.mode = mode
        self.useCase = useCase
        self.energyBackend = ASREnergyVoiceActivityBackend(threshold: energyThreshold)
        self.sileroThreshold = ASRVoiceActivityConfiguration.profile(for: useCase).onsetProbabilityThreshold
        self.omniDetector = OmniStreamVoiceActivityBackend(useCase: useCase)
    }

    func reset() async {
        await sileroDetector.reset()
        await omniDetector.reset()
        sileroFallbackWarningLogged = false
        omniDegradedWarningLogged = false
    }

    func releaseResources() async {
        await sileroDetector.unload()
        await omniDetector.releaseResources()
        sileroFallbackWarningLogged = false
        omniDegradedWarningLogged = false
    }

    func decision(for frame: ASRVoiceActivityAudioFrame) async -> RecordingVoiceActivityDecisionResult? {
        switch ASRVoiceActivityRuntimePolicy.effectiveBackend(mode: mode, useCase: useCase) {
        case .off:
            return nil
        case .energy:
            return await energyDecision(for: frame, source: .energy)
        case .mlxSilero:
            if let sileroDecision = await sileroDecision(for: frame) {
                return sileroDecision
            }
            return await energyDecision(for: frame, source: .fallbackEnergy)
        case .omniStream:
            if let omniDecision = await omniDecision(for: frame) {
                return omniDecision
            }
            return await energyDecision(for: frame, source: .fallbackEnergy)
        }
    }

    private func sileroDecision(for frame: ASRVoiceActivityAudioFrame) async -> RecordingVoiceActivityDecisionResult? {
        do {
            if let probability = try await sileroDetector.probability(
                samples: frame.samples,
                sampleRate: frame.sampleRate,
                streamID: "recording-\(useCase.rawValue)"
            ) {
                return RecordingVoiceActivityDecisionResult(
                    decision: ASRVoiceActivityFrameDecision(
                        startSeconds: frame.startSeconds,
                        endSeconds: frame.endSeconds,
                        isSpeech: probability >= sileroThreshold,
                        probability: probability
                    ),
                    source: .silero
                )
            }
        } catch {
            if shouldLogSileroFallback(error) {
                VoxtLog.asrWarning("Recording Silero VAD failed; falling back to energy VAD. error=\(error.localizedDescription)")
                sileroFallbackWarningLogged = true
            }
            await sileroDetector.reset()
        }
        return nil
    }

    private func omniDecision(for frame: ASRVoiceActivityAudioFrame) async -> RecordingVoiceActivityDecisionResult? {
        do {
            if let decision = try await omniDetector.decision(
                for: frame,
                streamID: "recording-\(useCase.rawValue)"
            ) {
                return RecordingVoiceActivityDecisionResult(
                    decision: decision,
                    source: .omni
                )
            }
        } catch {
            if !omniDegradedWarningLogged {
                VoxtLog.asrWarning("Recording OmniVAD unavailable; degrading current session to energy VAD. error=\(error.localizedDescription)")
                omniDegradedWarningLogged = true
            }
            await omniDetector.reset()
        }
        return nil
    }

    private func energyDecision(
        for frame: ASRVoiceActivityAudioFrame,
        source: RecordingVoiceActivityDecisionResult.Source
    ) async -> RecordingVoiceActivityDecisionResult? {
        guard let decision = try? await energyBackend.decision(for: frame) else {
            return nil
        }
        return RecordingVoiceActivityDecisionResult(decision: decision, source: source)
    }

    private func shouldLogSileroFallback(_ error: Error) -> Bool {
        if let modelError = error as? MeetingVADModelError {
            switch modelError {
            case .modelNotDownloaded, .runtimeUnavailable:
                return false
            }
        }
        return !sileroFallbackWarningLogged
    }
}
