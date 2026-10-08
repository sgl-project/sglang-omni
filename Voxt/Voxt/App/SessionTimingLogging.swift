// Timing snapshots are captured before post-delivery state changes.

import Foundation

extension AppDelegate {
    struct SessionTimingSummarySnapshot {
        let transcriptionCapturePipeline: TranscriptionCapturePipeline
        let captureStageLabels: [String]
        let asrProvider: String
        let asrModel: String
        let captureMetrics: TranscriptionCaptureMetrics?
        let recordingRequestedAt: Date?
        let recordingStartedAt: Date?
        let recordingStoppedAt: Date?
        let transcriptionResultReceivedAt: Date?
        let firstLiveASRPartialReceivedAt: Date?
        let sessionFinalOutputDeliveredAt: Date?
        let llmExecutions: [SessionLLMExecutionTiming]
    }

    func logSessionTimingSummaryIfPossible(
        snapshot: SessionTimingSummarySnapshot,
        deliveredText: String,
        outputMode: SessionOutputMode,
        didInject: Bool
    ) {
        let outputCompletedAt = snapshot.sessionFinalOutputDeliveredAt ?? Date()
        let firstLLMExecution = snapshot.llmExecutions.first
        let finalLLMExecution = snapshot.llmExecutions.last

        let requestToStartMs = resolvedDurationMs(from: snapshot.recordingRequestedAt, to: snapshot.recordingStartedAt)
        let startToStopMs = resolvedDurationMs(from: snapshot.recordingStartedAt, to: snapshot.recordingStoppedAt)
        let startToFirstLiveASRMs = resolvedDurationMs(
            from: snapshot.recordingStartedAt,
            to: snapshot.firstLiveASRPartialReceivedAt
        )
        let stopToASRMs = resolvedDurationMs(from: snapshot.recordingStoppedAt, to: snapshot.transcriptionResultReceivedAt)
        let asrToFirstChunkMs = resolvedDurationMs(
            from: snapshot.transcriptionResultReceivedAt,
            to: firstLLMExecution?.firstChunkAt
        )
        let asrToFirstCompleteMs = resolvedDurationMs(
            from: snapshot.transcriptionResultReceivedAt,
            to: firstLLMExecution?.completedAt
        )
        let asrToFinalCompleteMs = resolvedDurationMs(
            from: snapshot.transcriptionResultReceivedAt,
            to: finalLLMExecution?.completedAt
        )
        let asrToDeliveredMs = resolvedDurationMs(
            from: snapshot.transcriptionResultReceivedAt,
            to: outputCompletedAt
        )
        let wallClockCaptureMs = resolvedDurationMs(
            from: snapshot.recordingStartedAt,
            to: snapshot.recordingStoppedAt
        )
        let capturedAudioMs = snapshot.captureMetrics.map { Int($0.capturedAudioSeconds * 1000) }
        let captureGapMs = resolvedCaptureGapMs(
            wallClockCaptureMs: wallClockCaptureMs,
            capturedAudioMs: capturedAudioMs
        )
        let stopToDeliveredMs = resolvedDurationMs(from: snapshot.recordingStoppedAt, to: outputCompletedAt)
        let overallMs = resolvedDurationMs(from: snapshot.recordingRequestedAt, to: outputCompletedAt)

        let firstLLMSummary = sessionLLMSummaryLabel(firstLLMExecution)
        let finalLLMSummary = sessionLLMSummaryLabel(finalLLMExecution)

        if let captureGapMs, captureGapMs >= 350 {
            VoxtLog.inputWarning(
                "Transcription capture gap detected. pipeline=\(snapshot.transcriptionCapturePipeline.rawValue), captureGapMs=\(captureGapMs), capturedAudioMs=\(timingValueLabel(capturedAudioMs)), startToStopMs=\(timingValueLabel(startToStopMs))"
            )
        }

        VoxtLog.input(
            """
            Session timing summary. output=\(RecordingSessionSupport.outputLabel(for: outputMode)), pipeline=\(snapshot.transcriptionCapturePipeline.rawValue), stages=\(snapshot.captureStageLabels.joined(separator: ">")), asrProvider=\(snapshot.asrProvider), asrModel=\(snapshot.asrModel), llmCalls=\(snapshot.llmExecutions.count), deliveredChars=\(deliveredText.count), didInject=\(didInject), requestToStartMs=\(timingValueLabel(requestToStartMs)), startToStopMs=\(timingValueLabel(startToStopMs)), startToFirstLiveASRMs=\(timingValueLabel(startToFirstLiveASRMs)), capturedAudioMs=\(timingValueLabel(capturedAudioMs)), captureGapMs=\(timingValueLabel(captureGapMs)), stopToASRMs=\(timingValueLabel(stopToASRMs)), asrToFirstLLMChunkMs=\(timingValueLabel(asrToFirstChunkMs)), asrToFirstLLMCompleteMs=\(timingValueLabel(asrToFirstCompleteMs)), asrToFinalLLMCompleteMs=\(timingValueLabel(asrToFinalCompleteMs)), asrToDeliveredMs=\(timingValueLabel(asrToDeliveredMs)), stopToDeliveredMs=\(timingValueLabel(stopToDeliveredMs)), overallMs=\(timingValueLabel(overallMs)), firstLLM=\(firstLLMSummary), finalLLM=\(finalLLMSummary)
            """
        )
    }

    func sessionASRSummary(for outputMode: SessionOutputMode) -> (provider: String, model: String) {
        let selectionID: FeatureModelSelectionID
        switch outputMode {
        case .transcription:
            selectionID = transcriptionFeatureSettings.asrSelectionID
        case .translation:
            selectionID = translationFeatureSettings.asrSelectionID
        case .rewrite:
            selectionID = rewriteFeatureSettings.asrSelectionID
        }

        switch selectionID.asrSelection {
        case .dictation:
            return ("dictation", "builtin")
        case .mlx(let repo):
            let canonicalRepo = MLXModelManager.canonicalModelRepo(repo)
            return (MLXWhisperMigrationSupport.isWhisperRepo(canonicalRepo) ? "whisper-mlx" : "mlx", canonicalRepo)
        case .remote(let provider):
            let raw = UserDefaults.standard.string(forKey: AppPreferenceKey.remoteASRProviderConfigurations) ?? ""
            let stored = RemoteModelConfigurationStore.loadConfigurations(
                from: raw,
                sensitiveValueLoading: .metadataOnly
            )
            let configuration = RemoteModelConfigurationStore.resolvedASRConfiguration(provider: provider, stored: stored)
            return ("remote:\(provider.rawValue)", configuration.model)
        case .none:
            switch transcriptionEngine {
            case .dictation:
                return ("dictation", "builtin")
            case .mlxAudio:
                let canonicalRepo = MLXModelManager.canonicalModelRepo(mlxModelManager.currentModelRepo)
                return (MLXWhisperMigrationSupport.isWhisperRepo(canonicalRepo) ? "whisper-mlx" : "mlx", canonicalRepo)
            case .remote:
                return ("remote", "unknown")
            }
        }
    }

    private func resolvedDurationMs(from start: Date?, to end: Date?) -> Int? {
        guard let start, let end else { return nil }
        return max(Int(end.timeIntervalSince(start) * 1000), 0)
    }

    private func resolvedCaptureGapMs(
        wallClockCaptureMs: Int?,
        capturedAudioMs: Int?
    ) -> Int? {
        guard let wallClockCaptureMs, let capturedAudioMs else { return nil }
        return max(wallClockCaptureMs - capturedAudioMs, 0)
    }

    private func timingValueLabel(_ value: Int?) -> String {
        value.map(String.init) ?? "n/a"
    }

    private func sessionLLMSummaryLabel(_ execution: SessionLLMExecutionTiming?) -> String {
        guard let execution else { return "n/a" }
        let diagnostics = execution.diagnostics
        let firstChunk = diagnostics?.overallFirstChunkMs.map(String.init) ?? "n/a"
        let prefill = diagnostics?.prefillMs.map(String.init) ?? "n/a"
        let generation = diagnostics?.generationMs.map(String.init) ?? "n/a"
        let total = diagnostics.map { String($0.totalElapsedMs) } ?? timingValueLabel(
            resolvedDurationMs(from: execution.startedAt, to: execution.completedAt)
        )
        return
            "task=\(execution.taskLabel),provider=\(execution.providerLabel),firstChunkMs=\(firstChunk),prefillMs=\(prefill),generationMs=\(generation),totalElapsedMs=\(total)"
    }
}
