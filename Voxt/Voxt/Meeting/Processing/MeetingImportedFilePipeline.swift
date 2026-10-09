import Foundation

/// Resource owner for one file import. It never mutates the live meeting's
/// transcriber, engine selection or model-use fields.
@MainActor
final class MeetingImportedFilePipeline: MeetingImportedFileAnalyzing {
    private let modelManager: MLXModelManager
    private let engineContext: MeetingASREngineContext
    private let sourceIsPreparedAudio: Bool
    private let archivePreparedAudio: Bool
    private var transcriber: (any MeetingSegmentTranscribing)?
    private var holdsModelUse = false
    private var preparedAudioURL: URL?

    init(modelManager: MLXModelManager, engineContext: MeetingASREngineContext,
         sourceIsPreparedAudio: Bool = false, archivePreparedAudio: Bool = true) {
        self.modelManager = modelManager
        self.engineContext = engineContext
        self.sourceIsPreparedAudio = sourceIsPreparedAudio
        self.archivePreparedAudio = archivePreparedAudio
    }

    func analyze(
        at sourceURL: URL,
        progress: @escaping @MainActor @Sendable (MeetingFileAnalysisProgress) -> Void
    ) async throws -> MeetingSessionResult {
        try Task.checkCancellation()
        progress(MeetingFileAnalysisProgress(stage: .preparing, stageFraction: sourceIsPreparedAudio ? 1 : 0))
        var stage = "audio-validation"
        do {
            let sourceIsPreparedAudio = sourceIsPreparedAudio
            let traceTaskID = MeetingFileTaskContext.taskID
            let preparationTask = Task.detached(priority: .utility) {
                try await MeetingFileTaskContext.$taskID.withValue(traceTaskID) {
                    if sourceIsPreparedAudio {
                        return try MeetingImportedAudioFile.validatedPreparedFile(at: sourceURL)
                    }
                    return try await MeetingImportedAudioFile.prepare(from: sourceURL) { fraction in
                        await progress(
                            MeetingFileAnalysisProgress(
                                stage: .preparing,
                                stageFraction: fraction
                            )
                        )
                    }
                }
            }
            let importedAudio = try await withTaskCancellationHandler {
                try await preparationTask.value
            } onCancel: {
                preparationTask.cancel()
            }
            // Queue caches are borrowed read-only, never owned by this pipeline.
            // Only newly created temporary audio can be removed by cleanup.
            if !sourceIsPreparedAudio { preparedAudioURL = importedAudio.standardizedAudioURL }
            try Task.checkCancellation()

            progress(
                MeetingFileAnalysisProgress(
                    stage: .preparing,
                    stageFraction: 1,
                    mediaDurationSeconds: importedAudio.durationSeconds
                )
            )

            stage = "transcribing"
            let taskID = MeetingFileTaskContext.taskID
            let checkpointStore = MeetingFileAnalysisCheckpointStore.shared
            let modelFingerprint = Self.modelFingerprint(for: engineContext)
            let storedCheckpoint: MeetingFileASRCheckpoint?
            if let taskID {
                let candidate = await checkpointStore.load(taskID: taskID)
                storedCheckpoint = candidate?.preparedAudioSampleCount == importedAudio.sampleCount
                    && candidate?.descriptorCount == importedAudio.assetDescriptors.count
                    && candidate?.modelFingerprint == modelFingerprint
                    ? candidate
                    : nil
                if candidate != nil, storedCheckpoint == nil {
                    await checkpointStore.clear(taskID: taskID)
                    VoxtLog.meetingWarning("File ASR checkpoint discarded because the input or model changed. taskID=\(taskID)")
                }
            } else {
                storedCheckpoint = nil
            }
            let startingDescriptorIndex = storedCheckpoint?.completedDescriptorCount ?? 0
            let initialSegments = storedCheckpoint?.segments ?? []
            progress(MeetingFileAnalysisProgress(stage: .transcribing))
            let transcriptSegments: [MeetingTranscriptSegment]
            do {
                let importedTranscriber = try makeTranscriber()
                transcriber = importedTranscriber
                try Task.checkCancellation()

                transcriptSegments = try await MeetingFinalTranscriptionPass.transcribe(
                    descriptors: importedAudio.assetDescriptors,
                    loadAsset: { descriptor in
                        importedAudio.loadAsset(descriptor)
                    },
                    transcriber: importedTranscriber,
                    requiresCompleteTranscription: true,
                    processedDurationProgress: { fraction, processedDuration in
                        await progress(
                            MeetingFileAnalysisProgress(
                                stage: .transcribing,
                                stageFraction: fraction,
                                mediaDurationSeconds: importedAudio.durationSeconds,
                                processedMediaDurationSeconds: processedDuration
                            )
                        )
                    },
                    initialSegments: initialSegments,
                    startingDescriptorIndex: startingDescriptorIndex,
                    checkpoint: { segments, completedDescriptorCount in
                        guard let taskID else { return }
                        await checkpointStore.save(
                            MeetingFileASRCheckpoint(
                                schemaVersion: MeetingFileASRCheckpoint.currentSchemaVersion,
                                taskID: taskID,
                                preparedAudioSampleCount: importedAudio.sampleCount,
                                descriptorCount: importedAudio.assetDescriptors.count,
                                modelFingerprint: modelFingerprint,
                                completedDescriptorCount: completedDescriptorCount,
                                segments: segments,
                                updatedAt: Date()
                            )
                        )
                    }
                )
            }
            try Task.checkCancellation()
            guard !MeetingTranscriptFormatter.meaningfulSegments(for: transcriptSegments).isEmpty else {
                throw MeetingFileAnalysisError.noTranscript
            }

            await releaseASRResourcesAtStageBoundary()
            stage = "identifyingSpeakers"
            progress(MeetingFileAnalysisProgress(stage: .identifyingSpeakers))
            // The file engine acquires/releases a permit per small feed, not for
            // the entire recording. Keep native speaker state continuous inside it.
            let finalSegments = try await MeetingSpeakerAnalysisPipeline.analyzedFileSegments(
                from: transcriptSegments,
                descriptors: importedAudio.assetDescriptors,
                loadAsset: { descriptor in importedAudio.loadAsset(descriptor) },
                options: MeetingSpeakerDiarizationOptions.fromPreferences(),
                progress: { fraction in
                    await progress(MeetingFileAnalysisProgress(
                        stage: .identifyingSpeakers, stageFraction: fraction
                    ))
                }
            )
            try Task.checkCancellation()

            stage = "saving"
            progress(MeetingFileAnalysisProgress(stage: .saving))
            if sourceIsPreparedAudio, archivePreparedAudio {
                // History moves its input. Create an independent copy only after
                // analysis succeeds and only when an archive was requested.
                let traceID = MeetingFileTaskContext.taskID
                let archiveTask = Task.detached(priority: .utility) {
                    try await MeetingFileTaskContext.$taskID.withValue(traceID) {
                        try await MeetingImportedAudioFile.copyPreparedForAnalysis(from: sourceURL)
                    }
                }
                let archive = try await withTaskCancellationHandler {
                    try await archiveTask.value
                } onCancel: { archiveTask.cancel() }
                preparedAudioURL = archive.standardizedAudioURL
                try Task.checkCancellation()
            }
            let result = MeetingSessionResult(
                captureMode: .meeting,
                transcriptionEngine: engineContext.engine,
                transcriptionModelDescription: engineContext.historyModelDescription,
                segments: finalSegments,
                visibleSnapshotSegments: finalSegments,
                audioDurationSeconds: importedAudio.durationSeconds,
                archivedAudioURL: preparedAudioURL
            )
            return result
        } catch {
            if !(error is CancellationError) {
                VoxtLog.meetingError("File analysis stopped. stage=\(stage), \(MeetingFileTaskDiagnostics.errorSummary(error))")
            }
            if let preparedAudioURL {
                try? FileManager.default.removeItem(at: preparedAudioURL)
            }
            throw error
        }
    }

    private func releaseASRResourcesAtStageBoundary() async {
        await transcriber?.cancelPendingWork()
        transcriber = nil
        guard holdsModelUse else { return }
        holdsModelUse = false
        modelManager.endActiveUse()
        modelManager.releaseLoadedModelIfIdle(reason: "file-asr-stage-completed")
    }

    private static func modelFingerprint(for context: MeetingASREngineContext) -> String {
        [
            context.engine.rawValue,
            context.historyModelDescription,
            context.mlxModelRepo ?? "none",
            String(describing: context.resolvedMode)
        ].joined(separator: "|")
    }

    private func makeTranscriber() throws -> any MeetingSegmentTranscribing {
        switch engineContext.engine {
        case .mlxAudio:
            modelManager.beginActiveUse()
            holdsModelUse = true
            return MeetingMLXSegmentTranscriber(modelManager: modelManager, strictInferenceWorkClass: .fileASR)
        case .remote:
            return MeetingRemoteASRSegmentTranscriber()
        case .dictation:
            throw NSError(domain: "Voxt.Meeting", code: -1, userInfo: [NSLocalizedDescriptionKey: "Direct Dictation is not supported for Meeting Notes."])
        }
    }

    func cancel() async {
        await transcriber?.cancelPendingWork()
    }

    func finish(keepingResult: Bool) async {
        await transcriber?.cancelPendingWork()
        transcriber = nil
        if holdsModelUse {
            holdsModelUse = false
            modelManager.endActiveUse()
        }
        if !keepingResult || Task.isCancelled, let preparedAudioURL {
            try? FileManager.default.removeItem(at: preparedAudioURL)
            self.preparedAudioURL = nil
        }
    }
}
