import Foundation

enum MeetingFileTaskStatus: String, Codable, Hashable, Sendable {
    case queued
    case preparing
    case processing
    case waitingForResources
    case cancelling
    case completed
    case failed
    case cancelled

    var isTerminal: Bool {
        switch self {
        case .completed, .failed, .cancelled:
            return true
        case .queued, .preparing, .processing, .waitingForResources, .cancelling:
            return false
        }
    }
}

struct MeetingFileTask: Identifiable, Codable, Hashable, Sendable {
    let id: UUID
    let fileName: String
    var stagedFileName: String
    // Optional for v1 task migration. Publish only complete canonical WAVs.
    var preparedAudioVersion: Int? = nil
    var legacyStagedFileName: String? = nil
    let enqueuedAt: Date
    var startedAt: Date?
    var completedAt: Date?
    var status: MeetingFileTaskStatus
    var progressStage: MeetingFileAnalysisStage
    var progressFraction: Double
    var mediaDurationSeconds: TimeInterval?
    var processedMediaDurationSeconds: TimeInterval?
    /// Smoothed audio-seconds-per-wall-second measured during transcription.
    var processingSpeedSecondsPerSecond: Double?
    var speedSampleAt: Date?
    var speedSampleProcessedMediaDurationSeconds: TimeInterval?
    /// Conservative total-duration estimate captured while the task is running.
    /// It is intentionally monotonic: later progress observations may lower it,
    /// but never make it larger and cause the UI to oscillate.
    var estimatedTotalSeconds: TimeInterval?
    var errorMessage: String?
    var historyEntryID: UUID?

    var isTerminal: Bool { status.isTerminal }

    func elapsedSeconds(now: Date) -> TimeInterval {
        guard let startedAt else { return 0 }
        let end = completedAt ?? now
        return max(0, end.timeIntervalSince(startedAt))
    }

    func estimatedRemainingSeconds(now: Date) -> TimeInterval? {
        guard status == .processing else { return nil }
        let elapsed = elapsedSeconds(now: now)
        guard elapsed > 0 else { return nil }
        if let estimatedTotalSeconds, estimatedTotalSeconds > elapsed {
            return max(0, estimatedTotalSeconds - elapsed)
        }

        // If an old estimate was too optimistic and has already elapsed,
        // fall back to the current overall progress instead of displaying
        // zero while the task is still processing.
        guard progressFraction > 0 else { return nil }
        return max(0, elapsed * (1 - progressFraction) / progressFraction)
    }

    static func updatedEstimatedTotalSeconds(
        current: TimeInterval?,
        elapsed: TimeInterval,
        progressFraction: Double,
        mediaDurationSeconds: TimeInterval? = nil,
        processedMediaDurationSeconds: TimeInterval? = nil,
        stage: MeetingFileAnalysisStage? = nil,
        processingSpeed: Double? = nil
    ) -> TimeInterval? {
        guard elapsed > 0, progressFraction > 0 else { return current }

        // The first estimate is deliberately conservative. Once established,
        // only a faster observed rate can lower it; a slower phase never makes
        // the remaining-time label jump backwards.
        let observedTotal: TimeInterval
        if stage == nil || stage == .transcribing || stage == .identifyingSpeakers || stage == .saving,
           let mediaDurationSeconds,
           let processedMediaDurationSeconds,
           mediaDurationSeconds > 0,
           processedMediaDurationSeconds > 0 {
            let sampleSpeed = max(
                processingSpeed ?? processedMediaDurationSeconds / elapsed,
                0.001
            )
            let transcriptionTotal = mediaDurationSeconds / sampleSpeed
            // Transcription accounts for 63% of the overall task progress.
            // Include the later speaker-analysis and save stages so finishing
            // the audio pass does not make the task appear to have no time left.
            let transcriptionWeight = 0.63
            observedTotal = max(
                transcriptionTotal / transcriptionWeight,
                elapsed / progressFraction
            )
        } else {
            observedTotal = elapsed / progressFraction
        }
        let conservativeTotal = max(
            observedTotal * 1.35,
            elapsed + 30
        )
        // Treat a legacy or corrupted zero anchor as missing so it can be
        // recovered instead of permanently winning the minimum comparison.
        guard let current, current > 0 else { return conservativeTotal }
        return min(current, conservativeTotal)
    }

    static func queued(
        id: UUID = UUID(),
        fileName: String,
        stagedFileName: String,
        enqueuedAt: Date = Date()
    ) -> MeetingFileTask {
        MeetingFileTask(
            id: id,
            fileName: fileName,
            stagedFileName: stagedFileName,
            enqueuedAt: enqueuedAt,
            startedAt: nil,
            completedAt: nil,
            status: .queued,
            progressStage: .preparing,
            progressFraction: 0,
            mediaDurationSeconds: nil,
            processedMediaDurationSeconds: nil,
            processingSpeedSecondsPerSecond: nil,
            speedSampleAt: nil,
            speedSampleProcessedMediaDurationSeconds: nil,
            estimatedTotalSeconds: nil,
            errorMessage: nil,
            historyEntryID: nil
        )
    }

    func resetForRetry() -> MeetingFileTask {
        var retry = self
        retry.startedAt = nil
        retry.completedAt = nil
        retry.status = .queued
        retry.progressStage = .preparing
        retry.progressFraction = 0
        retry.processedMediaDurationSeconds = nil
        retry.processingSpeedSecondsPerSecond = nil
        retry.speedSampleAt = nil
        retry.speedSampleProcessedMediaDurationSeconds = nil
        retry.estimatedTotalSeconds = nil
        retry.errorMessage = nil
        retry.historyEntryID = nil
        return retry
    }
}

enum MeetingFileAnalysisStage: Codable, Equatable, Sendable {
    case preparing
    case transcribing
    case identifyingSpeakers
    case saving
}

struct MeetingFileAnalysisProgress: Equatable, Sendable {
    let stage: MeetingFileAnalysisStage
    let fractionCompleted: Double
    let mediaDurationSeconds: TimeInterval?
    let processedMediaDurationSeconds: TimeInterval?

    init(
        stage: MeetingFileAnalysisStage,
        stageFraction: Double = 0,
        mediaDurationSeconds: TimeInterval? = nil,
        processedMediaDurationSeconds: TimeInterval? = nil
    ) {
        let clampedStageFraction = min(max(stageFraction, 0), 1)
        self.stage = stage
        self.mediaDurationSeconds = Self.validDuration(mediaDurationSeconds)
        self.processedMediaDurationSeconds = Self.validDuration(processedMediaDurationSeconds)
        switch stage {
        case .preparing:
            fractionCompleted = clampedStageFraction * 0.15
        case .transcribing:
            fractionCompleted = 0.15 + clampedStageFraction * 0.63
        case .identifyingSpeakers:
            fractionCompleted = 0.78 + clampedStageFraction * 0.18
        case .saving:
            fractionCompleted = 0.96 + clampedStageFraction * 0.04
        }
    }

    private static func validDuration(_ value: TimeInterval?) -> TimeInterval? {
        guard let value, value.isFinite, value >= 0 else { return nil }
        return value
    }
}

enum MeetingFileAnalysisError: LocalizedError {
    case sessionAlreadyActive
    case noTranscript

    var errorDescription: String? {
        switch self {
        case .sessionAlreadyActive:
            return AppLocalization.localizedString("Finish the current recording before analyzing a meeting file.")
        case .noTranscript:
            return AppLocalization.localizedString("No speech could be transcribed from the selected file.")
        }
    }
}
