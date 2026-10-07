// MeetingMLXNativeLiveSession.swift
// Reuses visible MLX models' native streaming state for local meeting transcription.

import Foundation
import MLXAudioSTT

@MainActor
struct MeetingMLXNativeLiveSessionFactory: MeetingLiveSessionFactory {
    let modelManager: MLXModelManager

    func makeSession(
        for speaker: MeetingSpeaker,
        timelineOffsetSeconds: TimeInterval
    ) throws -> any MeetingLiveTranscribingSession {
        MeetingMLXNativeLiveSession(
            speaker: speaker,
            timelineOffsetSeconds: timelineOffsetSeconds,
            modelManager: modelManager
        )
    }
}

@MainActor
private final class MeetingMLXNativeLiveSession: MeetingLiveTranscribingSession {
    private static let silenceFinalizeSeconds: TimeInterval = 0.75
    private static let maximumLiveSegmentSeconds: TimeInterval = 8

    let speaker: MeetingSpeaker
    private let modelManager: MLXModelManager
    private let streamingTranscriber: MLXTranscriber

    private(set) var state: MeetingLiveSessionState = .connecting
    private var eventHandler: (@MainActor (MeetingTranscriptEvent) -> Void)?
    private var configuration: MLXMeetingNativeStreamingConfiguration?
    private var feedScheduler: MeetingMLXNativeFeedScheduler?
    private var eventTask: Task<Void, Never>?
    private var timelineOffsetSeconds: TimeInterval
    private var totalAudioSeconds: TimeInterval = 0
    private var currentSegmentID = UUID()
    private var currentSegmentStartSeconds: TimeInterval
    private var committedCumulativeText = ""
    private var latestCumulativeText = ""
    private var latestVisibleSegmentText = ""
    private var silenceDurationSeconds: TimeInterval = 0
    private var isCancelled = false
    private var defersSilenceFinalization = false
    private var didEmitStructuredEndedSegments = false

    init(
        speaker: MeetingSpeaker,
        timelineOffsetSeconds: TimeInterval,
        modelManager: MLXModelManager
    ) {
        self.speaker = speaker
        self.timelineOffsetSeconds = timelineOffsetSeconds
        self.currentSegmentStartSeconds = timelineOffsetSeconds
        self.modelManager = modelManager
        self.streamingTranscriber = MLXTranscriber(
            modelManager: modelManager,
            transcriptionPurpose: .meeting
        )
    }

    func start(
        timelineOffsetSeconds: TimeInterval,
        eventHandler: @escaping @MainActor (MeetingTranscriptEvent) -> Void
    ) async throws {
        self.timelineOffsetSeconds = timelineOffsetSeconds
        self.currentSegmentStartSeconds = timelineOffsetSeconds
        self.eventHandler = eventHandler
        state = .connecting

        let configuration = try await streamingTranscriber.makeMeetingNativeStreamingConfiguration()
        self.configuration = configuration
        defersSilenceFinalization = MeetingNativeLiveSegmentationPolicy.shouldDeferSilenceFinalization(
            timingGranularity: MLXModelCatalog.capability(for: modelManager.currentModelRepo).timingGranularity
        )
        didEmitStructuredEndedSegments = false
        let feedScheduler = MeetingMLXNativeFeedScheduler(session: configuration.session)
        self.feedScheduler = feedScheduler
        state = .active

        eventTask = Task { @MainActor [weak self, session = configuration.session] in
            for await event in session.events {
                guard !Task.isCancelled, let self else { return }
                self.handle(event)
                switch event {
                case .ended, .failed: return
                default: break
                }
            }
            guard !Task.isCancelled, let self, !self.isCancelled else { return }
            await self.failAndCancel(message: "The live stream closed without a final result. Recorded audio is preserved.")
        }
        VoxtLog.meeting(
            "Meeting native MLX streaming session started. repo=\(modelManager.currentModelRepo), speaker=\(speaker.rawValue), offset=\(String(format: "%.2f", timelineOffsetSeconds))",
            verbose: true
        )
    }

    func append(samples: [Float], sampleRate: Double) async {
        guard state == .active, !isCancelled, !samples.isEmpty else { return }
        let duration = Double(samples.count) / max(sampleRate, 1)
        totalAudioSeconds += duration

        let level = AudioLevelMeter.normalizedLevel(fromSamples: samples)
        let threshold: Float = speaker == .me ? 0.012 : 0.025
        if level >= threshold {
            silenceDurationSeconds = 0
        } else {
            silenceDurationSeconds += duration
        }

        if let feedScheduler {
            let accepted = await feedScheduler.submit(samples: samples, sampleRate: sampleRate)
            if !accepted {
                let message = await feedScheduler.failureMessage ?? "Live audio delivery stopped. Recorded audio is preserved."
                await failAndCancel(message: message)
                return
            }
        }

        let currentEnd = timelineOffsetSeconds + totalAudioSeconds
        // Keep one partial when the model can emit reliable timestamps on `.ended`
        // (for example Nemotron sentence segments). Silence-splitting would otherwise
        // publish text-only finals that duplicate or fight those timestamps.
        if !defersSilenceFinalization,
           MeetingNativeLiveSegmentationPolicy.shouldFinalizeBeforeStreamEnd(
            // MOSS provisional windows can rewrite already visible words when the
            // final window gains more context. Keep one partial segment until ended.
            streamCanReviseEarlierText: configuration?.mossVisibleOutputMode != nil,
            silenceDuration: silenceDurationSeconds,
            segmentDuration: currentEnd - currentSegmentStartSeconds,
            silenceThreshold: Self.silenceFinalizeSeconds,
            maximumSegmentDuration: Self.maximumLiveSegmentSeconds
           )
        {
            finalizeVisibleSegment(at: currentEnd)
        }
    }

    func finish() async {
        guard state != .stopping, state != .failed, !isCancelled else { return }
        state = .stopping
        // A broken provider must not hold meeting finalization forever. Cancelling
        // the consumer also wakes an AsyncStream awaiting a missing terminal event.
        let watchdog = Task { @MainActor [weak self] in
            do { try await Task.sleep(for: .seconds(120)) } catch { return }
            await self?.failAndCancel(message: "Live transcription timed out. Recorded audio is preserved for final processing.")
        }
        defer { watchdog.cancel() }
        do {
            try await feedScheduler?.finish()
            if let eventTask {
                let scheduler = feedScheduler
                await withTaskCancellationHandler {
                    await eventTask.value
                } onCancel: {
                    eventTask.cancel()
                    Task { await scheduler?.cancel() }
                }
            }
            try Task.checkCancellation()
            guard !isCancelled, state != .failed else { return }
            if !didEmitStructuredEndedSegments {
                finalizeVisibleSegment(at: timelineOffsetSeconds + totalAudioSeconds)
            }
            eventHandler?(.finished(speaker: speaker))
            release()
        } catch {
            await failAndCancel(message: error.localizedDescription)
        }
    }

    private func failAndCancel(message: String) async {
        guard !isCancelled else { return }
        isCancelled = true
        state = .failed
        await feedScheduler?.cancel()
        eventTask?.cancel()
        eventHandler?(.failed(speaker: speaker, message: message))
        release()
    }

    func cancel() async {
        guard !isCancelled else { return }
        isCancelled = true
        state = .stopping
        await feedScheduler?.cancel()
        eventTask?.cancel()
        eventHandler?(.finished(speaker: speaker))
        release()
    }

    private func handle(_ event: TranscriptionEvent) {
        switch event {
        case .displayUpdate(let confirmedText, let provisionalText):
            let cumulative = visibleCumulativeText(
                confirmedText: confirmedText,
                provisionalText: provisionalText
            )
            publishPartial(cumulative: cumulative)
        case .confirmed, .provisional:
            // displayUpdate carries the coherent cumulative confirmed + provisional view.
            break
        case .ended(let output):
            let cumulative = visibleFinalText(output.text)
            if finalizeWithEndedOutputIfPossible(output, cumulativeText: cumulative) {
                return
            }
            publishPartial(cumulative: cumulative)
            finalizeVisibleSegment(at: timelineOffsetSeconds + totalAudioSeconds)
        case .failed(let failure):
            state = .failed
            Task { @MainActor [self] in
                await failAndCancel(message: failure.localizedDescription)
            }
        case .stats:
            break
        }
    }

    private func visibleCumulativeText(confirmedText: String, provisionalText: String) -> String {
        guard let configuration else { return (confirmedText + provisionalText).trimmingCharacters(in: .whitespacesAndNewlines) }
        if configuration.qwenUsesAutomaticLanguageProtocol {
            let parts = MLXTranscriptionPlanning.qwenStreamingVisibleTextParts(
                confirmedText: confirmedText,
                provisionalText: provisionalText
            )
            return (parts.confirmedText + parts.provisionalText)
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let combined = confirmedText + provisionalText
        if let outputMode = configuration.mossVisibleOutputMode {
            return MossASRTranscriptRendering.renderedText(combined, outputMode: outputMode)
        }
        return combined.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func visibleFinalText(_ text: String) -> String {
        guard let configuration else { return text.trimmingCharacters(in: .whitespacesAndNewlines) }
        if configuration.qwenUsesAutomaticLanguageProtocol {
            return MLXTranscriptionPlanning.qwenStreamingVisibleText(
                text,
                suppressIncompleteWindowHeader: false
            )
        }
        if let outputMode = configuration.mossVisibleOutputMode {
            return MossASRTranscriptRendering.renderedText(text, outputMode: outputMode)
        }
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func publishPartial(cumulative: String) {
        let normalized = cumulative.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return }
        latestCumulativeText = normalized
        let visible = Self.uncommittedSuffix(
            cumulative: normalized,
            committed: committedCumulativeText
        )
        guard !visible.isEmpty, visible != latestVisibleSegmentText else { return }
        latestVisibleSegmentText = visible
        eventHandler?(
            .partial(
                segment(text: visible, endSeconds: timelineOffsetSeconds + totalAudioSeconds)
            )
        )
    }

    private func finalizeVisibleSegment(at endSeconds: TimeInterval) {
        let text = latestVisibleSegmentText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else {
            currentSegmentStartSeconds = endSeconds
            return
        }
        eventHandler?(.final(segment(text: text, endSeconds: endSeconds)))
        committedCumulativeText = latestCumulativeText
        latestVisibleSegmentText = ""
        currentSegmentID = UUID()
        currentSegmentStartSeconds = endSeconds
        silenceDurationSeconds = 0
    }

    @discardableResult
    private func finalizeWithEndedOutputIfPossible(_ output: STTOutput, cumulativeText: String) -> Bool {
        guard defersSilenceFinalization,
              let segments = output.segments,
              !segments.isEmpty
        else {
            return false
        }

        let capability = MLXModelCatalog.capability(for: modelManager.currentModelRepo)
        let structured = MeetingNativeLiveStructuredFinalization.meetingSegments(
            from: segments,
            timingGranularity: capability.timingGranularity,
            modelFamily: capability.family,
            timelineOffsetSeconds: timelineOffsetSeconds,
            speaker: speaker,
            audioSource: speaker == .me ? .microphone : .systemAudio,
            // Reuse the in-flight partial ID so overlay upsert replaces it instead of
            // appending a duplicate bubble with a fresh UUID.
            replacingSegmentID: currentSegmentID
        )
        guard !structured.isEmpty else { return false }

        // Drop the in-flight text-only partial; model timestamps replace it.
        latestVisibleSegmentText = ""
        for item in structured {
            eventHandler?(.final(item))
        }
        latestCumulativeText = cumulativeText
        committedCumulativeText = cumulativeText
        currentSegmentID = UUID()
        currentSegmentStartSeconds = timelineOffsetSeconds + totalAudioSeconds
        silenceDurationSeconds = 0
        didEmitStructuredEndedSegments = true
        VoxtLog.meeting(
            "Meeting native MLX ended with structured segments. speaker=\(speaker.rawValue), segmentCount=\(structured.count), timing=\(String(describing: capability.timingGranularity)), language=\(output.language ?? "nil")",
            verbose: true
        )
        return true
    }

    private func segment(text: String, endSeconds: TimeInterval) -> MeetingTranscriptSegment {
        MeetingTranscriptSegment(
            id: currentSegmentID,
            speaker: speaker,
            audioSource: speaker == .me ? .microphone : .systemAudio,
            startSeconds: currentSegmentStartSeconds,
            endSeconds: max(endSeconds, currentSegmentStartSeconds),
            text: text,
            preventsAdjacentMerge: true
        )
    }

    private func release() {
        eventTask?.cancel()
        eventTask = nil
        feedScheduler = nil
        configuration = nil
        eventHandler = nil
    }

    private nonisolated static func uncommittedSuffix(cumulative: String, committed: String) -> String {
        guard !committed.isEmpty else { return cumulative }
        if cumulative.hasPrefix(committed) {
            return String(cumulative.dropFirst(committed.count))
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let committedCharacters = Array(committed)
        let cumulativeCharacters = Array(cumulative)
        let maximumOverlap = min(committedCharacters.count, cumulativeCharacters.count)
        for overlap in stride(from: maximumOverlap, through: 1, by: -1) {
            if committedCharacters.suffix(overlap).elementsEqual(cumulativeCharacters.prefix(overlap)) {
                return String(cumulativeCharacters.dropFirst(overlap))
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }
        return cumulative
    }
}
