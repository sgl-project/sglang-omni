// MeetingSpeakerAnalysis.swift
// Provides Meeting Speaker Analysis for meeting transcript processing.

import Foundation

enum MeetingSpeakerAnalysisPipeline {
    /// File analysis must not silently save a successful history entry if speaker
    /// processing fails. Its completed ASR checkpoint remains available for retry.
    static func analyzedFileSegments(
        from segments: [MeetingTranscriptSegment],
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        options: MeetingSpeakerDiarizationOptions = .init(),
        engine: any MeetingSpeakerDiarizationEngine = SortformerMeetingSpeakerDiarizationEngine(),
        progress: (@Sendable (Double) async -> Void)? = nil
    ) async throws -> [MeetingTranscriptSegment] {
        let eligible = descriptors.filter { $0.durationSeconds >= options.minimumAudioDurationSeconds }
        let turns = try await engine.diarizeFile(
            descriptors: eligible, loadAsset: loadAsset, options: options, progress: progress
        )
        try Task.checkCancellation()
        return assembledSegments(from: segments, turns: turns, options: options)
    }

    static func analyzedSegmentsPreservingStructuredSpeakerData(
        from segments: [MeetingTranscriptSegment],
        assets: [MeetingAudioAsset],
        options: MeetingSpeakerDiarizationOptions = MeetingSpeakerDiarizationOptions(),
        engine: (any MeetingSpeakerDiarizationEngine)? = MeetingSpeakerDiarizationEngineFactory.makeDefault()
    ) async -> [MeetingTranscriptSegment] {
        await analyzedSegmentsPreservingStructuredSpeakerData(from: segments) { candidates in
            await analyzedSegments(from: candidates, assets: assets, options: options, engine: engine)
        }
    }

    static func analyzedSegmentsPreservingStructuredSpeakerData(
        from segments: [MeetingTranscriptSegment],
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        continuousAudioURL: URL? = nil,
        options: MeetingSpeakerDiarizationOptions = MeetingSpeakerDiarizationOptions(),
        engine: (any MeetingSpeakerDiarizationEngine)? = MeetingSpeakerDiarizationEngineFactory.makeDefault(),
        progress: (@Sendable (Double) async -> Void)? = nil
    ) async -> [MeetingTranscriptSegment] {
        await analyzedSegmentsPreservingStructuredSpeakerData(from: segments) { candidates in
            await analyzedSegments(
                from: candidates,
                descriptors: descriptors,
                loadAsset: loadAsset,
                continuousAudioURL: continuousAudioURL,
                options: options,
                engine: engine,
                progress: progress
            )
        }
    }

    static func analyzedSegments(
        from segments: [MeetingTranscriptSegment],
        assets: [MeetingAudioAsset],
        options: MeetingSpeakerDiarizationOptions = MeetingSpeakerDiarizationOptions(),
        engine: (any MeetingSpeakerDiarizationEngine)? = MeetingSpeakerDiarizationEngineFactory.makeDefault()
    ) async -> [MeetingTranscriptSegment] {
        guard !segments.isEmpty, !assets.isEmpty, let engine else {
            return segments
        }

        do {
            var turns: [MeetingSpeakerTurn] = []
            logDebug(
                "Meeting speaker analysis started. segments=\(segments.count), assets=\(assets.count), sensitivity=\(options.sensitivity.rawValue)",
                options: options
            )
            for asset in assets where asset.durationSeconds >= options.minimumAudioDurationSeconds {
                let assetTurns = try await engine.diarize(asset: asset, options: options)
                logDebug(
                    "Meeting speaker analysis asset raw turns. source=\(asset.source.rawValue), duration=\(String(format: "%.2f", asset.durationSeconds)), rawTurns=\(assetTurns.count), rawSpeakers=\(speakerCount(assetTurns))",
                    options: options
                )
                turns.append(contentsOf: assetTurns)
            }
            return assembledSegments(from: segments, turns: turns, options: options)
        } catch {
            VoxtLog.meetingWarning("Meeting speaker analysis failed.")
            return segments
        }
    }

    static func analyzedSegments(
        from segments: [MeetingTranscriptSegment],
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        continuousAudioURL: URL? = nil,
        options: MeetingSpeakerDiarizationOptions = MeetingSpeakerDiarizationOptions(),
        engine: (any MeetingSpeakerDiarizationEngine)? = MeetingSpeakerDiarizationEngineFactory.makeDefault(),
        progress: (@Sendable (Double) async -> Void)? = nil
    ) async -> [MeetingTranscriptSegment] {
        guard !segments.isEmpty, !descriptors.isEmpty, let engine else {
            return segments
        }

        do {
            let eligibleDescriptors = descriptors.filter {
                $0.durationSeconds >= options.minimumAudioDurationSeconds
            }
            logDebug(
                "Meeting speaker analysis started. segments=\(segments.count), assets=\(descriptors.count), sensitivity=\(options.sensitivity.rawValue)",
                options: options
            )
            if eligibleDescriptors.isEmpty {
                await progress?(1)
                return assembledSegments(from: segments, turns: [], options: options)
            }
            let turns = try await engine.diarizeSession(
                descriptors: eligibleDescriptors,
                loadAsset: loadAsset,
                continuousAudioURL: continuousAudioURL,
                options: options,
                progress: progress
            )
            guard !Task.isCancelled else { return segments }
            logDebug(
                "Meeting speaker analysis session raw turns. assets=\(eligibleDescriptors.count), rawTurns=\(turns.count), rawSpeakers=\(speakerCount(turns))",
                options: options
            )
            return assembledSegments(from: segments, turns: turns, options: options)
        } catch {
            VoxtLog.meetingWarning("Meeting speaker analysis failed.")
            return segments
        }
    }

    private static func assembledSegments(
        from segments: [MeetingTranscriptSegment],
        turns: [MeetingSpeakerTurn],
        options: MeetingSpeakerDiarizationOptions
    ) -> [MeetingTranscriptSegment] {
        let rawTurns = turns
        var turns = confidenceFilteredTurns(from: turns, options: options)
        let confidenceFilteredTurns = turns
        turns = MeetingSpeakerTurnSmoother.smooth(turns, options: options.smoothing)
        let smoothedTurns = turns
        turns = MeetingSpeakerTurnLabeler.label(turns)
        logDebug(
            "Meeting speaker analysis pipeline summary. rawTurns=\(rawTurns.count), rawSpeakers=\(speakerCount(rawTurns)), confidenceFilteredTurns=\(confidenceFilteredTurns.count), confidenceFilteredSpeakers=\(speakerCount(confidenceFilteredTurns)), smoothedTurns=\(smoothedTurns.count), smoothedSpeakers=\(speakerCount(smoothedTurns)), confidenceThreshold=\(String(format: "%.2f", options.minimumSpeakerConfidence))",
            options: options
        )
        guard !turns.isEmpty else { return segments }
        let assembled = MeetingSpeakerTranscriptAssembler.assemble(
            segments: segments,
            speakerTurns: turns,
            options: options.transcriptAssembly
        )
        let readableSegments = MeetingTranscriptPostProcessor.process(assembled)
        logDebug(
            "Meeting speaker analysis assembly summary. inputSegments=\(segments.count), assembledSegments=\(assembled.count), readableSegments=\(readableSegments.count), outputSpeakers=\(segmentSpeakerCount(readableSegments))",
            options: options
        )
        return readableSegments
    }

    private static func analyzedSegmentsPreservingStructuredSpeakerData(
        from segments: [MeetingTranscriptSegment],
        analyze: ([MeetingTranscriptSegment]) async -> [MeetingTranscriptSegment]
    ) async -> [MeetingTranscriptSegment] {
        let protectedSegments = segments.filter(hasStructuredSpeakerData)
        let candidates = segments.filter { !hasStructuredSpeakerData($0) }
        guard !candidates.isEmpty else {
            return MeetingTranscriptPostProcessor.process(protectedSegments)
        }

        let analyzedSegments = await analyze(candidates)
        return MeetingTranscriptPostProcessor.process(protectedSegments + analyzedSegments)
    }

    private static func hasStructuredSpeakerData(_ segment: MeetingTranscriptSegment) -> Bool {
        segment.speakerID?.hasPrefix("moss:") == true
    }

    private static func logDebug(_ message: @autoclosure () -> String, options: MeetingSpeakerDiarizationOptions) {
        guard options.debugLoggingEnabled else { return }
        VoxtLog.meeting(message())
    }

    private static func confidenceFilteredTurns(
        from turns: [MeetingSpeakerTurn],
        options: MeetingSpeakerDiarizationOptions
    ) -> [MeetingSpeakerTurn] {
        guard options.minimumSpeakerConfidence > 0 else { return turns }

        let filteredTurns = turns.filter { turn in
            guard let confidence = turn.confidence else { return true }
            return confidence >= options.minimumSpeakerConfidence
        }

        guard !filteredTurns.isEmpty else { return turns }

        let rawSpeakerCount = speakerCount(turns)
        guard rawSpeakerCount > 1 else { return filteredTurns }

        let filteredSpeakerCount = speakerCount(filteredTurns)
        guard filteredSpeakerCount < min(rawSpeakerCount, 2) else { return filteredTurns }

        logDebug(
            "Meeting speaker confidence filter would collapse speakers; preserving raw turns. rawSpeakers=\(rawSpeakerCount), filteredSpeakers=\(filteredSpeakerCount), threshold=\(String(format: "%.2f", options.minimumSpeakerConfidence))",
            options: options
        )
        return turns
    }

    private static func speakerCount(_ turns: [MeetingSpeakerTurn]) -> Int {
        Set(turns.map { "\($0.source.rawValue):\($0.speakerID)" }).count
    }

    private static func segmentSpeakerCount(_ segments: [MeetingTranscriptSegment]) -> Int {
        Set(segments.compactMap { segment in
            let trimmedID = segment.speakerID?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            return trimmedID.isEmpty ? nil : trimmedID
        }).count
    }
}
