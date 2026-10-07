// MeetingRealtimeDiarization.swift
// Provides Meeting Realtime Diarization for meeting session behavior.

import Foundation
import HuggingFace
import MLX
import MLXAudioVAD

enum MeetingSortformerModelStorage {
    static let repo = MeetingVADModelStorage.sortformerV2Repo

    static func modelDirectory(requireValid: Bool) -> URL? {
        for rootDirectory in ModelStorageDirectoryManager.resolvedReadableRootURLs() {
            guard let directory = MLXModelStorageSupport.cacheDirectory(
                for: repo,
                rootDirectory: rootDirectory
            ),
                  FileManager.default.fileExists(atPath: directory.path)
            else {
                continue
            }
            if requireValid && !isValidModelDirectory(directory) {
                continue
            }
            return directory
        }
        return nil
    }

    static func readableDirectories() -> [URL] {
        ModelStorageDirectoryManager.resolvedReadableRootURLs().compactMap {
            MLXModelStorageSupport.cacheDirectory(for: repo, rootDirectory: $0)
        }
    }

    static func validatedModelDirectory() async -> URL? {
        let directories = readableDirectories()
        return await Task.detached(priority: .utility) {
            directories.first { isValidModelDirectory($0) }
        }.value
    }

    static func writeModelDirectory() -> URL? {
        MLXModelStorageSupport.cacheDirectory(
            for: repo,
            rootDirectory: ModelStorageDirectoryManager.resolvedWriteRootURL()
        )
    }

    static func downloadTempDirectory() -> URL? {
        guard let repoID = HuggingFace.Repo.ID(rawValue: repo) else { return nil }
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        return ModelStorageDirectoryManager.resolvedWriteRootURL()
            .appendingPathComponent("mlx-audio", isDirectory: true)
            .appendingPathComponent("\(modelSubdir)-download", isDirectory: true)
    }

    nonisolated static func isValidModelDirectory(
        _ directory: URL,
        fileManager: FileManager = .default
    ) -> Bool {
        guard let config = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
              (try? JSONSerialization.jsonObject(with: config)) != nil,
              ModelWeightFileValidation.hasCompleteIndex(in: directory) else { return false }
        guard let entries = try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil
        ) else {
            return false
        }
        return entries.contains {
            guard $0.pathExtension == "safetensors",
                  let values = try? $0.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]) else { return false }
            return values.isRegularFile == true && (values.fileSize ?? 0) > 0
        }
    }

    static func clearHubCache(rootDirectory: URL = ModelStorageDirectoryManager.resolvedWriteRootURL()) {
        guard let repoID = HuggingFace.Repo.ID(rawValue: repo) else { return }
        MLXModelStorageSupport.clearHubCache(for: repoID, rootDirectory: rootDirectory)
    }
}

actor MeetingRealtimeDiarizationStage {
    private struct SlidingAudioWindow {
        private static let maxDurationSeconds: TimeInterval = 24
        private static let targetSampleRate = 16_000

        private var samples: [Float] = []
        private var startSeconds: TimeInterval = 0

        mutating func append(
            samples incomingSamples: [Float],
            startSeconds incomingStartSeconds: TimeInterval
        ) {
            guard !incomingSamples.isEmpty else { return }
            if samples.isEmpty {
                samples = incomingSamples
                startSeconds = incomingStartSeconds
            } else {
                let expectedStart = startSeconds + durationSeconds
                let gapSamples = Int(((incomingStartSeconds - expectedStart) * Double(Self.targetSampleRate)).rounded())
                if gapSamples > 0 {
                    samples.append(contentsOf: repeatElement(0, count: gapSamples))
                } else if gapSamples < 0 {
                    let overlap = min(-gapSamples, incomingSamples.count)
                    samples.removeLast(min(overlap, samples.count))
                }
                samples.append(contentsOf: incomingSamples)
            }
            trimToMaximumDuration()
        }

        var durationSeconds: TimeInterval {
            TimeInterval(samples.count) / TimeInterval(Self.targetSampleRate)
        }

        func snapshot() -> (samples: [Float], startSeconds: TimeInterval) {
            (samples, startSeconds)
        }

        private mutating func trimToMaximumDuration() {
            let maxSamples = max(Int(Self.maxDurationSeconds * Double(Self.targetSampleRate)), 1)
            guard samples.count > maxSamples else { return }
            let excess = samples.count - maxSamples
            samples.removeFirst(excess)
            startSeconds += TimeInterval(excess) / TimeInterval(Self.targetSampleRate)
        }
    }

    private var model: SortformerModel?
    private var audioWindows: [TranscriptAudioSource: SlidingAudioWindow] = [:]
    private var hasLoggedUnavailableModel = false
    private var hasLoggedRuntimeFailure = false

    func reset() {
        audioWindows.removeAll()
        model = nil
        hasLoggedUnavailableModel = false
        hasLoggedRuntimeFailure = false
    }

    func annotate(
        segment: MeetingTranscriptSegment,
        chunk: BufferedMeetingChunk,
        captureMode: MeetingCaptureMode
    ) async -> [MeetingTranscriptSegment] {
        let source = audioSource(for: chunk.speaker)
        guard captureMode.capabilities.shouldRunRealtimeDiarization(for: source),
              chunk.isFinal,
              chunk.endSeconds > chunk.startSeconds,
              !chunk.samples.isEmpty
        else {
            return [segment]
        }

        do {
            let turns = try await diarize(chunk: chunk, source: source)
            guard !turns.isEmpty else {
                return [segment.updatingSpeakerAnalysis(
                    speakerID: segment.speakerID,
                    speakerDisplayName: segment.speakerDisplayName,
                    audioSource: source,
                    speakerConfidence: segment.speakerConfidence
                )]
            }
            return await MainActor.run {
                let sourceTaggedSegment = segment.updatingSpeakerAnalysis(
                    speakerID: segment.speakerID,
                    speakerDisplayName: segment.speakerDisplayName,
                    audioSource: source,
                    speakerConfidence: segment.speakerConfidence
                )
                let assembled = MeetingSpeakerTranscriptAssembler.assemble(
                    segments: [sourceTaggedSegment],
                    speakerTurns: MeetingSpeakerTurnLabeler.label(turns),
                    options: MeetingSpeakerDiarizationSensitivity.balanced.transcriptAssemblyOptions
                )
                return preserveOriginalIDForFirstSegment(
                    assembled.isEmpty ? [sourceTaggedSegment] : assembled,
                    originalID: segment.id
                )
            }
        } catch MeetingVADModelError.modelNotDownloaded {
            if !hasLoggedUnavailableModel {
                VoxtLog.meeting("Meeting Sortformer v2 realtime diarization skipped: model is not downloaded.", verbose: true)
                hasLoggedUnavailableModel = true
            }
            return [segment]
        } catch {
            if !hasLoggedRuntimeFailure {
                VoxtLog.meetingWarning("Meeting Sortformer v2 realtime diarization failed; keeping source labels. error=\(error.localizedDescription)")
                hasLoggedRuntimeFailure = true
            }
            return [segment]
        }
    }

    func annotate(
        segment: MeetingTranscriptSegment,
        source: TranscriptAudioSource,
        asset: MeetingAudioAsset,
        captureMode: MeetingCaptureMode
    ) async -> [MeetingTranscriptSegment] {
        let speaker = source.defaultSpeaker
        let assetStartSeconds = asset.sessionStartOffset
        let assetDurationSeconds = asset.durationSeconds
        let chunk = BufferedMeetingChunk(
            segmentID: segment.id,
            speaker: speaker,
            startSeconds: assetStartSeconds,
            endSeconds: assetStartSeconds + assetDurationSeconds,
            sampleRate: asset.sampleRate,
            samples: asset.samples,
            isFinal: true,
            preventsAdjacentMerge: segment.preventsAdjacentMerge
        )
        let sourceTaggedSegment = segment.updatingSpeakerAnalysis(
            speaker: speaker,
            speakerID: segment.speakerID,
            speakerDisplayName: segment.speakerDisplayName,
            audioSource: source,
            speakerConfidence: segment.speakerConfidence
        )
        return await annotate(segment: sourceTaggedSegment, chunk: chunk, captureMode: captureMode)
    }

    private func diarize(
        chunk: BufferedMeetingChunk,
        source: TranscriptAudioSource
    ) async throws -> [MeetingSpeakerTurn] {
        let model = try await loadModelIfAvailable()
        let prepared = ASRVoiceActivitySampleRateConverter.resample(
            samples: chunk.samples,
            from: chunk.sampleRate,
            to: 16_000
        )
        guard !prepared.isEmpty else { return [] }

        var audioWindow = audioWindows[source] ?? SlidingAudioWindow()
        audioWindow.append(samples: prepared, startSeconds: chunk.startSeconds)
        audioWindows[source] = audioWindow
        let window = audioWindow.snapshot()

        let state = model.initStreamingState()
        let (output, nextState) = try await model.feed(
            chunk: MLXArray(window.samples),
            state: state,
            sampleRate: 16_000,
            threshold: 0.5,
            minDuration: 0.25,
            mergeGap: 0.18
        )
        _ = nextState

        let lowerBound = max(chunk.startSeconds - 0.2, window.startSeconds)
        let upperBound = chunk.endSeconds + 0.2

        return output.segments.map { item in
            let absoluteStart = window.startSeconds + TimeInterval(item.start)
            let absoluteEnd = window.startSeconds + TimeInterval(item.end)
            return MeetingSpeakerTurn(
                source: source,
                speakerID: "sortformer-\(item.speaker)",
                displayName: MeetingSpeakerDisplayNameFormatter.displayName(ordinal: item.speaker + 1),
                startSeconds: max(absoluteStart, lowerBound),
                endSeconds: min(absoluteEnd, upperBound),
                confidence: nil
            )
        }
        .filter { $0.endSeconds > $0.startSeconds }
        .filter { $0.endSeconds > lowerBound && $0.startSeconds < upperBound }
    }

    private func loadModelIfAvailable() async throws -> SortformerModel {
        if let model {
            return model
        }
        let directory = await MeetingSortformerModelStorage.validatedModelDirectory()
        guard let directory else {
            throw MeetingVADModelError.modelNotDownloaded
        }
        let loaded = try SortformerModel.fromModelDirectory(directory)
        model = loaded
        return loaded
    }

    private func audioSource(for speaker: MeetingSpeaker) -> TranscriptAudioSource {
        switch speaker {
        case .me:
            return .microphone
        case .them:
            return .systemAudio
        }
    }

    private nonisolated func preserveOriginalIDForFirstSegment(
        _ segments: [MeetingTranscriptSegment],
        originalID: UUID
    ) -> [MeetingTranscriptSegment] {
        guard let first = segments.first else { return segments }
        guard first.id != originalID else { return segments }

        var output = segments
        output[0] = MeetingTranscriptSegment(
            id: originalID,
            speaker: first.speaker,
            speakerID: first.speakerID,
            speakerDisplayName: first.speakerDisplayName,
            audioSource: first.audioSource,
            speakerConfidence: first.speakerConfidence,
            startSeconds: first.startSeconds,
            endSeconds: first.endSeconds,
            text: first.text,
            translatedText: first.translatedText,
            isTranslationPending: first.isTranslationPending,
            preventsAdjacentMerge: first.preventsAdjacentMerge
        )
        return output
    }
}
