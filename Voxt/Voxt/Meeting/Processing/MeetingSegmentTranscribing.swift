// MeetingSegmentTranscribing.swift
// Provides Meeting Segment Transcribing for meeting transcript processing.

import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXAudioSTT

protocol MeetingSegmentTranscribing {
    func transcribe(chunk: BufferedMeetingChunk) async -> MeetingTranscriptSegment?
    func transcribeSegments(chunk: BufferedMeetingChunk) async -> [MeetingTranscriptSegment]
    func transcribeSegmentsStrict(chunk: BufferedMeetingChunk) async throws -> [MeetingTranscriptSegment]
    func transcribeWholeAsset(_ asset: MeetingAudioAsset) async throws -> [MeetingTranscriptSegment]?
    func cancelPendingWork() async
}

extension MeetingSegmentTranscribing {
    func transcribeSegments(chunk: BufferedMeetingChunk) async -> [MeetingTranscriptSegment] {
        guard let segment = await transcribe(chunk: chunk) else { return [] }
        return [segment]
    }

    func transcribeSegmentsStrict(chunk: BufferedMeetingChunk) async throws -> [MeetingTranscriptSegment] {
        await transcribeSegments(chunk: chunk)
    }

    func transcribeWholeAsset(_ asset: MeetingAudioAsset) async throws -> [MeetingTranscriptSegment]? {
        nil
    }

    func cancelPendingWork() async {}
}

enum MeetingTranscriptSanitizer {
    nonisolated static func sanitizedText(
        _ rawText: String,
        prompt: String? = nil,
        contextualPhrases: [String] = [],
        dictionaryEntries: [DictionaryEntry] = []
    ) -> String {
        let withoutContextLeakage = MLXTranscriptionPlanning.removingKnownASRContextLeakage(from: rawText)
        let normalized = MeetingTranscriptTextPostProcessor.normalizedFinalText(withoutContextLeakage)
        let withoutPromptEcho = RecordingSessionSupport.textAfterSuppressingPromptEcho(normalized, prompt: prompt)
        guard !withoutPromptEcho.isEmpty else { return "" }

        if isLikelyHintOnlyEcho(
            withoutPromptEcho,
            contextualPhrases: contextualPhrases,
            dictionaryEntries: dictionaryEntries
        ) {
            return ""
        }

        return withoutPromptEcho
    }

    private nonisolated static func isLikelyHintOnlyEcho(
        _ text: String,
        contextualPhrases: [String],
        dictionaryEntries: [DictionaryEntry]
    ) -> Bool {
        let textKey = normalizedHintEchoKey(text)
        guard textKey.count >= 6 else { return false }

        let phraseKeys = hintPhraseKeys(
            contextualPhrases: contextualPhrases,
            dictionaryEntries: dictionaryEntries
        )
        guard !phraseKeys.isEmpty else { return false }

        if phraseKeys.contains(textKey), textKey.count >= 12 {
            return true
        }

        var remaining = textKey
        var matchedCount = 0
        var matchedCharacters = 0
        for phraseKey in phraseKeys.sorted(by: { $0.count > $1.count }) {
            guard phraseKey.count >= 3, remaining.contains(phraseKey) else { continue }
            let occurrences = remaining.components(separatedBy: phraseKey).count - 1
            guard occurrences > 0 else { continue }
            matchedCount += occurrences
            matchedCharacters += occurrences * phraseKey.count
            remaining = remaining.replacingOccurrences(of: phraseKey, with: "")
        }

        guard matchedCount >= 2 else { return false }
        let coverage = Double(matchedCharacters) / Double(max(textKey.count, 1))
        return coverage >= 0.72 && remaining.count <= max(6, textKey.count / 4)
    }

    private nonisolated static func hintPhraseKeys(
        contextualPhrases: [String],
        dictionaryEntries: [DictionaryEntry]
    ) -> Set<String> {
        var keys = Set<String>()

        func insert(_ value: String) {
            let key = normalizedHintEchoKey(value)
            guard key.count >= 2 else { return }
            keys.insert(key)
        }

        contextualPhrases.forEach(insert)
        for entry in dictionaryEntries where entry.status == .active {
            insert(entry.term)
            entry.replacementTerms.map(\.text).forEach(insert)
        }

        return keys
    }

    private nonisolated static func normalizedHintEchoKey(_ text: String) -> String {
        let normalized = text
            .folding(options: [.caseInsensitive, .diacriticInsensitive, .widthInsensitive], locale: nil)
            .lowercased()
        let dropped = CharacterSet.whitespacesAndNewlines
            .union(.punctuationCharacters)
            .union(.symbols)
        return String(normalized.unicodeScalars.filter { !dropped.contains($0) })
    }
}

actor MeetingRemoteTranscriptionGate {
    private var isBusy = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        if !isBusy {
            isBusy = true
            return
        }

        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func release() {
        if waiters.isEmpty {
            isBusy = false
            return
        }

        let next = waiters.removeFirst()
        next.resume()
    }

    func cancelAll() {
        isBusy = false
        let pendingWaiters = waiters
        waiters.removeAll()
        pendingWaiters.forEach { $0.resume() }
    }
}

enum MeetingMLXSegmentMapping {
    nonisolated static func shouldUseStructuredOutput(
        isFinalChunk: Bool,
        timingGranularity: MLXASRTimingGranularity
    ) -> Bool {
        isFinalChunk && timingGranularity.providesReliableSegments
    }

    nonisolated static func meetingSegments(
        from result: MLXBufferedTranscriptionResult,
        segmentID: UUID,
        speaker: MeetingSpeaker,
        audioSource: TranscriptAudioSource,
        startSeconds: TimeInterval,
        endSeconds: TimeInterval,
        usesStructuredOutput: Bool,
        modelFamily: MLXModelFamily,
        preventsAdjacentMerge: Bool,
        dictionaryEntries: [DictionaryEntry],
        speakerDisplayName: (String) -> String?
    ) -> [MeetingTranscriptSegment] {
        if usesStructuredOutput, !result.structuredSegments.isEmpty {
            return result.structuredSegments.enumerated().compactMap { index, segment in
                let text = MeetingTranscriptSanitizer.sanitizedText(
                    segment.text,
                    dictionaryEntries: dictionaryEntries
                )
                guard !text.isEmpty else { return nil }
                let boundedStart = min(max(startSeconds + segment.startSeconds, startSeconds), endSeconds)
                let boundedEnd = min(max(startSeconds + segment.endSeconds, boundedStart), endSeconds)
                // Only MOSS structured speaker IDs are diarization labels. Never promote
                // Nemotron/Parakeet timing metadata into speaker identity.
                let allowMossSpeaker = modelFamily == .mossTranscribeDiarize
                let speakerID = allowMossSpeaker ? segment.speakerID.map { "moss:\($0)" } : nil
                let displayName = allowMossSpeaker
                    ? segment.speakerID.flatMap(speakerDisplayName)
                    : nil
                return MeetingTranscriptSegment(
                    id: index == 0 ? segmentID : UUID(),
                    speaker: speaker,
                    speakerID: speakerID,
                    speakerDisplayName: displayName,
                    audioSource: audioSource,
                    startSeconds: boundedStart,
                    endSeconds: boundedEnd,
                    text: text,
                    preventsAdjacentMerge: true
                )
            }
        }

        if usesStructuredOutput, modelFamily == .mossTranscribeDiarize {
            VoxtLog.meetingWarning(
                "Meeting MOSS structured output was unavailable; raw protocol text was suppressed."
            )
            return []
        }

        let previewText = result.structuredSegments.isEmpty
            ? result.text
            : result.structuredSegments.map(\.text).joined(separator: " ")
        let sanitizedText = MeetingTranscriptSanitizer.sanitizedText(
            previewText,
            dictionaryEntries: dictionaryEntries
        )
        guard !sanitizedText.isEmpty else {
            VoxtLog.meetingWarning("Meeting MLX transcription suppressed because it matched ASR prompt or hint guidance.")
            return []
        }
        return [
            MeetingTranscriptSegment(
                id: segmentID,
                speaker: speaker,
                audioSource: audioSource,
                startSeconds: startSeconds,
                endSeconds: endSeconds,
                text: sanitizedText,
                preventsAdjacentMerge: preventsAdjacentMerge
            )
        ]
    }
}

@MainActor
final class MeetingMLXSegmentTranscriber: MeetingSegmentTranscribing {
    private let modelManager: MLXModelManager
    private let mlxTranscriber: MLXTranscriber
    private let strictInferenceWorkClass: MeetingLocalInferenceWorkClass

    init(
        modelManager: MLXModelManager,
        strictInferenceWorkClass: MeetingLocalInferenceWorkClass = .finalASR
    ) {
        self.modelManager = modelManager
        self.strictInferenceWorkClass = strictInferenceWorkClass
        self.mlxTranscriber = MLXTranscriber(
            modelManager: modelManager,
            transcriptionPurpose: .meeting,
            inferenceTaskPriority: strictInferenceWorkClass == .fileASR ? .utility : .userInitiated
        )
        self.mlxTranscriber.dictionaryEntryProvider = {
            guard let appDelegate = AppDelegate.shared else { return [] }
            return appDelegate.dictionaryStore.activeEntriesForRemoteRequest(
                activeGroupID: appDelegate.activeDictionaryGroupID()
            )
        }
    }

    func transcribe(chunk: BufferedMeetingChunk) async -> MeetingTranscriptSegment? {
        await transcribeSegments(chunk: chunk).first
    }

    func transcribeSegments(chunk: BufferedMeetingChunk) async -> [MeetingTranscriptSegment] {
        let workClass: MeetingLocalInferenceWorkClass = chunk.isFinal ? .liveASRFinal : .liveASRPartial
        let result: MLXBufferedTranscriptionResult?
        do {
            result = try await MeetingLocalInferenceCoordinator.shared.withPermit(workClass) { [mlxTranscriber] in
                await mlxTranscriber.transcribeMeetingChunkResult(
                    samples: chunk.samples,
                    sampleRate: chunk.sampleRate
                )
            }
        } catch {
            if !(error is CancellationError) {
                VoxtLog.meetingWarning("Meeting MLX inference deferred or rejected: \(error.localizedDescription)")
            }
            return []
        }
        guard let result else {
            return []
        }

        let capability = MLXModelCatalog.capability(for: modelManager.currentModelRepo)
        return meetingSegments(
            from: result,
            segmentID: chunk.segmentID,
            speaker: chunk.speaker,
            audioSource: chunk.speaker == .me ? .microphone : .systemAudio,
            startSeconds: chunk.startSeconds,
            endSeconds: chunk.endSeconds,
            usesStructuredOutput: MeetingMLXSegmentMapping.shouldUseStructuredOutput(
                isFinalChunk: chunk.isFinal,
                timingGranularity: capability.timingGranularity
            ),
            modelFamily: capability.family,
            preventsAdjacentMerge: chunk.preventsAdjacentMerge
        )
    }

    func transcribeSegmentsStrict(chunk: BufferedMeetingChunk) async throws -> [MeetingTranscriptSegment] {
        let startedAt = ContinuousClock.now
        let result: MLXBufferedTranscriptionResult?
        do {
            result = try await MeetingLocalInferenceCoordinator.shared.withPermit(strictInferenceWorkClass) { [mlxTranscriber] in
                try await mlxTranscriber.transcribeBufferedResult(
                    samples: chunk.samples,
                    sampleRate: chunk.sampleRate
                )
            }
        } catch {
            VoxtLog.meetingWarning(
                "Meeting strict ASR chunk failed. workClass=\(strictInferenceWorkClass.rawValue), repo=\(modelManager.currentModelRepo), audioStartSeconds=\(chunk.startSeconds), audioEndSeconds=\(chunk.endSeconds), elapsed=\(startedAt.duration(to: .now)), \(MeetingFileTaskDiagnostics.errorSummary(error))"
            )
            throw error
        }
        guard let result else { return [] }
        let capability = MLXModelCatalog.capability(for: modelManager.currentModelRepo)
        return meetingSegments(
            from: result,
            segmentID: chunk.segmentID,
            speaker: chunk.speaker,
            audioSource: chunk.speaker == .me ? .microphone : .systemAudio,
            startSeconds: chunk.startSeconds,
            endSeconds: chunk.endSeconds,
            usesStructuredOutput: MeetingMLXSegmentMapping.shouldUseStructuredOutput(
                isFinalChunk: chunk.isFinal,
                timingGranularity: capability.timingGranularity
            ),
            modelFamily: capability.family,
            preventsAdjacentMerge: chunk.preventsAdjacentMerge
        )
    }

    func transcribeWholeAsset(_ asset: MeetingAudioAsset) async throws -> [MeetingTranscriptSegment]? {
        guard MLXModelFamily.family(for: modelManager.currentModelRepo) == .mossTranscribeDiarize else {
            return nil
        }
        let result = try await MeetingLocalInferenceCoordinator.shared.withPermit(.finalASR) { [mlxTranscriber] in
            try await mlxTranscriber.transcribeBufferedResult(
                samples: asset.samples,
                sampleRate: asset.sampleRate
            )
        }
        guard let result else {
            return []
        }
        let capability = MLXModelCatalog.capability(for: modelManager.currentModelRepo)
        return meetingSegments(
            from: result,
            segmentID: UUID(),
            speaker: asset.source.defaultSpeaker,
            audioSource: asset.source,
            startSeconds: asset.sessionStartOffset,
            endSeconds: asset.sessionStartOffset + asset.durationSeconds,
            usesStructuredOutput: capability.timingGranularity.providesReliableSegments,
            modelFamily: capability.family,
            preventsAdjacentMerge: true
        )
    }

    private func meetingSegments(
        from result: MLXBufferedTranscriptionResult,
        segmentID: UUID,
        speaker: MeetingSpeaker,
        audioSource: TranscriptAudioSource,
        startSeconds: TimeInterval,
        endSeconds: TimeInterval,
        usesStructuredOutput: Bool,
        modelFamily: MLXModelFamily,
        preventsAdjacentMerge: Bool
    ) -> [MeetingTranscriptSegment] {
        MeetingMLXSegmentMapping.meetingSegments(
            from: result,
            segmentID: segmentID,
            speaker: speaker,
            audioSource: audioSource,
            startSeconds: startSeconds,
            endSeconds: endSeconds,
            usesStructuredOutput: usesStructuredOutput,
            modelFamily: modelFamily,
            preventsAdjacentMerge: preventsAdjacentMerge,
            dictionaryEntries: activeMeetingDictionaryEntries(),
            speakerDisplayName: { speakerID in
                self.speakerDisplayName(for: speakerID, source: audioSource)
            }
        )
    }

    private func speakerDisplayName(
        for speakerID: String,
        source: TranscriptAudioSource
    ) -> String? {
        guard source != .microphone else { return nil }
        let digits = speakerID.drop(while: { !$0.isNumber })
        guard let ordinal = Int(digits), ordinal > 0 else { return speakerID }
        return MeetingSpeakerDisplayNameFormatter.displayName(ordinal: ordinal)
    }
}

@MainActor
final class MeetingRemoteASRSegmentTranscriber: MeetingSegmentTranscribing {
    private let transcriptionGate = MeetingRemoteTranscriptionGate()
    private let remoteTranscriber: RemoteASRTranscriber = {
        let transcriber = RemoteASRTranscriber()
        transcriber.doubaoDictionaryEntryProvider = {
            guard let appDelegate = AppDelegate.shared else { return [] }
            return appDelegate.dictionaryStore.activeEntriesForRemoteRequest(
                activeGroupID: appDelegate.activeDictionaryGroupID()
            )
        }
        return transcriber
    }()
    private var isCancelled = false

    func cancelPendingWork() async {
        isCancelled = true
        await transcriptionGate.cancelAll()
    }

    func transcribe(chunk: BufferedMeetingChunk) async -> MeetingTranscriptSegment? {
        do {
            return try await transcribeStrict(chunk: chunk)
        } catch {
            if !(error is CancellationError) {
                VoxtLog.meetingError("Meeting Remote ASR transcription failed: \(error)")
            }
            return nil
        }
    }

    func transcribeSegmentsStrict(chunk: BufferedMeetingChunk) async throws -> [MeetingTranscriptSegment] {
        guard let segment = try await transcribeStrict(chunk: chunk) else { return [] }
        return [segment]
    }

    private func transcribeStrict(chunk: BufferedMeetingChunk) async throws -> MeetingTranscriptSegment? {
        guard !isCancelled else { return nil }
        await transcriptionGate.acquire()
        defer {
            Task {
                await transcriptionGate.release()
            }
        }
        guard !isCancelled else { return nil }

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("Voxt-Meeting-Chunk-\(UUID().uuidString)")
            .appendingPathExtension("wav")
        defer {
            try? FileManager.default.removeItem(at: tempURL)
        }
        try MeetingAudioChunkWAVExporter.write(
            samples: chunk.samples,
            sampleRate: Int(chunk.sampleRate.rounded()),
            to: tempURL
        )
        let text = try await transcribeWithRetry(tempURL)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let hintPayload = currentHintPayload()
        let sanitizedText = MeetingTranscriptSanitizer.sanitizedText(
            text,
            prompt: hintPayload.prompt,
            contextualPhrases: hintPayload.contextualPhrases,
            dictionaryEntries: activeMeetingDictionaryEntries()
        )
        guard !sanitizedText.isEmpty else {
            VoxtLog.meetingWarning("Meeting Remote ASR transcription suppressed because it matched ASR prompt or hint guidance.")
            return nil
        }
        return MeetingTranscriptSegment(
            id: chunk.segmentID,
            speaker: chunk.speaker,
            startSeconds: chunk.startSeconds,
            endSeconds: chunk.endSeconds,
            text: sanitizedText,
            preventsAdjacentMerge: chunk.preventsAdjacentMerge
        )
    }

    private func transcribeWithRetry(_ fileURL: URL) async throws -> String {
        let configuration = remoteTranscriber.currentMeetingConfiguration()
        let retryLimit = configuration.provider == .doubaoASR ? 2 : 1
        var attempt = 0
        var lastError: Error?

        while attempt < retryLimit {
            try Task.checkCancellation()
            if isCancelled {
                throw CancellationError()
            }
            do {
                return try await remoteTranscriber.transcribeMeetingAudioFile(fileURL)
            } catch {
                if error is CancellationError || isCancelled || Task.isCancelled {
                    throw CancellationError()
                }
                lastError = error
                attempt += 1
                guard attempt < retryLimit, shouldRetry(error, provider: configuration.provider) else {
                    throw error
                }
                VoxtLog.meetingWarning(
                    "Meeting Remote ASR chunk retry scheduled. provider=\(configuration.provider.rawValue), attempt=\(attempt + 1)"
                )
                try? await Task.sleep(for: .milliseconds(220))
            }
        }

        throw lastError ?? NSError(
            domain: "Voxt.Meeting",
            code: -1,
            userInfo: [NSLocalizedDescriptionKey: "Meeting Remote ASR transcription failed."]
        )
    }

    private func shouldRetry(_ error: Error, provider: RemoteASRProvider) -> Bool {
        let nsError = error as NSError
        if nsError.domain == NSPOSIXErrorDomain, nsError.code == 57 {
            return true
        }

        if nsError.domain == NSURLErrorDomain {
            return [
                NSURLErrorNetworkConnectionLost,
                NSURLErrorCannotConnectToHost,
                NSURLErrorNotConnectedToInternet,
                NSURLErrorTimedOut
            ].contains(nsError.code)
        }

        if provider == .doubaoASR {
            let description = nsError.localizedDescription.lowercased()
            return description.contains("socket is not connected")
        }

        return false
    }

    private func currentHintPayload() -> ResolvedASRHintPayload {
        let meetingConfiguration = remoteTranscriber.currentMeetingConfiguration()
        let settings = ASRHintSettingsStore.resolvedSettings(
            for: ASRHintTarget.from(engine: .remote, remoteProvider: meetingConfiguration.provider),
            rawValue: UserDefaults.standard.string(forKey: AppPreferenceKey.asrHintSettings)
        )
        let userLanguageCodes = UserMainLanguageOption.storedSelection(
            from: UserDefaults.standard.string(forKey: AppPreferenceKey.userMainLanguageCodes)
        )
        return ASRHintResolver.resolve(
            target: ASRHintTarget.from(engine: .remote, remoteProvider: meetingConfiguration.provider),
            settings: settings,
            userLanguageCodes: userLanguageCodes,
            mlxModelRepo: meetingConfiguration.configuration.model,
            dictionaryTerms: DictionaryEntryCollection.asrPromptTermsText(from: activeMeetingDictionaryEntries())
        )
    }
}

@MainActor
private func activeMeetingDictionaryEntries() -> [DictionaryEntry] {
    guard let appDelegate = AppDelegate.shared else { return [] }
    return appDelegate.dictionaryStore.activeEntriesForRemoteRequest(
        activeGroupID: appDelegate.activeDictionaryGroupID()
    )
}
