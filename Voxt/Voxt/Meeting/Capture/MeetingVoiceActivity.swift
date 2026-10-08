// MeetingVoiceActivity.swift
// Provides Meeting Voice Activity for meeting capture.

import Foundation
import HuggingFace
import MLX
import MLXAudioVAD

struct MeetingVoiceActivityDecision: Equatable, Sendable {
    let isSpeech: Bool
    let probability: Float?
    let source: Source

    enum Source: Equatable, Sendable {
        case off
        case server
        case energy
        case silero
        case omni
        case fallbackEnergy
    }
}

private extension MeetingVoiceActivityDecision.Source {
    nonisolated var telemetryName: String {
        switch self {
        case .off:
            return "off"
        case .server:
            return "server"
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

actor MeetingVoiceActivityDetector {
    private var sileroDetector = ASRSileroStreamingVoiceActivityDetector()
    private var omniDetector = OmniStreamVoiceActivityBackend(useCase: .meeting)
    private var mode = currentModeFromSettings()
    private var sileroSensitivity = currentSileroSensitivityFromSettings()
    private var sileroFallbackWarningLogged = false
    private var omniDegradedWarningLogged = false

    func refreshFromPreferences() async {
        let nextMode = Self.currentModeFromSettings()
        if nextMode != mode {
            await sileroDetector.reset()
            await omniDetector.reset()
        }
        mode = nextMode
        sileroSensitivity = Self.currentSileroSensitivityFromSettings()
        sileroFallbackWarningLogged = false
        omniDegradedWarningLogged = false
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

    private nonisolated static func currentModeFromSettings() -> LocalVADMode {
        LocalVADMode.stored()
    }

    private nonisolated static func currentSileroSensitivityFromSettings() -> MeetingSileroVADSensitivity {
        MeetingSileroVADSensitivity.stored()
    }

    func activity(
        samples: [Float],
        sampleRate: Double,
        speaker: MeetingSpeaker,
        fallbackLevel: Float,
        fallbackThreshold: Float,
        serverVADActive: Bool = false
    ) async -> MeetingVoiceActivityDecision {
        let startedAt = ProcessInfo.processInfo.systemUptime
        let fallback = MeetingVoiceActivityDecision(
            isSpeech: fallbackLevel >= fallbackThreshold,
            probability: nil,
            source: .energy
        )
        let frameBackend = ASRVoiceActivityRuntimePolicy.effectiveBackend(
            mode: mode,
            useCase: .meeting
        )
        if serverVADActive {
            return finishActivity(
                MeetingVoiceActivityDecision(
                    isSpeech: true,
                    probability: nil,
                    source: .server
                ),
                startedAt: startedAt,
                speaker: speaker,
                sampleCount: samples.count,
                frameBackend: frameBackend
            )
        }

        switch frameBackend {
        case .off:
            return finishActivity(
                MeetingVoiceActivityDecision(
                    isSpeech: true,
                    probability: nil,
                    source: .off
                ),
                startedAt: startedAt,
                speaker: speaker,
                sampleCount: samples.count,
                frameBackend: frameBackend
            )
        case .energy:
            return finishActivity(
                fallback,
                startedAt: startedAt,
                speaker: speaker,
                sampleCount: samples.count,
                frameBackend: frameBackend
            )
        case .mlxSilero:
            break
        case .omniStream:
            if let decision = await omniActivity(
                samples: samples,
                sampleRate: sampleRate,
                speaker: speaker
            ) {
                return finishActivity(
                    decision,
                    startedAt: startedAt,
                    speaker: speaker,
                    sampleCount: samples.count,
                    frameBackend: frameBackend
                )
            }
            return finishActivity(
                MeetingVoiceActivityDecision(
                    isSpeech: fallback.isSpeech,
                    probability: nil,
                    source: .fallbackEnergy
                ),
                startedAt: startedAt,
                speaker: speaker,
                sampleCount: samples.count,
                frameBackend: frameBackend
            )
        }

        if let decision = await mlxSileroActivity(
            samples: samples,
            sampleRate: sampleRate,
            speaker: speaker
        ) {
            return finishActivity(
                decision,
                startedAt: startedAt,
                speaker: speaker,
                sampleCount: samples.count,
                frameBackend: frameBackend
            )
        }

        return finishActivity(
            MeetingVoiceActivityDecision(
                isSpeech: fallback.isSpeech,
                probability: nil,
                source: .fallbackEnergy
            ),
            startedAt: startedAt,
            speaker: speaker,
            sampleCount: samples.count,
            frameBackend: frameBackend
        )
    }

    private func finishActivity(
        _ decision: MeetingVoiceActivityDecision,
        startedAt: TimeInterval,
        speaker: MeetingSpeaker,
        sampleCount: Int,
        frameBackend: ASRVoiceActivityBackendKind
    ) -> MeetingVoiceActivityDecision {
        let elapsedMilliseconds = max(0, (ProcessInfo.processInfo.systemUptime - startedAt) * 1000)
        let probabilityText = decision.probability.map { String(format: "%.3f", $0) } ?? "nil"
        VoxtLog.meeting(
            "Meeting VAD decision. mode=\(mode.rawValue), sileroSensitivity=\(sileroSensitivity.rawValue), frameBackend=\(frameBackend.rawValue), source=\(decision.source.telemetryName), speaker=\(speaker.rawValue), speech=\(decision.isSpeech), probability=\(probabilityText), sampleCount=\(sampleCount), elapsedMs=\(String(format: "%.2f", elapsedMilliseconds))",
            verbose: true
        )
        return decision
    }

    private func mlxSileroActivity(
        samples: [Float],
        sampleRate: Double,
        speaker: MeetingSpeaker
    ) async -> MeetingVoiceActivityDecision? {
        do {
            if let probability = try await sileroDetector.probability(
                samples: samples,
                sampleRate: sampleRate,
                streamID: speaker.rawValue
            ) {
                return MeetingVoiceActivityDecision(
                    isSpeech: probability >= sileroSensitivity.onsetProbabilityThreshold,
                    probability: probability,
                    source: .silero
                )
            }
        } catch {
            if shouldLogSileroFallback(error) {
                VoxtLog.meetingWarning("Meeting Silero VAD failed; falling back to energy VAD. error=\(error.localizedDescription)")
                sileroFallbackWarningLogged = true
            }
            await sileroDetector.reset()
        }
        return nil
    }

    private func omniActivity(
        samples: [Float],
        sampleRate: Double,
        speaker: MeetingSpeaker
    ) async -> MeetingVoiceActivityDecision? {
        do {
            let frame = ASRVoiceActivityAudioFrame(
                samples: samples,
                sampleRate: sampleRate,
                startSeconds: 0,
                endSeconds: Double(samples.count) / max(sampleRate, 1)
            )
            if let decision = try await omniDetector.decision(
                for: frame,
                streamID: "meeting-\(speaker.rawValue)"
            ) {
                return MeetingVoiceActivityDecision(
                    isSpeech: decision.isSpeech,
                    probability: decision.probability,
                    source: .omni
                )
            }
        } catch {
            if !omniDegradedWarningLogged {
                VoxtLog.meetingWarning("Meeting OmniVAD unavailable; degrading current session to energy VAD. error=\(error.localizedDescription)")
                omniDegradedWarningLogged = true
            }
            await omniDetector.reset()
        }
        return nil
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

actor ASRSileroStreamingVoiceActivityDetector {
    private let sampleRate = 16_000
    private let chunkSize = 512
    private var model: SileroVAD?
    private var states: [String: SileroVADStreamingState] = [:]
    private var pendingSamples: [String: [Float]] = [:]
    private var pendingSampleOffsets: [String: Int] = [:]
    private var lastProbabilities: [String: Float] = [:]

    func reset() {
        states.removeAll()
        pendingSamples.removeAll()
        pendingSampleOffsets.removeAll()
        lastProbabilities.removeAll()
    }

    func unload() {
        reset()
        model = nil
    }

    func probability(
        samples: [Float],
        sampleRate inputSampleRate: Double,
        streamID: String = "default"
    ) async throws -> Float? {
        guard !samples.isEmpty else { return nil }
        guard inputSampleRate.isFinite, inputSampleRate > 0 else { return nil }
        let model = try await loadModelIfAvailable()
        let prepared = ASRVoiceActivitySampleRateConverter.resample(
            samples: samples,
            from: inputSampleRate,
            to: Double(sampleRate)
        )
        guard !prepared.isEmpty else { return lastProbabilities[streamID] }

        var pending = pendingSamples[streamID] ?? []
        pending.append(contentsOf: prepared)
        var pendingOffset = min(pendingSampleOffsets[streamID] ?? 0, pending.count)
        var latestProbability: Float?

        while pending.count - pendingOffset >= chunkSize {
            let endOffset = pendingOffset + chunkSize
            let chunk = Array(pending[pendingOffset..<endOffset])
            pendingOffset = endOffset
            let state = states[streamID]
            let (probability, nextState) = try withError {
                let (probabilityArray, nextState) = try model.feed(
                    chunk: MLXArray(chunk),
                    state: state,
                    sampleRate: sampleRate
                )
                eval(probabilityArray)
                return (probabilityArray.asArray(Float.self).first, nextState)
            }
            if let probability {
                latestProbability = probability
            }
            states[streamID] = nextState
        }

        Self.compactPendingSamples(&pending, offset: &pendingOffset, chunkSize: chunkSize)
        pendingSamples[streamID] = pending
        pendingSampleOffsets[streamID] = pendingOffset
        if let latestProbability {
            lastProbabilities[streamID] = latestProbability
            return latestProbability
        }
        return nil
    }

    private nonisolated static func compactPendingSamples(
        _ samples: inout [Float],
        offset: inout Int,
        chunkSize: Int
    ) {
        if offset == samples.count {
            samples.removeAll(keepingCapacity: true)
            offset = 0
        } else if offset >= chunkSize * 8 {
            samples.removeFirst(offset)
            offset = 0
        }
    }

    private func loadModelIfAvailable() async throws -> SileroVAD {
        if let model {
            return model
        }
        let directory = try await SileroVADModelProvisioner.shared.ensureModelDirectory()
        let loaded = try SileroVADModelSupport.loadModel(from: directory)
        model = loaded
        return loaded
    }
}

actor ASRSileroOfflineVoiceActivityDetector: ASROfflineVoiceActivityBackend {
    private let sampleRate = 16_000
    private var model: SileroVAD?

    func unload() {
        model = nil
    }

    func speechRanges(samples: [Float], sampleRate inputSampleRate: Double) async throws -> [ASROfflineSpeechRange] {
        guard !samples.isEmpty, inputSampleRate.isFinite, inputSampleRate > 0 else { return [] }
        let model = try await loadModelIfAvailable()
        let prepared = ASRVoiceActivitySampleRateConverter.resample(
            samples: samples,
            from: inputSampleRate,
            to: Double(sampleRate)
        )
        guard !prepared.isEmpty else { return [] }

        let profile = MeetingSileroVADSensitivity.stored().configuration()
        let timestamps = try withError {
            try model.getSpeechTimestamps(
                MLXArray(prepared),
                sampleRate: sampleRate,
                threshold: profile.onsetProbabilityThreshold,
                minSpeechDurationMs: Int((profile.minSpeechSeconds * 1_000).rounded(.up)),
                minSilenceDurationMs: Int((profile.minSilenceSeconds * 1_000).rounded(.up)),
                speechPadMs: Int((profile.speechPadSeconds * 1_000).rounded(.up))
            )
        }
        return timestamps.compactMap { timestamp in
            let start = Double(timestamp.start) / Double(sampleRate)
            let end = Double(timestamp.end) / Double(sampleRate)
            guard end > start else { return nil }
            return ASROfflineSpeechRange(startSeconds: start, endSeconds: end)
        }
    }

    private func loadModelIfAvailable() async throws -> SileroVAD {
        if let model {
            return model
        }
        let directory = try await SileroVADModelProvisioner.shared.ensureModelDirectory()
        let loaded = try SileroVADModelSupport.loadModel(from: directory)
        model = loaded
        return loaded
    }
}

actor MeetingOfflineVoiceActivityDetector {
    private var sileroDetector: ASRSileroOfflineVoiceActivityDetector?
    private var omniDetector: OmniOfflineVoiceActivityBackend?
    private var sileroWarningLogged = false
    private var omniWarningLogged = false

    func releaseResources() async {
        await sileroDetector?.unload()
        await omniDetector?.releaseResources()
        sileroDetector = nil
        omniDetector = nil
        sileroWarningLogged = false
        omniWarningLogged = false
    }

    func speechRanges(
        samples: [Float],
        sampleRate: Double,
        fallbackThreshold: Float
    ) async -> [ASROfflineSpeechRange]? {
        let mode = LocalVADMode.stored()
        switch ASRVoiceActivityRuntimePolicy.effectiveBackend(mode: mode, useCase: .meeting) {
        case .off:
            return nil
        case .energy:
            return Self.energySpeechRanges(
                samples: samples,
                sampleRate: sampleRate,
                threshold: fallbackThreshold
            )
        case .mlxSilero:
            do {
                let detector = await resolvedSileroDetector()
                return try await detector.speechRanges(samples: samples, sampleRate: sampleRate)
            } catch {
                if !sileroWarningLogged {
                    VoxtLog.meetingWarning(
                        "Meeting final Silero VAD unavailable; preserving unvalidated transcript segments. error=\(error.localizedDescription)"
                    )
                    sileroWarningLogged = true
                }
                return nil
            }
        case .omniStream:
            do {
                let detector = await resolvedOmniDetector()
                return try await detector.speechRanges(samples: samples, sampleRate: sampleRate)
            } catch {
                if !omniWarningLogged {
                    VoxtLog.meetingWarning(
                        "Meeting final OmniVAD unavailable; preserving unvalidated transcript segments. error=\(error.localizedDescription)"
                    )
                    omniWarningLogged = true
                }
                return nil
            }
        }
    }

    private func resolvedSileroDetector() async -> ASRSileroOfflineVoiceActivityDetector {
        if let sileroDetector {
            return sileroDetector
        }
        let detector = await MainActor.run {
            ASRSileroOfflineVoiceActivityDetector()
        }
        sileroDetector = detector
        return detector
    }

    private func resolvedOmniDetector() async -> OmniOfflineVoiceActivityBackend {
        if let omniDetector {
            return omniDetector
        }
        let detector = await MainActor.run {
            OmniOfflineVoiceActivityBackend(useCase: .meeting)
        }
        omniDetector = detector
        return detector
    }

    private nonisolated static func energySpeechRanges(
        samples: [Float],
        sampleRate: Double,
        threshold: Float
    ) -> [ASROfflineSpeechRange] {
        guard !samples.isEmpty, sampleRate.isFinite, sampleRate > 0 else { return [] }
        let frameSampleCount = max(Int((sampleRate * 0.1).rounded()), 1)
        var segmenter = ASRVoiceActivitySegmenter(configuration: .meeting)
        var ranges: [ASROfflineSpeechRange] = []
        var startIndex = 0

        while startIndex < samples.count {
            let endIndex = min(startIndex + frameSampleCount, samples.count)
            let frameSamples = Array(samples[startIndex..<endIndex])
            let startSeconds = Double(startIndex) / sampleRate
            let endSeconds = Double(endIndex) / sampleRate
            let level = AudioLevelMeter.normalizedLevel(
                fromSamples: frameSamples,
                noiseGate: 0.002,
                gain: 12
            )
            let decision = ASRVoiceActivityFrameDecision(
                startSeconds: startSeconds,
                endSeconds: endSeconds,
                isSpeech: level >= threshold
            )
            ranges.append(contentsOf: speechRanges(from: segmenter.append(decision)))
            startIndex = endIndex
        }
        if let event = segmenter.finish(at: Double(samples.count) / sampleRate) {
            ranges.append(contentsOf: speechRanges(from: [event]))
        }
        return ranges
    }

    private nonisolated static func speechRanges(
        from events: [ASRVoiceActivityEvent]
    ) -> [ASROfflineSpeechRange] {
        events.compactMap { event in
            switch event {
            case .speechEnded(let segment), .speechForced(let segment):
                return ASROfflineSpeechRange(
                    startSeconds: segment.startSeconds,
                    endSeconds: segment.endSeconds
                )
            case .speechStarted, .speechRejected:
                return nil
            }
        }
    }
}

enum MeetingVADModelError: LocalizedError {
    case modelNotDownloaded
    case runtimeUnavailable(String)

    var errorDescription: String? {
        switch self {
        case .modelNotDownloaded:
            return AppLocalization.localizedString("VAD model is not downloaded.")
        case .runtimeUnavailable(let detail):
            return detail
        }
    }
}

enum MeetingVADModelStorage {
    static let sortformerV2Repo = "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"
    static let fallbackRemoteSizeText = "2 MB"
    static let sortformerFallbackRemoteSizeText = "120 MB"

    static let repo = SileroVADModelSupport.repo

    static func validatedModelDirectory() async -> URL? {
        let directories = ModelStorageDirectoryManager.resolvedReadableRootURLs().compactMap {
            MLXModelStorageSupport.cacheDirectory(for: repo, rootDirectory: $0)
        }
        return await Task.detached(priority: .utility) {
            directories.first { isValidModelDirectory($0) }
        }.value
    }

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

    static func writeModelDirectory() -> URL? {
        return MLXModelStorageSupport.cacheDirectory(
            for: repo,
            rootDirectory: ModelStorageDirectoryManager.resolvedWriteRootURL()
        )
    }

    static func downloadTempDirectory() -> URL? {
        guard let repoID = Repo.ID(rawValue: repo) else { return nil }
        let modelSubdir = repoID.description.replacingOccurrences(of: "/", with: "_")
        return ModelStorageDirectoryManager.resolvedWriteRootURL()
            .appendingPathComponent("mlx-audio", isDirectory: true)
            .appendingPathComponent("\(modelSubdir)-download", isDirectory: true)
    }

    nonisolated static func isValidModelDirectory(_ directory: URL, fileManager: FileManager = .default) -> Bool {
        isValidSileroModelDirectory(directory, fileManager: fileManager)
    }

    nonisolated private static func isValidSileroModelDirectory(_ directory: URL, fileManager: FileManager) -> Bool {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("config.json")),
              (try? JSONSerialization.jsonObject(with: data)) != nil,
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
        guard let repoID = Repo.ID(rawValue: repo) else { return }
        MLXModelStorageSupport.clearHubCache(for: repoID, rootDirectory: rootDirectory)
    }
}
