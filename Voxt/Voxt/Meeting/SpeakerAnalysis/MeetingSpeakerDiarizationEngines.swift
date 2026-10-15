// MeetingSpeakerDiarizationEngines.swift
// Provides Meeting Speaker Diarization Engines for meeting speaker analysis.

import Foundation
import MLX
import MLXAudioVAD

protocol MeetingSpeakerDiarizationEngine: Sendable {
    func diarize(
        asset: MeetingAudioAsset,
        options: MeetingSpeakerDiarizationOptions
    ) async throws -> [MeetingSpeakerTurn]

    func diarizeSession(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        continuousAudioURL: URL?,
        options: MeetingSpeakerDiarizationOptions,
        progress: (@Sendable (Double) async -> Void)?
    ) async throws -> [MeetingSpeakerTurn]

    func diarizeFile(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        options: MeetingSpeakerDiarizationOptions,
        progress: (@Sendable (Double) async -> Void)?
    ) async throws -> [MeetingSpeakerTurn]
}

extension MeetingSpeakerDiarizationEngine {
    func diarizeFile(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        options: MeetingSpeakerDiarizationOptions,
        progress: (@Sendable (Double) async -> Void)?
    ) async throws -> [MeetingSpeakerTurn] {
        try await diarizeSession(descriptors: descriptors, loadAsset: loadAsset,
                                continuousAudioURL: nil, options: options, progress: progress)
    }

    func diarizeSession(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        continuousAudioURL _: URL?,
        options: MeetingSpeakerDiarizationOptions,
        progress: (@Sendable (Double) async -> Void)?
    ) async throws -> [MeetingSpeakerTurn] {
        var turns: [MeetingSpeakerTurn] = []
        let descriptorCount = max(descriptors.count, 1)
        await progress?(0)
        for (index, descriptor) in descriptors.enumerated() {
            try Task.checkCancellation()
            if let asset = await loadAsset(descriptor) {
                turns.append(contentsOf: try await diarize(asset: asset, options: options))
            }
            await progress?(Double(index + 1) / Double(descriptorCount))
        }
        return turns
    }
}

enum MeetingSpeakerDiarizationEngineFactory {
    nonisolated private static let sharedSortformerEngine = SortformerMeetingSpeakerDiarizationEngine()

    nonisolated static func makeDefault(defaults _: UserDefaults = .standard) -> (any MeetingSpeakerDiarizationEngine)? {
        sharedSortformerEngine
    }
}

actor SortformerMeetingSpeakerDiarizationEngine: MeetingSpeakerDiarizationEngine {
    private var model: SortformerModel?
    // Note (Jiaxin Deng): the native path holds a server lease for as long as the Swift
    // path would hold its model.
    private var omniEndpoint: OmniServerEndpoint?
    private var omniConfig: SortformerConfig?

    func diarize(
        asset: MeetingAudioAsset,
        options _: MeetingSpeakerDiarizationOptions
    ) async throws -> [MeetingSpeakerTurn] {
        let descriptor = MeetingAudioAssetDescriptor(
            source: asset.source, sampleRate: asset.sampleRate,
            startSample: Int((asset.sessionStartOffset * asset.sampleRate).rounded()),
            sampleCount: asset.samples.count
        )
        return try await runSession(descriptors: [descriptor], loadAsset: { _ in asset },
                                    fileAnalysis: false, progress: nil)
    }

    func diarizeSession(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        continuousAudioURL _: URL?,
        options _: MeetingSpeakerDiarizationOptions,
        progress: (@Sendable (Double) async -> Void)?
    ) async throws -> [MeetingSpeakerTurn] {
        try await runSession(descriptors: descriptors, loadAsset: loadAsset,
                             fileAnalysis: false, progress: progress)
    }

    func diarizeFile(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        options _: MeetingSpeakerDiarizationOptions,
        progress: (@Sendable (Double) async -> Void)?
    ) async throws -> [MeetingSpeakerTurn] {
        // File callers own this engine, so its cached model is released on all exits.
        defer { model = nil }
        do {
            let turns = try await runSession(descriptors: descriptors, loadAsset: loadAsset,
                                             fileAnalysis: true, progress: progress)
            await releaseOmniEndpoint()
            return turns
        } catch {
            await releaseOmniEndpoint()
            throw error
        }
    }

    private func runSession(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        fileAnalysis: Bool,
        progress: (@Sendable (Double) async -> Void)?
    ) async throws -> [MeetingSpeakerTurn] {
        if OmniSortformerRuntime.isEnabled {
            return try await runOmniSession(descriptors: descriptors, loadAsset: loadAsset,
                                            fileAnalysis: fileAnalysis, progress: progress)
        }
        if fileAnalysis {
            try await MeetingLocalInferenceCoordinator.shared.withPermit(.fileSpeakerAnalysis) {
                try await self.prepareFileModel()
            }
        } else {
            _ = try await loadModelIfAvailable()
        }
        guard let model else { throw MeetingVADModelError.modelNotDownloaded }
        let config = model.config
        let policy = try MeetingSpeakerFeedPolicy(
            sampleRate: config.processorConfig.samplingRate,
            hopLength: config.processorConfig.hopLength,
            subsamplingFactor: config.fcEncoderConfig.subsamplingFactor,
            chunkFrames: config.modulesConfig.chunkLen,
            cacheFrames: config.modulesConfig.spkcacheLen,
            updateFrames: config.modulesConfig.spkcacheUpdatePeriod,
            usesAOSC: config.modulesConfig.useAosc
        )
        var state = model.initStreamingState()
        var previousDescriptor: MeetingAudioAssetDescriptor?
        var turns: [MeetingSpeakerTurn] = []
        let descriptorCount = max(descriptors.count, 1)
        await progress?(0)

        for (index, descriptor) in descriptors.enumerated() {
            try Task.checkCancellation()
            if let previousDescriptor {
                let expectedStart = previousDescriptor.sessionStartOffset + previousDescriptor.durationSeconds
                let isContinuous = descriptor.source == previousDescriptor.source
                    && abs(descriptor.sessionStartOffset - expectedStart) < 0.05
                if !isContinuous {
                    state = model.initStreamingState()
                }
            }
            previousDescriptor = descriptor

            guard let asset = await loadAsset(descriptor) else {
                throw MeetingSpeakerFeedError.audioUnavailable
            }
            // Prepared file PCM is already finite, mono and 16 kHz. Avoid the
            // converter's full-window sanitizing map/copy on this trusted path.
            let prepared = fileAnalysis && asset.sampleRate == 16_000
                ? asset.samples
                : ASRVoiceActivitySampleRateConverter.resample(
                    samples: asset.samples, from: asset.sampleRate, to: 16_000
                )
            guard !prepared.isEmpty else {
                throw MeetingSpeakerFeedError.audioUnavailable
            }

            var offset = 0
            while offset < prepared.count {
                try Task.checkCancellation()
                try policy.validate(fifoFrames: state.fifoLen, cacheFrames: state.spkcacheLen)
                let end = min(offset + policy.samplesPerFeed, prepared.count)
                let samples = Array(prepared[offset..<end])
                let inputState = state
                let audioOffset = asset.sessionStartOffset + Double(offset) / Double(policy.sampleRate)
                let result: (DiarizationOutput, StreamingState)
                if fileAnalysis {
                    result = try await MeetingLocalInferenceCoordinator.shared.withPermit(.fileSpeakerAnalysis) {
                        try await self.feed(samples: samples, state: inputState, policy: policy, enforceTimeLimit: true)
                    }
                } else {
                    result = try await feed(samples: samples, state: inputState, policy: policy)
                }
                try Task.checkCancellation() // feed's detached native work must exit first
                let (output, newState) = result
                try policy.validate(fifoFrames: newState.fifoLen, cacheFrames: newState.spkcacheLen)
                state = newState
                // feed offsets use subsampled frame counts; anchor to real samples
                // at every call so feature padding cannot accumulate timestamp drift.
                turns.append(contentsOf: output.segments.compactMap { item in
                    guard let range = policy.mappedRange(
                        start: Double(item.start), end: Double(item.end),
                        stateFrames: inputState.framesProcessed, audioOffset: audioOffset, sampleCount: samples.count
                    ) else { return nil }
                    return MeetingSpeakerTurn(
                        source: asset.source, speakerID: "sortformer-\(item.speaker)",
                        displayName: MeetingSpeakerDisplayNameFormatter.displayName(ordinal: item.speaker + 1),
                        startSeconds: range.lowerBound, endSeconds: range.upperBound, confidence: nil
                    )
                })
                offset = end
                await progress?((Double(index) + Double(offset) / Double(prepared.count)) / Double(descriptorCount))
            }
        }
        return turns
    }

    private func runOmniSession(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        fileAnalysis: Bool,
        progress: (@Sendable (Double) async -> Void)?
    ) async throws -> [MeetingSpeakerTurn] {
        let endpoint: OmniServerEndpoint
        if fileAnalysis {
            endpoint = try await MeetingLocalInferenceCoordinator.shared.withPermit(.fileSpeakerAnalysis) {
                try await self.omniEndpointIfAvailable()
            }
        } else {
            endpoint = try await omniEndpointIfAvailable()
        }
        guard let config = omniConfig else { throw MeetingVADModelError.modelNotDownloaded }
        let policy = try MeetingSpeakerFeedPolicy(
            sampleRate: config.processorConfig.samplingRate,
            hopLength: config.processorConfig.hopLength,
            subsamplingFactor: config.fcEncoderConfig.subsamplingFactor,
            chunkFrames: config.modulesConfig.chunkLen,
            cacheFrames: config.modulesConfig.spkcacheLen,
            updateFrames: config.modulesConfig.spkcacheUpdatePeriod,
            usesAOSC: config.modulesConfig.useAosc
        )
        let options = OmniDiarizationOptions(
            threshold: 0.5, minDuration: 0, mergeGap: 0.18,
            spkcacheMax: policy.cacheMaximumFrames, fifoMax: MeetingSpeakerFeedPolicy.fifoMaximumFrames
        )
        // Note (Jiaxin Deng): the server refuses these limits at the handshake, which reads
        // as a transport failure, so they are checked here first.
        let encoderFrames = max(options.spkcacheMax, config.modulesConfig.spkcacheLen) + options.fifoMax
            + config.modulesConfig.chunkLeftContext + config.modulesConfig.spkcacheUpdatePeriod
        guard encoderFrames <= config.tfEncoderConfig.maxSourcePositions else {
            throw MeetingSpeakerFeedError.invalidConfiguration
        }
        var stream = OmniDiarizationStream(endpoint: endpoint, options: options)
        var state = (fifo: 0, cache: 0, frames: 0)
        var previousDescriptor: MeetingAudioAssetDescriptor?
        var turns: [MeetingSpeakerTurn] = []
        let descriptorCount = max(descriptors.count, 1)
        await progress?(0)

        do {
            for (index, descriptor) in descriptors.enumerated() {
                try Task.checkCancellation()
                if let previousDescriptor {
                    let expectedStart = previousDescriptor.sessionStartOffset + previousDescriptor.durationSeconds
                    let isContinuous = descriptor.source == previousDescriptor.source
                        && abs(descriptor.sessionStartOffset - expectedStart) < 0.05
                    if !isContinuous {
                        await stream.close()
                        stream = OmniDiarizationStream(endpoint: endpoint, options: options)
                        state = (0, 0, 0)
                    }
                }
                previousDescriptor = descriptor

                guard let asset = await loadAsset(descriptor) else {
                    throw MeetingSpeakerFeedError.audioUnavailable
                }
                let prepared = fileAnalysis && asset.sampleRate == 16_000
                    ? asset.samples
                    : ASRVoiceActivitySampleRateConverter.resample(
                        samples: asset.samples, from: asset.sampleRate, to: 16_000
                    )
                guard !prepared.isEmpty else {
                    throw MeetingSpeakerFeedError.audioUnavailable
                }

                var offset = 0
                while offset < prepared.count {
                    try Task.checkCancellation()
                    try policy.validate(fifoFrames: state.fifo, cacheFrames: state.cache)
                    let end = min(offset + policy.samplesPerFeed, prepared.count)
                    let sampleCount = end - offset
                    var padded = Array(prepared[offset ..< end])
                    if padded.count < policy.frameSamples {
                        padded.append(contentsOf: repeatElement(0, count: policy.frameSamples - padded.count))
                    }
                    let samples = padded
                    let inputFrames = state.frames
                    let audioOffset = asset.sessionStartOffset + Double(offset) / Double(policy.sampleRate)
                    let currentStream = stream
                    let feed: OmniDiarizationFeed
                    if fileAnalysis {
                        feed = try await MeetingLocalInferenceCoordinator.shared.withPermit(.fileSpeakerAnalysis) {
                            let startedAt = ContinuousClock.now
                            let feed = try await currentStream.feed(samples16k: samples)
                            try policy.validateFeedDuration(startedAt.duration(to: .now))
                            return feed
                        }
                    } else {
                        feed = try await currentStream.feed(samples16k: samples)
                    }
                    try Task.checkCancellation()
                    try policy.validate(fifoFrames: feed.fifoLength, cacheFrames: feed.spkcacheLength)
                    state = (feed.fifoLength, feed.spkcacheLength, feed.framesProcessed)
                    turns.append(contentsOf: feed.segments.compactMap { item in
                        guard let range = policy.mappedRange(
                            start: Double(item.start), end: Double(item.end),
                            stateFrames: inputFrames, audioOffset: audioOffset, sampleCount: sampleCount
                        ) else { return nil }
                        return MeetingSpeakerTurn(
                            source: asset.source, speakerID: "sortformer-\(item.speaker)",
                            displayName: MeetingSpeakerDisplayNameFormatter.displayName(ordinal: item.speaker + 1),
                            startSeconds: range.lowerBound, endSeconds: range.upperBound, confidence: nil
                        )
                    })
                    offset = end
                    await progress?((Double(index) + Double(offset) / Double(prepared.count)) / Double(descriptorCount))
                }
            }
        } catch {
            await stream.close()
            await releaseOmniEndpoint(after: error, endpoint: endpoint)
            throw error
        }
        await stream.close()
        return turns
    }

    // Note (Jiaxin Deng): after a transport failure the server may be gone, so releasing lets
    // the next session restart it; an endpoint already replaced has nothing to release.
    private func releaseOmniEndpoint(after error: Error, endpoint: OmniServerEndpoint) async {
        let domain = (error as NSError).domain
        guard omniEndpoint == endpoint, domain == NSURLErrorDomain || domain == NSPOSIXErrorDomain else { return }
        await releaseOmniEndpoint()
    }

    private func omniEndpointIfAvailable() async throws -> OmniServerEndpoint {
        if let omniEndpoint {
            return omniEndpoint
        }
        guard let directory = await MeetingSortformerModelStorage.validatedModelDirectory() else {
            throw MeetingVADModelError.modelNotDownloaded
        }
        let config = try JSONDecoder().decode(
            SortformerConfig.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json"))
        )
        let endpoint = try await OmniSortformerRuntime.shared.acquire(modelDirectory: directory)
        if let omniEndpoint {
            // Note (Jiaxin Deng): another call acquired one while this one awaited.
            await OmniSortformerRuntime.shared.release()
            return omniEndpoint
        }
        omniEndpoint = endpoint
        omniConfig = config
        return endpoint
    }

    private func releaseOmniEndpoint() async {
        guard omniEndpoint != nil else { return }
        omniEndpoint = nil
        omniConfig = nil
        await OmniSortformerRuntime.shared.release()
    }

    private func prepareFileModel() async throws {
        _ = try await loadModelIfAvailable()
    }

    private func feed(samples: [Float], state: StreamingState, policy: MeetingSpeakerFeedPolicy, enforceTimeLimit: Bool = false) async throws -> (DiarizationOutput, StreamingState) {
        guard let model else { throw MeetingVADModelError.modelNotDownloaded }
        var padded = samples
        if padded.count < policy.frameSamples {
            padded.append(contentsOf: repeatElement(0, count: policy.frameSamples - padded.count))
        }
        let startedAt = ContinuousClock.now
        let result = try await model.feed(
            chunk: MLXArray(padded), state: state, sampleRate: policy.sampleRate,
            threshold: 0.5, minDuration: 0, mergeGap: 0.18,
            spkcacheMax: policy.cacheMaximumFrames, fifoMax: MeetingSpeakerFeedPolicy.fifoMaximumFrames
        )
        try Task.checkCancellation()
        if enforceTimeLimit {
            try policy.validateFeedDuration(startedAt.duration(to: .now))
        }
        return result
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
}
