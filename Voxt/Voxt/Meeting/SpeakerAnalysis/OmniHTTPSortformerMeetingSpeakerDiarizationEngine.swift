import Foundation

actor OmniHTTPSortformerMeetingSpeakerDiarizationEngine: MeetingSpeakerDiarizationEngine {
    private var loadedModel: MeetingSortformerModelLease?
    private var tail: Task<Void, Never>?

    func diarize(
        asset: MeetingAudioAsset, options _: MeetingSpeakerDiarizationOptions
    ) async throws -> [MeetingSpeakerTurn] {
        let descriptor = MeetingAudioAssetDescriptor(
            source: asset.source, sampleRate: asset.sampleRate,
            startSample: Int((asset.sessionStartOffset * asset.sampleRate).rounded()), sampleCount: asset.samples.count
        )
        return try await runSession(descriptors: [descriptor], loadAsset: { _ in asset }, fileAnalysis: false, progress: nil)
    }

    func diarizeSession(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        continuousAudioURL _: URL?, options _: MeetingSpeakerDiarizationOptions,
        progress: (@Sendable (Double) async -> Void)?
    ) async throws -> [MeetingSpeakerTurn] {
        try await runSession(descriptors: descriptors, loadAsset: loadAsset, fileAnalysis: false, progress: progress)
    }

    func diarizeFile(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        options _: MeetingSpeakerDiarizationOptions, progress: (@Sendable (Double) async -> Void)?
    ) async throws -> [MeetingSpeakerTurn] {
        try await runSession(descriptors: descriptors, loadAsset: loadAsset, fileAnalysis: true, progress: progress)
    }

    private func runSession(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        fileAnalysis: Bool, progress: (@Sendable (Double) async -> Void)?
    ) async throws -> [MeetingSpeakerTurn] {
        let previous = tail
        let operation = Task {
            await previous?.value
            try Task.checkCancellation()
            do {
                let turns = try await self.runSerialSession(
                    descriptors: descriptors, loadAsset: loadAsset, fileAnalysis: fileAnalysis, progress: progress
                )
                if fileAnalysis { await self.releaseModel() }
                return turns
            } catch {
                await self.releaseModel()
                throw error
            }
        }
        tail = Task { _ = try? await operation.value }
        return try await withTaskCancellationHandler {
            try await operation.value
        } onCancel: {
            operation.cancel()
        }
    }

    private func runSerialSession(
        descriptors: [MeetingAudioAssetDescriptor],
        loadAsset: @escaping @Sendable (MeetingAudioAssetDescriptor) async -> MeetingAudioAsset?,
        fileAnalysis: Bool, progress: (@Sendable (Double) async -> Void)?
    ) async throws -> [MeetingSpeakerTurn] {
        let model: MeetingSortformerModelLease
        if fileAnalysis {
            model = try await MeetingLocalInferenceCoordinator.shared.withPermit(.fileSpeakerAnalysis) {
                try await self.loadModelIfAvailable()
            }
        } else {
            model = try await loadModelIfAvailable()
        }
        let policy = try model.configuration.feedPolicy()
        let options = OmniDiarizationOptions(
            threshold: 0.5, minDuration: 0, mergeGap: 0.18,
            spkcacheMax: policy.cacheMaximumFrames, fifoMax: MeetingSpeakerFeedPolicy.fifoMaximumFrames
        )
        try model.configuration.validateEncoderCapacity(
            cacheMaximumFrameCount: options.spkcacheMax, fifoMaximumFrameCount: options.fifoMax
        )
        var stream = OmniDiarizationStream(endpoint: model.endpoint, options: options)
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
                        stream = OmniDiarizationStream(endpoint: model.endpoint, options: options)
                        state = (0, 0, 0)
                    }
                }
                previousDescriptor = descriptor
                guard let asset = await loadAsset(descriptor) else { throw MeetingSpeakerFeedError.audioUnavailable }
                let prepared = fileAnalysis && asset.sampleRate == 16_000
                    ? asset.samples
                    : ASRVoiceActivitySampleRateConverter.resample(samples: asset.samples, from: asset.sampleRate, to: 16_000)
                guard !prepared.isEmpty else { throw MeetingSpeakerFeedError.audioUnavailable }

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
                    let isLastFeed = index == descriptors.count - 1 && end == prepared.count
                    let feed: OmniDiarizationFeed
                    if fileAnalysis {
                        feed = try await MeetingLocalInferenceCoordinator.shared.withPermit(.fileSpeakerAnalysis) {
                            do {
                                let startedAt = ContinuousClock.now
                                let feed = try await currentStream.feed(samples16k: samples)
                                try Task.checkCancellation()
                                try policy.validateFeedDuration(startedAt.duration(to: .now))
                                try policy.validate(fifoFrames: feed.fifoLength, cacheFrames: feed.spkcacheLength)
                                if isLastFeed {
                                    await currentStream.close()
                                    await model.close()
                                }
                                return feed
                            } catch {
                                await currentStream.close()
                                // Note (Omni): a failed socket is not proof that Native stopped evaluating.
                                await model.close()
                                throw error
                            }
                        }
                    } else {
                        feed = try await currentStream.feed(samples16k: samples)
                    }
                    try Task.checkCancellation()
                    try policy.validate(fifoFrames: feed.fifoLength, cacheFrames: feed.spkcacheLength)
                    state = (feed.fifoLength, feed.spkcacheLength, feed.framesProcessed)
                    // Note (Omni): anchor each feed to real PCM so feature padding cannot accumulate timeline drift.
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
            await model.close()
            throw error
        }
        await stream.close()
        return turns
    }

    private func loadModelIfAvailable() async throws -> MeetingSortformerModelLease {
        if let loadedModel { return loadedModel }
        let model = try await MeetingSortformerModelLease.load()
        loadedModel = model
        return model
    }

    private func releaseModel() async {
        let original = loadedModel
        loadedModel = nil
        await original?.close()
    }
}
