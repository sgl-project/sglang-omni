import XCTest
@preconcurrency import MLX
import MLXAudioVAD
@testable import Voxt

@MainActor
final class SortformerBoundedFeedIntegrationTests: XCTestCase {
    func testFileWorkUnitBoundsOnlyUnusedCacheAndRestoresThePreviousSetting() throws {
        try ModelTestGate.requireEnabled("MLX file cache limit scope test")
        let previous = Memory.cacheLimit
        let liveLimit = Memory.memoryLimit
        let restore = MeetingFileInferenceCache.beginWorkUnit()
        XCTAssertEqual(Memory.cacheLimit, min(previous, MeetingFileInferenceCache.retainedCacheThresholdBytes))
        XCTAssertEqual(Memory.memoryLimit, liveLimit)
        restore()
        XCTAssertEqual(Memory.cacheLimit, previous)
        XCTAssertEqual(Memory.memoryLimit, liveLimit)
    }

    func testInstalledSortformerKeepsStreamingStateBoundedForTwentyMinutes() async throws {
        try ModelTestGate.requireEnabled("Sortformer bounded-feed integration test")
        ModelTestGate.configureStorageRoot(for: self)
        guard let directory = await MeetingSortformerModelStorage.validatedModelDirectory() else {
            throw XCTSkip("Install the configured Sortformer model before running this test")
        }
        let model = try SortformerModel.fromModelDirectory(directory)
        let config = model.config
        let policy = try MeetingSpeakerFeedPolicy(
            sampleRate: config.processorConfig.samplingRate, hopLength: config.processorConfig.hopLength,
            subsamplingFactor: config.fcEncoderConfig.subsamplingFactor,
            chunkFrames: config.modulesConfig.chunkLen, cacheFrames: config.modulesConfig.spkcacheLen,
            updateFrames: config.modulesConfig.spkcacheUpdatePeriod, usesAOSC: config.modulesConfig.useAosc
        )
        var state = model.initStreamingState()
        let samples = (0..<policy.samplesPerFeed).map {
            Float(sin(Double($0) * 2 * .pi * 220 / 16_000)) * 0.1
        }
        let feedCount = (20 * 60 * 16_000 + samples.count - 1) / samples.count
        for index in 0..<feedCount {
            let restore = MeetingFileInferenceCache.beginWorkUnit()
            defer {
                MeetingFileInferenceCache.trimIfNeeded(underPressure: false)
                restore()
            }
            let (_, nextState) = try await model.feed(
                chunk: MLXArray(samples), state: state, sampleRate: 16_000,
                spkcacheMax: policy.cacheMaximumFrames, fifoMax: MeetingSpeakerFeedPolicy.fifoMaximumFrames
            )
            try policy.validate(fifoFrames: nextState.fifoLen, cacheFrames: nextState.spkcacheLen)
            XCTAssertGreaterThan(nextState.framesProcessed, state.framesProcessed, "feed \(index)")
            state = nextState
            // The following feed consumes the same state across cache scopes.
        }
        // Synthetic audio proves a memory-state invariant, not diarization quality.
        // Real multi-speaker recordings must separately validate DER and timestamps.
    }
}
