import XCTest
@preconcurrency import MLX
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
        guard await MeetingSortformerModelStorage.validatedModelDirectory() != nil else {
            throw XCTSkip("Install the configured Sortformer model before running this test")
        }
        guard OmniSortformerRuntime.isEnabled else {
            throw XCTSkip("Configure the packaged Native runtime before running this test")
        }
        let model = try await MeetingSortformerModelLease.load()
        let policy = try model.configuration.feedPolicy()
        let stream = OmniDiarizationStream(endpoint: model.endpoint, options: OmniDiarizationOptions(
            spkcacheMax: policy.cacheMaximumFrames, fifoMax: MeetingSpeakerFeedPolicy.fifoMaximumFrames
        ))
        var framesProcessed = 0
        let samples = (0..<policy.samplesPerFeed).map {
            Float(sin(Double($0) * 2 * .pi * 220 / 16_000)) * 0.1
        }
        let feedCount = (20 * 60 * 16_000 + samples.count - 1) / samples.count
        do {
            for index in 0..<feedCount {
                let feed = try await stream.feed(samples16k: samples)
                try policy.validate(fifoFrames: feed.fifoLength, cacheFrames: feed.spkcacheLength)
                XCTAssertGreaterThan(feed.framesProcessed, framesProcessed, "feed \(index)")
                framesProcessed = feed.framesProcessed
            }
        } catch {
            await stream.close()
            await model.close()
            throw error
        }
        await stream.close()
        await model.close()
        // Synthetic audio proves a memory-state invariant, not diarization quality.
        // Real multi-speaker recordings must separately validate DER and timestamps.
    }
}
