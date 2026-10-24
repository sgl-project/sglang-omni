import Foundation

/// Sortformer AOSC removes at most spkcacheUpdatePeriod frames after each feed.
/// Feed no faster than it can retire history; a 60s IO window is NOT a feed size.
nonisolated struct MeetingSpeakerFeedPolicy: Sendable {
    static let fifoMaximumFrames = 188
    static let maximumCacheFrames = 4_096
    let samplesPerFeed: Int
    let sampleRate: Int
    let frameSamples: Int
    let cacheMaximumFrames: Int

    init(sampleRate: Int, hopLength: Int, subsamplingFactor: Int,
         chunkFrames: Int, cacheFrames: Int, updateFrames: Int, usesAOSC: Bool) throws {
        guard sampleRate == 16_000, hopLength > 0, hopLength <= sampleRate,
              subsamplingFactor > 0, subsamplingFactor <= 64,
              chunkFrames > 0, cacheFrames > 0, cacheFrames <= Self.maximumCacheFrames,
              !usesAOSC || updateFrames > 2 else {
            throw MeetingSpeakerFeedError.invalidConfiguration
        }
        let frameSamples = hopLength * subsamplingFactor
        let fiveSecondFrames = sampleRate * 5 / frameSamples
        // Leave headroom for feature-extractor edge padding.
        let frames = min(fiveSecondFrames, chunkFrames, usesAOSC ? updateFrames - 2 : chunkFrames)
        guard frames > 0 else { throw MeetingSpeakerFeedError.invalidConfiguration }
        self.samplesPerFeed = frames * frameSamples
        self.sampleRate = sampleRate
        self.frameSamples = frameSamples
        self.cacheMaximumFrames = cacheFrames
    }

    func mappedRange(start: Double, end: Double, stateFrames: Int,
                     audioOffset: TimeInterval, sampleCount: Int) -> Range<TimeInterval>? {
        guard start.isFinite, end.isFinite else { return nil }
        let stateOffset = Double(stateFrames * frameSamples) / Double(sampleRate)
        let duration = Double(sampleCount) / Double(sampleRate)
        let lower = audioOffset + min(duration, max(0, start - stateOffset))
        let upper = audioOffset + min(duration, max(0, end - stateOffset))
        return upper > lower ? lower..<upper : nil
    }

    func validateFeedDuration(_ duration: Duration) throws {
        // Cooperative stop AFTER feed returns; never abandon an in-flight Metal
        // operation or start a replacement task while native work is still active.
        guard duration <= .seconds(30) else { throw MeetingSpeakerFeedError.processingTooSlow }
    }

    func validate(fifoFrames: Int, cacheFrames: Int) throws {
        guard fifoFrames >= 0, fifoFrames <= Self.fifoMaximumFrames,
              cacheFrames >= 0, cacheFrames <= cacheMaximumFrames else {
            throw MeetingSpeakerFeedError.stateLimitExceeded
        }
    }
}

nonisolated enum MeetingSpeakerFeedError: LocalizedError {
    case invalidConfiguration
    case stateLimitExceeded
    case audioUnavailable
    case processingTooSlow

    var errorDescription: String? {
        switch self {
        case .invalidConfiguration:
            return AppLocalization.localizedString("The speaker model configuration cannot be processed safely.")
        case .stateLimitExceeded:
            return AppLocalization.localizedString("Speaker analysis stopped because its streaming state exceeded the safety limit. The transcription checkpoint is preserved.")
        case .audioUnavailable:
            return AppLocalization.localizedString("The speaker analysis audio window is unavailable.")
        case .processingTooSlow:
            return AppLocalization.localizedString("Speaker analysis stopped because a short audio block took too long. The transcription checkpoint is preserved; retry when system resources are available.")
        }
    }
}
