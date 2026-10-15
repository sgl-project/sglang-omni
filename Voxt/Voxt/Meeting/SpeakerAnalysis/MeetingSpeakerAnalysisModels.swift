// MeetingSpeakerAnalysisModels.swift
// Provides Meeting Speaker Analysis Models for meeting speaker analysis.

import Foundation

struct MeetingAudioAsset: Sendable {
    let source: TranscriptAudioSource
    let samples: [Float]
    let sampleRate: Double
    let sessionStartOffset: TimeInterval

    nonisolated var durationSeconds: TimeInterval {
        guard sampleRate > 0 else { return 0 }
        return TimeInterval(samples.count) / sampleRate
    }
}

struct MeetingAudioAssetDescriptor: Equatable, Sendable {
    let source: TranscriptAudioSource
    let sampleRate: Double
    let startSample: Int
    let sampleCount: Int

    nonisolated var sessionStartOffset: TimeInterval {
        guard sampleRate > 0 else { return 0 }
        return TimeInterval(startSample) / sampleRate
    }

    nonisolated var durationSeconds: TimeInterval {
        guard sampleRate > 0 else { return 0 }
        return TimeInterval(sampleCount) / sampleRate
    }
}

struct MeetingSpeakerTurn: Identifiable, Codable, Equatable, Sendable {
    let id: UUID
    let source: TranscriptAudioSource
    let speakerID: String
    let displayName: String
    let startSeconds: TimeInterval
    let endSeconds: TimeInterval
    let confidence: Double?

    nonisolated init(
        id: UUID = UUID(),
        source: TranscriptAudioSource,
        speakerID: String,
        displayName: String,
        startSeconds: TimeInterval,
        endSeconds: TimeInterval,
        confidence: Double? = nil
    ) {
        self.id = id
        self.source = source
        self.speakerID = speakerID
        self.displayName = displayName
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
        self.confidence = confidence
    }
}
