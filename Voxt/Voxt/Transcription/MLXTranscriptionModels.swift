// Value types shared by planning, dictation, replay diagnostics, and meetings.

import Foundation

struct MLXIntermediateCorrectionDecision: Equatable {
    let elapsedSeconds: Double
    let contextSampleCount: Int
}

struct MLXCorrectionCadence: Equatable {
    let correctionIntervalSeconds: Double
    let firstCorrectionMinimumSeconds: Double
    let intermediateContextWindowSeconds: Double
    let quickPassContextWindowSeconds: Double
}

struct MLXSequentialTranscriptMergeResult: Equatable {
    let text: String
    let overlapCount: Int
}

struct MLXRealtimeReplayEvent: Equatable {
    let elapsedSeconds: Double
    let text: String
    let isFinal: Bool
    let source: String
}

struct MLXRealtimeReplayDiagnostics: Equatable {
    let events: [MLXRealtimeReplayEvent]
    let trace: [String]
}

enum MLXTranscriptionPurpose: Sendable {
    case dictation
    case meeting

    var mossUsageScope: MossASRUsageScope {
        switch self {
        case .dictation: .dictation
        case .meeting: .meeting
        }
    }
}

struct MLXStructuredTranscriptSegment: Equatable, Sendable {
    let startSeconds: TimeInterval
    let endSeconds: TimeInterval
    let speakerID: String?
    let text: String

    nonisolated init(
        startSeconds: TimeInterval,
        endSeconds: TimeInterval,
        speakerID: String? = nil,
        text: String
    ) {
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
        self.speakerID = speakerID
        self.text = text
    }
}

struct MLXBufferedTranscriptionResult: Equatable, Sendable {
    let text: String
    let structuredSegments: [MLXStructuredTranscriptSegment]
}

struct MLXFinalizationSampleSelection: Equatable {
    let samples: [Float]
    let source: Source

    enum Source: Equatable {
        case full
        case voiceActivityFiltered
        case noSpeech

        var telemetryName: String {
            switch self {
            case .full:
                return "full"
            case .voiceActivityFiltered:
                return "voice-activity-filtered"
            case .noSpeech:
                return "no-speech"
            }
        }
    }
}

struct MLXFinalizationPlan: Equatable {
    let durationSeconds: Double
    let quickPassSampleCount: Int?

    var shouldRunQuickPass: Bool {
        quickPassSampleCount != nil
    }
}

enum MLXCorrectionPassKind: Equatable {
    case intermediate
    case postStopQuick
    case postStopFinal
}

enum MLXCorrectionPassSchedulingDecision: Equatable {
    case startImmediately
    case waitForInFlightPass
    case skipRequestedPass
    case interruptInFlightPass
}
