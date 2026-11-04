// MeetingLocalInferencePolicy.swift
// Defines admission priorities, resource deferrals, and coordinator results.

import Foundation

nonisolated enum MeetingLocalInferenceWorkClass: String, Sendable {
    case liveASRFeed
    case liveASRFinal
    case liveASRPartial
    case realtimeTranslation
    case fileASR
    case fileSpeakerAnalysis
    case finalASR
    case speakerAnalysis
    case detailTranslation
    case summary

    var priority: Int {
        switch self {
        case .liveASRFeed: return 100
        case .liveASRFinal: return 95
        case .liveASRPartial: return 85
        case .realtimeTranslation: return 70
        case .fileASR, .fileSpeakerAnalysis: return 10
        case .finalASR: return 60
        case .speakerAnalysis: return 50
        case .detailTranslation: return 40
        case .summary: return 20
        }
    }

    var isFileWork: Bool {
        self == .fileASR || self == .fileSpeakerAnalysis
    }

    var waitsWhileRecording: Bool {
        switch self {
        case .fileASR, .fileSpeakerAnalysis, .finalASR, .speakerAnalysis, .detailTranslation, .summary:
            return true
        case .liveASRFeed, .liveASRFinal, .liveASRPartial, .realtimeTranslation:
            return false
        }
    }

    var isThermallyDeferrable: Bool {
        switch self {
        case .liveASRFeed, .liveASRFinal, .finalASR:
            return false
        case .liveASRPartial, .realtimeTranslation, .fileASR, .fileSpeakerAnalysis, .speakerAnalysis, .detailTranslation, .summary:
            return true
        }
    }

    var isMemoryDeferrable: Bool {
        switch self {
        case .liveASRFeed, .liveASRFinal, .finalASR:
            return false
        case .liveASRPartial, .realtimeTranslation, .fileASR, .fileSpeakerAnalysis, .speakerAnalysis, .detailTranslation, .summary:
            return true
        }
    }
}

nonisolated struct MeetingLocalInferenceStatistics: Equatable, Sendable {
    var submittedCount = 0
    var completedCount = 0
    var cancelledCount = 0
    var overloadedCount = 0
    var thermalDeferralCount = 0
    var memoryDeferralCount = 0
    var peakQueuedCount = 0
    var totalWaitMilliseconds: Int64 = 0
}

nonisolated enum MeetingLocalInferenceCoordinatorError: LocalizedError, Sendable {
    case overloaded
    case thermallyConstrained
    case memoryConstrained

    var errorDescription: String? {
        switch self {
        case .overloaded:
            return "The local meeting inference queue reached its safety limit."
        case .thermallyConstrained:
            return "The Mac is too warm for this nonessential local inference task right now."
        case .memoryConstrained:
            return "The Mac is under memory pressure, so this nonessential local inference task was deferred."
        }
    }
}
