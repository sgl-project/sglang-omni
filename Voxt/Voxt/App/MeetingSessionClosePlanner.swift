// MeetingSessionClosePlanner.swift
// Resolves whether an active meeting session needs an explicit finish decision.

import Foundation

nonisolated enum MeetingSessionCloseDecision: Equatable, Sendable {
    case discard
    case confirmFinish
}

nonisolated enum MeetingSessionClosePlanner {
    static func resolve(
        hasTranscriptSegments: Bool,
        hasCapturedAudio: Bool
    ) -> MeetingSessionCloseDecision {
        hasTranscriptSegments || hasCapturedAudio ? .confirmFinish : .discard
    }
}
