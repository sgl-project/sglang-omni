// MeetingSpeakerTurnLabeler.swift
// Provides Meeting Speaker Turn Labeler for meeting speaker analysis.

import Foundation

enum MeetingSpeakerTurnLabeler {
    nonisolated static func label(
        _ turns: [MeetingSpeakerTurn]
    ) -> [MeetingSpeakerTurn] {
        var orderedKeys: [String] = []
        var seen = Set<String>()
        for turn in turns.sorted(by: speakerSort) {
            let key = speakerIdentityKey(for: turn)
            guard !seen.contains(key) else { continue }
            seen.insert(key)
            orderedKeys.append(key)
        }

        let nameByKey = Dictionary(uniqueKeysWithValues: orderedKeys.enumerated().map { index, key in
            (key, MeetingSpeakerDisplayNameFormatter.displayName(ordinal: index + 1))
        })

        return turns.map { turn in
            let key = speakerIdentityKey(for: turn)
            return MeetingSpeakerTurn(
                id: turn.id,
                source: turn.source,
                speakerID: turn.speakerID,
                displayName: nameByKey[key] ?? turn.displayName,
                startSeconds: turn.startSeconds,
                endSeconds: turn.endSeconds,
                confidence: turn.confidence
            )
        }
    }

    nonisolated private static func speakerIdentityKey(for turn: MeetingSpeakerTurn) -> String {
        switch turn.source {
        case .mixed:
            return turn.speakerID
        case .microphone, .systemAudio:
            return "\(turn.source.rawValue):\(turn.speakerID)"
        }
    }

    nonisolated private static func speakerSort(_ lhs: MeetingSpeakerTurn, _ rhs: MeetingSpeakerTurn) -> Bool {
        if lhs.startSeconds == rhs.startSeconds {
            if lhs.source == rhs.source {
                return lhs.speakerID < rhs.speakerID
            }
            return lhs.source.rawValue < rhs.source.rawValue
        }
        return lhs.startSeconds < rhs.startSeconds
    }
}
