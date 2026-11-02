// MeetingSpeakerTurnSmoother.swift
// Provides Meeting Speaker Turn Smoother for meeting transcript processing.

import Foundation

enum MeetingSpeakerTurnSmoother {
    struct Options: Equatable, Sendable {
        var minimumTurnDurationSeconds: TimeInterval = 0.5
        var sameSpeakerMergeGapSeconds: TimeInterval = 0.3

        nonisolated init(
            minimumTurnDurationSeconds: TimeInterval = 0.5,
            sameSpeakerMergeGapSeconds: TimeInterval = 0.3
        ) {
            self.minimumTurnDurationSeconds = minimumTurnDurationSeconds
            self.sameSpeakerMergeGapSeconds = sameSpeakerMergeGapSeconds
        }
    }

    static func smooth(
        _ turns: [MeetingSpeakerTurn],
        options: Options = Options()
    ) -> [MeetingSpeakerTurn] {
        let sorted = turns
            .filter { $0.endSeconds > $0.startSeconds }
            .sorted { lhs, rhs in
                if lhs.source == rhs.source {
                    return lhs.startSeconds < rhs.startSeconds
                }
                return lhs.source.rawValue < rhs.source.rawValue
            }

        let grouped = Dictionary(grouping: sorted, by: \.source)
        return grouped
            .flatMap { _, sourceTurns in
                absorbIsolatedShortTurns(
                    in: mergeSameSpeakerTurns(sourceTurns, options: options),
                    options: options
                )
            }
            .sorted { lhs, rhs in
                if lhs.startSeconds == rhs.startSeconds {
                    return lhs.speakerID < rhs.speakerID
                }
                return lhs.startSeconds < rhs.startSeconds
            }
    }

    private static func mergeSameSpeakerTurns(
        _ turns: [MeetingSpeakerTurn],
        options: Options
    ) -> [MeetingSpeakerTurn] {
        var output: [MeetingSpeakerTurn] = []
        for turn in turns {
            if let previous = output.last,
               previous.source == turn.source,
               previous.speakerID == turn.speakerID,
               turn.startSeconds - previous.endSeconds <= options.sameSpeakerMergeGapSeconds {
                output[output.count - 1] = merged(previous, turn)
            } else {
                output.append(turn)
            }
        }
        return output
    }

    private static func absorbIsolatedShortTurns(
        in turns: [MeetingSpeakerTurn],
        options: Options
    ) -> [MeetingSpeakerTurn] {
        guard turns.count >= 3 else { return turns }
        var output: [MeetingSpeakerTurn] = []
        var index = 0
        while index < turns.count {
            if index > 0,
               index < turns.count - 1,
               let previous = output.last {
                let current = turns[index]
                let next = turns[index + 1]
                let currentDuration = current.endSeconds - current.startSeconds
                if currentDuration < options.minimumTurnDurationSeconds,
                   previous.source == current.source,
                   current.source == next.source,
                   previous.speakerID == next.speakerID {
                    output[output.count - 1] = merged(merged(previous, current, speakerOverride: previous), next)
                    index += 2
                    continue
                }
            }

            output.append(turns[index])
            index += 1
        }
        return output
    }

    private static func merged(
        _ lhs: MeetingSpeakerTurn,
        _ rhs: MeetingSpeakerTurn,
        speakerOverride: MeetingSpeakerTurn? = nil
    ) -> MeetingSpeakerTurn {
        let speaker = speakerOverride ?? lhs
        return MeetingSpeakerTurn(
            id: lhs.id,
            source: speaker.source,
            speakerID: speaker.speakerID,
            displayName: speaker.displayName,
            startSeconds: min(lhs.startSeconds, rhs.startSeconds),
            endSeconds: max(lhs.endSeconds, rhs.endSeconds),
            confidence: [lhs.confidence, rhs.confidence].compactMap { $0 }.max()
        )
    }
}
