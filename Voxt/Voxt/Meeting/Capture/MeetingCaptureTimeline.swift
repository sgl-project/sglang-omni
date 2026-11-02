// MeetingCaptureTimeline.swift
// Maintains sample-clock continuity across independent capture-source epochs.

import Foundation

nonisolated struct MeetingCaptureTimelineTracker: Sendable {
    private var generationBySpeaker: [MeetingSpeaker: UInt64] = [:]
    private var cursorBySpeaker: [MeetingSpeaker: TimeInterval] = [:]
    private var minimumStartBySpeaker: [MeetingSpeaker: TimeInterval] = [:]

    mutating func beginEpoch(for speaker: MeetingSpeaker) -> UInt64 {
        let generation = (generationBySpeaker[speaker] ?? 0) &+ 1
        generationBySpeaker[speaker] = generation
        cursorBySpeaker[speaker] = nil
        minimumStartBySpeaker[speaker] = nil
        return generation
    }

    func isCurrent(_ generation: UInt64, for speaker: MeetingSpeaker) -> Bool {
        generationBySpeaker[speaker] == generation
    }

    /// Cleanup invalidates callbacks immediately, even before the next source starts.
    mutating func invalidateEpochs() {
        for speaker in Array(generationBySpeaker.keys) {
            generationBySpeaker[speaker] = (generationBySpeaker[speaker] ?? 0) &+ 1
        }
        resetCursors()
    }

    mutating func anchorEpoch(
        for speaker: MeetingSpeaker,
        generation: UInt64,
        minimumStartSeconds: TimeInterval
    ) {
        guard generationBySpeaker[speaker] == generation else { return }
        minimumStartBySpeaker[speaker] = max(minimumStartSeconds, 0)
    }

    mutating func nextRange(
        for speaker: MeetingSpeaker,
        generation: UInt64,
        durationSeconds: TimeInterval,
        fallbackEndSeconds: TimeInterval
    ) -> Range<TimeInterval>? {
        guard generationBySpeaker[speaker] == generation else { return nil }
        let duration = max(durationSeconds, 0)
        let start = cursorBySpeaker[speaker] ?? max(
            minimumStartBySpeaker[speaker] ?? 0,
            max(fallbackEndSeconds - duration, 0)
        )
        let end = start + duration
        cursorBySpeaker[speaker] = end
        return start..<end
    }

    mutating func resetCursors() {
        cursorBySpeaker.removeAll(keepingCapacity: false)
        minimumStartBySpeaker.removeAll(keepingCapacity: false)
    }
}
