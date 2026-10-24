// TranscriptModels.swift
// Provides Transcript Models for core app behavior.

import Foundation

enum TranscriptSpeaker: String, Codable, Hashable, Sendable {
    case me
    case them

    nonisolated var displayTitle: String {
        switch self {
        case .me:
            return "Me"
        case .them:
            return "Them"
        }
    }
}

enum TranscriptAudioSource: String, Codable, Hashable, Sendable {
    case microphone
    case systemAudio
    case mixed

    nonisolated var defaultSpeaker: TranscriptSpeaker {
        switch self {
        case .microphone:
            return .me
        case .systemAudio, .mixed:
            return .them
        }
    }
}

struct TranscriptSegment: Identifiable, Codable, Hashable, Sendable {
    let id: UUID
    let speaker: TranscriptSpeaker
    let speakerID: String?
    let speakerDisplayName: String?
    let audioSource: TranscriptAudioSource?
    let speakerConfidence: Double?
    let startSeconds: TimeInterval
    let endSeconds: TimeInterval?
    let text: String
    let translatedText: String?
    let isTranslationPending: Bool
    let preventsAdjacentMerge: Bool
    let isHighlighted: Bool

    nonisolated init(
        id: UUID = UUID(),
        speaker: TranscriptSpeaker,
        speakerID: String? = nil,
        speakerDisplayName: String? = nil,
        audioSource: TranscriptAudioSource? = nil,
        speakerConfidence: Double? = nil,
        startSeconds: TimeInterval,
        endSeconds: TimeInterval?,
        text: String,
        translatedText: String? = nil,
        isTranslationPending: Bool = false,
        preventsAdjacentMerge: Bool = false,
        isHighlighted: Bool = false
    ) {
        self.id = id
        self.speaker = speaker
        self.speakerID = speakerID
        self.speakerDisplayName = speakerDisplayName
        self.audioSource = audioSource
        self.speakerConfidence = speakerConfidence
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
        self.text = text
        self.translatedText = translatedText
        self.isTranslationPending = isTranslationPending
        self.preventsAdjacentMerge = preventsAdjacentMerge
        self.isHighlighted = isHighlighted
    }

    nonisolated func updatingTranslation(
        translatedText: String?,
        isTranslationPending: Bool
    ) -> TranscriptSegment {
        TranscriptSegment(
            id: id,
            speaker: speaker,
            speakerID: speakerID,
            speakerDisplayName: speakerDisplayName,
            audioSource: audioSource,
            speakerConfidence: speakerConfidence,
            startSeconds: startSeconds,
            endSeconds: endSeconds,
            text: text,
            translatedText: translatedText,
            isTranslationPending: isTranslationPending,
            preventsAdjacentMerge: preventsAdjacentMerge,
            isHighlighted: isHighlighted
        )
    }

    nonisolated func updatingText(
        _ text: String,
        translatedText: String? = nil,
        isTranslationPending: Bool = false
    ) -> TranscriptSegment {
        TranscriptSegment(
            id: id,
            speaker: speaker,
            speakerID: speakerID,
            speakerDisplayName: speakerDisplayName,
            audioSource: audioSource,
            speakerConfidence: speakerConfidence,
            startSeconds: startSeconds,
            endSeconds: endSeconds,
            text: text,
            translatedText: translatedText,
            isTranslationPending: isTranslationPending,
            preventsAdjacentMerge: preventsAdjacentMerge,
            isHighlighted: isHighlighted
        )
    }

    nonisolated func updatingHighlight(_ isHighlighted: Bool) -> TranscriptSegment {
        TranscriptSegment(
            id: id,
            speaker: speaker,
            speakerID: speakerID,
            speakerDisplayName: speakerDisplayName,
            audioSource: audioSource,
            speakerConfidence: speakerConfidence,
            startSeconds: startSeconds,
            endSeconds: endSeconds,
            text: text,
            translatedText: translatedText,
            isTranslationPending: isTranslationPending,
            preventsAdjacentMerge: preventsAdjacentMerge,
            isHighlighted: isHighlighted
        )
    }

    nonisolated func updatingSpeakerAnalysis(
        speaker: TranscriptSpeaker? = nil,
        speakerID: String?,
        speakerDisplayName: String?,
        audioSource: TranscriptAudioSource?,
        speakerConfidence: Double?
    ) -> TranscriptSegment {
        TranscriptSegment(
            id: id,
            speaker: speaker ?? self.speaker,
            speakerID: speakerID,
            speakerDisplayName: speakerDisplayName,
            audioSource: audioSource ?? self.audioSource,
            speakerConfidence: speakerConfidence,
            startSeconds: startSeconds,
            endSeconds: endSeconds,
            text: text,
            translatedText: translatedText,
            isTranslationPending: isTranslationPending,
            preventsAdjacentMerge: preventsAdjacentMerge,
            isHighlighted: isHighlighted
        )
    }

    nonisolated func updatingSpeakerDisplayName(_ displayName: String?) -> TranscriptSegment {
        TranscriptSegment(
            id: id,
            speaker: speaker,
            speakerID: speakerID,
            speakerDisplayName: displayName,
            audioSource: audioSource,
            speakerConfidence: speakerConfidence,
            startSeconds: startSeconds,
            endSeconds: endSeconds,
            text: text,
            translatedText: translatedText,
            isTranslationPending: isTranslationPending,
            preventsAdjacentMerge: preventsAdjacentMerge,
            isHighlighted: isHighlighted
        )
    }

    nonisolated var displaySpeakerTitle: String {
        let trimmedName = speakerDisplayName?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let trimmedName, !trimmedName.isEmpty {
            return trimmedName
        }
        return speaker.displayTitle
    }

    nonisolated var speakerIdentityKey: String {
        if let speakerID, !speakerID.isEmpty {
            return "\(audioSource?.rawValue ?? "unknown"):\(speakerID)"
        }
        return "speaker:\(speaker.rawValue)"
    }

    enum CodingKeys: String, CodingKey {
        case id
        case speaker
        case speakerID
        case speakerDisplayName
        case audioSource
        case speakerConfidence
        case startSeconds
        case endSeconds
        case text
        case translatedText
        case isTranslationPending
        case isHighlighted
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        speaker = try container.decode(TranscriptSpeaker.self, forKey: .speaker)
        speakerID = try container.decodeIfPresent(String.self, forKey: .speakerID)
        speakerDisplayName = try container.decodeIfPresent(String.self, forKey: .speakerDisplayName)
        audioSource = try container.decodeIfPresent(TranscriptAudioSource.self, forKey: .audioSource)
        speakerConfidence = try container.decodeIfPresent(Double.self, forKey: .speakerConfidence)
        startSeconds = try container.decode(TimeInterval.self, forKey: .startSeconds)
        endSeconds = try container.decodeIfPresent(TimeInterval.self, forKey: .endSeconds)
        text = try container.decode(String.self, forKey: .text)
        translatedText = try container.decodeIfPresent(String.self, forKey: .translatedText)
        isTranslationPending = try container.decodeIfPresent(Bool.self, forKey: .isTranslationPending) ?? false
        isHighlighted = try container.decodeIfPresent(Bool.self, forKey: .isHighlighted) ?? false
        preventsAdjacentMerge = false
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(speaker, forKey: .speaker)
        try container.encodeIfPresent(speakerID, forKey: .speakerID)
        try container.encodeIfPresent(speakerDisplayName, forKey: .speakerDisplayName)
        try container.encodeIfPresent(audioSource, forKey: .audioSource)
        try container.encodeIfPresent(speakerConfidence, forKey: .speakerConfidence)
        try container.encode(startSeconds, forKey: .startSeconds)
        try container.encodeIfPresent(endSeconds, forKey: .endSeconds)
        try container.encode(text, forKey: .text)
        try container.encodeIfPresent(translatedText, forKey: .translatedText)
        try container.encode(isTranslationPending, forKey: .isTranslationPending)
        try container.encode(isHighlighted, forKey: .isHighlighted)
    }
}

enum TranscriptFormatter {
    nonisolated static func meaningfulSegments(for segments: [TranscriptSegment]) -> [TranscriptSegment] {
        segments.filter { segment in
            let hasOriginalText = isMeaningfulText(segment.text)
            let hasTranslatedText = isMeaningfulText(segment.translatedText)
            return hasOriginalText || hasTranslatedText
        }
    }

    nonisolated static func mergedSegmentsForPersistence(
        primarySegments: [TranscriptSegment],
        fallbackSegments: [TranscriptSegment]
    ) -> [TranscriptSegment] {
        var mergedByID: [UUID: TranscriptSegment] = [:]

        for segment in meaningfulSegments(for: fallbackSegments) {
            mergedByID[segment.id] = segment
        }

        for segment in meaningfulSegments(for: primarySegments) {
            if let existing = mergedByID[segment.id] {
                mergedByID[segment.id] = mergedSegment(preferred: segment, fallback: existing)
            } else {
                mergedByID[segment.id] = segment
            }
        }

        let sorted = mergedByID.values.sorted { lhs, rhs in
            if lhs.startSeconds == rhs.startSeconds {
                return lhs.id.uuidString < rhs.id.uuidString
            }
            return lhs.startSeconds < rhs.startSeconds
        }
        return mergedAdjacentSegments(in: sorted)
    }

    nonisolated static func timestampString(for seconds: TimeInterval) -> String {
        let totalSeconds = max(Int(seconds.rounded(.down)), 0)
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let remainder = totalSeconds % 60
        if hours > 0 {
            return String(format: "%02d:%02d:%02d", hours, minutes, remainder)
        }
        return String(format: "%02d:%02d", minutes, remainder)
    }

    nonisolated static func copyString(for segment: TranscriptSegment) -> String {
        exportString(for: segment)
    }

    nonisolated static func joinedText(for segments: [TranscriptSegment]) -> String {
        segments
            .map { exportString(for: $0) }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    nonisolated static func llmInputText(for segments: [TranscriptSegment]) -> String {
        segments
            .compactMap { segment in
                let text = segment.text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty else { return nil }
                return "\(segment.displaySpeakerTitle)：\(text)"
            }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    nonisolated static func exportString(for segment: TranscriptSegment) -> String {
        var lines = ["\(timestampString(for: segment.startSeconds)) \(segment.displaySpeakerTitle) \(segment.text)"]
        if let translatedText = segment.translatedText?.trimmingCharacters(in: .whitespacesAndNewlines),
           !translatedText.isEmpty {
            lines.append("   -> \(translatedText)")
        }
        return lines.joined(separator: "\n")
    }

    nonisolated static func mergedAdjacentSegment(
        previous: TranscriptSegment,
        next: TranscriptSegment
    ) -> TranscriptSegment? {
        guard !previous.preventsAdjacentMerge, !next.preventsAdjacentMerge else { return nil }
        guard previous.speakerIdentityKey == next.speakerIdentityKey else { return nil }
        guard next.startSeconds >= previous.startSeconds else { return nil }
        let previousEnd = previous.endSeconds ?? previous.startSeconds
        guard next.startSeconds - previousEnd <= 2.0 else { return nil }

        return TranscriptSegment(
            id: previous.id,
            speaker: previous.speaker,
            speakerID: previous.speakerID,
            speakerDisplayName: previous.speakerDisplayName,
            audioSource: previous.audioSource,
            speakerConfidence: [previous.speakerConfidence, next.speakerConfidence]
                .compactMap { $0 }
                .max(),
            startSeconds: previous.startSeconds,
            endSeconds: max(previousEnd, next.endSeconds ?? next.startSeconds),
            text: mergedText(previous.text, next.text),
            translatedText: nil,
            isTranslationPending: false,
            preventsAdjacentMerge: false,
            isHighlighted: previous.isHighlighted || next.isHighlighted
        )
    }

    private nonisolated static func mergedAdjacentSegments(
        in segments: [TranscriptSegment]
    ) -> [TranscriptSegment] {
        var merged: [TranscriptSegment] = []
        for segment in segments {
            if let last = merged.last,
               let mergedSegment = mergedAdjacentSegment(previous: last, next: segment) {
                merged[merged.count - 1] = mergedSegment
            } else {
                merged.append(segment)
            }
        }
        return merged
    }

    private nonisolated static func mergedSegment(
        preferred: TranscriptSegment,
        fallback: TranscriptSegment
    ) -> TranscriptSegment {
        let preferredText = preferred.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let fallbackText = fallback.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let translatedText = preferred.translatedText?.trimmingCharacters(in: .whitespacesAndNewlines)
        let fallbackTranslatedText = fallback.translatedText?.trimmingCharacters(in: .whitespacesAndNewlines)

        return TranscriptSegment(
            id: preferred.id,
            speaker: preferred.speaker,
            speakerID: preferred.speakerID ?? fallback.speakerID,
            speakerDisplayName: preferred.speakerDisplayName ?? fallback.speakerDisplayName,
            audioSource: preferred.audioSource ?? fallback.audioSource,
            speakerConfidence: preferred.speakerConfidence ?? fallback.speakerConfidence,
            startSeconds: preferred.startSeconds,
            endSeconds: preferred.endSeconds ?? fallback.endSeconds,
            text: preferredText.isEmpty ? fallbackText : preferredText,
            translatedText: (translatedText?.isEmpty == false ? translatedText : fallbackTranslatedText),
            isTranslationPending: preferred.isTranslationPending && (translatedText?.isEmpty ?? true),
            preventsAdjacentMerge: preferred.preventsAdjacentMerge || fallback.preventsAdjacentMerge,
            isHighlighted: preferred.isHighlighted || fallback.isHighlighted
        )
    }

    private nonisolated static func mergedText(_ lhs: String, _ rhs: String) -> String {
        let left = lhs.trimmingCharacters(in: .whitespacesAndNewlines)
        let right = rhs.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !left.isEmpty else { return right }
        guard !right.isEmpty else { return left }

        let leftLast = left.unicodeScalars.last
        let rightFirst = right.unicodeScalars.first
        let separator = needsInlineSeparator(leftLast: leftLast, rightFirst: rightFirst) ? " " : ""
        return left + separator + right
    }

    private nonisolated static func needsInlineSeparator(
        leftLast: UnicodeScalar?,
        rightFirst: UnicodeScalar?
    ) -> Bool {
        guard let leftLast, let rightFirst else { return true }
        let punctuationScalars = CharacterSet(charactersIn: " \t\n\r,.!?;:，。！？；：、)]}\"'》】）")
        if punctuationScalars.contains(leftLast) || punctuationScalars.contains(rightFirst) {
            return false
        }
        let alphanumerics = CharacterSet.alphanumerics
        return alphanumerics.contains(leftLast) && alphanumerics.contains(rightFirst)
    }

    private nonisolated static func isMeaningfulText(_ value: String?) -> Bool {
        guard let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty
        else {
            return false
        }
        if RemoteASRTextSanitizer.isLikelyIdentifierText(trimmed) {
            return false
        }
        return true
    }
}
