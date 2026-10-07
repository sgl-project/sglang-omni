import Foundation

/// A speaker's session and callback token move together. Draining sessions remain
/// owned and callback-valid until finish returns; cancellation invalidates first.
@MainActor
final class MeetingLiveSessionRegistry {
    struct Entry {
        let speaker: MeetingSpeaker
        let token: UUID
        let session: any MeetingLiveTranscribingSession
    }

    private var entries: [UUID: Entry] = [:]
    private var activeTokens: [MeetingSpeaker: UUID] = [:]
    private var currentTokens: [MeetingSpeaker: UUID] = [:]

    var isEmpty: Bool { entries.isEmpty }

    subscript(speaker: MeetingSpeaker) -> (any MeetingLiveTranscribingSession)? {
        guard let token = activeTokens[speaker] else { return nil }
        return entries[token]?.session
    }

    @discardableResult
    func insert(_ session: any MeetingLiveTranscribingSession, for speaker: MeetingSpeaker) -> Entry {
        let entry = Entry(speaker: speaker, token: UUID(), session: session)
        entries[entry.token] = entry
        activeTokens[speaker] = entry.token
        currentTokens[speaker] = entry.token
        return entry
    }

    func accepts(_ token: UUID, for speaker: MeetingSpeaker) -> Bool {
        currentTokens[speaker] == token && entries[token]?.speaker == speaker
    }

    func accepts(_ event: MeetingTranscriptEvent, token: UUID) -> Bool {
        let speaker: MeetingSpeaker
        switch event {
        case .partial(let segment), .final(let segment): speaker = segment.speaker
        case .failed(let value, _), .finished(let value): speaker = value
        }
        return accepts(token, for: speaker)
    }

    func beginFinishing(_ speaker: MeetingSpeaker) -> Entry? {
        guard let token = activeTokens.removeValue(forKey: speaker) else { return nil }
        return entries[token]
    }

    func beginFinishingAll() -> [Entry] {
        let result = activeTokens.values.compactMap { entries[$0] }
        activeTokens.removeAll()
        return result.sorted { $0.speaker.rawValue < $1.speaker.rawValue }
    }

    @discardableResult
    func remove(_ token: UUID) -> Entry? {
        guard let entry = entries.removeValue(forKey: token) else { return nil }
        if activeTokens[entry.speaker] == token { activeTokens[entry.speaker] = nil }
        if currentTokens[entry.speaker] == token { currentTokens[entry.speaker] = nil }
        return entry
    }

    func removeAll() -> [Entry] {
        let result = Array(entries.values)
        entries.removeAll()
        activeTokens.removeAll()
        currentTokens.removeAll()
        return result
    }
}
