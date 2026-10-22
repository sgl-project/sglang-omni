// OnboardingPracticeState.swift
// Correlates tutorial feedback with real sessions, never with keystrokes or edited text.

import Foundation

enum OnboardingPracticeKind: Equatable {
    case transcription, voiceTranslation, selectedTextTranslation
}

enum OnboardingSessionEvent {
    case started(id: UUID, kind: OnboardingPracticeKind, windowNumber: Int?)
    case processing(id: UUID)
    case delivered(id: UUID, text: String, succeeded: Bool)
    case ended(id: UUID, message: String)

    static let notification = Notification.Name("voxt.onboarding.sessionEvent")

    func post() {
        NotificationCenter.default.post(name: Self.notification, object: self)
    }
}

struct OnboardingPracticeState {
    enum Phase: Equatable {
        case ready, starting, listening, processing, succeeded, failed
    }

    private(set) var sessionID: UUID?
    private(set) var phase: Phase = .ready
    private(set) var result = ""
    private(set) var message = ""

    var isBusy: Bool { [.starting, .listening, .processing].contains(phase) }

    mutating func receive(
        _ event: OnboardingSessionEvent,
        expectedKind: OnboardingPracticeKind,
        windowNumber: Int?
    ) {
        switch event {
        case let .started(id, kind, sourceWindow):
            guard kind == expectedKind, let windowNumber, sourceWindow == windowNumber else { return }
            sessionID = id
            phase = kind == .selectedTextTranslation ? .processing : .starting
            result = ""
            message = ""
        case let .processing(id):
            guard sessionID == id, isBusy else { return }
            phase = .processing
        case let .delivered(id, text, succeeded):
            guard sessionID == id, isBusy else { return }
            result = text.trimmingCharacters(in: .whitespacesAndNewlines)
            phase = succeeded && !result.isEmpty ? .succeeded : .failed
        case let .ended(id, status):
            guard sessionID == id, isBusy else { return }
            phase = .failed
            message = status
        }
    }

    mutating func startedListening(sessionID: UUID) {
        guard self.sessionID == sessionID, phase == .starting else { return }
        phase = .listening
    }
}

/// Only explicit selections are committed. Loading or browsing the guide is read-only.
struct OnboardingModelDraft {
    var speech: FeatureModelSelectionID?
    var translation: FeatureModelSelectionID?

    var hasChanges: Bool { speech != nil || translation != nil }

    func applying(to current: FeatureSettings) -> FeatureSettings {
        var updated = current
        if let speech { updated.transcription.asrSelectionID = speech }
        if let translation { updated.translation.modelSelectionID = translation }
        return updated
    }
}
