import Foundation
import Testing
@testable import VoxtMobileKit

/// Personal style examples: meaning-preserving before/after pairs the user explicitly keeps.
/// The local classifier only admits review candidates; it never claims the meaning was kept.
struct PersonalStyleTests {
    @Test func shorteningAndInsertingPolitenessWithinThePassageCanBeReviewed() {
        #expect(PersonalStyleExample.candidate(original: "Please send the file today.", edited: "Please send the file.") != nil)
        #expect(PersonalStyleExample.candidate(original: "Thanks for the file.", edited: "Many Thanks for the file.") != nil)
    }

    @Test func clearedOrChangedConsentCannotRestoreAnOldCandidate() throws {
        let (store, defaults, name) = makeStore()
        defer { defaults.removePersistentDomain(forName: name) }
        store.isEnabled = true
        let revision = store.consentRevision
        let example = PersonalStyleExample(before: "Thanks for the file.", after: "Many Thanks for the file.")
        store.clear()
        #expect(store.consentRevision != revision)
        #expect(throws: PersonalStyleStore.KeepError.staleConsent) { try store.keep(example, expectedRevision: revision) }
        let current = store.consentRevision
        let kept = try store.keep(example, expectedRevision: current)
        store.remove(kept.id)
        #expect(throws: PersonalStyleStore.KeepError.staleConsent) { try store.keep(example, expectedRevision: current) }
        let beforeToggle = store.consentRevision
        store.isEnabled = false
        store.isEnabled = true
        #expect(throws: PersonalStyleStore.KeepError.staleConsent) { try store.keep(example, expectedRevision: beforeToggle) }
        #expect(store.records.isEmpty)
    }
    // MARK: Classifier gates

    @Test func multiwordAndPunctuatedPhrasingEditsAreCandidates() {
        // Em dash variant: the changed span "Cheers —" contains whitespace.
        #expect(PersonalStyleExample.candidate(
            original: "Thanks, I'll send the file.",
            edited: "Cheers — I'll send the file."
        ) == PersonalStyleExample(before: "Thanks, I'll send the file.", after: "Cheers — I'll send the file."))
        #expect(PersonalStyleExample.candidate(original: "I would like to go", edited: "I want to go") != nil)
    }

    @Test func wordSpellingStaysOnTheVocabularyPath() {
        // No whitespace in the changed span: reserved for /v1/corrections.
        #expect(PersonalStyleExample.candidate(original: "Use gray today.", edited: "Use grey today.") == nil)
        #expect(PersonalStyleExample.candidate(original: "请用星浪处理。", edited: "请用星朗处理。") == nil)
        // Comma variant: "Thanks," → "Cheers," has no whitespace in the diff, so vocabulary keeps it.
        #expect(PersonalStyleExample.candidate(original: "Thanks, I'll send the file.", edited: "Cheers, I'll send the file.") == nil)
    }

    @Test func appendDeletionPunctuationAndDigitsAreNotCandidates() {
        #expect(PersonalStyleExample.candidate(original: "Use Voxt.", edited: "Use Voxt. Thanks!") == nil, "append-only")
        #expect(PersonalStyleExample.candidate(original: "Use Voxt today.", edited: "") == nil, "full deletion")
        #expect(PersonalStyleExample.candidate(original: "Use Voxt.", edited: "Use Voxt!") == nil, "punctuation only")
        #expect(PersonalStyleExample.candidate(original: "Pay $50.", edited: "Pay $60.") == nil, "digit change")
        #expect(PersonalStyleExample.candidate(original: "Pay $50.", edited: "Pay fifty dollars.") == nil, "digit disappears")
        #expect(PersonalStyleExample.candidate(original: "Pay fifty dollars.", edited: "Pay $50.") == nil, "digit appears")
    }

    @Test func separatedPunctuationEditsAroundUnchangedWordsAreNotCandidates() {
        // Two punctuation swaps with the same words in between: the merged changed span
        // contains letters, but the lexical content is identical — this is not wording.
        #expect(PersonalStyleExample.candidate(original: "Hello, world.", edited: "Hello — world!") == nil)
        #expect(PersonalStyleExample.candidate(original: "Yes, we can.", edited: "Yes — we can!") == nil)
        // Same shape, one real word swap: still a candidate.
        #expect(PersonalStyleExample.candidate(original: "Hello, brave world.", edited: "Hello — new world!") != nil)
    }

    @Test func boundsAndIdentityAreRejected() {
        let long = String(repeating: "a ", count: 100)
        #expect(PersonalStyleExample.candidate(original: long, edited: long + "x") == nil, "over 120 chars: nil for now, no trimming yet")
        #expect(PersonalStyleExample.candidate(original: "same", edited: "same") == nil)
        #expect(PersonalStyleExample.candidate(original: "", edited: "new") == nil)
        // Fact change without digits stays a review candidate — the gate must NOT claim to block it.
        #expect(PersonalStyleExample.candidate(original: "Meet on Friday.", edited: "Meet on Monday.") == nil, "single-word swap stays vocabulary-shaped")
        #expect(PersonalStyleExample.candidate(original: "I will come soon for sure.", edited: "I will come next year for sure.") != nil,
                "multiword fact change remains eligible; the reviewer discards it")
    }

    @Test func isValidEnforcesBoundsNonemptyDifferentAndEqualDigits() {
        #expect(PersonalStyleExample(before: "a b", after: "c d").isValid)
        #expect(!PersonalStyleExample(before: "", after: "c d").isValid)
        #expect(!PersonalStyleExample(before: "a b", after: "a b").isValid)
        #expect(!PersonalStyleExample(before: "call at 3", after: "call at 4").isValid)
        #expect(!PersonalStyleExample(before: String(repeating: "x", count: 121), after: "y z").isValid)
    }

    // MARK: Store

    private func makeStore() -> (PersonalStyleStore, UserDefaults, String) {
        let name = "personal-style.\(UUID().uuidString)"
        let defaults = UserDefaults(suiteName: name)!
        defaults.removePersistentDomain(forName: name)
        return (PersonalStyleStore(defaults: defaults), defaults, name)
    }

    @Test func defaultOffAndConsentRevisionTurnsWithTheSwitch() throws {
        let (store, defaults, name) = makeStore()
        defer { defaults.removePersistentDomain(forName: name) }
        #expect(!store.isEnabled)
        #expect(store.consentRevision.isEmpty)
        let example = PersonalStyleExample(before: "Thanks, I'll send the file.", after: "Cheers — I'll send the file.")
        #expect(throws: PersonalStyleStore.KeepError.self) { try store.keep(example) }
        store.isEnabled = true
        let first = store.consentRevision
        #expect(!first.isEmpty)
        store.isEnabled = false
        #expect(store.consentRevision != first)
    }

    @Test func keepPersistsUpToEightAndTheNinthFails() throws {
        let (store, defaults, name) = makeStore()
        defer { defaults.removePersistentDomain(forName: name) }
        store.isEnabled = true
        for index in 1...8 {
            try store.keep(PersonalStyleExample(before: "before \(index) x", after: "after \(index) x"))
        }
        #expect(store.records.count == 8)
        #expect(throws: PersonalStyleStore.KeepError.self) {
            try store.keep(PersonalStyleExample(before: "before 9 x", after: "after 9 x"))
        }
        #expect(store.records.count == 8, "the 9th Keep fails; no silent eviction")
        // Persistence: a fresh store on the same suite sees the same records.
        #expect(PersonalStyleStore(defaults: defaults).records.count == 8)
        // Invalid examples never persist.
        #expect(throws: PersonalStyleStore.KeepError.self) {
            try store.keep(PersonalStyleExample(before: "same", after: "same"))
        }
    }

    @Test func removeDeletesOnlyThatRecordAndClearEmpties() throws {
        let (store, defaults, name) = makeStore()
        defer { defaults.removePersistentDomain(forName: name) }
        store.isEnabled = true
        let first = try store.keep(PersonalStyleExample(before: "a b", after: "c d"))
        let second = try store.keep(PersonalStyleExample(before: "e f", after: "g h"))
        store.remove(first.id)
        #expect(store.records.map(\.id) == [second.id])
        store.clear()
        #expect(store.records.isEmpty)
    }

    @Test func examplesFollowSwitchAndPrecedence() throws {
        let (store, defaults, name) = makeStore()
        defer { defaults.removePersistentDomain(forName: name) }
        let example = PersonalStyleExample(before: "Thanks, I'll send the file.", after: "Cheers — I'll send the file.")
        store.isEnabled = true
        try store.keep(example)
        #expect(store.examples(for: .dictate, style: .automatic) == [example])
        #expect(store.examples(for: .edit, style: .automatic) == [example])
        #expect(store.examples(for: .write, style: .automatic) == [example])
        #expect(store.examples(for: .translate, style: .automatic) == [], "translate never carries examples")
        #expect(store.examples(for: .dictate, style: .formal) == [], "explicit style omits examples")
        store.isEnabled = false
        #expect(store.examples(for: .dictate, style: .automatic) == [], "off sends nothing…")
        #expect(store.records.count == 1, "…but active records stay on the phone")
    }

    @Test func duplicatePairsAreRejected() throws {
        let (store, defaults, name) = makeStore()
        defer { defaults.removePersistentDomain(forName: name) }
        store.isEnabled = true
        let example = PersonalStyleExample(before: "Thanks, I'll send the file.", after: "Cheers — I'll send the file.")
        try store.keep(example)
        #expect(throws: PersonalStyleStore.KeepError.self) { try store.keep(example) }
        #expect(store.records.count == 1)
    }

    // MARK: Routing: style candidate first, else vocabulary-shaped edits keep the old path

    @Test func vocabularyCandidateMatchesTheOldObserveGate() {
        let grayEdit = CorrectionLearningRequest(original: "Use gray today.", edited: "Use grey today.")
        #expect(grayEdit.isVocabularyCandidate)
        #expect(PersonalStyleExample.candidate(original: grayEdit.original, edited: grayEdit.edited) == nil)

        let cheers = CorrectionLearningRequest(original: "Thanks, I'll send the file.", edited: "Cheers — I'll send the file.")
        #expect(!cheers.isVocabularyCandidate, "the extra em dash changes word count, so the full vocabulary gate rejects it despite detect returning a pair")
        #expect(PersonalStyleExample.candidate(original: cheers.original, edited: cheers.edited) != nil)

        let append = CorrectionLearningRequest(original: "Use Voxt.", edited: "Use Voxt. Thanks!")
        #expect(!append.isVocabularyCandidate)
        #expect(PersonalStyleExample.candidate(original: append.original, edited: append.edited) == nil)

        // detect nil (>4 words) but style-eligible: candidate covers what vocabulary cannot.
        let clause = CorrectionLearningRequest(
            original: "Thank you so much for everything, Jack",
            edited: "Cheers, Jack"
        )
        #expect(!clause.isVocabularyCandidate)
        #expect(PersonalStyleExample.candidate(original: clause.original, edited: clause.edited) != nil)
    }
}

/// observeEdit: the generalized observer returns the stable bounded raw edit; observe keeps
/// the old word-only shaping as a wrapper. Both share field/boundary/caret/expiry guards.
struct ObserveEditTests {
    private let id = UUID()
    private let start = Date(timeIntervalSince1970: 1000)

    @Test func rawEditNeverExceedsThePassageBound() {
        var session = CorrectionLearningSession(documentID: id, before: "", inserted: "Short text.", after: "", now: start)
        let oversized = String(repeating: "x", count: 2_001) + "."
        _ = session.observeEdit(documentID: id, before: oversized, after: "", selectedText: "", now: start)
        #expect(session.observeEdit(documentID: id, before: oversized, after: "", selectedText: "", now: start.addingTimeInterval(2)) == nil)
    }

    @Test func observeEditReturnsRawEditThatObserveWouldRejectAsAppend() {
        var session = CorrectionLearningSession(documentID: id, before: "", inserted: "Use Voice.", after: "", now: start)
        #expect(session.observeEdit(documentID: id, before: "Use Voice. More words.", after: "", selectedText: "", now: start) == nil)
        let request = session.observeEdit(documentID: id, before: "Use Voice. More words.", after: "", selectedText: "", now: start.addingTimeInterval(2))
        #expect(request == CorrectionLearningRequest(original: "Use Voice.", edited: "Use Voice. More words."))
    }

    @Test func observeStillRejectsTheSameEditAndEndsTheSession() {
        var session = CorrectionLearningSession(documentID: id, before: "", inserted: "Use Voice.", after: "", now: start)
        _ = session.observe(documentID: id, before: "Use Voice. More words.", after: "", selectedText: "", now: start)
        #expect(session.observe(documentID: id, before: "Use Voice. More words.", after: "", selectedText: "", now: start.addingTimeInterval(2)) == nil)
        // Session is dead: even a later valid correction produces nothing.
        _ = session.observe(documentID: id, before: "Use Voxt. More words.", after: "", selectedText: "", now: start.addingTimeInterval(3))
        #expect(session.observe(documentID: id, before: "Use Voxt. More words.", after: "", selectedText: "", now: start.addingTimeInterval(5)) == nil)
    }

    @Test func observeEditKeepsFieldSelectionCaretAndExpiryProtections() {
        // Wrong field.
        var other = CorrectionLearningSession(documentID: id, before: "prefix ", inserted: "Voice", after: " suffix", now: start)
        _ = other.observeEdit(documentID: UUID(), before: "prefix Voxt", after: " suffix", selectedText: "", now: start)
        #expect(other.observeEdit(documentID: UUID(), before: "prefix Voxt", after: " suffix", selectedText: "", now: start.addingTimeInterval(2)) == nil)
        // Active selection.
        var selected = CorrectionLearningSession(documentID: id, before: "prefix ", inserted: "Voice", after: " suffix", now: start)
        _ = selected.observeEdit(documentID: id, before: "prefix Vox", after: " suffix", selectedText: "t", now: start)
        #expect(selected.observeEdit(documentID: id, before: "prefix Vox", after: " suffix", selectedText: "t", now: start.addingTimeInterval(2)) == nil)
        // Caret still inside word characters: stability never starts.
        var caret = CorrectionLearningSession(documentID: id, before: "", inserted: "Use Voice.", after: "", now: start)
        for t in [0.0, 2.0, 5.0] {
            #expect(caret.observeEdit(documentID: id, before: "Use Voxt", after: ".", selectedText: "", now: start.addingTimeInterval(t)) == nil)
        }
        // Expiry after two minutes.
        var expired = CorrectionLearningSession(documentID: id, before: "", inserted: "Use Voice.", after: "", now: start)
        _ = expired.observeEdit(documentID: id, before: "Use Voxt.", after: "", selectedText: "", now: start.addingTimeInterval(119))
        #expect(expired.observeEdit(documentID: id, before: "Use Voxt.", after: "", selectedText: "", now: start.addingTimeInterval(121)) == nil)
    }

    @Test func observeEditAdvancesTheBaselineSoChainedEditsStayComparable() {
        var session = CorrectionLearningSession(documentID: id, before: "", inserted: "Use Voice.", after: "", now: start)
        _ = session.observeEdit(documentID: id, before: "Use Voice. More words.", after: "", selectedText: "", now: start)
        let first = session.observeEdit(documentID: id, before: "Use Voice. More words.", after: "", selectedText: "", now: start.addingTimeInterval(2))
        #expect(first != nil)
        _ = session.observeEdit(documentID: id, before: "Use Voxt. More words.", after: "", selectedText: "", now: start.addingTimeInterval(3))
        let second = session.observeEdit(documentID: id, before: "Use Voxt. More words.", after: "", selectedText: "", now: start.addingTimeInterval(5))
        #expect(second == CorrectionLearningRequest(original: "Use Voice. More words.", edited: "Use Voxt. More words."),
                "second edit is measured against the accepted intermediate text")
    }
}
