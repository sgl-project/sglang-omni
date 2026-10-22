import Foundation
import Testing
@testable import VoxtMobileKit

/// Regression tests for correction learning.
/// The first two cover a bug fixed in CorrectionLearning.swift: a completed
/// correction followed by continued typing merged into one candidate. The third documents the
/// deliberate boundary rule (caret must leave word characters, including CJK, until
/// an IME completion signal exists) and passes.
///
/// Out of scope: the cross-process dictionary lost-write risk lacked a
/// deterministic reproduction and is reported as analysis only.
struct CorrectionLearningReviewTests {
    /// Dictation inserts "Use Voice." → user fixes Voice→Voxt → keeps typing " more."
    /// The caret rule blocks submission while the caret is on letters, so the stable diff
    /// spans correction + append. The append must be ignored, not merged into the request.
    @Test func correctionFollowedByContinuedTypingMustNotMergeIntoOneCandidate() {
        let id = UUID(), start = Date(timeIntervalSince1970: 1000)
        var session = CorrectionLearningSession(documentID: id, before: "", inserted: "Use Voice.", after: "", now: start)
        // Caret on letters while fixing and typing: blocked, pending keeps resetting.
        #expect(session.observe(documentID: id, before: "Use Voxt", after: ".", selectedText: "", now: start) == nil)
        #expect(session.observe(documentID: id, before: "Use Voxt more", after: ".", selectedText: "", now: start.addingTimeInterval(1)) == nil)
        // Caret reaches the period; the merged diff goes stable.
        #expect(session.observe(documentID: id, before: "Use Voxt more.", after: "", selectedText: "", now: start.addingTimeInterval(2)) == nil)
        let request = session.observe(documentID: id, before: "Use Voxt more.", after: "", selectedText: "", now: start.addingTimeInterval(4))
        #expect(request == nil, "correction + later typing must not merge into one learning request")
    }

    /// Defense in depth: a replacement that swallows appended words ("Voice" → "Voxt more")
    /// must never enter the dictionary.
    @Test func validationMustRejectReplacementCoveringAppendedWords() {
        let request = CorrectionLearningRequest(original: "Use Voice.", edited: "Use Voxt more.")
        #expect(request.validated(.init(original: "Voice", replacement: "Voxt more")) == nil,
                "a replacement covering the correction plus appended text pollutes the dictionary")
    }

    /// Deliberate boundary rule, not expected support: learning only starts once the caret
    /// leaves word characters. Han characters are letters, so a Chinese name correction is
    /// observed only after the caret moves to punctuation (or another non-letter) — e.g.
    /// the user fixes 星浪→星朗, then taps after the closing 。. Recorded limitation: fixing
    /// a name and dismissing the keyboard without that caret move learns nothing.
    @Test func learningRequiresCaretToLeaveWordCharactersIncludingCJK() {
        let id = UUID(), start = Date(timeIntervalSince1970: 1000)
        var session = CorrectionLearningSession(documentID: id, before: "", inserted: "明天和星浪开会。", after: "", now: start)
        // Caret right after 朗: Han is a letter, so observation is blocked indefinitely.
        #expect(session.observe(documentID: id, before: "明天和星朗", after: "开会。", selectedText: "", now: start) == nil)
        #expect(session.observe(documentID: id, before: "明天和星朗", after: "开会。", selectedText: "", now: start.addingTimeInterval(2)) == nil)
        // Caret moves after 。: the first read only starts the stability window.
        #expect(session.observe(documentID: id, before: "明天和星朗开会。", after: "", selectedText: "", now: start.addingTimeInterval(3)) == nil)
        let request = session.observe(documentID: id, before: "明天和星朗开会。", after: "", selectedText: "", now: start.addingTimeInterval(5))
        #expect(request == CorrectionLearningRequest(original: "明天和星浪开会。", edited: "明天和星朗开会。"))
    }
}
