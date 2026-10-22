import Foundation
import Testing
@testable import VoxtMobileKit

struct CorrectionLearningTests {
    @Test func stableCorrectionIsLearnedOnlyInsideTheOriginalField() throws {
        let document = UUID()
        let start = Date(timeIntervalSince1970: 1_000)
        var session = CorrectionLearningSession(documentID: document, before: "Dear team, ", inserted: "use Voice tomorrow.", after: " Thanks.", now: start)
        #expect(session.observe(documentID: document, before: "Dear team, use Voxt tomorrow.", after: " Thanks.", selectedText: "", now: start) == nil)
        #expect(session.observe(documentID: document, before: "Dear team, use Voxt tomorrow.", after: " Thanks.", selectedText: "", now: start.addingTimeInterval(1)) == nil)
        let correction = session.observe(documentID: document, before: "Dear team, use Voxt tomorrow.", after: " Thanks.", selectedText: "", now: start.addingTimeInterval(2))
        #expect(correction == CorrectionLearningRequest(original: "use Voice tomorrow.", edited: "use Voxt tomorrow."))
        #expect(session.observe(documentID: document, before: "Dear team, use Voxt tomorrow.", after: " Thanks.", selectedText: "", now: start.addingTimeInterval(4)) == nil)
    }

    @Test func doesNotLearnFromOtherFieldsSelectionsTruncatedContextOrExpiredSession() {
        let id = UUID(), start = Date(timeIntervalSince1970: 1_000)
        for sample in [
            (UUID(), "prefix Voxt", " suffix", "", 2.0),
            (id, "prefix Vox", " suffix", "t", 2.0),
            (id, "Voxt", " suffix", "", 2.0),
            (id, "prefix Voxt", " suffix", "", 121.0)
        ] {
            var session = CorrectionLearningSession(documentID: id, before: "prefix ", inserted: "Voice", after: " suffix", now: start)
            _ = session.observe(documentID: sample.0, before: sample.1, after: sample.2, selectedText: sample.3, now: start.addingTimeInterval(sample.4))
            #expect(session.observe(documentID: sample.0, before: sample.1, after: sample.2, selectedText: sample.3, now: start.addingTimeInterval(sample.4 + 3)) == nil)
        }
    }

    @Test func candidateDistinguishesWordCorrectionsFromAppendDeletionAndPunctuation() {
        #expect(LearnedCorrection.detect(original: "Use Voice today.", edited: "Use Voxt today.") == LearnedCorrection(original: "Voice", replacement: "Voxt"))
        #expect(CorrectionLearningRequest(original: "请用星浪处理。", edited: "请用星朗处理。").validated(.init(original: "星浪", replacement: "星朗")) == LearnedCorrection(original: "星浪", replacement: "星朗"))
        #expect(LearnedCorrection.detect(original: "Use Qwen2 today.", edited: "Use Qwen3 today.") == LearnedCorrection(original: "Qwen2", replacement: "Qwen3"))
        #expect(LearnedCorrection.detect(original: "Use Voxt.", edited: "Use Voxt. Thanks!") == nil)
        #expect(LearnedCorrection.detect(original: "Use Voxt.", edited: "Use Voxt!") == nil)
        #expect(LearnedCorrection.detect(original: "Pay $50.", edited: "Pay $60.") == nil)
        #expect(LearnedCorrection.detect(original: "Use Voxt today.", edited: "Use today.") == nil)
        #expect(LearnedCorrection.detect(original: "Use Voxt today.", edited: "") == nil)
        #expect(LearnedCorrection.detect(original: "one two three four five six", edited: "a completely different long new sentence") == nil)
    }

    @Test func boundedInsertionCanBeLearnedInsideALongerField() {
        let id = UUID(), start = Date(timeIntervalSince1970: 1000)
        let prefix = String(repeating: "a", count: 1500) + " "
        let suffix = " " + String(repeating: "b", count: 600)
        var session = CorrectionLearningSession(documentID: id, before: prefix, inserted: "Use Voice.", after: suffix, now: start)
        _ = session.observe(documentID: id, before: prefix + "Use Voxt.", after: suffix, selectedText: "", now: start)
        #expect(session.observe(documentID: id, before: prefix + "Use Voxt.", after: suffix, selectedText: "", now: start.addingTimeInterval(2)) == .init(original: "Use Voice.", edited: "Use Voxt."))
    }

    @Test func incompleteContractionsAndSeparateWordChangesAreNotLearned() {
        let id = UUID(), start = Date(timeIntervalSince1970: 1000)
        for ending in ["can'", "can’", "can-"] {
            var session = CorrectionLearningSession(documentID: id, before: "", inserted: "cannot", after: "", now: start)
            _ = session.observe(documentID: id, before: ending, after: "", selectedText: "", now: start)
            #expect(session.observe(documentID: id, before: ending, after: "", selectedText: "", now: start.addingTimeInterval(2)) == nil)
        }
        let rewrite = CorrectionLearningRequest(original: "See Alice about the invoice", edited: "See Bob about the bill")
        #expect(rewrite.validated(.init(original: "Alice about the invoice", replacement: "Bob about the bill")) == nil)
    }

    @Test func leadingLetterCorrectionWorksButIgnoredRewriteEndsObservation() {
        #expect(LearnedCorrection.detect(original: "say oxt today.", edited: "say Voxt today.") == .init(original: "oxt", replacement: "Voxt"))
        #expect(LearnedCorrection.detect(original: "Voxt today.", edited: "oxt today.") == .init(original: "Voxt", replacement: "oxt"))
        let id = UUID(), start = Date(timeIntervalSince1970: 1000)
        var session = CorrectionLearningSession(documentID: id, before: "", inserted: "Use Voice.", after: "", now: start)
        _ = session.observe(documentID: id, before: "Use Voice. More words.", after: "", selectedText: "", now: start)
        #expect(session.observe(documentID: id, before: "Use Voice. More words.", after: "", selectedText: "", now: start.addingTimeInterval(2)) == nil)
        _ = session.observe(documentID: id, before: "Use Voice. More Voxt.", after: "", selectedText: "", now: start.addingTimeInterval(3))
        #expect(session.observe(documentID: id, before: "Use Voice. More Voxt.", after: "", selectedText: "", now: start.addingTimeInterval(5)) == nil)
    }

    @Test func everyStillStoredLearnedWordKeepsItsUndoRecord() throws {
        let name = "voxt-learning-capacity.\(UUID())", defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let store = CorrectionLearningStore(defaults: defaults)
        store.isEnabled = true
        let first = try #require(try store.learn(.init(original: "First", replacement: "Term0")))
        for index in 1...100 { try store.learn(.init(original: "Before", replacement: "Term\(index)")) }
        #expect(store.records.contains(where: { $0.id == first.id }))
        store.undo(first.id)
        #expect(!PersonalDictionary(defaults: defaults).vocabulary.contains("Term0"))
    }

    @Test func optInLearningPersistsAndUndoProtectsManualEdits() throws {
        let name = "voxt-learning.\(UUID())", defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let store = CorrectionLearningStore(defaults: defaults)
        let dictionary = PersonalDictionary(defaults: defaults)
        #expect(!store.isEnabled)
        #expect(try store.learn(.init(original: "Voice", replacement: "Voxt")) == nil)
        #expect(dictionary.entries.isEmpty)
        store.isEnabled = true
        let learned = try #require(try store.learn(.init(original: "Voice", replacement: "Voxt")))
        #expect(dictionary.vocabulary == ["Voxt"])
        #expect(CorrectionLearningStore(defaults: defaults).records == [learned])
        #expect(try store.learn(.init(original: "Voice", replacement: "Voxt")) == nil)
        #expect(store.records.count == 1)
        try dictionary.rename(learned.dictionaryEntryID, to: "Voxt Mobile")
        store.undo(learned.id)
        #expect(dictionary.vocabulary == ["Voxt Mobile"])
        #expect(store.records.isEmpty)
        let second = try #require(try store.learn(.init(original: "Quen", replacement: "Qwen3")))
        store.undo(second.id)
        #expect(dictionary.vocabulary == ["Voxt Mobile"])
        store.isEnabled = false
        #expect(!CorrectionLearningStore(defaults: defaults).isEnabled)
    }
}

struct CorrectionAnalysisTests {
    @Test func rejectsUnrelatedWordsPartialChineseNamesAndBroadRewrites() {
        let request = CorrectionLearningRequest(original: "请用星浪处理。", edited: "请用星朗处理。")
        #expect(request.validated(.init(original: "星浪", replacement: "星朗")) == .init(original: "星浪", replacement: "星朗"))
        #expect(request.validated(.init(original: "浪", replacement: "朗")) == nil)
        #expect(request.validated(.init(original: "星浪", replacement: "幻觉")) == nil)
        #expect(request.validated(.init(original: "处理", replacement: "处理")) == nil)
        #expect(CorrectionLearningRequest(original: "Use Voice now.", edited: "Use Voxt later.").validated(.init(original: "Voice", replacement: "Voxt")) == nil)
        #expect(CorrectionLearningRequest(original: "Use Voice now.", edited: "Use Voxt now.").validated(.init(original: "oice", replacement: "oxt")) == nil)
        #expect(CorrectionLearningRequest(original: "Qwen2", edited: "Qwen3").validated(.init(original: "Qwen2", replacement: "Qwen3")) != nil)
        #expect(CorrectionLearningRequest(original: "50", edited: "60").validated(.init(original: "50", replacement: "60")) == nil)
    }

    @Test func unfinishedWordIsNeverSubmittedAndFurtherEditsInvalidateResponse() {
        let id = UUID(), start = Date(timeIntervalSince1970: 1000)
        var session = CorrectionLearningSession(documentID: id, before: "", inserted: "Use Voice.", after: "", now: start)
        #expect(session.observe(documentID: id, before: "Use Vox", after: "", selectedText: "", now: start) == nil)
        #expect(session.observe(documentID: id, before: "Use Vox", after: "", selectedText: "", now: start.addingTimeInterval(3)) == nil)
        #expect(session.observe(documentID: id, before: "Use Voxt.", after: "", selectedText: "", now: start.addingTimeInterval(4)) == nil)
        let request = session.observe(documentID: id, before: "Use Voxt.", after: "", selectedText: "", now: start.addingTimeInterval(6))
        #expect(request == .init(original: "Use Voice.", edited: "Use Voxt."))
        #expect(session.accepts(request!, documentID: id, before: "Use Voxt.", after: "", selectedText: "", now: start.addingTimeInterval(7)))
        #expect(!session.accepts(request!, documentID: id, before: "Use Voxtral.", after: "", selectedText: "", now: start.addingTimeInterval(7)))
        #expect(!session.accepts(request!, documentID: UUID(), before: "Use Voxt.", after: "", selectedText: "", now: start.addingTimeInterval(7)))
    }

    @Test @MainActor func channelConsumesOnceExpiresAndNeverPublishesAfterOptOut() async throws {
        let name = "learning-channel.\(UUID())", defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        let channel = CorrectionLearningChannel(defaults: defaults, notifier: .none)
        let store = CorrectionLearningStore(defaults: defaults)
        let request = CorrectionLearningRequest(original: "Use Voice.", edited: "Use Voxt.")
        let start = Date()
        let id = channel.submit(request, now: start)
        #expect(channel.takeRequest(now: start.addingTimeInterval(121)) == nil)
        #expect(channel.takeRequest() == nil)
        _ = channel.submit(request)
        let disabled = CorrectionLearningProcessor(channel: channel, store: store) { _ in
            Issue.record("Disabled learning must never call model")
            return nil
        }
        await disabled.process()
        #expect(channel.takeResult(for: id) == nil)
        store.isEnabled = true
        let validID = channel.submit(request)
        let processor = CorrectionLearningProcessor(channel: channel, store: store) { candidate in
            #expect(candidate == request)
            return .init(original: "Voice", replacement: "Voxt")
        }
        await processor.process()
        #expect(channel.takeRequest() == nil)
        let result = try #require(channel.takeResult(for: validID))
        #expect(result.correction == .init(original: "Voice", replacement: "Voxt"))
        #expect(channel.takeResult(for: validID) == nil)
        let staleID = channel.submit(request)
        let turningOff = CorrectionLearningProcessor(channel: channel, store: store) { _ in
            store.isEnabled = false
            return .init(original: "Voice", replacement: "Voxt")
        }
        await turningOff.process()
        #expect(channel.takeResult(for: staleID) == nil)
        store.isEnabled = true
        let restartedID = channel.submit(request)
        let restarting = CorrectionLearningProcessor(channel: channel, store: store) { _ in
            store.isEnabled = false
            store.isEnabled = true
            return .init(original: "Voice", replacement: "Voxt")
        }
        await restarting.process()
        #expect(channel.takeResult(for: restartedID) == nil)

    }
}

struct CorrectionEditTrackerTests {
    @Test func authorizesOnlyEditsStartingFromTheObservedInsertion() {
        var tracker = CorrectionEditTracker(text: "Re: Voice today.")
        let edit1 = tracker.replace(before: "Re: ", selected: "Voice", after: " today.", with: "Voxt")
        #expect(edit1)
        #expect(tracker.matches(before: "Re: Voxt", selected: "", after: " today."))
        // A second line with identical boundaries is not the tracked insertion.
        #expect(!tracker.matches(before: "Re: Qwen", selected: "", after: " today."))
        let edit2 = !tracker.replace(before: "Re: Qwen", selected: "", after: " today.", with: "3")
        #expect(edit2)
        #expect(!tracker.matches(before: "Re: Voxt", selected: "", after: " today."))
    }
    @Test func deletionAndTypingPreserveAnExactExpectedPassage() {
        var tracker = CorrectionEditTracker(text: "Use Voice.")
        let edit3 = tracker.delete(before: "Use Voice", selected: "", after: ".")
        #expect(edit3)
        #expect(tracker.matches(before: "Use Voic", selected: "", after: "."))
        let edit4 = tracker.replace(before: "Use Voic", selected: "", after: ".", with: "e")
        #expect(edit4)
        #expect(tracker.matches(before: "Use Voice.", selected: "", after: ""))
    }
}
