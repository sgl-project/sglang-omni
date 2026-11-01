import Foundation
import Testing
@testable import TypingHost

@Suite struct JapaneseCompositionTests {
    private let field = FieldContext(documentID: UUID(), before: "Message: ", after: "", selected: "")

    @Test func oldCandidatesCannotReplaceNewReading() throws {
        var input = MarkedComposition()
        let old = try #require({ input.update(reading: "かい", context: field) }())
        let current = try #require({ input.update(reading: "かいぎ", context: field) }())
        #expect(!{ input.receive(["貝"], for: old) }())
        #expect({ input.receive(["会議", "会議", "", "かいぎ"], for: current) }())
        #expect(input.candidates == ["かいぎ", "会議"])
        #expect({ input.commit("貝", context: field) }() == nil)
        #expect({ input.commit("会議", context: field) }() == "会議")
        #expect(input.reading.isEmpty)
        #expect(!{ input.receive(["会議"], for: current) }())
    }

    @Test func changedFieldCaretOrSelectionInvalidatesComposition() throws {
        for changed in [
            FieldContext(documentID: UUID(), before: field.before, after: "", selected: ""),
            FieldContext(documentID: field.documentID, before: "Message:", after: " ", selected: ""),
            FieldContext(documentID: field.documentID, before: field.before, after: "", selected: "x")
        ] {
            var input = MarkedComposition()
            let query = try #require({ input.update(reading: "かいぎ", context: field) }())
            #expect({ input.receive(["会議"], for: query) }())
            #expect({ input.commit("会議", context: changed) }() == nil)
            #expect(input.reading.isEmpty)
        }
    }

    @Test func capturesTheContextActuallyReturnedAfterMarking() throws {
        var input = MarkedComposition()
        let query = try #require({ input.update(reading: "かいぎ", context: field) }())
        let marked = FieldContext(documentID: field.documentID, before: "Message: かいぎ", after: "", selected: "")
        #expect({ input.didMark(context: marked, for: query) }())
        #expect(input.matches(marked))
        #expect(!input.matches(field))
        #expect({ input.commit("かいぎ", context: marked) }() == "かいぎ")
    }

    @Test func cancelledOrDetachedQueriesCannotRestoreCandidates() throws {
        var input = MarkedComposition()
        let query = try #require({ input.update(reading: "か", context: field) }())
        input.reset()
        #expect(!{ input.didMark(context: field, for: query) }())
        #expect(!{ input.receive(["蚊"], for: query) }())
        #expect({ input.commit("か", context: field) }() == nil)
        #expect({ input.update(reading: "か", context: nil) }() == nil)
    }

    @Test func delayedMarkedTextAcknowledgementAcceptsOnlyOurExactWrite() throws {
        // iOS 26.3: setMarkedText returns the old context, then
        // textDidChange exposes the inserted kana about 25ms later.
        var input = MarkedComposition()
        let first = try #require({ input.update(reading: "か", context: field) }())
        #expect({ input.didMark(context: field, for: first) }())
        #expect(input.visibleReadingCount(in: field) == 0)
        let marked = FieldContext(documentID: field.documentID, before: "Message: か", after: "", selected: "")
        #expect({ input.acknowledgeMarkedContext(marked) }())
        #expect(input.visibleReadingCount(in: marked) == 1)
        let next = try #require({ input.update(reading: "かいぎ", context: marked) }())
        #expect({ input.didMark(context: marked, for: next) }())
        let complete = FieldContext(documentID: field.documentID, before: "Message: かいぎ", after: "", selected: "")
        #expect({ input.acknowledgeMarkedContext(complete) }())
        #expect(input.visibleReadingCount(in: complete) == 3)
        #expect(!{ input.acknowledgeMarkedContext(marked) }(), "A previous acknowledgement cannot be replayed")
        #expect(!{ input.acknowledgeMarkedContext(field) }(), "Removing the reading is an external edit")
        let moved = FieldContext(documentID: field.documentID, before: "Message: かい", after: "ぎ", selected: "")
        #expect(!{ input.acknowledgeMarkedContext(moved) }())
        #expect(input.visibleReadingCount(in: moved) == nil)
        #expect({ input.receive(["会議"], for: next) }())
        #expect({ input.commit("会議", context: complete) }() == "会議")
    }

    @Test func readingLimitAndUnavailableContextDoNotMutateOtherText() throws {
        var input = MarkedComposition()
        #expect({ input.update(reading: String(repeating: "か", count: 128), context: field) }() != nil)
        #expect({ input.update(reading: String(repeating: "か", count: 129), context: field) }() == nil)
        #expect(input.reading.count == 128)
        #expect({ input.update(reading: "か", context: nil) }() == nil)
        #expect(input.reading.isEmpty)
    }
}
