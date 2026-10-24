import Foundation
import Testing
@testable import TypingHost

/// A composing layout can finish one piece and keep composing the next
/// (Hangul: 한 is inserted while ㄱ is marked). These are the two host steps.
@Suite struct CompositionContinuationTests {
    private let field = FieldContext(documentID: UUID(), before: "Note: ", after: "", selected: "")

    @Test func engineTextThatIsNotACandidateCommitsOnlyInTheVerifiedField() throws {
        var composition = MarkedComposition()
        let update = composition.update(reading: "한", context: field)
        let query = try #require(update)
        let marked = FieldContext(documentID: field.documentID, before: "Note: 한", after: "", selected: "")
        let didMark = composition.didMark(context: marked, for: query)
        #expect(didMark)
        let candidate = composition.commit("한", context: marked)
        #expect(candidate == "한", "the reading itself is a candidate")
        _ = composition.update(reading: "한", context: field)
        let withSpace = composition.commitText("한 ", context: field)
        #expect(withSpace == "한 ")
        #expect(composition.reading.isEmpty)
        _ = composition.update(reading: "가", context: field)
        let moved = FieldContext(documentID: field.documentID, before: "Other", after: "", selected: "")
        let refused = composition.commitText("가", context: moved)
        #expect(refused == nil)
        #expect(composition.reading.isEmpty, "a changed field drops the composition")
        let empty = composition.commitText("x", context: field)
        #expect(empty == nil, "nothing composed, nothing to commit")
    }

    @Test func aContinuationRunsBeforeKeysAlreadyWaiting() {
        var queue = CompositionWriteQueue<String>()
        let write = UUID()
        queue.beginWrite(id: write)
        let key = queue.enqueue("next key")
        let continuation = queue.enqueueFirst("mark ㄱ")
        #expect(key && continuation)
        let blocked = queue.next()
        #expect(blocked == nil, "still waiting for the commit write")
        let acknowledged = queue.acknowledge(id: write)
        #expect(acknowledged)
        let first = queue.next(), second = queue.next()
        #expect(first == "mark ㄱ")
        #expect(second == "next key")
        for index in 0..<128 { _ = queue.enqueue("\(index)") }
        let over = queue.enqueueFirst("over")
        #expect(!over, "same bound as enqueue")
    }
}
