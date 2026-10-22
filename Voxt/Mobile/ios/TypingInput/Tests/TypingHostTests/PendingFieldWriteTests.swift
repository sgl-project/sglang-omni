import Foundation
import Testing
@testable import TypingHost

@Suite struct JapanesePendingWriteTests {
    @Test func omittedReplacementMarkCanBeUnmarkedButIsNotAcknowledged() {
        let document = UUID()
        let origin = FieldContext(documentID: document, before: "Before か", after: " after", selected: "")
        let write = PendingFieldWrite(id: UUID(), origin: origin, replacing: 1, text: "かい")
        let omitted = FieldContext(documentID: document, before: "Before ", after: " after", selected: "")
        #expect(write.canUnmarkForAcknowledgement(origin))
        #expect(write.canUnmarkForAcknowledgement(omitted))
        #expect(omitted != write.expected)
        #expect(write.expected.before == "Before かい")
    }
    @Test func changedFieldSelectionAndCaretCannotTriggerRecovery() {
        let document = UUID()
        let origin = FieldContext(documentID: document, before: "aか", after: "b", selected: "")
        let write = PendingFieldWrite(id: UUID(), origin: origin, replacing: 1, text: "かい")
        for context in [
            FieldContext(documentID: UUID(), before: "a", after: "b", selected: ""),
            FieldContext(documentID: document, before: "a", after: "かb", selected: ""),
            FieldContext(documentID: document, before: "a", after: "b", selected: "か"),
            FieldContext(documentID: document, before: "changed", after: "b", selected: "")
        ] { #expect(!write.canUnmarkForAcknowledgement(context)) }
        #expect(!write.canUnmarkForAcknowledgement(nil))
    }
}
