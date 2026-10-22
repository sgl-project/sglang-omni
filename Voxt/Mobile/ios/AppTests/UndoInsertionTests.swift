import XCTest
@testable import VoxtMobile
import VoxtMobileKit

private struct Field: CompositionTextContext {
    var documentIdentifier: UUID
    var selectedText: String?
    var documentContextBeforeInput: String?
    var documentContextAfterInput: String?

    init(_ document: UUID, selected: String = "", before: String, after: String = "") {
        documentIdentifier = document
        selectedText = selected
        documentContextBeforeInput = before
        documentContextAfterInput = after
    }
}

/// After every insert, ↶ restores the field while the inserted text is still
/// exactly at the cursor (the same check as editing the last result).
final class UndoInsertionTests: XCTestCase {
    private let document = UUID()

    private func dictated(_ text: String, before: String = "Hi ", after: String = "") throws -> KeyboardComposer {
        let name = "undo-insertion.\(UUID().uuidString)"
        addTeardownBlock { UserDefaults().removePersistentDomain(forName: name) }
        var composer = KeyboardComposer(defaults: UserDefaults(suiteName: name)!)
        let field = Field(document, before: before, after: after)
        let command = try composer.begin(context: field, intent: nil)
        XCTAssertEqual(composer.receive(current: field, text: text, resultID: "r", requestID: command.requestID), .insert(text))
        return composer
    }

    func testAJustDictatedResultCanBeUndoneOnce() throws {
        var composer = try dictated("Running late.")
        let atCursor = Field(document, before: "Hi Running late.")
        XCTAssertTrue(composer.canUndo(context: atCursor))
        XCTAssertEqual(composer.undo(context: atCursor), .replace(oldText: "Running late.", newText: ""))
        XCTAssertFalse(composer.canUndo(context: Field(document, before: "Hi ")), "one undo only")
    }

    func testUndoAlsoRemovesTheSpaceDictationAdded() throws {
        let name = "undo-insertion.\(UUID().uuidString)"
        addTeardownBlock { UserDefaults().removePersistentDomain(forName: name) }
        var composer = KeyboardComposer(defaults: UserDefaults(suiteName: name)!)
        let field = Field(document, before: "Hi.")
        let command = try composer.begin(context: field, intent: nil)
        XCTAssertEqual(composer.receive(current: field, text: "Running late.", resultID: "r", requestID: command.requestID), .insert(" Running late."))
        XCTAssertEqual(composer.undo(context: Field(document, before: "Hi. Running late.")), .replace(oldText: " Running late.", newText: ""))
    }

    func testTypingAfterTheResultOrMovingTheCursorEndsUndo() throws {
        let composer = try dictated("Running late.", after: " Bye")
        XCTAssertFalse(composer.canUndo(context: Field(document, before: "Hi Running late. x", after: " Bye")), "the user typed")
        XCTAssertFalse(composer.canUndo(context: Field(document, before: "Hi Running", after: " late. Bye")), "the cursor moved")
        XCTAssertFalse(composer.canUndo(context: Field(UUID(), before: "Hi Running late.", after: " Bye")), "another field")
        XCTAssertFalse(composer.canUndo(context: nil))
        XCTAssertTrue(composer.canUndo(context: Field(document, before: "Hi Running late.", after: " Bye")))
    }

    func testAnEditUndoesToTheTextBeforeTheEdit() throws {
        var composer = try dictated("See you Monday.")
        let edited = Field(document, before: "Hi See you Monday.")
        let edit = try composer.begin(context: edited, intent: .editLastResult)
        XCTAssertEqual(composer.receive(current: edited, text: "See you Tuesday.", resultID: "r2", requestID: edit.requestID),
                       .replace(oldText: "See you Monday.", newText: "See you Tuesday."))
        let after = Field(document, before: "Hi See you Tuesday.")
        XCTAssertTrue(composer.canUndo(context: after))
        XCTAssertEqual(composer.undo(context: after), .replace(oldText: "See you Tuesday.", newText: "See you Monday."),
                       "an edit restores the text it replaced, not an empty field")
    }
}
