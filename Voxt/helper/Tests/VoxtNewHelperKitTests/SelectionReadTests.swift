import XCTest
@testable import VoxtNewHelperKit

/// When textio.selection stops at a caret instead of trying ⌘C, and what
/// counts as read text at all.
final class SelectionReadTests: XCTestCase {
    func testHasText() {
        XCTAssertFalse(SelectionRead.hasText(""))
        // A Lark block selection in Chrome reads from AX as one blank.
        XCTAssertFalse(SelectionRead.hasText(" "))
        XCTAssertFalse(SelectionRead.hasText("  \n\t "))
        XCTAssertFalse(SelectionRead.hasText("\u{00A0}"), "a non-breaking space alone is not text either")
        // An image-only block selection in Chrome reads from AX as the
        // object replacement character.
        XCTAssertFalse(SelectionRead.hasText("\u{FFFC}"))
        XCTAssertFalse(SelectionRead.hasText(" \u{FFFC} "))
        XCTAssertTrue(SelectionRead.hasText("\u{FFFC}caption"), "an image with its text still has the text")
        XCTAssertTrue(SelectionRead.hasText("a"))
        XCTAssertTrue(SelectionRead.hasText("  summary of the selection  "), "real text keeps its whitespace")
        XCTAssertTrue(SelectionRead.hasText("图"))
    }

    func testEndsAtCaret() {
        let caret = TextRange(location: 12, length: 0)
        XCTAssertTrue(SelectionRead.endsAtCaret(range: caret, editable: true, blankSelection: false), "a caret in a text field: nothing selected")
        // A page selected with ⌘A in Chrome, a PDF page in Acrobat: the
        // focused element is not a field and its range is empty (#3835).
        XCTAssertFalse(SelectionRead.endsAtCaret(range: caret, editable: false, blankSelection: false))
        XCTAssertFalse(SelectionRead.endsAtCaret(range: TextRange(location: 0, length: 0), editable: false, blankSelection: false))
        XCTAssertFalse(SelectionRead.endsAtCaret(range: nil, editable: true, blankSelection: false), "no range: ⌘C")
        XCTAssertFalse(SelectionRead.endsAtCaret(range: nil, editable: false, blankSelection: false))
        XCTAssertFalse(SelectionRead.endsAtCaret(range: TextRange(location: 0, length: 5), editable: true, blankSelection: false), "a range the app could not give as text: ⌘C")
        // AX reported a blank selected text (a Lark block selection):
        // something is selected, so the caret rule must not stop the ⌘C.
        XCTAssertFalse(SelectionRead.endsAtCaret(range: caret, editable: true, blankSelection: true), "a blank AX selection is a selection: ⌘C")
    }
}
