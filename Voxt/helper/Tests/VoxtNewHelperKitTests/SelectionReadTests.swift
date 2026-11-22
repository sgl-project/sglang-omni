import CoreGraphics
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

    /// A caret-only answer waits for a selection made with ⌘A in the same
    /// motion as the shortcut to settle (#4467), then concludes. The app
    /// sets the budget in textio.selection's params.
    func testCaretPending() {
        let settle = SelectionRead.defaultCaretSettleMs
        XCTAssertTrue(SelectionRead.caretPending(elapsedMs: 0, settleMs: settle), "the first reading can predate the ⌘A")
        XCTAssertTrue(SelectionRead.caretPending(elapsedMs: settle - 1, settleMs: settle))
        XCTAssertFalse(SelectionRead.caretPending(elapsedMs: settle, settleMs: settle), "a caret that stays a caret is the answer")
        XCTAssertFalse(SelectionRead.caretPending(elapsedMs: settle * 2, settleMs: settle))
        XCTAssertFalse(SelectionRead.caretPending(elapsedMs: 10, settleMs: 10), "the app's budget, not the helper's")
    }

    /// The ⌘C waits for a modifier still held from the shortcut: the app
    /// reads Control+⌘C as no copy (#3835).
    func testBlocksCopy() {
        XCTAssertTrue(SelectionRead.blocksCopy(.maskControl), "Control of a Control+C shortcut")
        XCTAssertTrue(SelectionRead.blocksCopy([.maskControl, .maskCommand]))
        XCTAssertTrue(SelectionRead.blocksCopy(.maskShift))
        XCTAssertTrue(SelectionRead.blocksCopy(.maskAlternate))
        XCTAssertFalse(SelectionRead.blocksCopy(.maskCommand), "⌘ is ⌘C's own modifier: a Command+C shortcut posts at once")
        XCTAssertFalse(SelectionRead.blocksCopy([]))
    }
}
