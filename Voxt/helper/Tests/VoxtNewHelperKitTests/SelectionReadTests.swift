import XCTest
@testable import VoxtNewHelperKit

/// When textio.selection stops at a caret instead of trying ⌘C.
final class SelectionReadTests: XCTestCase {
    func testEndsAtCaret() {
        let caret = TextRange(location: 12, length: 0)
        XCTAssertTrue(SelectionRead.endsAtCaret(range: caret, editable: true), "a caret in a text field: nothing selected")
        // A page selected with ⌘A in Chrome, a PDF page in Acrobat: the
        // focused element is not a field and its range is empty (#3835).
        XCTAssertFalse(SelectionRead.endsAtCaret(range: caret, editable: false))
        XCTAssertFalse(SelectionRead.endsAtCaret(range: TextRange(location: 0, length: 0), editable: false))
        XCTAssertFalse(SelectionRead.endsAtCaret(range: nil, editable: true), "no range: ⌘C")
        XCTAssertFalse(SelectionRead.endsAtCaret(range: nil, editable: false))
        XCTAssertFalse(SelectionRead.endsAtCaret(range: TextRange(location: 0, length: 5), editable: true), "a range the app could not give as text: ⌘C")
    }
}
