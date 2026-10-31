import XCTest
@testable import VoxtNewHelperKit

/// textio.paste's check: a change in what both readings tell is an
/// insert; nothing to compare is unknown, never unchanged.
final class PasteCheckTests: XCTestCase {
    private let caret0 = TextRange(location: 0, length: 0)

    func testCompare() {
        let empty = FieldReading(length: 0, digest: PasteChecker.digest(""), selection: caret0)
        let typed = FieldReading(length: 2, digest: PasteChecker.digest("Hi"), selection: TextRange(location: 2, length: 0))
        XCTAssertEqual(PasteChecker.compare(empty, typed), .changed)
        XCTAssertEqual(PasteChecker.compare(empty, empty), .same)

        // A selection replaced by text of the same length: the value's digest.
        let ab = FieldReading(length: 2, digest: PasteChecker.digest("ab"), selection: TextRange(location: 0, length: 2))
        let cd = FieldReading(length: 2, digest: PasteChecker.digest("cd"), selection: TextRange(location: 0, length: 2))
        XCTAssertEqual(PasteChecker.compare(ab, cd), .changed)

        // A web editor with a caret and no value: the caret moves.
        let caret = FieldReading(length: nil, selection: TextRange(location: 5, length: 0))
        XCTAssertEqual(PasteChecker.compare(caret, FieldReading(length: nil, selection: TextRange(location: 9, length: 0))), .changed)
        XCTAssertEqual(PasteChecker.compare(caret, caret), .same)

        // A web area: only the caret's text marker.
        let page = FieldReading(length: nil, selection: nil, marker: 11)
        XCTAssertTrue(page.readable)
        XCTAssertEqual(PasteChecker.compare(page, FieldReading(length: nil, selection: nil, marker: 12)), .changed)
        XCTAssertEqual(PasteChecker.compare(page, page), .same, "a page that ignored ⌘V")
        XCTAssertEqual(page.state, FieldState(length: 0, selection: nil, marker: true))

        // Nothing to compare.
        let blank = FieldReading(length: nil, selection: nil)
        XCTAssertEqual(PasteChecker.compare(blank, blank), .unreadable)
        XCTAssertEqual(PasteChecker.compare(empty, nil), .unreadable, "the element is gone")
        XCTAssertEqual(PasteChecker.compare(FieldReading(length: 3, selection: nil), FieldReading(length: nil, selection: caret0)), .unreadable,
                       "no reading in common")
    }

    func testOutcome() {
        let empty = FieldReading(length: 0, selection: caret0)
        XCTAssertEqual(PasteChecker.outcome(sawChange: true, before: empty, last: empty, focusMoved: false), PasteCheck.inserted)
        XCTAssertEqual(PasteChecker.outcome(sawChange: false, before: empty, last: empty, focusMoved: false), PasteCheck.unchanged)
        XCTAssertEqual(PasteChecker.outcome(sawChange: false, before: empty, last: empty, focusMoved: true), PasteCheck.unknown,
                       "another element has the focus: the text may be there")
        XCTAssertEqual(PasteChecker.outcome(sawChange: false, before: FieldReading(length: nil, selection: nil), last: nil, focusMoved: false),
                       PasteCheck.unknown, "the element tells nothing")
    }

    func testState() {
        XCTAssertNil(FieldReading(length: nil, selection: nil).state)
        XCTAssertEqual(FieldReading(length: nil, selection: caret0).state, FieldState(length: 0, selection: caret0))
        XCTAssertEqual(FieldReading(length: 7, digest: 1, selection: nil).state, FieldState(length: 7, selection: nil))
    }
}
