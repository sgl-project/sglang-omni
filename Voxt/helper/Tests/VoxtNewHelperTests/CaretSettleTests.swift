import CoreGraphics
import Foundation
@testable import VoxtNewHelper
@testable import VoxtNewHelperKit
import XCTest

/// TextIO.selection's caret-settle loop with scripted accessibility
/// readings and a fake clock (#4467): a selection made with ⌘A in the
/// shortcut's motion shows up readings late, and a real caret answers
/// only after the app's settle budget is spent.
final class CaretSettleTests: XCTestCase {
    /// Scripted readings and a clock the sleeps advance.
    private final class Fake: @unchecked Sendable {
        var readings: [SelectionReading?]
        var now = 0.0
        var reads = 0

        init(_ readings: [SelectionReading?]) { self.readings = readings }

        func world() -> SelectionWorld {
            SelectionWorld(
                read: {
                    self.reads += 1
                    guard !self.readings.isEmpty else { return nil }
                    return self.readings.removeFirst()
                },
                now: { self.now },
                sleep: { ns in self.now += Double(ns) / 1_000_000 }
            )
        }
    }

    private let caret = SelectionReading(role: "AXTextField", selected: nil,
                                         range: CFRange(location: 3, length: 0),
                                         ranged: nil, marker: nil, editable: true)

    /// The first readings are the caret from before the ⌘A; the selection
    /// shows up two readings later and is the answer.
    func testCaretSettleReadsLateSelection() async throws {
        let selected = SelectionReading(role: "AXTextField", selected: "hello world",
                                        range: CFRange(location: 0, length: 11),
                                        ranged: "hello world", marker: nil, editable: true)
        let f = Fake([caret, caret, selected])
        var p = SelectionParams()
        p.caretSettleMs = 600
        p.caretPollMs = 60
        let r = try await TextIO.selection(p, world: f.world())
        XCTAssertEqual(r.text, "hello world")
        XCTAssertEqual(r.via, SelectionVia.ax)
        XCTAssertEqual(f.reads, 3, "the read polled until the selection showed")
        XCTAssertTrue(r.steps?.contains("caret-settle") ?? false, "steps \(r.steps ?? [])")
    }

    /// A caretSettleMs of 0 on the wire disables the settle: the
    /// dictionary's AX-only probe, which must answer inside its own
    /// 300 ms, gets the caret's answer at once.
    func testCaretSettleZeroDisables() async throws {
        let f = Fake([caret, caret])
        var p = SelectionParams()
        p.caretSettleMs = 0
        p.caretPollMs = 60
        let r = try await TextIO.selection(p, world: f.world())
        XCTAssertNil(r.text)
        XCTAssertEqual(r.via, SelectionVia.none)
        XCTAssertEqual(f.reads, 1, "no polling with a zero settle")
        XCTAssertEqual(f.now, 0)
        XCTAssertFalse(r.steps?.contains("caret-settle") ?? true)
        XCTAssertEqual(r.steps?.last, "caret-only")
    }

    /// A page's AX lags the ⌘A the same way a field's does (#3835): an
    /// empty, non-editable AXWebArea reading settles too, and the
    /// selection that shows up a reading later is the answer.
    func testEmptyWebAreaSettlesIntoSelection() async throws {
        let webCaret = SelectionReading(role: "AXWebArea", selected: nil,
                                        range: CFRange(location: 0, length: 0),
                                        ranged: nil, marker: nil, editable: false)
        let page = SelectionReading(role: "AXWebArea", selected: "page text",
                                    range: CFRange(location: 0, length: 9),
                                    ranged: "page text", marker: nil, editable: false)
        let f = Fake([webCaret, page])
        var p = SelectionParams()
        p.caretSettleMs = 600
        p.caretPollMs = 60
        let r = try await TextIO.selection(p, world: f.world())
        XCTAssertEqual(r.text, "page text")
        XCTAssertEqual(r.via, SelectionVia.ax)
        XCTAssertEqual(f.reads, 2, "the empty page reading polled once more")
        XCTAssertTrue(r.steps?.contains("empty-settle") ?? false, "steps \(r.steps ?? [])")
    }

    /// An empty, non-editable reading that stays empty (a PDF view whose
    /// AX shows nothing of the selection, #3835) goes on to ⌘C after the
    /// settle — while a genuine editable caret never does.
    func testEmptyPDFViewSettlesThenCopies() async throws {
        let pdfCaret = SelectionReading(role: "AXScrollArea", selected: nil,
                                        range: CFRange(location: 0, length: 0),
                                        ranged: nil, marker: nil, editable: false)
        let f = Fake([pdfCaret, pdfCaret, pdfCaret])
        var p = SelectionParams()
        p.caretSettleMs = 100
        p.caretPollMs = 60
        p.allowCopy = true
        let copy = CopyWorld(
            snapshot: { ([], 1) },
            changeCount: { 2 },
            heldFlags: { [] },
            postCopy: {},
            readAndRestore: { _ in ("copied page", ["public.utf8-plain-text"], [:], nil) },
            now: { 0 },
            sleep: { _ in }
        )
        let r = try await TextIO.selection(p, world: f.world(), copyWorld: copy)
        XCTAssertEqual(r.text, "copied page")
        XCTAssertEqual(r.via, SelectionVia.copy)
        XCTAssertEqual(f.reads, 3, "the settle ran out before the ⌘C")
        XCTAssertTrue(r.steps?.contains("empty-settle") ?? false, "steps \(r.steps ?? [])")
    }

    /// A nil range is an empty reading too (fixture c23 reads an
    /// AXGroup's range as nil): a non-editable AXWebArea with no range
    /// settles, and the selection that shows up later is the answer.
    func testNilRangeWebAreaSettlesIntoSelection() async throws {
        let webEmpty = SelectionReading(role: "AXWebArea", selected: nil,
                                        range: nil, ranged: nil, marker: nil, editable: false)
        let page = SelectionReading(role: "AXWebArea", selected: "page text",
                                    range: CFRange(location: 0, length: 9),
                                    ranged: "page text", marker: nil, editable: false)
        let f = Fake([webEmpty, page])
        var p = SelectionParams()
        p.caretSettleMs = 600
        p.caretPollMs = 60
        let r = try await TextIO.selection(p, world: f.world())
        XCTAssertEqual(r.text, "page text")
        XCTAssertEqual(r.via, SelectionVia.ax)
        XCTAssertEqual(f.reads, 2, "the nil-range reading polled once more")
        XCTAssertTrue(r.steps?.contains("empty-settle") ?? false, "steps \(r.steps ?? [])")
    }

    /// focus=nil does not settle: nothing is focused (the desktop), a
    /// state, not a reading that can lag the ⌘A. The answer comes from
    /// the first reading (here: no copy allowed, so none).
    func testFocusNilDoesNotSettle() async throws {
        let page = SelectionReading(role: "AXWebArea", selected: "page text",
                                    range: CFRange(location: 0, length: 9),
                                    ranged: "page text", marker: nil, editable: false)
        let f = Fake([nil, page])
        var p = SelectionParams()
        p.caretSettleMs = 600
        p.caretPollMs = 60
        let r = try await TextIO.selection(p, world: f.world())
        XCTAssertNil(r.text)
        XCTAssertEqual(r.via, SelectionVia.none)
        XCTAssertEqual(f.reads, 1, "a missing focus is final, not polled")
        XCTAssertEqual(f.now, 0)
        XCTAssertEqual(r.steps, ["focus=nil"])
    }

    /// A caret that stays a caret is the answer once the app's settle
    /// budget is spent: caret-only, none, no ⌘C.
    func testCaretSettleGivesUpAfterTheBudget() async throws {
        let f = Fake([caret, caret, caret, caret])
        var p = SelectionParams()
        p.caretSettleMs = 100
        p.caretPollMs = 60
        p.allowCopy = true
        let r = try await TextIO.selection(p, world: f.world())
        XCTAssertNil(r.text)
        XCTAssertEqual(r.via, SelectionVia.none)
        XCTAssertEqual(f.reads, 3, "the third reading is past the budget")
        XCTAssertEqual(f.now, 120, "two polls of 60 ms spend the 100 ms budget")
        XCTAssertEqual(r.steps?.first, "focus=AXTextField")
        XCTAssertEqual(r.steps?.last, "caret-only")
    }
}
