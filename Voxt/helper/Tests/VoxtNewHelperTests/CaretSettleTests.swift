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
