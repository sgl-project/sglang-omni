import AppKit
import CoreGraphics
import Foundation
@testable import VoxtNewHelper
import VoxtNewHelperKit
import XCTest

/// TextIO.copySelection at its boundary: the clipboard, the held keys and
/// the posted ⌘C are faked, the ordering is real. #3835 (a shortcut
/// modifier still held turns the helper's ⌘C into no copy) and the
/// clipboard snapshot's place after the modifier wait (a change during
/// the wait is the user's newer clipboard, not the ⌘C's result).
final class CopySelectionTests: XCTestCase {
    /// The world copySelection runs against, scripted by each test.
    private final class Fake: @unchecked Sendable {
        var items: [[NSPasteboard.PasteboardType: Data]] = [[.string: Data("user text".utf8)]]
        var changes = 5
        var now = 0.0
        var flags: CGEventFlags = .maskControl
        var sleepCount = 0
        /// What happens on each sleep (another process writing, the user
        /// lifting a key), by sleep number.
        var onSleep: ((Int, Fake) -> Void)?
        /// What the app puts on the clipboard for the posted ⌘C; nil: it
        /// copied nothing.
        var copyText: String?
        var postedWithFlags: CGEventFlags?
        var nowAtPost: Double?
        var readCalls = 0
        var restoredSnap: [[NSPasteboard.PasteboardType: Data]]?

        func write(_ text: String) {
            items = [[.string: Data(text.utf8)]]
            changes += 1
        }

        func world() -> CopyWorld {
            CopyWorld(
                snapshot: { (self.items, self.changes) },
                changeCount: { self.changes },
                heldFlags: { self.flags },
                postCopy: {
                    self.postedWithFlags = self.flags
                    self.nowAtPost = self.now
                    // As the apps in the reports: a ⌘C posted with a
                    // blocking modifier still held copies nothing
                    // (Chrome reads Control+⌘C as no copy, #3835).
                    if self.flags.isEmpty, let text = self.copyText { self.write(text) }
                },
                readAndRestore: { snap in
                    self.readCalls += 1
                    self.restoredSnap = snap
                    let text = self.items.first?[.string].flatMap { String(data: $0, encoding: .utf8) }
                    return (text, ["public.utf8-plain-text"], [:], nil)
                },
                now: { self.now },
                sleep: { ns in
                    self.sleepCount += 1
                    self.now += Double(ns) / 1_000_000
                    self.onSleep?(self.sleepCount, self)
                }
            )
        }
    }

    /// Another process writes to the clipboard during the modifier wait
    /// and the ⌘C copies nothing: that write is not the selection, and
    /// nothing is restored over it.
    func testClipboardChangeDuringWaitIsNotTheCopy() async {
        let f = Fake()
        f.onSleep = { n, f in
            if n == 1 { f.write("external") } // another process, mid-wait
            if n == 2 { f.flags = [] } // the user lifts Control
        }
        let r = await TextIO.copySelection(timeoutMs: 50, modifierWaitMs: 800, world: f.world())
        XCTAssertNil(r.text, "the external write is not the ⌘C's result: \(r.note)")
        XCTAssertEqual(r.note, "copy: clipboard unchanged")
        XCTAssertEqual(f.readCalls, 0, "nothing read, nothing restored")
        XCTAssertNil(f.restoredSnap)
    }

    /// The snapshot taken after the wait is what the restore puts back:
    /// a clipboard written during the wait survives the ⌘C.
    func testRestoreKeepsClipboardWrittenDuringWait() async {
        let f = Fake()
        f.copyText = "selected text"
        f.onSleep = { n, f in
            if n == 1 { f.write("external") }
            if n == 2 { f.flags = [] }
        }
        let r = await TextIO.copySelection(timeoutMs: 50, modifierWaitMs: 800, world: f.world())
        XCTAssertEqual(r.text, "selected text", r.note)
        XCTAssertEqual(r.note, "copy: changed")
        let restored = f.restoredSnap?.first?[.string].flatMap { String(data: $0, encoding: .utf8) }
        XCTAssertEqual(restored, "external", "the restore puts back the snapshot from after the wait, not over it")
    }

    /// The ⌘C is posted only once the shortcut's modifiers have lifted:
    /// the app reads ⌘C, not Control+⌘C (#3835).
    func testCopyWaitsForModifierRelease() async {
        let f = Fake()
        f.copyText = "selected text"
        f.onSleep = { n, f in
            if n == 3 { f.flags = [] }
        }
        let r = await TextIO.copySelection(timeoutMs: 50, modifierWaitMs: 800, world: f.world())
        XCTAssertEqual(f.postedWithFlags, [], "posted while Control was still held")
        XCTAssertEqual(r.note, "copy: changed", "no held= once the keys lifted")
        XCTAssertEqual(r.text, "selected text")
    }

    /// A modifier that never lifts (a hold-to-record shortcut keeps its
    /// key down) blocks the ⌘C: posting it would reach the app as a
    /// chord that copies nothing (#3835), so after modifierWaitMs the
    /// copy is skipped and no contaminated key is posted.
    func testCopySkippedWhileModifierHeld() async {
        let f = Fake()
        f.copyText = "selected text"
        let r = await TextIO.copySelection(timeoutMs: 50, modifierWaitMs: 100, world: f.world())
        XCTAssertNil(f.postedWithFlags, "no contaminated keystroke")
        XCTAssertNil(r.text)
        XCTAssertEqual(r.note, "copy: skipped held=control")
        XCTAssertEqual(f.now, 100, "the wait is modifierWaitMs, no more")
        XCTAssertEqual(f.readCalls, 0, "the clipboard is never touched")
    }
}
