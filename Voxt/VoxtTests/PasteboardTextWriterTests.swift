import AppKit
import XCTest
@testable import Voxt

@MainActor
final class PasteboardTextWriterTests: XCTestCase {
    func testUnchangedTemporaryWriteRestoresPreviousTextOnce() throws {
        try withPasteboard { pasteboard in
            writeExternal("original", to: pasteboard)
            let writer = PasteboardTextWriter()
            let restoration = try XCTUnwrap(writer.write("result", to: pasteboard, restorePrevious: true))
            XCTAssertEqual(pasteboard.string(forType: .string), "result")
            XCTAssertTrue(writer.restoreIfUnchanged(restoration))
            XCTAssertEqual(pasteboard.string(forType: .string), "original")
            let restoredChangeCount = pasteboard.changeCount
            XCTAssertFalse(writer.restoreIfUnchanged(restoration))
            XCTAssertEqual(pasteboard.changeCount, restoredChangeCount)
        }
    }

    func testLaterExternalCopyIsNotOverwritten() throws {
        try withPasteboard { pasteboard in
            writeExternal("original", to: pasteboard)
            let writer = PasteboardTextWriter()
            let restoration = try XCTUnwrap(writer.write("result", to: pasteboard, restorePrevious: true))
            writeExternal("new user copy", to: pasteboard)
            let externalChangeCount = pasteboard.changeCount
            XCTAssertFalse(writer.restoreIfUnchanged(restoration))
            XCTAssertEqual(pasteboard.string(forType: .string), "new user copy")
            XCTAssertEqual(pasteboard.changeCount, externalChangeCount)
        }
    }

    func testExternalCopyOfIdenticalTextStillInvalidatesOwnership() throws {
        try withPasteboard { pasteboard in
            writeExternal("original", to: pasteboard)
            let writer = PasteboardTextWriter()
            let restoration = try XCTUnwrap(writer.write("result", to: pasteboard, restorePrevious: true))
            writeExternal("result", to: pasteboard)
            XCTAssertFalse(writer.restoreIfUnchanged(restoration))
            XCTAssertEqual(pasteboard.string(forType: .string), "result")
        }
    }

    func testOverlappingPastesCarryOriginalBaselineAndRetireOldRestoration() throws {
        try withPasteboard { pasteboard in
            writeExternal("original", to: pasteboard)
            let writer = PasteboardTextWriter()
            let first = try XCTUnwrap(writer.write("first", to: pasteboard, restorePrevious: true))
            let second = try XCTUnwrap(writer.write("second", to: pasteboard, restorePrevious: true))
            XCTAssertFalse(writer.restoreIfUnchanged(first))
            XCTAssertEqual(pasteboard.string(forType: .string), "second")
            XCTAssertTrue(writer.restoreIfUnchanged(second))
            XCTAssertEqual(pasteboard.string(forType: .string), "original")
        }
    }

    func testExternalCopyBetweenPastesBecomesNewBaseline() throws {
        try withPasteboard { pasteboard in
            let writer = PasteboardTextWriter()
            let first = try XCTUnwrap(writer.write("first", to: pasteboard, restorePrevious: true))
            writeExternal("new user copy", to: pasteboard)
            let second = try XCTUnwrap(writer.write("second", to: pasteboard, restorePrevious: true))
            XCTAssertFalse(writer.restoreIfUnchanged(first))
            XCTAssertTrue(writer.restoreIfUnchanged(second))
            XCTAssertEqual(pasteboard.string(forType: .string), "new user copy")
        }
    }

    func testRetainedResultInvalidatesEarlierTemporaryWrite() throws {
        try withPasteboard { pasteboard in
            writeExternal("original", to: pasteboard)
            let writer = PasteboardTextWriter()
            let first = try XCTUnwrap(writer.write("first", to: pasteboard, restorePrevious: true))
            XCTAssertNil(writer.write("keep result", to: pasteboard, restorePrevious: false))
            XCTAssertFalse(writer.restoreIfUnchanged(first))
            XCTAssertEqual(pasteboard.string(forType: .string), "keep result")
        }
    }

    func testOverlappingPastesPreserveEmptyBaselineRatherThanTemporaryText() throws {
        try withPasteboard { pasteboard in
            pasteboard.clearContents()
            let writer = PasteboardTextWriter()
            _ = writer.write("first", to: pasteboard, restorePrevious: true)
            let second = try XCTUnwrap(writer.write("second", to: pasteboard, restorePrevious: true))
            XCTAssertTrue(writer.restoreIfUnchanged(second))
            XCTAssertNil(pasteboard.string(forType: .string))
        }
    }

    func testQueuedInjectionSnapshotsClipboardWhenItActuallyWrites() throws {
        try withPasteboard { pasteboard in
            writeExternal("before focus restoration", to: pasteboard)
            let writer = PasteboardTextWriter()
            var restoration: PasteboardTextWriter.Restoration?
            let transaction = TextInjectionTransaction(
                isValid: { true },
                inject: { done in
                    restoration = writer.write("result", to: pasteboard, restorePrevious: true)
                    done(true)
                }
            )
            writeExternal("copied while waiting", to: pasteboard)
            transaction.perform()
            XCTAssertTrue(writer.restoreIfUnchanged(try XCTUnwrap(restoration)))
            XCTAssertEqual(pasteboard.string(forType: .string), "copied while waiting")
        }
    }

    private func withPasteboard(_ body: (NSPasteboard) throws -> Void) rethrows {
        // Never read or mutate the user's general pasteboard in tests.
        let pasteboard = NSPasteboard(name: .init("VoxtTests.\(UUID().uuidString)"))
        defer { pasteboard.releaseGlobally() }
        try body(pasteboard)
    }

    private func writeExternal(_ text: String, to pasteboard: NSPasteboard) {
        pasteboard.clearContents()
        XCTAssertTrue(pasteboard.setString(text, forType: .string))
    }
}
