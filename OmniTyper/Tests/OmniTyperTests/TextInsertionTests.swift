// SPDX-License-Identifier: Apache-2.0
import AppKit
import ApplicationServices
import Testing
@testable import OmniTyper

struct TextInsertionTests {
    @Test @MainActor func opaqueInputCanPasteOnlyIntoTheCapturedWindow() throws {
        let window = AXUIElementCreateApplication(101)
        func target(window: AXUIElement?, element: AXUIElement? = nil,
                    range: CFRange? = nil, value: String? = nil) -> InsertionTarget {
            InsertionTarget(applicationName: "Test", bundleID: "test", selectedText: "",
                            application: .current, element: element, range: range, value: value,
                            window: window, windowTitle: "Test", document: nil)
        }

        let opaque = target(window: window)
        try opaque.validate(against: target(window: window))
        #expect(throws: Failure.self) {
            try opaque.validate(against: target(window: AXUIElementCreateApplication(102)))
        }
        #expect(throws: Failure.self) {
            try opaque.validate(against: target(window: nil))
        }
        #expect(throws: Failure.self) {
            try target(window: nil).validate(against: target(window: nil))
        }

        let field = AXUIElementCreateApplication(103)
        let accessible = target(window: window, element: field, range: CFRange(location: 2, length: 0), value: "hi")
        try accessible.validate(against: accessible)
        #expect(throws: Failure.self) {
            try accessible.validate(against: target(window: window, element: field,
                                                    range: CFRange(location: 0, length: 0), value: "hi"))
        }
        #expect(throws: Failure.self) {
            try accessible.validate(against: target(window: window, element: field,
                                                    range: CFRange(location: 2, length: 0), value: "changed"))
        }
        #expect(throws: Failure.self) { try accessible.validate(against: opaque) }
    }

    @Test func unchangedCharacterCountReportsAnIgnoredPaste() {
        #expect(TextInsertion.pasteWasIgnored(before: 12, after: 12, inserted: 5, replaced: 0))
        #expect(!TextInsertion.pasteWasIgnored(before: 12, after: 17, inserted: 5, replaced: 0))
        // Note (Yifei Leng): Replacing a selection of equal length leaves the count unchanged on success.
        #expect(!TextInsertion.pasteWasIgnored(before: 12, after: 12, inserted: 5, replaced: 5))
    }

    @Test func unknownCharacterCountNeverReportsAnIgnoredPaste() {
        // Note (Yifei Leng): cmux exposes an AXTextArea without AXNumberOfCharacters whose AXValue stays empty.
        #expect(!TextInsertion.pasteWasIgnored(before: nil, after: nil, inserted: 43, replaced: 0))
        #expect(!TextInsertion.pasteWasIgnored(before: 0, after: nil, inserted: 43, replaced: 0))
        #expect(!TextInsertion.pasteWasIgnored(before: nil, after: 0, inserted: 43, replaced: 0))
    }
}
