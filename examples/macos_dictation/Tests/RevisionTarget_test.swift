import AppKit
import Foundation

@MainActor
private final class Editor {
    var draft = TextDraft(value: "前文：", selection: NSRange(location: 3, length: 0))
    var focused = true
    var rejectsSelection = false
    var writes = 0
    let board = NSPasteboard(name: .init("omni-revision-test-\(UUID())"))
    var onPaste: (() -> Void)?

    func anchor(_ text: String) -> EditableTextAnchor {
        EditableTextAnchor(text: text, before: draft, read: { self.draft }, validateIdentity: {
            guard self.focused else { throw DictationError("Wrong input") }
        }, select: { range in
            guard !self.rejectsSelection else { throw DictationError("Selection is read only") }
            self.draft = TextDraft(value: self.draft.value, selection: range)
        }, makePaste: {
            ClipboardPaste(pasteboard: self.board, post: { _, _ in
                let text = self.board.string(forType: .string)!
                self.draft = TextDraft(value: try self.draft.inserting(text),
                                      selection: NSRange(location: self.draft.selection.location + text.utf16.count, length: 0))
                self.writes += 1
                self.onPaste?()
            })
        }, deleteSelection: {
            self.draft = TextDraft(value: try self.draft.inserting(""),
                                  selection: NSRange(location: self.draft.selection.location, length: 0))
            self.writes += 1
        })
    }

    func insert(_ text: String) throws {
        draft = TextDraft(value: try draft.inserting(text),
                          selection: NSRange(location: draft.selection.location + text.utf16.count, length: 0))
    }
}

@main
private enum RevisionTargetTests {
    @MainActor
    static func main() async throws {
        let text = "明天和张三开会。"
        let revised = "明天和张珊开会。"
        let editor = Editor()
        defer { editor.board.releaseGlobally() }
        let anchor = editor.anchor(text)
        try editor.insert(text)
        let confirmed = await anchor.confirmInsertion()
        precondition(confirmed)
        try editor.insert("后文由用户追加。")
        try anchor.prepare(original: text)
        let applied = try await anchor.apply(TextRevision(original: text, corrected: revised))
        precondition(applied && editor.draft.value == "前文：\(revised)后文由用户追加。")
        precondition(editor.board.string(forType: .string) == revised, "Clipboard should hold the full revision")
        anchor.cancel()
        try anchor.prepare(original: revised)
        let shortened = try await anchor.apply(TextRevision(original: revised, corrected: "明天开会。"))
        precondition(shortened && editor.draft.value == "前文：明天开会。后文由用户追加。")
        precondition(editor.writes == 2, "Each revision must write once")

        let copied = Editor()
        defer { copied.board.releaseGlobally() }
        let copiedTarget = copied.anchor(text)
        try copied.insert(text)
        let copiedConfirmed = await copiedTarget.confirmInsertion()
        precondition(copiedConfirmed)
        try copiedTarget.prepare(original: text)
        copied.onPaste = {
            copied.board.clearContents()
            copied.board.setString("user's newer copy", forType: .string)
        }
        let copiedApplied = try await copiedTarget.apply(TextRevision(original: text, corrected: revised))
        precondition(copiedApplied && copied.board.string(forType: .string) == "user's newer copy")
        copiedTarget.cancel()

        for scenario in ["originalEdited", "duringCorrection", "otherField", "readOnly", "cancel"] {
            let editor = Editor()
            defer { editor.board.releaseGlobally() }
            let target = editor.anchor(text)
            try editor.insert(text)
            let confirmed = await target.confirmInsertion()
            precondition(confirmed)
            if scenario == "originalEdited" {
                editor.draft = TextDraft(value: "用户已改写全部", selection: NSRange(location: 0, length: 0))
            }
            do {
                try target.prepare(original: text)
                if scenario == "duringCorrection" { try editor.insert("用户新输入") }
                if scenario == "otherField" { editor.focused = false }
                if scenario == "readOnly" { editor.rejectsSelection = true }
                if scenario == "cancel" { target.cancel() }
                _ = try await target.apply(TextRevision(original: text, corrected: revised))
                preconditionFailure("Unsafe edit must fail: \(scenario)")
            } catch { }
            precondition(editor.writes == 0, "Must not write when validation fails")
            target.cancel()
        }
        print("PASS: exact span replacement, trailing text, successive revisions, clipboard and invalidated targets")
    }
}
