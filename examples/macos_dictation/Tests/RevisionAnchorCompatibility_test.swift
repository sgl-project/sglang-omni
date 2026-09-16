import AppKit
import Darwin

@MainActor
private final class DraftEditor {
    var draft: TextDraft
    var focused = true
    var writes = 0
    let board = NSPasteboard(name: .init("omni-anchor-compatibility-\(UUID())"))
    init(_ draft: TextDraft) { self.draft = draft }
    func anchor(text: String, before: TextDraft) -> EditableTextAnchor {
        EditableTextAnchor(text: text, before: before, read: { self.draft }, validateIdentity: {
            guard self.focused else { throw DictationError("Input focus changed") }
        }, select: { self.draft = TextDraft(value: self.draft.value, selection: $0) }, makePaste: {
            ClipboardPaste(pasteboard: self.board, post: { _, _ in
                self.draft = TextDraft(value: try self.draft.inserting(self.board.string(forType: .string)!),
                                       selection: NSRange(location: 0, length: 0))
                self.writes += 1
            })
        }, deleteSelection: { throw DictationError("Deletion not used in this fixture") })
    }
}

@main
private enum RevisionAnchorCompatibilityTests {
    @MainActor
    static func main() async {
        do { try await run() }
        catch { fputs("FAIL: \(error.localizedDescription)\n", stderr); exit(1) }
    }

    @MainActor
    static func run() async throws {
        let text = "大家好，我们是S.G.浪组合。", revised = "大家好，我们是SGLang组合。"
        for (name, beforeValue, actualValue) in [
            ("unchanged AX caret", "", text),
            ("editor trailing newline", "", text + "\n"),
            ("empty editor placeholder", "\n", text),
        ] {
            let editor = DraftEditor(TextDraft(value: actualValue, selection: NSRange(location: 0, length: 0)))
            defer { editor.board.releaseGlobally() }
            let anchor = editor.anchor(text: text, before: TextDraft(value: beforeValue, selection: NSRange(location: 0, length: 0)))
            defer { anchor.cancel() }
            guard await anchor.confirmInsertion() else { throw DictationError("Must locate exact pasted text: \(name)") }
            try anchor.prepare(original: text)
            let applied = try await anchor.apply(TextRevision(original: text, corrected: revised))
            let expected = revised + (actualValue.hasSuffix("\n") ? "\n" : "")
            guard applied, editor.draft.value == expected, editor.writes == 1 else {
                throw DictationError("Revision must preserve editor suffix: \(name)")
            }
        }
        // A stale selection or editor placeholder can make the predicted draft longer
        // than the delivered paragraph. The entire field still identifies it exactly.
        for start in [0, 5] {
            for suffix in ["", "\n"] {
                let editor = DraftEditor(TextDraft(value: text + suffix, selection: NSRange(location: 0, length: 0)))
                defer { editor.board.releaseGlobally() }
                let before = TextDraft(value: "输入占位符", selection: NSRange(location: start, length: 0))
                let anchor = editor.anchor(text: text, before: before)
                defer { anchor.cancel() }
                guard await anchor.confirmInsertion() else {
                    throw DictationError("Must locate a paragraph that occupies the entire field despite a stale pre-paste snapshot")
                }
                editor.draft = TextDraft(value: editor.draft.value + "用户追加的后文。", selection: NSRange(location: 0, length: 0))
                try anchor.prepare(original: text)
                let corrected = "大家好，我们是S.G.Lang组合。"
                let applied = try await anchor.apply(TextRevision(original: text, corrected: corrected))
                guard applied, editor.draft.value == corrected + suffix + "用户追加的后文。", editor.writes == 1 else {
                    throw DictationError("Whole-field confirmation must replace only the changed fragment")
                }
            }
        }
        for (name, capturedText, value, focused) in [
            ("different content", text, "用户新文字", true),
            ("moved span", text, "前缀" + text, true),
            ("focus changed", text, text, false),
            ("inserted newline is missing", text + "\n", text, true),
            ("duplicate paragraph", text, text + text, true),
            ("changed paragraph", text, "大家好，我们是其他组合。", true),
        ] {
            let editor = DraftEditor(TextDraft(value: value, selection: NSRange(location: 0, length: 0)))
            defer { editor.board.releaseGlobally() }
            editor.focused = focused
            let anchor = editor.anchor(text: capturedText, before: TextDraft(value: "", selection: NSRange(location: 0, length: 0)))
            defer { anchor.cancel() }
            guard !(await anchor.confirmInsertion()), editor.writes == 0 else {
                throw DictationError("Must reject: \(name)")
            }
        }
        let unchanged = DraftEditor(TextDraft(value: text, selection: NSRange(location: 0, length: 0)))
        defer { unchanged.board.releaseGlobally() }
        let unpasted = unchanged.anchor(text: text, before: unchanged.draft)
        defer { unpasted.cancel() }
        guard !(await unpasted.confirmInsertion()), unchanged.writes == 0 else {
            throw DictationError("An unchanged pre-existing paragraph must not confirm an unobserved paste")
        }
        print("PASS: exact-span and whole-field confirmation with stale snapshot/caret and editor newline; ambiguous or unchanged drafts rejected")
    }
}
