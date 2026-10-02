import AppKit
import Darwin

@MainActor
private final class DelayedEditor {
    var draft: TextDraft
    var focused = true
    var selections = 0
    var writes = 0
    var selectionTask: Task<Void, Never>?
    var afterSelection: (() -> Void)?
    let board = NSPasteboard.withUniqueName()

    init(_ value: String) { draft = TextDraft(value: value, selection: NSRange(location: value.utf16.count, length: 0)) }

    func anchor(text: String, scenario: String) -> EditableTextAnchor {
        EditableTextAnchor(text: text, before: TextDraft(value: "", selection: NSRange(location: 0, length: 0)),
            read: { self.draft }, validateIdentity: {
                guard self.focused else { throw DictationError("Focus changed") }
            }, select: { range in
                self.selections += 1
                self.selectionTask = Task { @MainActor in
                    try? await Task.sleep(nanoseconds: 60_000_000)
                    guard !Task.isCancelled else { return }
                    switch scenario {
                    case "never": return
                    case "focus": self.focused = false
                    case "edited": self.draft = TextDraft(value: "用户新输入", selection: NSRange(location: 0, length: 0))
                    case "cancel": self.afterSelection?()
                    default: self.draft = TextDraft(value: self.draft.value, selection: range)
                    }
                }
            }, makePaste: {
                ClipboardPaste(pasteboard: self.board, post: { _, _ in
                    let replacement = self.board.string(forType: .string)!
                    self.draft = TextDraft(value: try self.draft.inserting(replacement), selection: NSRange(location: 0, length: 0))
                    self.writes += 1
                })
            }, deleteSelection: { throw DictationError("Deletion not used in this test") })
    }
}

@main
private enum RevisionSelectionDelayTests {
    @MainActor
    static func main() async {
        do { try await run() }
        catch { fputs("FAIL: \(error.localizedDescription)\n", stderr); exit(1) }
    }

    @MainActor
    static func run() async throws {
        let original = "大家好，我们是S G浪组合。", corrected = "大家好，我们是S G Lang组合。"
        for scenario in ["delayed", "never", "focus", "edited", "cancel"] {
            let editor = DelayedEditor(original)
            defer { editor.selectionTask?.cancel(); editor.board.releaseGlobally() }
            let anchor = editor.anchor(text: original, scenario: scenario)
            defer { anchor.cancel() }
            editor.afterSelection = { anchor.cancel() }
            guard await anchor.confirmInsertion() else { throw DictationError("Initial paste must confirm") }
            try anchor.prepare(original: original)
            var applied = false
            var message = ""
            do { applied = try await anchor.apply(TextRevision(original: original, corrected: corrected)) }
            catch { message = error.localizedDescription }
            if scenario == "delayed" {
                guard applied, editor.valueMatches(corrected), editor.writes == 1, editor.selections == 1,
                      editor.board.string(forType: .string) == corrected else {
                    throw DictationError("Must await AX selection acknowledgement and paste once: \(message)")
                }
            } else {
                guard !applied, editor.writes == 0, editor.selections == 1 else {
                    throw DictationError("Must not write for \(scenario)")
                }
                if scenario == "never", !message.contains("选区") { throw DictationError("Selection timeout needs an actionable reason") }
            }
            editor.afterSelection = nil
        }
        print("PASS: delayed selection acknowledgement, one paste, selection timeout, focus/content changes and cancellation")
    }
}

private extension DelayedEditor {
    func valueMatches(_ expected: String) -> Bool { draft.value.utf16.elementsEqual(expected.utf16) }
}
