#if canImport(DictationCore)
import DictationCore
#endif
import AppKit
import Carbon

/// A single paste with a temporary clipboard lease. Never emits Enter.
@MainActor
final class ClipboardPaste {
    private static let ownerType = NSPasteboard.PasteboardType("local.omni.dictation.clipboard-owner")
    private let pasteboard: NSPasteboard
    private let post: (pid_t?, [CGEvent]) throws -> Void
    private let owner = UUID().uuidString
    private var original: [NSPasteboardItem] = []
    private var ownedRevision: Int?
    private var dispatched = false
    private var used = false
    private var finished = false

    init(pasteboard: NSPasteboard, post: @escaping (pid_t?, [CGEvent]) throws -> Void = { pid, events in
        for event in events {
            if let pid { event.postToPid(pid) }
            else { event.post(tap: .cgSessionEventTap) }
        }
    }) {
        self.pasteboard = pasteboard
        self.post = post
    }

    func perform(text: String, pid: pid_t? = nil, validate: () throws -> Void) throws {
        guard !used, !text.isEmpty else { throw DictationError("本轮粘贴已经处理或文字为空。") }
        used = true
        guard let source = CGEventSource(stateID: .privateState),
              let down = CGEvent(keyboardEventSource: source, virtualKey: CGKeyCode(kVK_ANSI_V), keyDown: true),
              let up = CGEvent(keyboardEventSource: source, virtualKey: CGKeyCode(kVK_ANSI_V), keyDown: false) else {
            throw DictationError("无法创建粘贴事件，请手动复制。")
        }
        // Do not inherit recording-hotkey modifiers or emit return/newline keys.
        down.flags = .maskCommand
        up.flags = .maskCommand

        let revision = pasteboard.changeCount
        original = try snapshot()
        try validate()
        guard pasteboard.changeCount == revision else {
            throw DictationError("准备期间剪贴板发生变化，本轮未粘贴。")
        }
        let item = NSPasteboardItem()
        item.setString(text, forType: .string)
        item.setString(owner, forType: Self.ownerType)
        let clearedRevision = pasteboard.clearContents()
        guard pasteboard.writeObjects([item]) else {
            // Restore only if no other owner has taken over the clipboard we cleared.
            if pasteboard.changeCount == clearedRevision { _ = restoreOriginal() }
            throw DictationError("无法准备剪贴板，请手动复制。")
        }
        ownedRevision = pasteboard.changeCount
        try validate()
        guard stillOwnsClipboard else { throw DictationError("粘贴前剪贴板发生变化，本轮未粘贴。") }
        // Posting has no acknowledgement. A throwing transport could have posted a partial sequence.
        dispatched = true
        try post(pid, [down, up])
    }

    /// Called once on success, failure or cancellation. An unconfirmed queued paste must never
    /// later consume the user's restored old clipboard. Keep the dictated text in that case.
    func finish(confirmed: Bool) -> String? {
        guard !finished else { return nil }
        finished = true
        defer { original = []; ownedRevision = nil }
        guard stillOwnsClipboard else { return nil }
        if dispatched && !confirmed {
            return "剪贴板暂保留本轮文字；请先检查输入框，避免重复粘贴。"
        }
        return restoreOriginal() ? nil : "文字回填处理已结束，但原剪贴板未能完整恢复。"
    }

    private var stillOwnsClipboard: Bool {
        guard let ownedRevision, pasteboard.changeCount == ownedRevision else { return false }
        return pasteboard.pasteboardItems?.count == 1 && pasteboard.string(forType: Self.ownerType) == owner
    }

    private func snapshot() throws -> [NSPasteboardItem] {
        guard let items = pasteboard.pasteboardItems else {
            guard pasteboard.types?.isEmpty != false else { throw DictationError("无法保存现有剪贴板，本轮只提供手动复制。") }
            return []
        }
        return try items.map { source in
            let copy = NSPasteboardItem()
            for type in source.types {
                guard let data = source.data(forType: type), copy.setData(data, forType: type) else {
                    throw DictationError("现有剪贴板含无法保存的格式，本轮只提供手动复制。")
                }
            }
            return copy
        }
    }

    private func restoreOriginal() -> Bool {
        pasteboard.clearContents()
        return original.isEmpty || pasteboard.writeObjects(original)
    }
}
