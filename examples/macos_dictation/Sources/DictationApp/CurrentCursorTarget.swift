#if canImport(DictationCore)
import DictationCore
#endif
import AppKit
import ApplicationServices

/// Lets the system route Cmd+V to the focus at completion without reading editor state.
@MainActor
final class CurrentCursorTarget: DictationTextTarget {
    let applicationName = "当前光标"
    let supportsConfirmation = false
    let dismissesFeedbackAfterPaste = true
    let preparationMessage = "识别完成后粘贴到当前光标，可切换输入位置。"
    private let paste: ClipboardPaste
    private let canPost: () -> Bool

    convenience init() {
        self.init(paste: ClipboardPaste(pasteboard: .general),
                  canPost: { AXIsProcessTrusted() && CGPreflightPostEventAccess() })
    }

    init(paste: ClipboardPaste, canPost: @escaping () -> Bool) {
        self.paste = paste
        self.canPost = canPost
    }

    func validate() throws {
        guard canPost() else { throw DictationError("请在设置中授权辅助功能，再重新开始录音。") }
    }

    func insert(_ text: String) throws {
        try validate()
        try paste.perform(text: text, validate: { try self.validate() })
    }

    func confirms(_ text: String) -> Bool { false }
    func stopObserving() {}
    func finishInsertion(confirmed: Bool) -> String? { paste.finish(confirmed: false) }
}
