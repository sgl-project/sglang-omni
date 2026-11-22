import CoreGraphics
import Foundation

// textio.selection's rule for when Accessibility's answer is the whole
// answer, and for the keystrokes around the shortcut: the selection is
// often made with ⌘A in the same motion as the shortcut, so the reading
// can lag the keys, and a modifier still held from the shortcut turns the
// helper's ⌘C into another key. The reading itself is the helper's
// TextIO; the rule is here, pure, so that it is tested.

public enum SelectionRead {
    /// Whether an Accessibility reading holds the selection's text. A
    /// block selection in a web editor (Lark docs in Chrome) reads as one
    /// blank space: a selection the element could not flatten, not text.
    /// The read goes on (to ⌘C) instead of answering with it.
    public static func hasText(_ s: String) -> Bool {
        s.contains { !$0.isWhitespace }
    }

    /// Whether an empty selected range ends the read before ⌘C: only in a
    /// field that takes text, where the caret is all there is. A web
    /// page's text or a PDF page (Chrome's web area, Acrobat's view) has a
    /// range of its own that shows nothing of the selection, so ⌘C reads it
    /// there (#3835). blankSelection: an AX reading gave a blank selected
    /// text, so something is selected and the caret rule does not apply.
    public static func endsAtCaret(range: TextRange?, editable: Bool, blankSelection: Bool) -> Bool {
        editable && range?.length == 0 && !blankSelection
    }

    /// The budgets when the app's params name none: how long a
    /// caret-only answer waits for the selection to settle, and the
    /// pause between readings — ⌘A and the shortcut leave in one motion,
    /// and the accessibility reading lags the keys, so the first reading
    /// can be the caret from before the ⌘A (#4467). A real caret just
    /// answers this much later.
    public static let defaultCaretSettleMs = 600.0
    public static let defaultCaretPollMs = 60.0

    /// Whether the caret-only answer still waits for the selection.
    public static func caretPending(elapsedMs: Double, settleMs: Double) -> Bool {
        elapsedMs < settleMs
    }

    /// Whether held modifiers turn the helper's ⌘C into another key in
    /// the app: Control, Option or Shift still down from the shortcut
    /// (Chrome reads Control+⌘C as no copy, #3835). ⌘ is ⌘C's own
    /// modifier, so a shortcut like Command+C need not wait.
    public static func blocksCopy(_ f: CGEventFlags) -> Bool {
        !f.isDisjoint(with: [.maskControl, .maskAlternate, .maskShift])
    }

    /// How long the ⌘C waits for those keys to lift before posting
    /// anyway, when the app's params name none.
    public static let defaultModifierWaitMs = 800.0
}
