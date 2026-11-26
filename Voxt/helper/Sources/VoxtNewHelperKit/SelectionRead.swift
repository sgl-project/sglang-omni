import Foundation

// textio.selection's rule for when Accessibility's answer is the whole
// answer. The reading itself is the helper's TextIO; the rule is here,
// pure, so that it is tested.

public enum SelectionRead {
    /// Whether an Accessibility reading holds the selection's text. A
    /// block selection in a web editor (Lark docs in Chrome) reads as one
    /// blank space, an image-only block selection as the object
    /// replacement character U+FFFC: a selection the element could not
    /// flatten, not text. The read goes on (to ⌘C) instead of answering
    /// with it.
    public static func hasText(_ s: String) -> Bool {
        s.contains { !$0.isWhitespace && $0 != "\u{FFFC}" }
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
}
