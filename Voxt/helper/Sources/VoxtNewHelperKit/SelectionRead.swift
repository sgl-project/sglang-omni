import Foundation

// textio.selection's rule for when Accessibility's answer is the whole
// answer. The reading itself is the helper's TextIO; the rule is here,
// pure, so that it is tested.

public enum SelectionRead {
    /// Whether an empty selected range ends the read before ⌘C: only in a
    /// field that takes text, where the caret is all there is. A web
    /// page's text or a PDF page (Chrome's web area, Acrobat's view) has a
    /// range of its own that shows nothing of the selection, so ⌘C reads it
    /// there (#3835).
    public static func endsAtCaret(range: TextRange?, editable: Bool) -> Bool {
        editable && range?.length == 0
    }
}
