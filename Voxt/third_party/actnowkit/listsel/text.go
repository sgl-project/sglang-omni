package listsel

import "strings"

// Items returns the items of items whose key is selected, in their order:
// the selected lines of the lines shown, to copy.
func Items[K comparable, T any](m *Model[K], items []T, key func(T) K) []T {
	if m.Empty() {
		return nil
	}
	out := make([]T, 0, m.Count())
	for _, it := range items {
		if m.IsSelected(key(it)) {
			out = append(out, it)
		}
	}
	return out
}

// Text formats items one per line, as the clipboard takes copied lines:
// format gives a line (without a newline). Several lines each end in a
// newline, as a file's do; one line has none, to paste into a sentence or
// a search field.
func Text[T any](items []T, format func(T) string) string {
	if len(items) == 1 {
		return format(items[0])
	}
	var b strings.Builder
	for _, it := range items {
		b.WriteString(format(it))
		b.WriteByte('\n')
	}
	return b.String()
}

// Wording is a reference text in English (the default) and Simplified
// Chinese, with a printf verb (%d) where the number of rows goes.
// actnowkit does no i18n: apps copy these into their own catalogues.
type Wording struct{ EN, ZH string }

// The wordings of the log views' selection (notices/log-selection-spec.md
// in the ActNow project). The English plurals (%d line / %d lines) are the
// app's catalogue's concern.
var (
	// TextSelected: the header while rows are selected.
	TextSelected = Wording{EN: "%d lines selected", ZH: "已选中 %d 行"}
	// TextCopy: the header's button that copies the rows selected.
	TextCopy = Wording{EN: "Copy", ZH: "复制"}
	// TextDeselect: the header's button that clears the selection.
	TextDeselect = Wording{EN: "Deselect", ZH: "取消选择"}
	// TextCopyLine: the context menu's item for the row clicked.
	TextCopyLine = Wording{EN: "Copy this line", ZH: "复制这一行"}
	// TextCopySelected: the context menu's item for the rows selected.
	TextCopySelected = Wording{EN: "Copy %d selected lines", ZH: "复制选中的 %d 行"}
	// TextCopyShown: the context menu's item (and the header's button
	// without a selection) for every row shown.
	TextCopyShown = Wording{EN: "Copy all lines shown", ZH: "复制全部显示的行"}
	// TextCopied: the toast after copying.
	TextCopied = Wording{EN: "Copied %d lines", ZH: "已复制 %d 行"}
)
