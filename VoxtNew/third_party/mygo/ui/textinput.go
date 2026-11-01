package ui

import (
	"slices"

	transfer "github.com/egoist/mygo/transfer"
)

// TextInput creates a single-line text input editing *value.
func coreTextInput(c *context, value *string) *node { return textInput(c, value, false) }

// TextArea creates a multi-line text input editing *value.
func coreTextArea(c *context, value *string) *node { return textInput(c, value, true) }

func textInput(c *context, value *string, multiline bool) *node {
	t := c.theme
	e := textInputBase(c, value, multiline)
	e.Padding(t.Space(1.5), t.Space(2.5)).Radius(t.Radius).Background(t.Surface).Border(1, t.Border)
	if multiline {
		e.MinHeight(t.Space(20))
	}
	e.styleFn = func(e *node) { inputBorder(t, e, e) }
	return e
}

func textInputBase(c *context, value *string, multiline bool) *node {
	e := c.newElement(kindInput)
	e.flags |= flagEditable | flagFocusable | flagHover
	e.widget = "TextInput"
	if multiline {
		e.widget = "TextArea"
		// It scrolls its text as a scroll container does its children.
		e.flags |= flagScrollY
	}
	st := e.st
	if st.editor == nil || st.editor.document != nil {
		st.editor = newEditor(c.rt.now)
		st.editor.setText(*value)
		st.editor.caret, st.editor.anchor = st.editor.buf.n, st.editor.buf.n
	}
	ed := st.editor
	ed.value = value

	if ed.client == nil {
		ed.client = &widgetTextInput{ed: ed, rt: c.rt, id: e.id}
	}
	e.textClient = ed.client
	ed.multiline = multiline
	if multiline && ed.area == nil {
		ed.area = &area{reveal: true}
	}
	// The input shares the string of *value: the same string is equal at
	// once, whatever its length.
	if !ed.nativeDirty && ed.buf.s != *value {
		ed.setText(*value)
		ed.compose = ""
	}
	ed.rebuild()
	ed.ranges = ed.ranges[:0]
	e.onValueInput(stringInput)

	return e
}

// Placeholder shows s in an empty text input.
func (e *node) Placeholder(s string) *node {
	if e.st.editor != nil {
		e.st.editor.placeholder = s
	}
	return e
}

// ReadOnly makes a text input show its text without letting the user
// change it: the text can still be selected and copied, from the keyboard
// too, as it takes the focus, without a caret; assistive technology reads
// it as read-only.
func (e *node) ReadOnly(on bool) *node {
	if ed := e.st.editor; ed != nil && e.flags&flagEditable != 0 {
		ed.readOnly = on
		if on {
			ed.compose = ""
		}
	}
	return e
}

// OnPaste has fn see a paste into a text input, by Cmd+V, Ctrl+V or a
// Paste menu item, before it is inserted, on the main thread as the paste
// comes: text is the clipboard's text, empty when it holds none, as with
// an image, which the app reads from the clipboard itself. fn reports
// whether it took the paste; a paste taken changes nothing, its selection
// and undo history included, as when a web page prevents a paste's
// default and attaches a large block instead of inserting it. Like
// ReadOnly, it holds for the input that comes until the next frame.
func (e *node) OnPaste(fn func(text string) bool) *node {
	if ed := e.st.editor; ed != nil && e.flags&flagEditable != 0 {
		ed.onPaste = fn
	}
	return e
}

// OnTextKey has fn see a key pressed in the text input before the input
// takes it, on the main thread as the key comes, with the text typed
// before it in place, as AppKit's doCommandBySelector and a web page's
// keydown on a text field: Enter that sends a message, Up that edits the
// last one, Tab that accepts a completion, Escape that cancels, Cmd+Enter.
// fn reports whether it took the key; one taken changes nothing but what
// fn sets: a text area inserts no newline, a single-line input does not
// submit, Tab does not move the focus. One left goes on as it would
// without fn.
//
// fn sees the keys that type no text: those with Cmd or Ctrl, and Enter,
// Tab, Escape, Backspace, Delete, the arrows and the other keys that move
// the caret, as pressed (macOS's Control keys that move the caret too).
// While an input method composes, its keys are its own. Keys that the
// window or an element around the focus handles as a Shortcut go to the
// shortcut first, as with HandleInput, unless the input edits with them,
// as with Enter and the arrows. Like ReadOnly, OnTextKey holds for the
// input that comes until the next frame.
//
// fn runs between frames, where an Element kept from the view is no longer
// valid: it reads and replaces the text through k instead, as a chat
// composer sends with Enter, clears what it sent, and breaks the line with
// Shift+Enter:
//
//	ui.TextArea(c, &app.draft).OnTextKey(func(k ui.TextKey) bool {
//		if k.Key != ui.KeyEnter || k.Mods != 0 {
//			return false // Shift+Enter breaks the line
//		}
//		app.send(k.Text())
//		k.SetText("")
//		return true
//	})
func (e *node) OnTextKey(fn func(k TextKey) bool) *node {
	if ed := e.st.editor; ed != nil && e.flags&flagEditable != 0 {
		ed.onTextKey = fn
	}
	return e
}

// A TextKey is a key pressed in a text input, as OnTextKey's fn sees it.
// It is valid while fn runs: Text and SetText panic after fn returns.
type TextKey struct {
	Key  Key
	Mods Modifiers
	ed   *editor
	seq  uint64
}

// valid panics unless k's fn is running.
func (k TextKey) valid() *editor {
	if k.ed == nil || !k.ed.keying || k.ed.keySeq != k.seq {
		panic("ui: a TextKey is valid only while its OnTextKey fn runs")
	}
	return k.ed
}

// Text returns the text of the input as the key comes, with what was typed
// before it, which may be ahead of the bound value: that takes what is
// typed as the next frame is built.
func (k TextKey) Text() string { return k.valid().String() }

// SetText replaces the text of the input, and its bound value, as the app
// does by setting the value as the view is built: a composer clears the
// draft it sent, Up puts the last message back. Writing the bound value
// from fn is not enough, as the editor writes what it holds over it.
func (k TextKey) SetText(s string) {
	ed := k.valid()
	if ed.document != nil {
		// A TextBuffer takes the text as when the app sets it.
		ed.document.Set(s)
		ed.loadBuffer(ed.document.Snapshot())
		ed.bufferDirty = false
	} else {
		ed.setText(s)
		ed.compose = ""
		if ed.value != nil {
			*ed.value = s
		}
	}
	if w := ed.client; w != nil {
		// What the view built before the key shows the old text.
		w.rt.consumed = true
		w.rt.requestFrame()
	}
}

// Lines makes a text area as high as its text, wrapped at its width, from
// min lines up to max, past which it scrolls, as a message field grows with
// what is typed: TextArea(c, &draft).Lines(1, 8). Without it, a text area is
// as high as its paragraphs, three lines at least, unless given a height.
func (e *node) Lines(min, max int) *node {
	if ed := e.st.editor; ed != nil && ed.area != nil {
		ed.lines = [2]int{min, max}
	}
	return e
}

// TextSelection returns the selection of a text input as offsets in runes
// into its text, the caret where start equals end, as the app reads it to
// complete the word being typed.
func (e *node) TextSelection() (start, end int) {
	ed := e.st.editor
	if ed == nil {
		return 0, 0
	}
	return ed.selection()
}

// TextAnchorCaret returns the selection of a text input as offsets in runes
// into its text, from where it was started to its caret: the anchor is after
// the caret when the selection was made backwards.
func (e *node) TextAnchorCaret() (anchor, caret int) {
	ed := e.st.editor
	if ed == nil {
		return 0, 0
	}
	return ed.anchor, ed.caret
}

// SetTextSelection selects the runes of a text input from start to end, or
// puts the caret at start when they are equal, and scrolls it into view: the
// caret after a word the app completed. The offsets are kept within the text.
func (e *node) SetTextSelection(start, end int) *node {
	ed := e.st.editor
	if ed == nil {
		return e
	}
	n := ed.buf.n
	start, end = min(max(start, 0), n), min(max(end, 0), n)
	ed.editorSelection = editorSelection{anchor: start, caret: end}
	ed.hasDesired = false
	if ed.area != nil {
		ed.area.reveal = true
	}
	e.c.rt.requestFrame()
	return e
}

// Composing reports whether an input method composes text in a text
// input, as Pinyin before a candidate is chosen: its value holds the text
// once composed. Keys typed meanwhile are the input method's, as Enter
// choosing a candidate: they submit nothing and press no shortcut.
func (e *node) Composing() bool {
	ed := e.st.editor
	return ed != nil && ed.compose != ""
}

// Password hides what a text input holds, in the frames that call it: an
// input that stops calling it shows its text again, as a field's eye
// button does. It does nothing to a text area: as on every platform, only
// single-line fields hide their text.
func (e *node) Password() *node {
	if ed := e.st.editor; ed != nil && !ed.multiline {
		ed.password = true
	}
	return e
}

// Selectable lets the user select and copy text by dragging, double-clicking
// a word, triple-clicking a line, or using Shift with the arrows. Its context
// menu has Copy and Select All; Cmd+C copies the selection.
//
// On a container, its Text and RichText descendants share one selection.
// Copy joins their selected text with newlines, in the order they were built.
// Nested Selectable containers have independent selections. Text inside
// controls, such as buttons and text inputs, keeps the control's interaction.
// Unselectable excludes a subtree. On a Text or RichText alone, the selection
// stays within that paragraph. Inline elements share their paragraph's
// selection; set Selectable on the paragraph.
func (e *node) Selectable() *node {
	if e.isInline() {
		return e
	}
	if e.kind == kindText || e.kind == kindBox {
		e.flags &^= flagUnselectable
		e.flags |= flagSelectable
	}
	return e
}

// SelectedText is the part of a paragraph that the selection of a
// Selectable container holds.
type SelectedText struct {
	// ID is the paragraph's element's (Element.ID).
	ID uint64
	// Text is the paragraph's text, and Start and End the runes of it the
	// selection holds.
	Text       string
	Start, End int
}

// SelectedTexts returns the parts of the paragraphs that the selection of
// a Selectable container holds, in the order they were built, or none,
// as of the last frame: to copy them otherwise, as a context menu's item
// does.
func (e *node) SelectedTexts() []SelectedText {
	return e.c.rt.selectedTexts(e.id)
}

// CopyData sets what Copy, its shortcut and the context menu's put on
// the clipboard for the selection of a Selectable container: fn gets its
// parts, as SelectedTexts returns them, and returns the data, as a
// document copies its paragraphs with blank lines between them as text
// (transfer.TextData), and as HTML too to keep their formatting.
func (e *node) CopyData(fn func(parts []SelectedText) transfer.Data) *node {
	e.copyData = fn
	return e
}

// Unselectable excludes a paragraph or container subtree from a surrounding
// Selectable container. A Selectable container nested inside it can provide
// a selection of its own. For inline elements, set it on their paragraph.
func (e *node) Unselectable() *node {
	if e.isInline() {
		return e
	}
	e.flags &^= flagSelectable
	e.flags |= flagUnselectable
	return e
}

// TextRange styles runes Start up to End of a text input's text: their
// color, when its alpha is not zero, and their weight, when Weight is set,
// as a message field shows the mentions in what is typed.
type TextRange struct {
	Start, End int
	Color      Color
	Weight     int
}

// TextRanges styles runs of a text input's text, in the frames that call
// it. Ranges may overlap and nest, as the attributes of an attributed
// string do: where they do, each of Color and Weight is that of the last
// range given that sets it, across the calls of the frame. They follow the
// text as it is: an app that finds them in the text finds them again as it
// changes. A password shows none; an input method's composition shows
// unstyled between them.
func (e *node) TextRanges(ranges ...TextRange) *node {
	ed := e.st.editor
	if ed == nil || e.flags&flagEditable == 0 {
		return e
	}
	for _, r := range ranges {
		if r.End > r.Start {
			ed.ranges = append(ed.ranges, r)
			ed.apart = false
		}
	}
	return e
}

// rangeMerge is what sortRanges reuses from frame to frame.
type rangeMerge struct {
	given          []TextRange
	points, starts []int
	all, colors    rangeHeap
	weights        rangeHeap
}

// sortRanges splits the ranges into runs apart, sorted by start, where
// each of Color and Weight is that of the last range given setting it, as
// TextRanges says, once the frame has given them all: a frame calling
// TextRanges for each range merges them once. It takes O(n log n) for n
// ranges: a sweep over their ends, with the ranges covering it on heaps
// by the order they were given in.
func (ed *editor) sortRanges() {
	m := &ed.merge
	given := append(m.given[:0], ed.ranges...)
	points, starts := m.points[:0], m.starts[:0]
	for i, r := range given {
		points = append(points, r.Start, r.End)
		starts = append(starts, i)
	}
	slices.Sort(points)
	points = slices.Compact(points)
	slices.SortFunc(starts, func(a, b int) int { return given[a].Start - given[b].Start })
	m.all, m.colors, m.weights = m.all[:0], m.colors[:0], m.weights[:0]
	out, next := ed.ranges[:0], 0
	for k := 0; k+1 < len(points); k++ {
		from, to := points[k], points[k+1]
		for ; next < len(starts) && given[starts[next]].Start <= from; next++ {
			i := starts[next]
			m.all.push(i)
			if given[i].Color.A > 0 {
				m.colors.push(i)
			}
			if given[i].Weight != 0 {
				m.weights.push(i)
			}
		}
		if _, ok := m.all.top(given, from); !ok {
			continue // between ranges
		}
		r := TextRange{Start: from, End: to}
		if i, ok := m.colors.top(given, from); ok {
			r.Color = given[i].Color
		}
		if i, ok := m.weights.top(given, from); ok {
			r.Weight = given[i].Weight
		}
		if n := len(out); n > 0 && out[n-1].End == from && out[n-1].Color == r.Color && out[n-1].Weight == r.Weight {
			out[n-1].End = to // a run as one, as an attributed string's
			continue
		}
		out = append(out, r)
	}
	m.given, m.points, m.starts = given[:0], points[:0], starts[:0]
	ed.ranges, ed.apart = out, true
}

// rangeHeap is a heap of ranges by the order they were given in, the last
// on top.
type rangeHeap []int

func (h *rangeHeap) push(i int) {
	*h = append(*h, i)
	s := *h
	for j := len(s) - 1; j > 0; {
		p := (j - 1) / 2
		if s[p] >= s[j] {
			break
		}
		s[p], s[j] = s[j], s[p]
		j = p
	}
}

// top returns the last range given that covers at, taking off those that
// end by it, which cover nothing after it either.
func (h *rangeHeap) top(given []TextRange, at int) (int, bool) {
	s := *h
	for len(s) > 0 && given[s[0]].End <= at {
		n := len(s) - 1
		s[0] = s[n]
		s = s[:n]
		for j := 0; ; {
			c := 2*j + 1
			if c >= n {
				break
			}
			if c+1 < n && s[c+1] > s[c] {
				c++
			}
			if s[j] >= s[c] {
				break
			}
			s[j], s[c] = s[c], s[j]
			j = c
		}
	}
	*h = s
	if len(s) == 0 {
		return 0, false
	}
	return s[0], true
}
