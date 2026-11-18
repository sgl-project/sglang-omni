package ui

import (
	"strings"
	"time"
	"unicode/utf8"

	"github.com/egoist/mygo/internal/text"
)

type editKind uint8

const (
	editKey editKind = iota
	editInsert
	editCompose
	editCommand
)

type editEvent struct {
	kind  editKind
	mods  Modifiers
	key   Key
	text  string
	caret int
	// replace makes an insertion or a composition take the runes from to
	// to instead of the selection, as an input method asked.
	replace  bool
	from, to int
}

// editor is the state of a text input: the text, the caret, the selection,
// the composition of an input method, undo history and what the last frame
// laid out.
type editor struct {
	buf       buffer
	graphemes graphemes
	editorSelection
	compose            string
	composeCaret       int
	composeSelected    TextInputRange
	compositionActive  bool // one undo transaction for preedit and its commit
	client             *widgetTextInput
	value              *string
	nativeDirty        bool
	nativeValue        string // bound value before the next widget build publishes input
	document           *TextBuffer
	published          TextSnapshot
	bufferDirty        bool
	nativeBufferBefore TextSnapshot
	queue              []editEvent
	multiline          bool
	readOnly           bool   // selectable text: selected and copied, not edited
	source             string // the text of selectable text
	password           bool
	// onPaste sees a paste before it is inserted (OnPaste).
	onPaste func(text string) bool
	// onTextKey sees a key before the editor takes it (OnTextKey).
	onTextKey func(k TextKey) bool
	// keying is set while onTextKey runs, as the TextKey numbered keySeq
	// is valid.
	keying bool
	keySeq uint64
	// leaveEmptyBackspace leaves Backspace to shortcuts while the text is
	// empty, as a token field's input does to take out a token.
	leaveEmptyBackspace bool
	// shown are the options the build before set, of the input the user
	// typed in, which its edits apply by (see withShown).
	shown editorOptions
	// lines are a text area's least and most lines (Lines), as high as its
	// wrapped text between them; zero for the paragraphs, at least three.
	lines       [2]int
	placeholder string
	// ranges style runs of the text (TextRanges): as given until apart is
	// set, then merged into runs apart, sorted by Start (sortRanges).
	ranges []TextRange
	apart  bool
	merge  rangeMerge
	// layout is what the last frame laid out of a single-line input or a
	// selectable text; area lays out a text area (textarea.go).
	layout           *text.Layout
	layoutVersion    uint64
	layoutGeneration uint64
	layoutCompose    string
	layoutPassword   bool
	area             *area
	// display maps runes of the text to runes of the layout, which shows
	// bullets for passwords and holds the composition.
	scrollX          float32
	originX, originY float32 // content box, relative to the element
	desiredX         float32
	hasDesired       bool
	undo, redo       []undoStep
	lastEdit         time.Time
	// now is the time, the engine's (engine.now), for what depends on it.
	now       func() time.Time
	coalesce  bool
	dragging  bool
	dragUnit  int // 1 rune, 2 word, 3 line
	dragStart [2]int
	pressMods Modifiers
	contentW  float32
}

func newEditor(now func() time.Time) *editor { return &editor{now: now} }

func (ed *editor) String() string { return ed.buf.string() }

// edit replaces the runes from a to z with s, and tells the area.
func (ed *editor) edit(a, z int, s string) {
	first, old, after := ed.buf.replace(a, z, s)
	if ed.area != nil {
		ed.area.edited(&ed.buf, first, old, after)
	}
}

// editorOptions are the options of a text input that its fluent methods
// set in each build and its edits follow.
type editorOptions struct {
	readOnly, password bool
	onPaste            func(text string) bool
	onTextKey          func(k TextKey) bool
}

// rebuild starts a build of the input: what the build before set becomes
// the options its pending edits apply by, and this build's start from
// the defaults.
func (ed *editor) rebuild() {
	ed.shown = editorOptions{ed.readOnly, ed.password, ed.onPaste, ed.onTextKey}
	ed.readOnly, ed.password, ed.lines, ed.onPaste, ed.onTextKey = false, false, [2]int{}, nil, nil
}

// withShown runs fn, which applies the user's edits, with the options of
// the build the user typed in: input reaches an input as its constructor
// returns, before this build's ReadOnly or OnPaste, and follows what the
// user saw.
func (ed *editor) withShown(fn func()) {
	now := editorOptions{ed.readOnly, ed.password, ed.onPaste, ed.onTextKey}
	o := ed.shown
	ed.readOnly, ed.password, ed.onPaste, ed.onTextKey = o.readOnly, o.password, o.onPaste, o.onTextKey
	fn()
	ed.readOnly, ed.password, ed.onPaste, ed.onTextKey = now.readOnly, now.password, now.onPaste, now.onTextKey
}

func (ed *editor) insert(s string) {
	if ed.readOnly {
		return
	}
	s = strings.ReplaceAll(s, "\r\n", "\n")
	a, b := ed.selection()
	ranges := ed.selectedRanges()
	ed.record(a == b && utf8.RuneCountInString(s) == 1)
	if len(ranges) == 0 {
		ed.replace(ed.caret, ed.caret, s)
	} else {
		for i := len(ranges) - 1; i >= 0; i-- {
			r := ranges[i]
			insert := ""
			if i == 0 {
				insert = s
			}
			ed.replace(r.Start, r.End, insert)
		}
	}
	ed.hasDesired = false
}

func (ed *editor) deleteRange(a, b int) {
	if a == b || ed.readOnly {
		return
	}
	ed.record(false)
	ed.replace(a, b, "")
	ed.hasDesired = false
}

func (ed *editor) commitCompose() {
	if ed.compose != "" {
		s := ed.compose
		ed.compose = ""
		ed.replace(ed.caret, ed.caret, s)
		if ed.client != nil {
			ed.client.publish()
		}
	}
	ed.composeCaret, ed.composeSelected = 0, TextInputRange{}
	ed.compositionActive = false
}

// process applies the input queued for the editor.
func (ed *editor) process(c *context, e *node) {
	st := e.st
	// The queue is taken first: a command may draw a nested frame, as GTK's
	// clipboard read does, which must not apply the same input again.
	queue := ed.queue
	ed.queue = nil
	for _, ev := range queue {
		if ev.replace {
			ed.anchor, ed.caret = min(ev.from, ed.buf.n), min(ev.to, ed.buf.n)
		}
		switch ev.kind {
		case editKey:
			ed.commitCompose()
			ed.key(c, st, ev)
		case editInsert:
			ed.compose = ""
			ed.insert(ev.text)
		case editCompose:
			if ed.readOnly {
				break
			}
			if ed.area != nil {
				ed.area.reveal = true
			}
			ed.compose = ev.text
			ed.composeCaret = max(0, min(ev.caret, utf8.RuneCountInString(ev.text)))
			if ev.text != "" {
				a, b := ed.selection()
				if a != b {
					ed.deleteRange(a, b)
				}
			}
		case editCommand:
			ed.commitCompose()
			ed.command(c, ev.text)
		}
	}
	clear(queue)
	if ed.queue == nil {
		ed.queue = queue[:0]
	}
	if ed.dragging && st.pressed {
		rt := c.rt
		x, y := st.local(rt.pointerX, rt.pointerY)
		ed.drag(x, y)
	}
}
