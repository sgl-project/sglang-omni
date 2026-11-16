package richtext

import (
	"sync"
	"time"
)

// Document is what an Editor edits: a Text, its
// selection with the attributes typing gets there, and the history of
// undo, which the app keeps from frame to frame, as it keeps a TextBuffer.
// Its methods are safe from any goroutine: a toolbar, a menu or a timer
// formats and edits it while the editor shows it, and the editor follows
// what changed at its next frame (schedule changes from other goroutines
// with Window.Update, so that a frame comes).
//
// Commands act on the selection as the editor's shortcuts do: on a range
// they change its runs or paragraphs, as one step of undo; on a caret,
// character formats change only the typing attributes, which the next
// typed text gets and the next move of the caret forgets, as ⌘B does in
// NSTextView.
type Document struct {
	mu      sync.Mutex
	text    Text
	anchor  int
	caret   int
	typing  *Attributes // nil: derived from the text at the caret
	version uint64
	// selVersion counts changes of the selection made by the document's
	// methods, which the editor then takes.
	selVersion uint64
	undo, redo []undoStep
	lastEdit   time.Time
	coalesce   bool // typing may join the last step
	// log is what changed since the versions the editor last saw, for it
	// to lay out only the paragraphs that changed.
	log []change
}

// change is a change of a document: the runes from start to oldEnd
// became those from start to newEnd, at version.
type change struct {
	version               uint64
	start, oldEnd, newEnd int
}

// undoStep is a step of undo: the texts before and after it, which share
// all they did not change, the selections, and the region it changed.
type undoStep struct {
	before, after         Text
	selBefore, selAfter   [2]int
	start, oldEnd, newEnd int
}

// NewDocument returns a document of text, its caret at the end.
func NewDocument(text Text) *Document {
	n := text.Len()
	return &Document{text: text, anchor: n, caret: n, version: 1}
}

func (d *Document) lock() {
	d.mu.Lock()
	if d.version == 0 {
		d.version = 1 // the zero value is an empty document
	}
}

// Text returns the text of the document, which later edits leave as it is.
func (d *Document) Text() Text {
	d.lock()
	defer d.mu.Unlock()
	return d.text
}

// Version counts the changes of the document's text.
func (d *Document) Version() uint64 {
	d.lock()
	defer d.mu.Unlock()
	return d.version
}

// SetText replaces the text, as when the app opens another note: it is not
// a step of undo, and clears the history. The selection goes to the end.
func (d *Document) SetText(text Text) {
	d.lock()
	defer d.mu.Unlock()
	old := d.text.Len()
	d.text = text
	d.undo, d.redo = nil, nil
	d.coalesce = false
	d.changed(0, old, text.Len())
	n := text.Len()
	d.setSelection(n, n)
}

// Selection returns the selection: anchor is where it started, caret where
// it ends and the caret is, equal for a caret.
func (d *Document) Selection() (anchor, caret int) {
	d.lock()
	defer d.mu.Unlock()
	return d.anchor, d.caret
}

// SetSelection selects the runes between anchor and caret, or puts the
// caret at caret when they are equal; the editor scrolls it into view.
func (d *Document) SetSelection(anchor, caret int) {
	d.lock()
	defer d.mu.Unlock()
	d.setSelection(anchor, caret)
}

func (d *Document) setSelection(anchor, caret int) {
	n := d.text.Len()
	d.anchor, d.caret = max(0, min(anchor, n)), max(0, min(caret, n))
	d.typing = nil
	d.coalesce = false
	d.selVersion++
}

func (d *Document) span() (int, int) {
	return min(d.anchor, d.caret), max(d.anchor, d.caret)
}

// TypingAttributes returns the attributes typing gets at the caret: those
// a command set at the caret, else those of the character before it (after
// it at the start of a paragraph), or of the first selected character,
// without a link, code or tag at their edges, as NSTextView derives them.
func (d *Document) TypingAttributes() Attributes {
	d.lock()
	defer d.mu.Unlock()
	return d.typingAttributes()
}

func (d *Document) typingAttributes() Attributes {
	if d.typing != nil {
		return *d.typing
	}
	a, z := d.span()
	return typingAttributes(d.text, a, z)
}

// typingAttributes returns the attributes typing gets in place of the runes
// from start to end of t.
func typingAttributes(t Text, start, end int) Attributes {
	if start < end {
		a := t.AttributesAt(start)
		a.Attachment = nil
		return a
	}
	ps, _ := t.ParagraphRange(start)
	if start == ps {
		return t.AttributesAt(start).extending()
	}
	before, after := t.AttributesAt(start-1), t.AttributesAt(start)
	if _, pe := t.ParagraphRange(start); start == pe {
		// The newline: a link or code ends with the line, as a selection
		// of whole lines formats newlines too.
		after = Attributes{}
	}
	a := before.extending()
	// What does not extend holds inside: between two runes of a link.
	if before.Link == after.Link {
		a.Link = before.Link
	}
	if before.Code && after.Code {
		a.Code = true
	}
	if before.Tag == after.Tag {
		a.Tag = before.Tag
	}
	return a
}

// SetTypingAttributes sets the attributes the next typed text gets, until
// the caret moves.
func (d *Document) SetTypingAttributes(a Attributes) {
	d.lock()
	defer d.mu.Unlock()
	a.Attachment = nil
	d.typing = &a
}

// changed notes that the runes from start to oldEnd became those from start
// to newEnd.
func (d *Document) changed(start, oldEnd, newEnd int) {
	d.version++
	d.log = append(d.log, change{d.version, start, oldEnd, newEnd})
	if len(d.log) > 256 {
		// The editor reloads a text it is too far behind.
		d.log = append(d.log[:0], d.log[len(d.log)-128:]...)
	}
}

// since returns the change from version v to the document's, as one
// region, and whether the log holds all of it.
func (d *Document) since(v uint64) (change, bool) {
	if v == d.version {
		return change{version: v}, true
	}
	var c change
	found := false
	for _, l := range d.log {
		if l.version <= v {
			continue
		}
		if !found {
			if l.version != v+1 {
				return change{}, false
			}
			c, found = l, true
			continue
		}
		c = composeChanges(c, l)
	}
	return c, found && c.version == d.version
}

// composeChanges returns the change a then b make together.
func composeChanges(a, b change) change {
	// Map a's new end through b, and b's old end back through a.
	end := a.newEnd
	switch {
	case end > b.start && end < b.oldEnd:
		end = b.newEnd
	case end >= b.oldEnd:
		end += b.newEnd - b.oldEnd
	}
	old := b.oldEnd
	switch {
	case old > a.start && old < a.newEnd:
		old = a.oldEnd
	case old >= a.newEnd:
		old -= a.newEnd - a.oldEnd
	}
	return change{
		version: b.version,
		start:   min(a.start, b.start),
		oldEnd:  max(a.oldEnd, old),
		newEnd:  max(end, b.newEnd),
	}
}

// Edit changes a document as one step of undo, in
// Document.Edit. Offsets are into the text as the edit's changes
// so far left it, and the selection follows them.
type Edit struct {
	d    *Document
	step *undoStep
}

// Text returns the text as the edit's changes so far left it.
func (e *Edit) Text() Text { return e.d.text }

// Selection returns the selection as the edit's changes so far moved it.
func (e *Edit) Selection() (anchor, caret int) { return e.d.anchor, e.d.caret }

// Select selects the runes between anchor and caret.
func (e *Edit) Select(anchor, caret int) { e.d.setSelection(anchor, caret) }

// Replace replaces the runes from start to end with text.
func (e *Edit) Replace(start, end int, text Text) {
	n := e.d.text.Len()
	start, end = max(0, min(start, end, n)), max(0, min(max(start, end), n))
	e.d.apply(e.step, start, end-start, text.Len(), e.d.text.Replace(start, end, text))
}

// ReplaceString replaces the runes from start to end with s, which takes
// attributes a.
func (e *Edit) ReplaceString(start, end int, s string, a Attributes) {
	n := e.d.text.Len()
	start, end = max(0, min(start, end, n)), max(0, min(max(start, end), n))
	t := e.d.text.ReplaceString(start, end, s, a)
	e.d.apply(e.step, start, end-start, t.Len()-n+end-start, t)
}

// Format changes the attributes of the runes from start to end.
func (e *Edit) Format(start, end int, fn func(a *Attributes)) {
	n := e.d.text.Len()
	start, end = max(0, min(start, end, n)), max(0, min(max(start, end), n))
	t := e.d.text.Format(start, end, fn)
	if !t.Same(e.d.text) {
		e.d.apply(e.step, start, end-start, end-start, t)
	}
}

// FormatParagraphs changes the styles of the paragraphs from the one
// holding start to the one holding end (see Text.FormatParagraphs).
func (e *Edit) FormatParagraphs(start, end int, fn func(p *ParagraphStyle)) {
	t := e.d.text
	pa, pz := t.paragraphSpan(start, end)
	a := t.paraStart(pa)
	z := t.paraStart(pz) + seqAt(t.seq(), pz).n
	if f := t.FormatParagraphs(start, end, fn); !f.Same(t) {
		e.d.apply(e.step, a, z-a, z-a, f)
	}
}

// apply makes t, in which the runes from start to start+removed of the text
// are now start to start+inserted, the document's text, as part of step.
func (d *Document) apply(step *undoStep, start, removed, inserted int, t Text) {
	formatted := t.text == d.text.text // the text is the same: only formats changed
	d.text = t
	if step != nil {
		if step.start < 0 {
			step.start, step.oldEnd, step.newEnd = start, start+removed, start+inserted
		} else {
			c := composeChanges(change{start: step.start, oldEnd: step.oldEnd, newEnd: step.newEnd},
				change{start: start, oldEnd: start + removed, newEnd: start + inserted})
			step.start, step.oldEnd, step.newEnd = c.start, c.oldEnd, c.newEnd
		}
		step.after = t
	}
	d.changed(start, start+removed, start+inserted)
	if formatted {
		return
	}
	a, z := d.span()
	if a == z {
		a = mapPosition(a, start, removed, inserted, false)
		z = a
	} else {
		a, z = mapPosition(a, start, removed, inserted, false), mapPosition(z, start, removed, inserted, true)
	}
	if d.caret >= d.anchor {
		d.anchor, d.caret = a, z
	} else {
		d.anchor, d.caret = z, a
	}
	d.typing = nil
}

// mapPosition returns where p goes as the runes from start to
// start+removed become inserted others, as AttributedString tracks indices:
// a caret or the start of a range where text is inserted goes after it,
// the end of a range does not grow, and what was replaced collapses to
// its edges.
func mapPosition(p, start, removed, inserted int, upper bool) int {
	end := start + removed
	switch {
	case p < start:
		return p
	case removed == 0 && p == start:
		if upper {
			return p
		}
		return p + inserted
	case p < end:
		if upper {
			return start + inserted
		}
		return start
	}
	return p + inserted - removed
}

// Edit changes the document with fn, as one step of undo, which undoing
// takes back whole with the selection before it:
//
//	doc.Edit(func(e *richtext.Edit) {
//		e.ReplaceString(0, 0, "Dear Ann,\n", richtext.Attributes{})
//		e.Format(0, 8, func(a *richtext.Attributes) { a.Italic = true })
//	})
func (d *Document) Edit(fn func(e *Edit)) {
	d.lock()
	defer d.mu.Unlock()
	d.edit(false, fn)
}

// edit runs fn as a step of undo, joined to the last one when join is set
// and the last edit was typing a moment ago.
func (d *Document) edit(join bool, fn func(e *Edit)) {
	now := time.Now()
	var step *undoStep
	if join && d.coalesce && now.Sub(d.lastEdit) < time.Second && len(d.undo) > 0 {
		step = &d.undo[len(d.undo)-1]
	} else {
		d.undo = append(d.undo, undoStep{before: d.text, after: d.text, selBefore: [2]int{d.anchor, d.caret}, start: -1})
		step = &d.undo[len(d.undo)-1]
	}
	fn(&Edit{d: d, step: step})
	if step.start < 0 {
		// Nothing changed.
		d.undo = d.undo[:len(d.undo)-1]
		return
	}
	if len(d.undo) > 200 {
		d.undo[0] = undoStep{}
		d.undo = d.undo[1:]
	}
	step.selAfter = [2]int{d.anchor, d.caret}
	clear(d.redo)
	d.redo = d.redo[:0]
	d.coalesce = join
	d.lastEdit = now
}

// CanUndo and CanRedo report whether there is a step to undo or redo.
func (d *Document) CanUndo() bool {
	d.lock()
	defer d.mu.Unlock()
	return len(d.undo) > 0
}

func (d *Document) CanRedo() bool {
	d.lock()
	defer d.mu.Unlock()
	return len(d.redo) > 0
}

// Undo takes back the last step of undo, and Redo the last step undone,
// restoring the text, its attributes and the selection.
func (d *Document) Undo() {
	d.lock()
	defer d.mu.Unlock()
	d.takeBack(false)
}

func (d *Document) Redo() {
	d.lock()
	defer d.mu.Unlock()
	d.takeBack(true)
}

func (d *Document) takeBack(redo bool) {
	from, to := &d.undo, &d.redo
	if redo {
		from, to = to, from
	}
	n := len(*from)
	if n == 0 {
		return
	}
	s := (*from)[n-1]
	(*from)[n-1] = undoStep{}
	*from = (*from)[:n-1]
	if redo {
		d.text = s.after
		d.changed(s.start, s.oldEnd, s.newEnd)
		d.setSelection(s.selAfter[0], s.selAfter[1])
	} else {
		d.text = s.before
		d.changed(s.start, s.newEnd, s.oldEnd)
		d.setSelection(s.selBefore[0], s.selBefore[1])
	}
	*to = append(*to, s)
}

// BreakUndoCoalescing ends the step typing adds to, as at a save, so that
// what is typed next undoes on its own.
func (d *Document) BreakUndoCoalescing() {
	d.lock()
	defer d.mu.Unlock()
	d.coalesce = false
}
