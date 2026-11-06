package editor

import (
	"io"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/egoist/mygo/internal/diff"
	"github.com/egoist/mygo/internal/rope"
	"github.com/egoist/mygo/ui"
)

// Buffer is the text an editor edits, with its undo history and its
// language. Several editors, and both sides of a DiffView, can show one
// buffer: an edit in one shows in the others, their carets and
// selections moving with the text around them, and undo takes back the
// buffer's last edit, whichever editor made it, as in Zed.
//
// The text is a persistent tree of chunks: an edit copies a few
// kilobytes and a path of the tree, whatever the size of the document,
// and each version shares the rest with the one before, so undo history
// keeps versions rather than copies, and frames and other goroutines
// read a version (Snapshot) without locks. Offsets count bytes of UTF-8,
// and a Pos a line and the bytes into it.
//
// Lines end in \n, as in Zed: a text set with \r\n line endings keeps
// \n ones, and LineEnding says which it had, which WriteTo writes.
//
// Methods are safe from any goroutine; schedule changes with
// Window.Update so the window shows them.
type Buffer struct {
	mu      sync.Mutex
	text    rope.Text
	version uint64
	// log is the changes of the last versions, and the text each made,
	// which editors, the highlighter and diffs follow to keep their state
	// of the text.
	log      []logEntry
	language *Language
	saved    rope.Text
	undo     []transaction
	redo     []transaction
	// open counts the Transact calls running, which group edits into
	// openTx.
	open     int
	openTx   *transaction
	lastEdit time.Time
	crlf     bool // the text's line endings are \r\n
	syntax   *syntaxState
	// wake are the windows showing the buffer, which a highlighter
	// working in the background has draw again.
	wake []ui.Services
}

// Edit replaces the bytes Start to End of a Buffer's text with Text.
type Edit struct {
	Start, End int
	Text       string
}

// Pos is a place in a text: a line, from 0, and a column in bytes from
// the start of the line.
type Pos struct{ Line, Col int }

// Less reports whether p comes before q.
func (p Pos) Less(q Pos) bool { return p.Line < q.Line || p.Line == q.Line && p.Col < q.Col }

// change is a range of the text an edit replaced, in bytes and lines,
// before (old) and after (new) the edit. The changes of a version are in
// order and do not overlap; newLine is oldLine moved by the lines the
// changes before it added.
type change struct {
	oldStart, oldEnd, newStart, newEnd int
	oldLine, oldEndLine                int
	newLine, newEndLine                int
}

type logEntry struct {
	version uint64
	text    rope.Text
	changes []change
}

// transaction is a step of undo: the text before and after, the changes
// of each version it made, and where the editor making it had its
// selections before and after.
type transaction struct {
	before, after rope.Text
	changes       [][]change
	// texts are the texts the versions of the step made, one a changes,
	// which undo and redo log as they go through them, so that each
	// entry's text is its changes' (as tree-sitter needs).
	texts     []rope.Text
	owner     any
	selBefore any
	selAfter  any
	kind      editKind
}

// editKind groups edits into undo steps: typing in a row is one step,
// as Zed groups edits within 300 ms.
type editKind uint8

const (
	editOther editKind = iota
	editTyping
	editDeleting
	// editCompose is the text of an input method's composition and its
	// commit, which make one step however long they take.
	editCompose
)

const (
	logMax       = 256
	undoMax      = 10000
	groupTimeout = 300 * time.Millisecond
)

// NewBuffer returns a buffer holding text, which is its saved text.
func NewBuffer(text string) *Buffer {
	b := &Buffer{text: rope.New(lf(text)), crlf: strings.Contains(text, "\r\n")}
	b.saved = b.text
	return b
}

// lf returns s with \n line endings.
func lf(s string) string {
	if strings.Contains(s, "\r\n") {
		return strings.ReplaceAll(s, "\r\n", "\n")
	}
	return s
}

// LineEnding returns the line ending the text had as it was set, \r\n or
// \n, which WriteTo writes; SetLineEnding changes it.
func (b *Buffer) LineEnding() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.crlf {
		return "\r\n"
	}
	return "\n"
}

func (b *Buffer) SetLineEnding(ending string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.crlf = ending == "\r\n"
}

// writeText writes t to w with the line ending of crlf.
func writeText(t rope.Text, crlf bool, w io.Writer) (int64, error) {
	if !crlf {
		return rope.WriteTo(t.Root(), w)
	}
	var n int64
	var err error
	t.Chunks(0, t.Len(), func(s string) bool {
		var k int
		k, err = io.WriteString(w, strings.ReplaceAll(s, "\n", "\r\n"))
		n += int64(k)
		return err == nil
	})
	return n, err
}

// snapshot returns the current text and version.
func (b *Buffer) snapshot() (rope.Text, uint64) {
	if b == nil {
		return rope.Text{}, 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.text, b.version
}

// Snapshot returns the text as it is now, which stays as it is while
// the buffer changes: read it from any goroutine, as to save the text or
// to send it to a language server.
func (b *Buffer) Snapshot() Snapshot {
	b.mu.Lock()
	defer b.mu.Unlock()
	return Snapshot{b.text, b.version, b.crlf}
}

// String returns the text.
func (b *Buffer) String() string { r, _ := b.snapshot(); return r.String() }

// WriteTo writes the text to w, a chunk at a time, as for saving it, with
// its LineEnding.
func (b *Buffer) WriteTo(w io.Writer) (int64, error) {
	s := b.Snapshot()
	return s.WriteTo(w)
}

// Len returns the length of the text in bytes.
func (b *Buffer) Len() int { r, _ := b.snapshot(); return r.Len() }

// LineCount returns how many lines the text has: one more than its
// newlines.
func (b *Buffer) LineCount() int { r, _ := b.snapshot(); return r.LineCount() }

// Line returns line i, without its newline.
func (b *Buffer) Line(i int) string { r, _ := b.snapshot(); return r.Line(i) }

// Slice returns the bytes start to end of the text.
func (b *Buffer) Slice(start, end int) string { r, _ := b.snapshot(); return r.Slice(start, end) }

// Pos returns the line and column of byte offset.
func (b *Buffer) Pos(offset int) Pos { r, _ := b.snapshot(); return posOf(r, offset) }

// Offset returns the byte offset of p, its column clamped to its line.
func (b *Buffer) Offset(p Pos) int { r, _ := b.snapshot(); return offsetOf(r, p) }

// Version returns a number that changes with every change of the text.
func (b *Buffer) Version() uint64 { _, v := b.snapshot(); return v }

// Language returns the buffer's language, nil for plain text.
func (b *Buffer) Language() *Language {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.language
}

// SetLanguage sets the language that highlights the text, and that
// comments, brackets and indentation follow; nil is plain text.
func (b *Buffer) SetLanguage(l *Language) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.language != l {
		b.language = l
		b.syntax = nil
	}
}

// Modified reports whether the text differs from the saved one: the
// text the buffer was made or Reset with, or had when MarkSaved was
// called. Undoing back to it makes the buffer unmodified again.
func (b *Buffer) Modified() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return !b.text.Same(b.saved) && b.text.String() != b.saved.String()
}

// MarkSaved makes the current text the saved one.
func (b *Buffer) MarkSaved() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.saved = b.text
}

// Reset replaces the text and forgets the undo history, as for a file
// loaded anew, and makes it the saved text.
func (b *Buffer) Reset(text string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	old := b.text
	b.text = rope.New(lf(text))
	b.crlf = strings.Contains(text, "\r\n")
	b.saved = b.text
	b.undo, b.redo, b.openTx = nil, nil, nil
	b.record([]change{wholeChange(old, b.text)})
}

// Set replaces the whole text as one edit, which Undo takes back.
func (b *Buffer) Set(text string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.edit([]Edit{{Start: 0, End: b.text.Len(), Text: text}}, editOther, nil, nil)
}

// Edit applies edits to the text as one step of undo. Their offsets are
// in the text before any of them; overlapping edits are cut to follow
// the one before.
func (b *Buffer) Edit(edits ...Edit) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.edit(edits, editOther, nil, nil)
}

// Transact groups the edits fn makes, through Edit and the editors
// showing the buffer, into one step of undo.
func (b *Buffer) Transact(fn func()) {
	b.mu.Lock()
	b.open++
	b.mu.Unlock()
	defer func() {
		b.mu.Lock()
		b.open--
		if b.open == 0 {
			b.openTx = nil
		}
		b.mu.Unlock()
	}()
	fn()
}

// CanUndo and CanRedo report whether Undo and Redo have a step to take.
func (b *Buffer) CanUndo() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.undo) > 0
}

func (b *Buffer) CanRedo() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	return len(b.redo) > 0
}

// Undo takes back the last step of edits, and reports whether there was
// one.
func (b *Buffer) Undo() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	_, ok := b.undoStep()
	return ok
}

// Redo makes the step Undo took back again, and reports whether there
// was one.
func (b *Buffer) Redo() bool {
	b.mu.Lock()
	defer b.mu.Unlock()
	_, ok := b.redoStep()
	return ok
}

// undoStep takes back the last transaction; b.mu is held.
func (b *Buffer) undoStep() (transaction, bool) {
	n := len(b.undo)
	if n == 0 {
		return transaction{}, false
	}
	tx := b.undo[n-1]
	b.undo = b.undo[:n-1]
	b.redo = append(b.redo, tx)
	b.openTx = nil
	b.lastEdit = time.Time{}
	for i := len(tx.changes) - 1; i >= 0; i-- {
		b.text = tx.before
		if i > 0 {
			b.text = tx.texts[i-1]
		}
		b.record(invertChanges(tx.changes[i]))
	}
	b.text = tx.before
	return tx, true
}

// redoStep makes the last transaction undone again; b.mu is held.
func (b *Buffer) redoStep() (transaction, bool) {
	n := len(b.redo)
	if n == 0 {
		return transaction{}, false
	}
	tx := b.redo[n-1]
	b.redo = b.redo[:n-1]
	b.undo = append(b.undo, tx)
	b.openTx = nil
	b.lastEdit = time.Time{}
	for i, c := range tx.changes {
		b.text = tx.texts[i]
		b.record(c)
	}
	b.text = tx.after
	return tx, true
}

// edit applies edits as a transaction, or into the open one; b.mu is
// held. It returns the changes, nil when nothing changed.
func (b *Buffer) edit(edits []Edit, kind editKind, owner, selBefore any) []change {
	if len(edits) == 0 {
		return nil
	}
	if !slices.IsSortedFunc(edits, func(x, y Edit) int { return x.Start - y.Start }) {
		edits = slices.Clone(edits)
		slices.SortStableFunc(edits, func(x, y Edit) int { return x.Start - y.Start })
	}
	cloned := false
	for i, e := range edits {
		if strings.Contains(e.Text, "\r\n") {
			if !cloned {
				edits, cloned = slices.Clone(edits), true
			}
			edits[i].Text = lf(e.Text)
		}
	}
	before := b.text
	text := before
	changes := make([]change, 0, len(edits))
	delta, lineDelta, last := 0, 0, 0
	for _, e := range edits {
		start := max(last, min(e.Start, before.Len()))
		end := max(start, min(e.End, before.Len()))
		last = end
		if start == end && e.Text == "" || end-start == len(e.Text) && before.Slice(start, end) == e.Text {
			continue
		}
		c := change{oldStart: start, oldEnd: end}
		c.oldLine = before.LineOf(start)
		c.oldEndLine = c.oldLine
		if end > start {
			c.oldEndLine = before.LineOf(end)
		}
		c.newStart = start + delta
		c.newEnd = c.newStart + len(e.Text)
		c.newLine = c.oldLine + lineDelta
		c.newEndLine = c.newLine + strings.Count(e.Text, "\n")
		text = text.Replace(c.newStart, c.newStart+end-start, e.Text)
		delta += len(e.Text) - (end - start)
		lineDelta += (c.newEndLine - c.newLine) - (c.oldEndLine - c.oldLine)
		changes = append(changes, c)
	}
	if len(changes) == 0 {
		return nil
	}
	b.text = text
	b.record(changes)
	b.redo = nil
	now := time.Now()
	switch tx := b.openTx; {
	case tx != nil:
		tx.after = text
		tx.changes = append(tx.changes, changes)
		tx.texts = append(tx.texts, text)
	case b.open == 0 && kind != editOther && len(b.undo) > 0 && b.undo[len(b.undo)-1].kind == kind &&
		b.undo[len(b.undo)-1].owner == owner && (now.Sub(b.lastEdit) < groupTimeout || kind == editCompose && !b.lastEdit.IsZero()):
		tx := &b.undo[len(b.undo)-1]
		tx.after = text
		tx.changes = append(tx.changes, changes)
		tx.texts = append(tx.texts, text)
	default:
		b.undo = append(b.undo, transaction{before: before, after: text, changes: [][]change{changes}, texts: []rope.Text{text},
			owner: owner, selBefore: selBefore, kind: kind})
		if len(b.undo) > undoMax {
			b.undo = slices.Delete(b.undo, 0, len(b.undo)-undoMax)
		}
		if b.open > 0 {
			b.openTx = &b.undo[len(b.undo)-1]
		}
	}
	b.lastEdit = now
	return changes
}

// endGroup ends the group of typing, so that the next edit makes a step
// of undo of its own, as moving the caret does.
func (b *Buffer) endGroup() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.lastEdit = time.Time{}
}

// setSelectionsAfter notes where the owner of the last transaction has
// its selections after it.
func (b *Buffer) setSelectionsAfter(owner, sel any) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if n := len(b.undo); n > 0 && b.undo[n-1].owner == owner {
		b.undo[n-1].selAfter = sel
	}
}

// record logs the changes of a new version, made of the current text;
// b.mu is held.
func (b *Buffer) record(changes []change) {
	b.version++
	b.log = append(b.log, logEntry{version: b.version, text: b.text, changes: changes})
	if len(b.log) > logMax {
		b.log = slices.Delete(b.log, 0, len(b.log)-logMax/2)
	}
}

// changesSince returns the changes of each version after since, the
// version they lead to, and false when the log no longer goes back that
// far.
func (b *Buffer) changesSince(since uint64) ([][]change, uint64, bool) {
	entries, version, ok := b.entriesSince(since)
	out := make([][]change, len(entries))
	for i, e := range entries {
		out[i] = e.changes
	}
	return out, version, ok
}

// entriesSince returns the log's entries after version since.
// changesTo returns the changes of the versions after since, the version
// they lead to and its text, read at once, so that the text is the
// changes' and not a later one's; ok false when the log no longer
// reaches back to since.
func (b *Buffer) changesTo(since uint64) (changes [][]change, version uint64, text rope.Text, ok bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	entries, version, ok := b.entriesSinceLocked(since)
	changes = make([][]change, len(entries))
	for i, e := range entries {
		changes[i] = e.changes
	}
	return changes, version, b.text, ok
}

// changesBetween returns the changes from old to new by their lines, for
// an editor too far behind the log to follow the edits themselves.
func changesBetween(old, new rope.Text) []change {
	lines := func(t rope.Text) []string {
		out := make([]string, 0, t.LineCount())
		t.EachLine(0, t.LineCount(), func(_, _ int, s string) bool {
			out = append(out, s)
			return true
		})
		return out
	}
	at := func(t rope.Text, line int) int {
		if line >= t.LineCount() {
			return t.Len()
		}
		return t.LineStart(line)
	}
	var out []change
	for _, h := range diff.Lines(lines(old), lines(new)) {
		c := change{oldStart: at(old, h.A0), oldEnd: at(old, h.A1), newStart: at(new, h.B0), newEnd: at(new, h.B1)}
		c.oldLine, c.oldEndLine = old.LineOf(c.oldStart), old.LineOf(c.oldEnd)
		c.newLine, c.newEndLine = new.LineOf(c.newStart), new.LineOf(c.newEnd)
		out = append(out, c)
	}
	return out
}

func (b *Buffer) entriesSince(since uint64) ([]logEntry, uint64, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.entriesSinceLocked(since)
}

func (b *Buffer) entriesSinceLocked(since uint64) ([]logEntry, uint64, bool) {
	if since == b.version {
		return nil, b.version, true
	}
	if len(b.log) == 0 || b.log[0].version > since+1 || since > b.version {
		return nil, b.version, false
	}
	i, _ := slices.BinarySearchFunc(b.log, since+1, func(e logEntry, v uint64) int {
		switch {
		case e.version < v:
			return -1
		case e.version > v:
			return 1
		}
		return 0
	})
	return slices.Clone(b.log[i:]), b.version, true
}

// watch notes a window showing the buffer, for a highlighter working in
// the background to draw it again.
func (b *Buffer) watch(s ui.Services) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if !slices.Contains(b.wake, s) {
		b.wake = append(b.wake, s)
		if len(b.wake) > 8 {
			b.wake = b.wake[1:]
		}
	}
}

// invalidate draws the windows showing the buffer again, from any
// goroutine.
func (b *Buffer) invalidate() {
	b.mu.Lock()
	wake := slices.Clone(b.wake)
	b.mu.Unlock()
	for _, s := range wake {
		s.Invalidate()
	}
}

// wholeChange is the change replacing all of old with new.
func wholeChange(old, new rope.Text) change {
	return change{oldEnd: old.Len(), newEnd: new.Len(), oldEndLine: old.LineCount() - 1, newEndLine: new.LineCount() - 1}
}

func invertChanges(cs []change) []change {
	out := make([]change, len(cs))
	for i, c := range cs {
		out[i] = change{oldStart: c.newStart, oldEnd: c.newEnd, newStart: c.oldStart, newEnd: c.oldEnd,
			oldLine: c.newLine, oldEndLine: c.newEndLine, newLine: c.oldLine, newEndLine: c.oldEndLine}
	}
	return out
}

// bias says which side of an insertion at an offset the offset moves
// to.
type bias uint8

const (
	biasLeft  bias = iota // stays before text inserted at it
	biasRight             // moves after text inserted at it
)

// mapOffset moves an offset of the text before changes to the text after
// them. An offset inside a replaced range goes to its start (left) or end
// (right).
func mapOffset(off int, changes []change, b bias) int {
	// The last change starting at or before off.
	i, _ := slices.BinarySearchFunc(changes, off, func(c change, o int) int {
		if c.oldStart <= o {
			return -1
		}
		return 1
	})
	if i == 0 {
		return off
	}
	if b == biasLeft {
		// Before every insertion at off, not only the last.
		for i > 1 && changes[i-2].oldStart == off {
			i--
		}
	}
	c := changes[i-1]
	switch {
	case off > c.oldEnd || off == c.oldEnd && c.oldEnd > c.oldStart:
		return off + c.newEnd - c.oldEnd
	case off == c.oldStart && b == biasLeft:
		return c.newStart
	case off == c.oldStart && c.oldStart == c.oldEnd:
		return c.newEnd
	case b == biasLeft:
		return c.newStart
	}
	return c.newEnd
}

// mapLine moves a line of the text before changes to the text after
// them; a line the changes replaced goes to where they start.
func mapLine(line int, changes []change) int {
	delta := 0
	for _, c := range changes {
		if line < c.oldLine {
			break
		}
		if line <= c.oldEndLine {
			if line == c.oldLine {
				return c.newLine
			}
			return min(c.newLine+(line-c.oldLine), c.newEndLine)
		}
		delta = c.newEndLine - c.oldEndLine
	}
	return line + delta
}

// posOf returns the line and column, in bytes, of byte off of t.
func posOf(t rope.Text, off int) Pos {
	off = max(0, min(off, t.Len()))
	line := t.LineOf(off)
	return Pos{Line: line, Col: off - t.LineStart(line)}
}

// offsetOf returns the byte of p in t, its column clamped to its line.
func offsetOf(t rope.Text, p Pos) int {
	if p.Line < 0 {
		return 0
	}
	if p.Line >= t.LineCount() {
		return t.Len()
	}
	a := t.LineStart(p.Line)
	return a + max(0, min(p.Col, t.LineEnd(p.Line)-a))
}

// Snapshot is a version of a buffer's text, which stays as it is while
// the buffer changes: read it from any goroutine.
type Snapshot struct {
	t       rope.Text
	version uint64
	crlf    bool
}

// Version returns the buffer's version the snapshot is of.
func (s Snapshot) Version() uint64 { return s.version }

// Len, LineCount and String return the length in bytes, the number of
// lines and the text of the snapshot.
func (s Snapshot) Len() int       { return s.t.Len() }
func (s Snapshot) LineCount() int { return s.t.LineCount() }
func (s Snapshot) String() string { return s.t.String() }

// Line returns line i, without its newline.
func (s Snapshot) Line(i int) string { return s.t.Line(i) }

// LineStart and LineEnd return the bytes line i starts and ends at,
// before its newline.
func (s Snapshot) LineStart(i int) int { return s.t.LineStart(i) }
func (s Snapshot) LineEnd(i int) int   { return s.t.LineEnd(i) }

// LineOf returns the line holding byte offset.
func (s Snapshot) LineOf(offset int) int { return s.t.LineOf(offset) }

// Pos returns the line and column of byte offset, and Offset the byte of
// a line and column.
func (s Snapshot) Pos(offset int) Pos { return posOf(s.t, offset) }
func (s Snapshot) Offset(p Pos) int   { return offsetOf(s.t, p) }

// Slice returns bytes start to end.
func (s Snapshot) Slice(start, end int) string { return s.t.Slice(start, end) }

// Chunks calls fn with bytes start to end, a piece at a time, until fn
// returns false: what a parser reads text with.
func (s Snapshot) Chunks(start, end int, fn func(text string) bool) { s.t.Chunks(start, end, fn) }

// WriteTo writes the text to w.
func (s Snapshot) WriteTo(w io.Writer) (int64, error) { return writeText(s.t, s.crlf, w) }

// Change is a range of a text an edit replaced: bytes OldStart to OldEnd
// of the text before the edit, which are bytes NewStart to NewEnd of the
// text after it. An edit's changes are in order and do not overlap, so
// NewStart is OldStart moved by what the changes before it added and
// removed. An incremental parser, as tree-sitter's, takes them from the
// last: each starts at OldStart, and ends at OldEnd before and at
// OldStart+NewEnd-NewStart after, as the changes before it have not
// moved it yet.
type Change struct {
	OldStart, OldEnd, NewStart, NewEnd int
}

func publicChanges(cs []change) []Change {
	out := make([]Change, len(cs))
	for i, c := range cs {
		out[i] = Change{c.oldStart, c.oldEnd, c.newStart, c.newEnd}
	}
	return out
}
