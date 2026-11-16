package ui

import (
	"io"
	"sync"

	"github.com/egoist/mygo/internal/rope"
)

// TextBuffer stores editable UTF-8 text in an indexed, persistent tree.
// Small edits copy bounded chunks and tree paths, not the whole document.
// Rune offsets are used by Slice/Replace, matching Element.TextSelection;
// UTF16Offset/RuneOffset and TextForRange bridge native input coordinates.
// The zero value is an empty buffer. Methods are safe from any goroutine.
// Schedule edits with Window.Update to invalidate a view showing the buffer.
type TextBuffer struct {
	mu      sync.RWMutex
	root    *rope.Node
	version uint64
}

// TextSnapshot is an immutable, inexpensive view of a buffer. It shares
// unchanged chunks and can be read concurrently with further edits.
type TextSnapshot struct {
	root    *rope.Node
	version uint64
}

// NewTextBuffer creates indexed storage owning its text chunks. A small
// surviving fragment cannot pin the original large input string.
func NewTextBuffer(s string) *TextBuffer { return &TextBuffer{root: rope.Build(s)} }

func (b *TextBuffer) Snapshot() TextSnapshot {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return TextSnapshot{root: b.root, version: b.version}
}

func (b *TextBuffer) Len() int        { return b.Snapshot().Len() }
func (b *TextBuffer) ByteLen() int    { return b.Snapshot().ByteLen() }
func (b *TextBuffer) UTF16Len() int   { return b.Snapshot().UTF16Len() }
func (b *TextBuffer) LineCount() int  { return b.Snapshot().LineCount() }
func (b *TextBuffer) Version() uint64 { return b.Snapshot().version }

// String materializes the document for export. Controls do not call it
// during ordinary editing; Slice and WriteTo avoid that full allocation.
func (b *TextBuffer) String() string              { return b.Snapshot().String() }
func (b *TextBuffer) Slice(start, end int) string { return b.Snapshot().Slice(start, end) }
func (b *TextBuffer) UTF16Offset(index int) int   { return b.Snapshot().UTF16Offset(index) }
func (b *TextBuffer) RuneOffset(index int) int    { return b.Snapshot().RuneOffset(index) }
func (b *TextBuffer) TextForRange(r TextInputRange) (string, TextInputRange) {
	return b.Snapshot().TextForRange(r)
}
func (b *TextBuffer) WriteTo(w io.Writer) (int64, error) { return b.Snapshot().WriteTo(w) }

// Set replaces the document. Existing snapshots retain their own version.
func (b *TextBuffer) Set(s string) { b.Restore(TextSnapshot{root: rope.Build(s)}) }

// Restore adopts a snapshot in constant time and advances the version.
func (b *TextBuffer) Restore(s TextSnapshot) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.root = s.root
	b.version++
}

// Replace replaces a half-open rune range, clamping it to the document.
func (b *TextBuffer) Replace(start, end int, s string) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.root = rope.Replace(b.root, start, end, s)
	b.version++
}

// ReplaceUTF16 adjusts surrogate interiors down to whole runes, replaces
// the range, and returns the actual old UTF-16 range it replaced.
func (b *TextBuffer) ReplaceUTF16(r TextInputRange, s string) TextInputRange {
	b.mu.Lock()
	defer b.mu.Unlock()
	view := TextSnapshot{root: b.root}
	a, z := view.RuneOffset(min(r.Start, r.End)), view.RuneOffset(max(r.Start, r.End))
	actual := TextInputRange{Start: view.UTF16Offset(a), End: view.UTF16Offset(z)}
	b.root = rope.Replace(b.root, a, z, s)
	b.version++
	return actual
}

// compareRestore prevents a widget from overwriting a concurrent program edit.
func (b *TextBuffer) compareRestore(before TextSnapshot, root *rope.Node) (TextSnapshot, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.root != before.root || b.version != before.version {
		return TextSnapshot{root: b.root, version: b.version}, false
	}
	b.root = root
	b.version++
	return TextSnapshot{root: root, version: b.version}, true
}

func (s TextSnapshot) Len() int        { return rope.Sum(s.root).Runes }
func (s TextSnapshot) ByteLen() int    { return rope.Sum(s.root).Bytes }
func (s TextSnapshot) UTF16Len() int   { return rope.Sum(s.root).Units }
func (s TextSnapshot) LineCount() int  { return rope.Sum(s.root).Lines + 1 }
func (s TextSnapshot) Version() uint64 { return s.version }
func (s TextSnapshot) String() string  { return s.Slice(0, s.Len()) }
func (s TextSnapshot) Slice(start, end int) string {
	a, z := max(0, min(start, end, s.Len())), max(0, min(max(start, end), s.Len()))
	return sliceBytes(s.root, rope.ByteOfRune(s.root, a), rope.ByteOfRune(s.root, z))
}
func (s TextSnapshot) UTF16Offset(index int) int {
	return rope.Prefix(s.root, rope.ByteOfRune(s.root, index)).Units
}
func (s TextSnapshot) RuneOffset(index int) int {
	return rope.RuneOfUnit(s.root, max(0, min(index, s.UTF16Len())))
}
func (s TextSnapshot) TextForRange(r TextInputRange) (string, TextInputRange) {
	a, z := s.RuneOffset(min(r.Start, r.End)), s.RuneOffset(max(r.Start, r.End))
	return nativeText(s.Slice(a, z)), TextInputRange{Start: s.UTF16Offset(a), End: s.UTF16Offset(z)}
}

// LineRange returns a line's rune offsets, excluding its following newline.
// The terminal empty line after a newline is counted; line indices clamp.
func (s TextSnapshot) LineRange(line int) (int, int) {
	line = max(0, min(line, s.LineCount()-1))
	a := rope.Prefix(s.root, rope.LineStart(s.root, line)).Runes
	z := s.Len()
	if line+1 < s.LineCount() {
		z = rope.Prefix(s.root, rope.LineStart(s.root, line+1)).Runes - 1
	}
	return a, z
}
func (s TextSnapshot) Line(line int) string { a, z := s.LineRange(line); return s.Slice(a, z) }
func (s TextSnapshot) LineAt(index int) int {
	return rope.Prefix(s.root, rope.ByteOfRune(s.root, index)).Lines
}

func (s TextSnapshot) WriteTo(w io.Writer) (int64, error) { return rope.WriteTo(s.root, w) }

// sliceBytes returns bytes a to b of a tree.
func sliceBytes(n *rope.Node, a, b int) string {
	var t rope.Text
	if n != nil {
		t = rope.TextOf(n)
	}
	return t.Slice(a, b)
}
