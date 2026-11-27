package rope

import (
	"strings"
	"unicode/utf8"
)

// Text is a version of a text, read by bytes and lines, as code editing
// counts. Its zero value is the empty text.
type Text struct{ n *Node }

// New returns the text s.
func New(s string) Text { return Text{Build(s)} }

// TextOf returns the text of a tree.
func TextOf(n *Node) Text { return Text{n} }

// Root returns the tree of the text.
func (t Text) Root() *Node { return t.n }

// Same reports whether t and u are the same version, without comparing
// their bytes.
func (t Text) Same(u Text) bool { return t.n == u.n }

func (t Text) Len() int       { return Sum(t.n).Bytes }
func (t Text) LineCount() int { return Sum(t.n).Lines + 1 }

// LineStart returns the byte line starts at, clamped.
func (t Text) LineStart(line int) int { return LineStart(t.n, line) }

// LineEnd returns the byte line ends at, before its newline.
func (t Text) LineEnd(line int) int {
	if line+1 < t.LineCount() {
		return LineStart(t.n, line+1) - 1
	}
	return t.Len()
}

// LineOf returns the line holding byte off.
func (t Text) LineOf(off int) int { return Prefix(t.n, off).Lines }

// Slice returns bytes a to b.
func (t Text) Slice(a, b int) string {
	a, b = max(0, min(a, t.Len())), max(0, min(b, t.Len()))
	if a >= b {
		return ""
	}
	var out strings.Builder
	out.Grow(b - a)
	Read(t.n, a, b, &out)
	return out.String()
}

// Line returns line i, without its newline.
func (t Text) Line(i int) string { return t.Slice(t.LineStart(i), t.LineEnd(i)) }

func (t Text) String() string { return t.Slice(0, t.Len()) }

// ByteAt returns the byte at off, or 0 past the end.
func (t Text) ByteAt(off int) byte {
	n := t.n
	if off < 0 || off >= Sum(n).Bytes {
		return 0
	}
	for n.left != nil {
		if off < n.left.sum.Bytes {
			n = n.left
		} else {
			off -= n.left.sum.Bytes
			n = n.right
		}
	}
	return n.text[off]
}

// RuneAt returns the rune starting at off and its size, 0 and 0 at the
// end.
func (t Text) RuneAt(off int) (rune, int) {
	if off < 0 || off >= t.Len() {
		return 0, 0
	}
	if b := t.ByteAt(off); b < utf8.RuneSelf {
		return rune(b), 1
	}
	return utf8.DecodeRuneInString(t.Slice(off, off+utf8.UTFMax))
}

// RuneBefore returns the rune ending at off and its size.
func (t Text) RuneBefore(off int) (rune, int) {
	if off <= 0 {
		return 0, 0
	}
	if b := t.ByteAt(off - 1); b < utf8.RuneSelf {
		return rune(b), 1
	}
	return utf8.DecodeLastRuneInString(t.Slice(max(0, off-utf8.UTFMax), off))
}

// UTF16 returns how many UTF-16 units come before byte off.
func (t Text) UTF16(off int) int { return Prefix(t.n, off).Units }

// FromUTF16 returns the byte of UTF-16 unit u, rounded down to a rune.
func (t Text) FromUTF16(u int) int { return ByteOfUnit(t.n, u) }

// Chunks calls fn with bytes a to b, a piece at a time, until fn returns
// false.
func (t Text) Chunks(a, b int, fn func(s string) bool) { Chunks(t.n, a, b, fn) }

// Replace returns the text with bytes a to b replaced by s.
func (t Text) Replace(a, b int, s string) Text { return Text{ReplaceBytes(t.n, a, b, s)} }

// EachLine calls fn with lines from to to, exclusive, and the byte each
// starts at, reading them a batch at a time rather than finding each in
// the tree, until fn returns false.
func (t Text) EachLine(from, to int, fn func(i, start int, s string) bool) {
	to = min(to, t.LineCount())
	for i := max(0, from); i < to; {
		end := min(i+256, to)
		a := t.LineStart(i)
		text := t.Slice(a, t.LineEnd(end-1))
		for i < end {
			n := strings.IndexByte(text, '\n')
			if n < 0 {
				n = len(text)
			}
			if !fn(i, a, text[:n]) {
				return
			}
			a += n + 1
			if n < len(text) {
				text = text[n+1:]
			} else {
				text = ""
			}
			i++
		}
	}
}

// EachLineBack calls fn with lines from down to to, inclusive, as
// EachLine does going up.
func (t Text) EachLineBack(from, to int, fn func(i, start int, s string) bool) {
	from = min(from, t.LineCount()-1)
	to = max(0, to)
	for i := from; i >= to; {
		first := max(i-255, to)
		a := t.LineStart(first)
		text := t.Slice(a, t.LineEnd(i))
		// Where the batch's lines start, to go back through them.
		starts := make([]int, 0, i-first+1)
		for at := 0; ; {
			starts = append(starts, at)
			n := strings.IndexByte(text[at:], '\n')
			if n < 0 {
				break
			}
			at += n + 1
		}
		for k := len(starts) - 1; k >= 0 && i >= first; k-- {
			end := len(text)
			if k+1 < len(starts) {
				end = starts[k+1] - 1
			}
			if !fn(i, a+starts[k], text[starts[k]:end]) {
				return
			}
			i--
		}
	}
}
