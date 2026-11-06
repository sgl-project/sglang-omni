package editor

import (
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/egoist/mygo/internal/rope"
)

// Selection is a selection of an editor: the text from Anchor to
// Head, byte offsets, with the caret at Head. Anchor == Head is a caret
// alone.
type Selection struct {
	Anchor, Head int
}

// Start and End return the selection's edges in order.
func (s Selection) Start() int { return min(s.Anchor, s.Head) }
func (s Selection) End() int   { return max(s.Anchor, s.Head) }

// Empty reports whether the selection is a caret alone.
func (s Selection) Empty() bool { return s.Anchor == s.Head }

// sel is a selection with the x the caret keeps going up and down,
// and an id ordering selections by when they were made: the newest is
// the primary one, which input methods and scrolling follow.
type sel struct {
	anchor, head int
	goal         float32 // -1 when none
	id           int
}

func (s sel) start() int  { return min(s.anchor, s.head) }
func (s sel) end() int    { return max(s.anchor, s.head) }
func (s sel) empty() bool { return s.anchor == s.head }

func (s *sel) collapse(at int) { s.anchor, s.head, s.goal = at, at, -1 }

// mergeSels sorts selections and merges those that overlap, as Zed's
// selections collection does: overlapping ones merge, as does a caret
// touching a selection, while two selections that only touch stay apart.
func mergeSels(sels []sel) []sel {
	if len(sels) < 2 {
		return sels
	}
	slices.SortStableFunc(sels, func(a, b sel) int {
		if a.start() != b.start() {
			return a.start() - b.start()
		}
		return a.end() - b.end()
	})
	out := sels[:1]
	for _, s := range sels[1:] {
		last := &out[len(out)-1]
		touch := s.start() == last.end() && (s.empty() || last.empty())
		if s.start() < last.end() || touch || s.start() == last.start() {
			// Keep the newer's identity and direction.
			start, end := last.start(), max(last.end(), s.end())
			reversed := last.head < last.anchor
			id := last.id
			if s.id > id {
				id, reversed = s.id, s.head < s.anchor
			}
			*last = sel{anchor: start, head: end, goal: -1, id: id}
			if reversed {
				last.anchor, last.head = end, start
			}
			continue
		}
		out = append(out, s)
	}
	return out
}

// newest returns the index of the newest selection.
func newest(sels []sel) int {
	best := 0
	for i, s := range sels {
		if s.id > sels[best].id {
			best = i
		}
	}
	return best
}

// Character classes, as Zed's CharClassifier: words, punctuation and
// whitespace. extra holds characters other than letters, digits and _
// that words of the language hold.
type charClass uint8

const (
	classSpace charClass = iota
	classPunct
	classWord
)

func classOf(r rune, extra string) charClass {
	switch {
	case r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) || extra != "" && strings.ContainsRune(extra, r):
		return classWord
	case unicode.IsSpace(r):
		return classSpace
	}
	return classPunct
}

// wordRight returns where the word after off ends: past spaces, then the
// run of characters of one class, as alt-right does. sub stops at the
// ends of subwords too: at underscores and where lower case turns upper.
func wordRight(r rope.Text, off int, extra string, sub bool) int {
	n := r.Len()
	// Past whitespace on the line; a newline is a step of its own.
	c, size := r.RuneAt(off)
	if off < n && c == '\n' {
		return off + 1
	}
	for off < n && c != '\n' && unicode.IsSpace(c) {
		off += size
		c, size = r.RuneAt(off)
	}
	if off >= n || c == '\n' {
		return off
	}
	cls := classOf(c, extra)
	prev := c
	off += size
	for off < n {
		c, size = r.RuneAt(off)
		if classOf(c, extra) != cls || c == '\n' {
			break
		}
		if sub && cls == classWord && subwordBreak(prev, c, r, off+size) {
			break
		}
		prev = c
		off += size
	}
	return off
}

// wordLeft returns where the word before off starts.
func wordLeft(r rope.Text, off int, extra string, sub bool) int {
	c, size := r.RuneBefore(off)
	if off > 0 && c == '\n' {
		return off - 1
	}
	for off > 0 && c != '\n' && unicode.IsSpace(c) {
		off -= size
		c, size = r.RuneBefore(off)
	}
	if off <= 0 || c == '\n' {
		return off
	}
	cls := classOf(c, extra)
	next := c
	off -= size
	for off > 0 {
		c, size = r.RuneBefore(off)
		if classOf(c, extra) != cls || c == '\n' {
			break
		}
		if sub && cls == classWord && subwordBreakBack(c, next) {
			break
		}
		next = c
		off -= size
	}
	return off
}

// subwordBreak reports whether a subword ends between prev and c going
// right: at an underscore, a hyphen, or lower case turning upper.
func subwordBreak(prev, c rune, r rope.Text, after int) bool {
	if prev == '_' && c != '_' || c == '_' && prev != '_' {
		return true
	}
	if unicode.IsLower(prev) && unicode.IsUpper(c) {
		return true
	}
	// "HTTPServer": a capital before a capital and lower case starts a word.
	if unicode.IsUpper(prev) && unicode.IsUpper(c) {
		n, _ := r.RuneAt(after)
		return unicode.IsLower(n)
	}
	return false
}

func subwordBreakBack(c, next rune) bool {
	if c == '_' && next != '_' || next == '_' && c != '_' {
		return true
	}
	return unicode.IsLower(c) && unicode.IsUpper(next)
}

// wordAt returns the word around off, or the run of its class: what a
// double click selects.
func wordAt(r rope.Text, off int, extra string) (int, int) {
	line := r.LineOf(off)
	a, z := r.LineStart(line), r.LineEnd(line)
	s := r.Slice(a, z)
	i := off - a
	if i > len(s) {
		i = len(s)
	}
	// The class of the character at off, or before it at the end of a
	// word.
	var cls charClass
	switch {
	case i < len(s):
		c, _ := utf8.DecodeRuneInString(s[i:])
		cls = classOf(c, extra)
		if cls != classWord && i > 0 {
			if p, _ := utf8.DecodeLastRuneInString(s[:i]); classOf(p, extra) == classWord {
				cls = classWord
			}
		}
	case i > 0:
		c, _ := utf8.DecodeLastRuneInString(s[:i])
		cls = classOf(c, extra)
	default:
		return off, off
	}
	start, end := i, i
	for start > 0 {
		c, size := utf8.DecodeLastRuneInString(s[:start])
		if classOf(c, extra) != cls {
			break
		}
		start -= size
	}
	for end < len(s) {
		c, size := utf8.DecodeRuneInString(s[end:])
		if classOf(c, extra) != cls {
			break
		}
		end += size
	}
	return a + start, a + end
}

// graphemeRight and graphemeLeft step over a whole character with its
// combining marks, joiners and variation selectors, and over a CRLF.
func graphemeRight(r rope.Text, off int) int {
	n := r.Len()
	if off >= n {
		return n
	}
	c, size := r.RuneAt(off)
	if isRegional(c) && regionalBefore(r, off)%2 == 0 {
		// A flag: two regional indicators.
		if d, dsize := r.RuneAt(off + size); isRegional(d) {
			size += dsize
		}
	}
	off += size
	if c == '\r' {
		if d, _ := r.RuneAt(off); d == '\n' {
			return off + 1
		}
	}
	for off < n {
		c, size := r.RuneAt(off)
		if c == 0x200d { // a joiner and what it joins
			off += size
			if off < n {
				_, size = r.RuneAt(off)
				off += size
			}
			continue
		}
		if !extends(c) {
			break
		}
		off += size
	}
	return off
}

func graphemeLeft(r rope.Text, off int) int {
	if off <= 0 {
		return 0
	}
	for off > 0 {
		c, size := r.RuneBefore(off)
		off -= size
		if c == '\n' {
			if d, _ := r.RuneBefore(off); d == '\r' {
				off--
			}
			return off
		}
		if extends(c) {
			continue
		}
		if p, size := r.RuneBefore(off); p == 0x200d {
			off -= size
			continue
		}
		if isRegional(c) && regionalBefore(r, off)%2 == 1 {
			// The second of a flag's regional indicators.
			_, size := r.RuneBefore(off)
			off -= size
		}
		return off
	}
	return 0
}

// isRegional reports whether c is a regional indicator, two of which make
// a flag.
func isRegional(c rune) bool { return c >= 0x1f1e6 && c <= 0x1f1ff }

// regionalBefore counts the regional indicators right before off, which
// pair up from the first.
func regionalBefore(r rope.Text, off int) int {
	n := 0
	for off > 0 {
		c, size := r.RuneBefore(off)
		if !isRegional(c) {
			break
		}
		n++
		off -= size
	}
	return n
}

// extends reports whether c goes with the character before it.
func extends(c rune) bool {
	return unicode.Is(unicode.Mn, c) || unicode.Is(unicode.Me, c) || unicode.Is(unicode.Mc, c) || c >= 0xfe00 && c <= 0xfe0f ||
		c >= 0x1f3fb && c <= 0x1f3ff || c >= 0xe0020 && c <= 0xe007f
}

// indentOf returns the leading whitespace of s.
func indentOf(s string) string {
	return s[:len(s)-len(strings.TrimLeft(s, " \t"))]
}

// indentWidth returns the columns of an indentation, tabs to tab stops.
func indentWidth(s string, tab int) int {
	col := 0
	for _, c := range s {
		switch c {
		case '\t':
			col += tab - col%tab
		case ' ':
			col++
		default:
			return col
		}
	}
	return col
}
