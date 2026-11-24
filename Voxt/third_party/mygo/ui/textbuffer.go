package ui

import (
	"encoding/binary"
	"math/bits"
	"slices"
	"sort"
	"strings"
	"unicode"
	"unicode/utf16"
	"unicode/utf8"

	"github.com/egoist/mygo/internal/rope"
	"github.com/egoist/mygo/internal/text"
)

// buffer is the text of a text input: the string the app's value holds,
// which it shares, and the paragraphs of the text, the runes between
// newlines. Edits make a new string, a copy of the bytes, and move the
// paragraphs after them: no edit converts the whole text, which takes
// milliseconds for a few megabytes.
type buffer struct {
	s     string
	n     int // runes
	units int // UTF-16 units, for indexed native input queries
	paras []paragraph
	// version counts the edits.
	version uint64
	indexed bool
	root    *rope.Node
}

func (b *buffer) string() string {
	if b.indexed {
		return (TextSnapshot{root: b.root}).String()
	}
	return b.s
}
func (b *buffer) byteLen() int {
	if b.indexed {
		return rope.Sum(b.root).Bytes
	}
	return len(b.s)
}
func (b *buffer) start(p int) int {
	if b.indexed {
		return rope.Prefix(b.root, rope.LineStart(b.root, p)).Runes
	}
	return b.paras[p].rune
}
func (b *buffer) setSnapshot(s TextSnapshot) {
	b.s = ""
	b.indexed, b.root = true, s.root
	b.n, b.units = s.Len(), s.UTF16Len()
	clear(b.paras)
	n := s.LineCount()
	if cap(b.paras) < n || cap(b.paras) > 4*n {
		b.paras = make([]paragraph, n)
	} else {
		b.paras = b.paras[:n]
	}
	b.version++
}

// paragraph is a paragraph of a buffer: where it starts, in runes and in
// bytes, and what a text area made of it (textarea.go): its layout, at
// the width and in the style of the text area's last layout, with the
// composition of an input method it showed, and its height, 0 until it was
// laid out.
type paragraph struct {
	rune, byte, units int
	layout            *text.Layout
	compose           string
	// spans are the weights of the TextRanges in the paragraph, as its
	// layout took them.
	spans string
	h     float32
}

func (b *buffer) set(s string) {
	b.indexed, b.root = false, nil
	b.s = s
	clear(b.paras) // the layouts of the old text
	n := strings.Count(s, "\n") + 1
	if cap(b.paras) < n || cap(b.paras) > 4*n {
		b.paras = make([]paragraph, n)
	} else {
		b.paras = b.paras[:n]
	}
	b.paras[0] = paragraph{}
	r, units, from := 0, 0, 0
	for p := 1; p < n; p++ {
		end := from + strings.IndexByte(s[from:], '\n') + 1
		part := s[from:end]
		nr := countRunes(part)
		r += nr
		units += unitsForRunes(part, nr)
		b.paras[p] = paragraph{rune: r, byte: end, units: units}
		from = end
	}
	nr := countRunes(s[from:])
	b.n = r + nr
	b.units = units + unitsForRunes(s[from:], nr)
	b.version++
}

// countRunes counts ASCII in words, as most code and logs consist of it,
// and leaves non-ASCII and malformed UTF-8 to the standard decoder.
func countRunes(s string) int {
	all := s
	for len(s) >= 8 {
		if binary.LittleEndian.Uint64([]byte(s))&0x8080808080808080 != 0 {
			return utf8.RuneCountInString(all)
		}
		s = s[8:]
	}
	for i := range len(s) {
		if s[i] >= utf8.RuneSelf {
			return utf8.RuneCountInString(all)
		}
	}
	return len(all)
}

func countUnits(s string) int {
	return unitsForRunes(s, countRunes(s))
}

func unitsForRunes(s string, n int) int {
	if n == len(s) {
		return n
	}
	for _, r := range s {
		if r > 0xffff {
			n++
		}
	}
	return n
}

// Native offsets are indexed by paragraph, so queries near a large file's
// end do not scan the whole document. Surrogate interiors round down.
func (b *buffer) utf16At(index int) int {
	if b.indexed {
		return (TextSnapshot{root: b.root}).UTF16Offset(index)
	}
	index = max(0, min(index, b.n))
	p := b.para(index)
	pr := b.paras[p]
	return pr.units + countUnits(b.s[pr.byte:b.byteOf(index)])
}

func (b *buffer) runeAtUTF16(index int) int {
	if b.indexed {
		return (TextSnapshot{root: b.root}).RuneOffset(index)
	}
	index = max(0, min(index, b.units))
	p := sort.Search(len(b.paras), func(p int) bool { return b.paras[p].units > index }) - 1
	pr := b.paras[p]
	nr, nb := b.next(p)
	if nb-pr.byte == nr-pr.rune {
		return pr.rune + index - pr.units
	}
	at, units := pr.rune, pr.units
	for _, r := range b.s[pr.byte:nb] {
		if units+utf16.RuneLen(r) > index {
			break
		}
		units += utf16.RuneLen(r)
		at++
	}
	return at
}

// para returns the paragraph holding rune i; the newline ending a
// paragraph is its.
func (b *buffer) para(i int) int {
	if b.indexed {
		return (TextSnapshot{root: b.root}).LineAt(i)
	}
	return sort.Search(len(b.paras), func(k int) bool { return b.paras[k].rune > i }) - 1
}

// next returns where the paragraph after p starts, after its newline, or
// the end of the text.
func (b *buffer) next(p int) (rune, byte int) {
	if b.indexed {
		off := rope.LineStart(b.root, p+1)
		return rope.Prefix(b.root, off).Runes, off
	}
	if p+1 < len(b.paras) {
		return b.paras[p+1].rune, b.paras[p+1].byte
	}
	return b.n, len(b.s)
}

// end returns the rune ending paragraph p, its newline or the end of the
// text.
func (b *buffer) end(p int) int {
	if p+1 < len(b.paras) {
		return b.start(p+1) - 1
	}
	return b.n
}

// byteOf returns the byte of the text rune i starts at.
func (b *buffer) byteOf(i int) int {
	if b.indexed {
		return rope.ByteOfRune(b.root, i)
	}
	i = max(0, min(i, b.n))
	p := b.para(i)
	pr := &b.paras[p]
	nr, nb := b.next(p)
	if nb-pr.byte == nr-pr.rune {
		return pr.byte + i - pr.rune // ASCII
	}
	off := pr.byte
	for k := pr.rune; k < i; k++ {
		_, size := utf8.DecodeRuneInString(b.s[off:])
		off += size
	}
	return off
}

// slice returns the runes from a to z.
func (b *buffer) slice(a, z int) string {
	if b.indexed {
		return (TextSnapshot{root: b.root}).Slice(a, z)
	}
	return b.s[b.byteOf(a):b.byteOf(z)]
}

// runeOffset returns the byte of s rune i starts at.
func runeOffset(s string, i int) int {
	for at := range s {
		if i == 0 {
			return at
		}
		i--
	}
	return len(s)
}

// text returns paragraph p, without its newline.
func (b *buffer) text(p int) string {
	if b.indexed {
		return (TextSnapshot{root: b.root}).Line(p)
	}
	_, nb := b.next(p)
	if p+1 < len(b.paras) {
		nb-- // the newline
	}
	return b.s[b.paras[p].byte:nb]
}

// replace replaces the runes from a to z with s. The paragraphs from the
// one holding a to the one holding z give way to those of the new text,
// which have no layout yet: it returns the first, those there were, and
// how many there are.
func (b *buffer) replace(a, z int, s string) (first int, old []paragraph, after int) {
	if b.indexed {
		pa, pz := b.para(a), b.para(z)
		old = slices.Clone(b.paras[pa : pz+1])
		b.root = rope.Replace(b.root, a, z, s)
		sum := rope.Sum(b.root)
		b.n, b.units = sum.Runes, sum.Units
		added := make([]paragraph, strings.Count(s, "\n"))
		b.paras = slices.Replace(b.paras, pa+1, pz+1, added...)
		b.paras[pa] = paragraph{}
		b.version++
		return pa, old, len(added) + 1
	}
	ba, bz := b.byteOf(a), b.byteOf(z)
	pa, pz := b.para(a), b.para(z)
	old = slices.Clone(b.paras[pa : pz+1])
	runes := utf8.RuneCountInString(s)
	var added []paragraph
	r, units := a, b.utf16At(a)
	for i, c := range s {
		r++
		units += utf16.RuneLen(c)
		if c == '\n' {
			added = append(added, paragraph{rune: r, byte: ba + i + 1, units: units})
		}
	}
	dr, db := runes-(z-a), len(s)-(bz-ba)
	du := countUnits(s) - (b.utf16At(z) - b.utf16At(a))
	b.s = b.s[:ba] + s + b.s[bz:]
	b.n += dr
	b.units += du
	b.paras = slices.Replace(b.paras, pa+1, pz+1, added...)
	b.paras[pa] = paragraph{rune: b.paras[pa].rune, byte: b.paras[pa].byte, units: b.paras[pa].units}
	for k := pa + 1 + len(added); k < len(b.paras); k++ {
		b.paras[k].rune += dr
		b.paras[k].byte += db
		b.paras[k].units += du
	}
	b.version++
	return pa, old, len(added) + 1
}

func isWordRune(r rune) bool { return unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' }

// nextWord returns the end of the word after i, skipping what is between,
// as text.Boundaries does.
func (b *buffer) nextWord(i int) int {
	if b.indexed {
		word := false
		for i < b.n {
			r := b.runeAt(i)
			if isWordRune(r) {
				word = true
			} else if word {
				break
			}
			i++
		}
		return i
	}
	off := b.byteOf(i)
	word := false
	for i < b.n {
		r, size := utf8.DecodeRuneInString(b.s[off:])
		if isWordRune(r) {
			word = true
		} else if word {
			break
		}
		i, off = i+1, off+size
	}
	return i
}

// prevWord returns the start of the word before i.
func (b *buffer) prevWord(i int) int {
	if b.indexed {
		word := false
		for i > 0 {
			r := b.runeAt(i - 1)
			if isWordRune(r) {
				word = true
			} else if word {
				break
			}
			i--
		}
		return i
	}
	off := b.byteOf(i)
	word := false
	for i > 0 {
		r, size := utf8.DecodeLastRuneInString(b.s[:off])
		if isWordRune(r) {
			word = true
		} else if word {
			break
		}
		i, off = i-1, off-size
	}
	return i
}

// wordAt returns the word, or the run of other runes, around i, within
// its line.
func (b *buffer) wordAt(i int) (start, end int) {
	if b.indexed {
		if b.n == 0 {
			return 0, 0
		}
		i = max(0, min(i, b.n-1))
		word := isWordRune(b.runeAt(i))
		start, end = i, i
		for start > 0 {
			r := b.runeAt(start - 1)
			if r == '\n' || isWordRune(r) != word {
				break
			}
			start--
		}
		for end < b.n {
			r := b.runeAt(end)
			if r == '\n' || isWordRune(r) != word {
				break
			}
			end++
		}
		return start, end
	}
	if b.n == 0 {
		return 0, 0
	}
	i = min(i, b.n-1)
	off := b.byteOf(i)
	r, _ := utf8.DecodeRuneInString(b.s[off:])
	word := isWordRune(r)
	start, end = i, i
	for o := off; start > 0; {
		r, size := utf8.DecodeLastRuneInString(b.s[:o])
		if isWordRune(r) != word || r == '\n' {
			break
		}
		start, o = start-1, o-size
	}
	for o := off; end < b.n; {
		r, size := utf8.DecodeRuneInString(b.s[o:])
		if isWordRune(r) != word || r == '\n' {
			break
		}
		end, o = end+1, o+size
	}
	return start, end
}

func (b *buffer) runeAt(i int) rune {
	r, _ := rope.TextOf(b.root).RuneAt(rope.ByteOfRune(b.root, i))
	return r
}

// graphemes finds the grapheme boundaries of a buffer a paragraph at a
// time: they never cross a newline, but for the carriage return before
// one, which the paragraph holds with its newline.
type graphemes struct {
	b       text.Boundaries
	para    int
	version uint64
	valid   bool
}

// of prepares g for paragraph p of b, with its newline, and returns where
// the paragraph starts.
func (g *graphemes) of(b *buffer, p int) int {
	if !g.valid || g.para != p || g.version != b.version {
		nr, _ := b.next(p)
		g.b.Reset([]rune(b.slice(b.start(p), nr)))
		g.para, g.version, g.valid = p, b.version, true
	}
	return b.start(p)
}

// next returns the end of the grapheme starting at or containing i.
func (g *graphemes) next(b *buffer, i int) int {
	if i >= b.n {
		return b.n
	}
	start := g.of(b, b.para(i))
	return start + g.b.NextGrapheme(i-start)
}

// prev returns the start of the grapheme before i.
func (g *graphemes) prev(b *buffer, i int) int {
	if i <= 0 {
		return 0
	}
	p := b.para(i)
	if b.start(p) == i {
		p-- // the newline ending the paragraph before
	}
	start := g.of(b, p)
	return start + g.b.PrevGrapheme(i-start)
}

// heights sums the heights of the paragraphs of a text area, measured as
// they are laid out and estimated for the others, in two Fenwick trees:
// the heights measured, and how many are not, which an estimate of their
// height multiplies. The top of a paragraph, and the paragraph at a
// height, take O(log n) for any estimate.
type heights struct {
	measured []float64 // 1-based
	unknown  []int32
	est      float64
}

// reset sums the heights of paras.
func (h *heights) reset(paras []paragraph) {
	n := len(paras)
	// Opening a small file after a large one must not keep its trees.
	if cap(h.measured) > 4*(n+1) {
		h.measured, h.unknown = nil, nil
	}
	h.measured = slices.Grow(h.measured[:0], n+1)[:n+1]
	h.unknown = slices.Grow(h.unknown[:0], n+1)[:n+1]
	h.measured[0], h.unknown[0] = 0, 0
	for i, p := range paras {
		if p.h > 0 {
			h.measured[i+1], h.unknown[i+1] = float64(p.h), 0
		} else {
			h.measured[i+1], h.unknown[i+1] = 0, 1
		}
	}
	for i := 1; i <= n; i++ {
		if j := i + i&-i; j <= n {
			h.measured[j] += h.measured[i]
			h.unknown[j] += h.unknown[i]
		}
	}
}

// add changes the height measured of paragraph p by dm, and the count of
// those not measured by du.
func (h *heights) add(p int, dm float64, du int32) {
	for i := p + 1; i < len(h.measured); i += i & -i {
		h.measured[i] += dm
		h.unknown[i] += du
	}
}

// top returns the top of paragraph p: the height of those before it.
func (h *heights) top(p int) float64 {
	var m float64
	var u int32
	for i := min(p, len(h.measured)-1); i > 0; i -= i & -i {
		m += h.measured[i]
		u += h.unknown[i]
	}
	return m + float64(u)*h.est
}

// at returns the paragraph at y, the last for y past the end.
func (h *heights) at(y float64) int {
	n := len(h.measured) - 1
	if n <= 0 {
		return 0
	}
	pos := 0
	var sum float64
	for step := 1 << (bits.Len(uint(n)) - 1); step > 0; step >>= 1 {
		if next := pos + step; next <= n {
			if s := sum + h.measured[next] + float64(h.unknown[next])*h.est; s <= y {
				pos, sum = next, s
			}
		}
	}
	return min(pos, n-1)
}

// totals returns the sum of the heights measured and how many are not.
func (h *heights) totals() (float64, int) {
	var m float64
	var u int32
	for i := len(h.measured) - 1; i > 0; i -= i & -i {
		m += h.measured[i]
		u += h.unknown[i]
	}
	return m, int(u)
}
