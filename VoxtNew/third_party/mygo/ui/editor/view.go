package editor

import (
	"math"
	"slices"
	"sort"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/egoist/mygo/internal/rope"
	"github.com/egoist/mygo/internal/text"
	"github.com/egoist/mygo/ui"
)

// view shows the lines of a Buffer: it keeps the display model in step
// with the text, lays out the lines in view a row at a time, shaping each
// row's text once and keeping it while frames draw it, and paints them
// with the colors of their tokens. An editor has one; a diff one per
// side.
type view struct {
	buf      *Buffer
	text     rope.Text
	version  uint64
	hl       Highlighter // the buffer's, as of the last frame
	hlGen    int         // counts the highlighters the view had
	readOnly bool

	font       ui.Font
	style      text.Style
	lineHeight float32
	ascent     float32
	descent    float32
	charWidth  float32 // the advance of "0", for columns
	tabSize    int
	wrap       bool
	wrapWidth  float32
	scale      float32
	theme      *Theme

	heights heights
	// inlays are texts shown in lines that are not part of them, sorted
	// by offset.
	inlays []inlay

	lines   lineCache
	shaped  shapeCache
	widths  charWidths
	maxLine float32 // the widest line laid out, for scrolling sideways
	wrapAt  int     // the next line to measure the wraps of
	// approx tells that a frame showed tokens from states the
	// highlighter has not confirmed yet.
	approx     bool
	tokScratch []Token
}

// inlay is text shown in a line at an offset, as an inlay hint or a
// suggestion.
type inlay struct {
	at   int
	text string
	kind int // 0 a hint, 1 a suggestion
}

// lineLayout is a line laid out for display.
type lineLayout struct {
	n     int
	start int    // the byte of the text it starts at
	text  string // the line, without its newline
	disp  string // as displayed: tabs expanded, inlays inserted
	segs  []dispSeg
	rows  []rowLayout
	toks  []Token
	exact bool // toks from a confirmed state
	width float32
	used  uint64
}

// dispSeg maps a run of the displayed text to the line's: text, a tab
// expanded to spaces, or an inlay.
type dispSeg struct {
	disp     int // the displayed byte it starts at
	from, to int // the bytes of the line it shows
	kind     uint8
	inlay    int
	ghost    bool // a suggestion, which a caret at it shows before
}

const (
	segText uint8 = iota
	segTab
	segInlay
)

// rowLayout is a row of a line: a run of its displayed text, from its
// wrap indent, in pieces shaped apart for very long lines.
type rowLayout struct {
	from, to int
	indent   float32
	pieces   []rowPiece
	width    float32
}

type rowPiece struct {
	from, to int     // displayed bytes
	x        float32 // from the row's start
	l        *text.Layout
	ascii    bool
	width    float32
}

const pieceBytes = 2048

type lineCache struct {
	m     map[int]*lineLayout
	frame uint64
}

// shapeCache keeps the layouts of row texts two frames, as GPUI's
// line layout cache: those of the last frame, which the next one takes
// over as it uses them, and those of this one.
type shapeCache struct {
	style     text.Style
	prev, cur map[string]*text.Layout
}

func (c *shapeCache) get(s string, style text.Style) *text.Layout {
	if c.style != style {
		c.style, c.prev, c.cur = style, nil, nil
	}
	if c.cur == nil {
		c.cur = map[string]*text.Layout{}
	}
	if l, ok := c.cur[s]; ok {
		return l
	}
	if l, ok := c.prev[s]; ok {
		c.cur[s] = l
		return l
	}
	l := text.Shared().Shape(text.Params{Text: s, Style: style, NoWrap: true, KeepSpaces: true})
	c.cur[s] = l
	return l
}

func (c *shapeCache) endFrame() {
	c.prev, c.cur = c.cur, c.prev
	if c.cur != nil {
		clear(c.cur)
	}
}

// charWidths measures characters for soft wraps and long lines
// without shaping them, as Zed's LineWrapper sums cached advances.
type charWidths struct {
	style text.Style
	ascii [128]float32
	other map[rune]float32
	mono  float32 // the advance of every printable ASCII character, if one
}

func (w *charWidths) reset(style text.Style) {
	if w.style == style && w.other != nil {
		return
	}
	*w = charWidths{style: style, other: map[rune]float32{}}
	for i := range w.ascii {
		w.ascii[i] = -1
	}
	l := text.Shared().Shape(text.Params{Text: asciiPrintableCode, Style: style, NoWrap: true, KeepSpaces: true})
	mono := float32(-1)
	if len(l.Lines) == 1 {
		line := l.Lines[0]
		ok := len(line.Glyphs) == len(asciiPrintableCode)
		for _, g := range line.Glyphs {
			if g.Cluster < len(asciiPrintableCode) {
				w.ascii[asciiPrintableCode[g.Cluster]] = g.Advance
				if mono < 0 {
					mono = g.Advance
				} else if math.Abs(float64(mono-g.Advance)) > 0.01 {
					ok = false
				}
			}
		}
		if ok {
			w.mono = mono
		}
	}
}

const asciiPrintableCode = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"

func (w *charWidths) of(r rune) float32 {
	if r < 128 {
		if v := w.ascii[r]; v >= 0 {
			return v
		}
		if r < ' ' {
			return 0
		}
	}
	if v, ok := w.other[r]; ok {
		return v
	}
	l := text.Shared().Shape(text.Params{Text: string(r), Style: w.style, NoWrap: true, KeepSpaces: true})
	v := l.Width
	if len(w.other) < 4096 {
		w.other[r] = v
	}
	return v
}

// width returns the width of s.
func (w *charWidths) width(s string) float32 {
	if w.mono > 0 && isASCII(s) {
		return w.mono * float32(len(s))
	}
	var x float32
	for _, r := range s {
		x += w.of(r)
	}
	return x
}

func isASCII(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}

// setStyle sets the font and the line height, and the scale of the
// display, which line heights snap to.
func (v *view) setStyle(f ui.Font, lineHeight, scale float32, tabSize int) {
	if f.Family == "" {
		f.Family = "monospace"
	}
	if f.Size <= 0 {
		f.Size = 13
	}
	style := styleOf(f)
	if lineHeight <= 0 {
		lineHeight = 1.5
	}
	lh := float32(math.Round(float64(f.Size*lineHeight*scale))) / scale
	tabSize = max(1, min(tabSize, 16))
	if v.style == style && v.lineHeight == lh && v.tabSize == tabSize && v.scale == scale {
		return
	}
	fontChanged := v.style != style || v.scale != scale
	v.font, v.style, v.lineHeight, v.tabSize, v.scale = f, style, lh, tabSize, scale
	v.heights.lh = lh
	v.heights.dirty = true
	a, d, _ := text.Shared().Metrics(style)
	v.ascent, v.descent = a, d
	v.widths.reset(style)
	v.charWidth = v.widths.of('0')
	if fontChanged {
		v.maxLine = 0
	}
	v.lines.m = nil
	if v.wrap {
		v.heights.wraps.reset(v.text.LineCount(), true)
		v.wrapAt = 0
	}
}

// setWrap turns soft wrapping on at width, or off.
func (v *view) setWrap(on bool, width float32) {
	width = float32(math.Floor(float64(width)))
	if on == v.wrap && (!on || width == v.wrapWidth) {
		return
	}
	v.wrap, v.wrapWidth = on, width
	v.heights.wraps.reset(v.text.LineCount(), on)
	v.heights.dirty = true
	v.wrapAt = 0
	v.lines.m = nil
}

// setBuffer shows another buffer.
func (v *view) setBuffer(b *Buffer) {
	if v.buf == b {
		return
	}
	v.buf = b
	v.text, v.version = b.snapshot()
	v.hl = nil
	v.lines.m = nil
	v.maxLine = 0
	v.heights.setLines(v.text.LineCount())
	v.heights.wraps.reset(v.text.LineCount(), v.wrap)
	v.wrapAt = 0
}

// sync follows the buffer's edits, calling moved with the changes of
// each version so the owner moves what it keeps of the text. It returns
// false when the buffer changed past what its log keeps, and everything
// was taken anew.
func (v *view) sync(moved func(cs []change)) bool {
	if v.buf == nil {
		return true
	}
	changes, version, text, ok := v.buf.changesTo(v.version)
	if version == v.version && ok {
		v.text = text
		return true
	}
	old := v.text
	v.text, v.version = text, version
	v.lines.m = nil
	// The highlighter hears of the edits before a line of the text is
	// laid out, as moving the caret into view does between frames.
	v.followHighlighter()
	if !ok {
		// Too far behind the buffer's log to follow its edits: the
		// changes of the lines between the text it had and the text.
		cs := changesBetween(old, text)
		v.heights.setLines(text.LineCount())
		v.heights.wraps.reset(text.LineCount(), v.wrap)
		v.wrapAt = 0
		for i := range v.inlays {
			v.inlays[i].at = mapOffset(v.inlays[i].at, cs, biasRight)
		}
		if moved != nil {
			moved(cs)
		}
		return false
	}
	for _, cs := range changes {
		for _, c := range cs {
			// Lines from the change's first are measured anew.
			v.heights.wraps.splice(c.newLine, c.oldEndLine-c.oldLine+1, c.newEndLine-c.newLine+1)
			v.wrapAt = min(v.wrapAt, c.newLine)
		}
		for i := range v.inlays {
			v.inlays[i].at = mapOffset(v.inlays[i].at, cs, biasRight)
		}
		if moved != nil {
			moved(cs)
		}
	}
	v.heights.setLines(text.LineCount())
	v.heights.dirty = true
	return true
}

// advanceSyntax has the buffer's highlighter work toward the tokens of
// the lines up to upTo, for at most budget, and reports whether they are
// ready.
func (v *view) advanceSyntax(upTo int, budget time.Duration) bool {
	if v.buf == nil {
		return true
	}
	if hl := v.followHighlighter(); hl != nil {
		return hl.Highlight(upTo, budget)
	}
	return true
}

// followHighlighter has the buffer's highlighter follow its edits, and
// returns it, nil for plain text.
func (v *view) followHighlighter() Highlighter {
	hl := v.buf.highlighter()
	if hl != v.hl {
		// Another language: the lines laid out with the old one's tokens
		// go, and what was found of its brackets.
		v.hl = hl
		v.lines.m = nil
		v.hlGen++
	}
	return hl
}

// line returns line n laid out.
func (v *view) line(n int) *lineLayout {
	if v.lines.m == nil {
		v.lines.m = map[int]*lineLayout{}
	}
	if l, ok := v.lines.m[n]; ok {
		l.used = v.lines.frame
		if !l.exact {
			// A guess: the highlighter may know better now. The layout
			// does not depend on the tokens, which only color it.
			l.toks, l.exact = v.tokens(n, l.text, l.toks[:0])
		}
		return l
	}
	l := &lineLayout{n: n, used: v.lines.frame}
	l.start = v.text.LineStart(n)
	l.text = v.text.Slice(l.start, v.text.LineEnd(n))
	l.toks, l.exact = v.tokens(n, l.text, nil)
	v.display(l)
	v.layoutRows(l)
	v.lines.m[n] = l
	if l.width > v.maxLine {
		v.maxLine = l.width
	}
	return l
}

// endFrame forgets the lines and layouts no recent frame used.
func (v *view) endFrame() {
	v.lines.frame++
	if len(v.lines.m) > 512 {
		for k, l := range v.lines.m {
			if v.lines.frame-l.used > 2 {
				delete(v.lines.m, k)
			}
		}
	}
	v.shaped.endFrame()
}

// tokens returns the tokens of line n, whose text is s, and whether they
// are final.
func (v *view) tokens(n int, s string, dst []Token) ([]Token, bool) {
	if v.hl == nil {
		return dst[:0], true
	}
	toks, ok := v.hl.Tokens(n, s, dst[:0])
	if !ok {
		v.approx = true
	}
	return toks, ok
}

// display makes the displayed text of l: tabs expanded to the next tab
// stop, by columns counted in characters, and inlays inserted.
func (v *view) display(l *lineLayout) {
	s := l.text
	inl := v.inlaysIn(l.start, l.start+len(s))
	if strings.IndexByte(s, '\t') < 0 && len(inl) == 0 {
		l.disp, l.segs = s, nil
		return
	}
	var b strings.Builder
	b.Grow(len(s) + 8)
	col := 0
	textFrom := 0
	ii := 0
	flush := func(to int) {
		if to > textFrom {
			l.segs = append(l.segs, dispSeg{disp: b.Len(), from: textFrom, to: to, kind: segText})
			b.WriteString(s[textFrom:to])
		}
		textFrom = to
	}
	emitInlays := func(at int) {
		for ii < len(inl) && inl[ii].at-l.start <= at {
			flush(at)
			in := inl[ii]
			l.segs = append(l.segs, dispSeg{disp: b.Len(), from: at, to: at, kind: segInlay, inlay: ii, ghost: in.kind == 1})
			b.WriteString(in.text)
			col += utf8.RuneCountInString(in.text)
			ii++
		}
	}
	for i, r := range s {
		emitInlays(i)
		if r == '\t' {
			flush(i)
			w := v.tabSize - col%v.tabSize
			if col >= maxTabColumn {
				w = 1
			}
			l.segs = append(l.segs, dispSeg{disp: b.Len(), from: i, to: i + 1, kind: segTab})
			b.WriteString(spaces(w))
			col += w
			textFrom = i + 1
			continue
		}
		col++
	}
	emitInlays(len(s))
	flush(len(s))
	l.disp = b.String()
}

// maxTabColumn is where tabs stop expanding, which bounds the work
// of a line, as Zed's MAX_EXPANSION_COLUMN.
const maxTabColumn = 256

func spaces(n int) string {
	const s = "                "
	if n <= len(s) {
		return s[:n]
	}
	return strings.Repeat(" ", n)
}

func (v *view) inlaysIn(a, b int) []inlay {
	i := sort.Search(len(v.inlays), func(i int) bool { return v.inlays[i].at >= a })
	j := i
	for j < len(v.inlays) && v.inlays[j].at <= b {
		j++
	}
	return v.inlays[i:j]
}

// toDisp maps a byte of the line to its displayed byte; after places an
// offset at an inlay after it.
func (l *lineLayout) toDisp(col int, after bool) int {
	if l.segs == nil {
		return col
	}
	out := len(l.disp)
	for i := len(l.segs) - 1; i >= 0; i-- {
		s := l.segs[i]
		switch s.kind {
		case segText:
			if col >= s.from && col <= s.to {
				return s.disp + col - s.from
			}
		case segTab:
			if col == s.from {
				return s.disp
			}
			if col == s.to {
				return s.disp + segLen(l, i)
			}
		case segInlay:
			if col == s.from {
				if after && !s.ghost {
					return s.disp + segLen(l, i)
				}
				out = s.disp
				continue
			}
		}
		if s.from < col {
			break
		}
	}
	return out
}

func segLen(l *lineLayout, i int) int {
	if i+1 < len(l.segs) {
		return l.segs[i+1].disp - l.segs[i].disp
	}
	return len(l.disp) - l.segs[i].disp
}

// fromDisp maps a displayed byte to the byte of the line, the nearest
// edge for a byte inside a tab or an inlay.
func (l *lineLayout) fromDisp(d int) int {
	if l.segs == nil {
		return min(d, len(l.text))
	}
	i := sort.Search(len(l.segs), func(i int) bool { return l.segs[i].disp > d }) - 1
	if i < 0 {
		return 0
	}
	s := l.segs[i]
	switch s.kind {
	case segText:
		return min(s.from+d-s.disp, s.to)
	case segTab:
		if d-s.disp > segLen(l, i)/2 {
			return s.to
		}
		return s.from
	}
	return s.from
}

// layoutRows wraps the displayed text of l into rows and shapes them.
func (v *view) layoutRows(l *lineLayout) {
	l.rows = l.rows[:0]
	breaks := v.wrapBreaks(l.disp)
	indent := float32(0)
	if len(breaks) > 0 {
		indent = v.wrapIndent(l.disp)
	}
	from := 0
	for i := 0; i <= len(breaks); i++ {
		to := len(l.disp)
		if i < len(breaks) {
			to = breaks[i]
		}
		row := rowLayout{from: from, to: to}
		if i > 0 {
			row.indent = indent
		}
		// The rows of a wrapped line are shaped as they show, not all
		// of a long line's at once.
		v.shapeRow(l.disp, &row, len(breaks) > 0)
		l.rows = append(l.rows, row)
		l.width = max(l.width, row.indent+row.width)
		from = to
	}
	if v.wrap {
		v.heights.wraps.set(l.n, len(l.rows))
	}
}

// shapeRow shapes a row, in pieces of at most pieceBytes, placed by
// their widths: only the first piece of the row is shaped at once when
// the row is long, the others as they show (piece).
func (v *view) shapeRow(s string, row *rowLayout, lazy bool) {
	text := s[row.from:row.to]
	if len(text) <= pieceBytes && lazy {
		w := v.widths.width(text)
		row.pieces = append(row.pieces[:0], rowPiece{from: row.from, to: row.to, ascii: isASCII(text), width: w})
		row.width = w
		return
	}
	if len(text) <= pieceBytes {
		l := v.shaped.get(text, v.style)
		row.pieces = append(row.pieces[:0], rowPiece{from: row.from, to: row.to, l: l, ascii: isASCII(text), width: l.Width})
		row.width = l.Width
		return
	}
	row.pieces = row.pieces[:0]
	x := float32(0)
	for a := row.from; a < row.to; {
		b := min(a+pieceBytes, row.to)
		for b < row.to && !utf8.RuneStart(s[b]) {
			b++
		}
		w := v.widths.width(s[a:b])
		row.pieces = append(row.pieces, rowPiece{from: a, to: b, x: x, ascii: isASCII(s[a:b]), width: w})
		x += w
		a = b
	}
	row.width = x
}

// piece returns the layout of a row's piece, shaping it the first time.
func (v *view) piece(l *lineLayout, p *rowPiece) *text.Layout {
	if p.l == nil {
		p.l = v.shaped.get(l.disp[p.from:p.to], v.style)
	}
	return p.l
}

// wrapBreaks returns where the displayed text s breaks into rows when
// wrapping, as Zed's LineWrapper: summing the advances of characters,
// breaking after whitespace or between a word and what is not, anywhere
// in CJK, and inside a word too long for a row.
func (v *view) wrapBreaks(s string) []int {
	if !v.wrap || v.wrapWidth <= 0 {
		return nil
	}
	width := v.wrapWidth
	if v.widths.mono > 0 && float32(len(s))*v.widths.mono <= width {
		return nil
	}
	var breaks []int
	indent := float32(0)
	x := float32(0)
	start := 0
	last := -1 // the last place to break
	var prev rune
	for i, r := range s {
		if i > start && canBreak(prev, r) {
			last = i
		}
		w := v.widths.of(r)
		if x+w > width && i > start {
			at := i
			if last > start {
				at = last
			}
			breaks = append(breaks, at)
			if indent == 0 {
				indent = v.wrapIndent(s)
			}
			// Measure the new row from its start, after the indent.
			x = indent + v.widths.width(s[at:i])
			start, last = at, -1
		}
		x += w
		prev = r
	}
	return breaks
}

// wrapIndent is how far rows after the first are indented: as far as the
// line, but no more than half the wrap width.
func (v *view) wrapIndent(s string) float32 {
	n := len(s) - len(strings.TrimLeft(s, " "))
	w := v.widths.width(s[:n])
	return min(w, v.wrapWidth/2)
}

func canBreak(prev, r rune) bool {
	if prev == ' ' || prev == '\t' {
		return r != ' ' && r != '\t'
	}
	if isCJK(r) || isCJK(prev) {
		return true
	}
	return isWordRune(prev) != isWordRune(r) && r != ' ' && r != '\t' && !strings.ContainsRune(".,;:)]}>'\"", r)
}

func isCJK(r rune) bool {
	return r >= 0x2e80 && r <= 0x9fff || r >= 0xac00 && r <= 0xd7af || r >= 0xf900 && r <= 0xfaff || r >= 0xff00 && r <= 0xffef
}

// measureWraps measures the wraps of lines not measured yet, from those
// around near, for at most budget, and reports whether all are measured.
func (v *view) measureWraps(near int, budget time.Duration) bool {
	w := &v.heights.wraps
	if !w.on {
		return true
	}
	n := v.text.LineCount()
	if w.known >= n {
		return true
	}
	start := time.Now()
	measure := func(i, at int, s string) bool {
		if w.extra[i] != wrapUnknown {
			return true
		}
		disp := s
		if strings.IndexByte(s, '\t') >= 0 || len(v.inlaysIn(at, at+len(s))) > 0 {
			l := &lineLayout{n: i, start: at, text: s}
			v.display(l)
			disp = l.disp
		}
		w.set(i, len(v.wrapBreaks(disp))+1)
		return true
	}
	// Those around near first, so what shows is right at once.
	v.text.EachLine(near-200, near+200, measure)
	count := 0
	v.text.EachLine(v.wrapAt, n, func(i, at int, s string) bool {
		measure(i, at, s)
		v.wrapAt = i + 1
		count++
		return count&255 != 0 || time.Since(start) <= budget
	})
	return w.known >= n
}

// rowOf returns the row of l holding displayed byte d, the row it ends
// rather than the next for a byte at a wrap when upstream.
func (l *lineLayout) rowOf(d int, upstream bool) int {
	for i := range l.rows {
		r := &l.rows[i]
		if d < r.to || i == len(l.rows)-1 || d == r.to && upstream {
			return i
		}
	}
	return 0
}

// xOf returns the x of displayed byte d in row r, from the row's start.
func (v *view) xOf(l *lineLayout, r *rowLayout, d int) float32 {
	d = max(r.from, min(d, r.to))
	for i := range r.pieces {
		p := &r.pieces[i]
		if d < p.to || i == len(r.pieces)-1 {
			lay := v.piece(l, p)
			idx := d - p.from
			if !p.ascii {
				idx = utf8.RuneCountInString(l.disp[p.from:d])
			}
			x, _, _ := lay.Caret(idx)
			return r.indent + p.x + x
		}
	}
	return r.indent
}

// dispAt returns the displayed byte in row r nearest to x, from the row's
// start.
func (v *view) dispAt(l *lineLayout, r *rowLayout, x float32) int {
	x -= r.indent
	d := r.from
	for i := range r.pieces {
		p := &r.pieces[i]
		if x < p.x+p.width || i == len(r.pieces)-1 {
			lay := v.piece(l, p)
			idx := lay.IndexAt(x-p.x, 0)
			if p.ascii {
				d = min(p.from+idx, p.to)
			} else {
				d = p.from + byteOfRune(l.disp[p.from:p.to], idx)
			}
			break
		}
	}
	// The end of a row a line wraps after is the start of the next, where
	// a caret shows: on this row, it is before the row's last character,
	// as Zed clips there.
	if d >= r.to && r.to < len(l.disp) && r.to > r.from {
		_, size := utf8.DecodeLastRuneInString(l.disp[r.from:r.to])
		d = r.to - size
	}
	return d
}

// byteOfRune returns the byte of s rune i starts at.
func byteOfRune(s string, i int) int {
	for at := range s {
		if i == 0 {
			return at
		}
		i--
	}
	return len(s)
}

// runColors colors a row's glyphs: the color of each run of the
// displayed text.
type runColors struct {
	ends   []int // displayed bytes each run ends at
	colors []ui.Color
}

// colorsOf makes the colors of l's tokens, by displayed bytes, with
// inlays muted.
func (v *view) colorsOf(l *lineLayout, out *runColors) {
	out.ends, out.colors = out.ends[:0], out.colors[:0]
	for _, t := range l.toks {
		out.ends = append(out.ends, l.toDisp(t.End, false))
		out.colors = append(out.colors, v.theme.TokenColor(t.Kind))
	}
	for i, sg := range l.segs {
		if sg.kind == segInlay {
			out.overlay(sg.disp, sg.disp+segLen(l, i), v.theme.LineNumber)
		}
	}
}

// overlay colors displayed bytes a to b c.
func (cc *runColors) overlay(a, b int, c ui.Color) {
	if a >= b {
		return
	}
	var ends []int
	var colors []ui.Color
	add := func(e int, c ui.Color) {
		if n := len(ends); n > 0 && ends[n-1] >= e {
			return
		}
		ends, colors = append(ends, e), append(colors, c)
	}
	prev := 0
	for i, e := range cc.ends {
		col := cc.colors[i]
		if e <= a || prev >= b {
			add(e, col)
		} else {
			if prev < a {
				add(a, col)
			}
			add(min(e, b), c)
			if e > b {
				add(e, col)
			}
		}
		prev = e
	}
	if prev < b {
		if prev < a {
			add(a, ui.Color{})
		}
		add(b, c)
	}
	cc.ends, cc.colors = ends, colors
}

func (cc *runColors) at(d int, def ui.Color) ui.Color {
	i := sort.SearchInts(cc.ends, d+1)
	if i < len(cc.colors) && cc.colors[i].A > 0 {
		return cc.colors[i]
	}
	return def
}

// paintRow paints the glyphs of row r of l, its start at x and its top at
// y, colored by cc, or by override when given.
func (v *view) paintRow(p *ui.Painter, l *lineLayout, r *rowLayout, x, y float32, cc *runColors, override func(d int, c ui.Color) ui.Color) {
	clip := p.Visible()
	baseline := y + (v.lineHeight-(v.ascent+v.descent))/2 + v.ascent
	def := v.theme.Text
	for pi := range r.pieces {
		pc := &r.pieces[pi]
		px := x + r.indent + pc.x
		if px+pc.width < clip.X-64 || px > clip.X+clip.W+64 {
			continue
		}
		lay := v.piece(l, pc)
		var bytesOf []int
		if !pc.ascii {
			bytesOf = runeBytes(l.disp[pc.from:pc.to])
		}
		p.TextLine(lay, px, baseline, func(cluster int) ui.Color {
			d := pc.from + cluster
			if bytesOf != nil && cluster < len(bytesOf) {
				d = pc.from + bytesOf[cluster]
			}
			c := cc.at(d, def)
			if override != nil {
				c = override(d, c)
			}
			return c
		})
	}
}

// runeBytes returns the byte each rune of s starts at.
func runeBytes(s string) []int {
	out := make([]int, 0, len(s))
	for i := range s {
		out = append(out, i)
	}
	return out
}

// rangeRects returns the rectangles, relative to row r's start, covering
// displayed bytes a to b of the row.
func (v *view) rangeRects(l *lineLayout, r *rowLayout, a, b int, out []ui.Rect) []ui.Rect {
	a, b = max(a, r.from), min(b, r.to)
	if a > b {
		return out
	}
	if a == b {
		return out
	}
	for pi := range r.pieces {
		pc := &r.pieces[pi]
		pa, pb := max(a, pc.from), min(b, pc.to)
		if pa >= pb {
			continue
		}
		lay := v.piece(l, pc)
		ra, rb := pa-pc.from, pb-pc.from
		if !pc.ascii {
			ra = utf8.RuneCountInString(l.disp[pc.from:pa])
			rb = ra + utf8.RuneCountInString(l.disp[pa:pb])
		}
		for _, rc := range lay.SelectionVisual(ra, rb, false) {
			out = append(out, ui.Rect{X: r.indent + pc.x + rc.X, Y: 0, W: rc.W, H: v.lineHeight})
		}
	}
	return out
}

// mergeRects joins rectangles that touch, which selections of runs
// return in pieces.
func mergeRects(rs []ui.Rect) []ui.Rect {
	if len(rs) < 2 {
		return rs
	}
	slices.SortFunc(rs, func(a, b ui.Rect) int {
		if a.X < b.X {
			return -1
		}
		if a.X > b.X {
			return 1
		}
		return 0
	})
	out := rs[:1]
	for _, r := range rs[1:] {
		last := &out[len(out)-1]
		if r.X <= last.X+last.W+0.5 {
			last.W = max(last.W, r.X+r.W-last.X)
			continue
		}
		out = append(out, r)
	}
	return out
}

// styleOf returns the text system's style of a font.
func styleOf(f ui.Font) text.Style {
	return text.Style{Family: f.Family, Size: f.Size, Weight: f.Weight, Italic: f.Italic, Features: f.Features}
}

func isWordRune(r rune) bool { return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r) }
