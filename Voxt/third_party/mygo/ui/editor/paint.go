package editor

import (
	"math"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
)

// blinkInterval is how long the caret shows and hides, as Zed's.
const blinkInterval = 500 * time.Millisecond

// frame is what painting a frame of an editor works out once: the
// lines in view and what highlights them.
type frame struct {
	first, last int
	matches     []diffRange
	current     diffRange
	occurrence  string
	brackets    [2]Range
	bracketsOK  bool
	caretsOn    bool
	activeLines map[int]bool
	activeGuide struct{ level, from, to int }
	colors      runColors
	rects       []ui.Rect
}

// layout sets the view's geometry for a frame of size w by h, before
// painting, from the last frame's scale.
func (ed *editor) layout(w, h, scale float32) {
	v := &ed.view
	v.setStyle(ed.o.font, ed.o.lineHeight, scale, ed.tabSize())
	ed.gutterW = ed.gutterWidth()
	pad := float32(4)
	if ed.gutterW == 0 {
		pad = 8
	}
	ed.textX = ed.gutterW + pad
	if ed.o.signs {
		ed.textX += 2 * v.charWidth
	}
	ed.viewH = h
	wrap := ed.o.wrap != ed.toggledWrap
	ww := w - ed.textX - scrollbarWidth - 4
	if ed.o.wrapColumn > 0 {
		ww = min(ww, float32(ed.o.wrapColumn)*v.charWidth)
	}
	v.setWrap(wrap, max(ww, v.charWidth*8))
	if wrap {
		// Wrapped lines fit: nothing is sideways.
		ed.scrollX = 0
	}
	v.heights.setLines(v.text.LineCount())
	hidden := ed.hiddenSpans()
	var blocks []block
	if ed.o.decor != nil {
		if ds, ok := ed.o.decor.(*diffSide); ok {
			hidden = mergeSpans(append(hidden, ds.hidden()...))
			blocks = ds.blocks()
		}
	}
	v.heights.setHidden(hidden)
	v.heights.setBlocks(blocks)
}

func mergeSpans(spans []lineSpan) []lineSpan {
	sort.Slice(spans, func(i, j int) bool { return spans[i].from < spans[j].from })
	out := spans[:0]
	for _, s := range spans {
		if n := len(out); n > 0 && s.from <= out[n-1].to+1 {
			out[n-1].to = max(out[n-1].to, s.to)
			continue
		}
		out = append(out, s)
	}
	return out
}

// paint draws the editor in r.
func (ed *editor) paint(p *ui.Painter, r ui.Rect) {
	v := &ed.view
	ed.frameNow = p.Now()
	ed.r = r
	ed.layout(r.W, r.H, p.Scale())
	more := false
	if !v.measureWraps(ed.anchorLine, 4*time.Millisecond) {
		more = true
	}
	if ed.o.sharedScroll == nil {
		ed.restoreScroll()
		ed.doReveal()
	}
	if ed.autoscroll() {
		p.AnimationFrame()
	}
	ct := ed.theme
	// The lines in view: those the window shows of the element, for an
	// editor as high as its content.
	clip := p.Visible()
	y0 := ed.scrollY + float64(max(0, clip.Y-r.Y))
	y1 := ed.scrollY + float64(min(r.H, clip.Y+clip.H-r.Y))
	var f frame
	f.first, f.last = ed.visibleLines(y0, y1)
	ed.shown = [2]int{f.first, f.last}
	if !v.advanceSyntax(f.last+1, 4*time.Millisecond) {
		more = true
	}
	v.approx = false
	if ct.Background.A > 0 {
		p.Fill(r, ct.Background, 0)
	}
	ed.prepare(&f)
	if ed.focused {
		phase := ed.frameNow.Sub(ed.blinkStart)
		f.caretsOn = (phase/blinkInterval)%2 == 0 || ed.gesture.active
		if !ed.gesture.active {
			p.After(blinkInterval - phase%blinkInterval)
		}
	} else if ed.o.cursorShape == CursorBlock {
		// Outlined, not blinking, as Vim's and Zed's blocks without the
		// keyboard.
		f.caretsOn = true
	}
	ed.paintBlocks(p, r, &f)
	text := ui.Rect{X: r.X + ed.gutterW, Y: r.Y, W: r.W - ed.gutterW, H: r.H}
	p.Clip(text, 0, func() {
		ed.paintLines(p, r, &f)
	})
	ed.paintGutter(p, r, &f)
	ed.paintScrollbars(p, r, &f)
	if v.approx || more {
		p.AnimationFrame()
	}
	v.endFrame()
}

// prepare works out what highlights the lines in view.
func (ed *editor) prepare(f *frame) {
	v := &ed.view
	t := &v.text
	a, z := t.LineStart(f.first), t.LineEnd(f.last)
	if ed.find.open {
		ed.find.search(ed)
		m := ed.find.matches
		i := sort.Search(len(m), func(i int) bool { return m[i].end >= a })
		j := sort.Search(len(m), func(i int) bool { return m[i].start > z })
		f.matches = m[i:j]
		if ed.find.current < len(m) {
			f.current = m[ed.find.current]
		}
	}
	p := ed.primary()
	if !p.empty() && p.end()-p.start() <= 200 && t.LineOf(p.start()) == t.LineOf(p.end()) {
		if s := t.Slice(p.start(), p.end()); strings.TrimSpace(s) != "" {
			f.occurrence = s
		}
	}
	if p.empty() && len(ed.sels) == 1 {
		if o, c, ok := ed.enclosingBrackets(p.head); ok {
			f.brackets, f.bracketsOK = [2]Range{o, c}, true
		}
	}
	f.activeLines = map[int]bool{}
	for _, s := range ed.sels {
		f.activeLines[t.LineOf(s.head)] = true
	}
	f.activeGuide.level = -1
	if ed.o.indentGuides && len(ed.sels) == 1 {
		ed.activeGuide(f, t.LineOf(p.head))
	}
}

// activeGuide finds the guide of the block holding line, which shows
// stronger.
func (ed *editor) activeGuide(f *frame, line int) {
	t := &ed.view.text
	tab := ed.tabSize()
	in := ed.lineIndent(line)
	if in <= 0 {
		return
	}
	level := (in - 1) / tab
	from, to := line, line
	for from > 0 && line-from < 2000 {
		if ed.lineIndent(from-1) <= level*tab {
			break
		}
		from--
	}
	for to+1 < t.LineCount() && to-line < 2000 {
		if ed.lineIndent(to+1) <= level*tab {
			break
		}
		to++
	}
	f.activeGuide.level, f.activeGuide.from, f.activeGuide.to = level, from, to
}

// lineIndent returns the columns of a line's indentation, a blank line
// taking the larger of the lines around it.
func (ed *editor) lineIndent(line int) int {
	t := &ed.view.text
	tab := ed.tabSize()
	s := t.Line(line)
	if strings.TrimSpace(s) != "" {
		return indentWidth(s, tab)
	}
	before, after := 0, 0
	for i := line - 1; i >= 0 && line-i < 50; i-- {
		if s := t.Line(i); strings.TrimSpace(s) != "" {
			before = indentWidth(s, tab)
			break
		}
	}
	for i := line + 1; i < t.LineCount() && i-line < 50; i++ {
		if s := t.Line(i); strings.TrimSpace(s) != "" {
			after = indentWidth(s, tab)
			break
		}
	}
	return max(before, after)
}

// paintLines paints the lines in view and their blocks.
func (ed *editor) paintLines(p *ui.Painter, r ui.Rect, f *frame) {
	v := &ed.view
	h := &v.heights
	if ed.view.text.Len() == 0 && ed.o.placeholder != "" {
		y := r.Y + float32(-ed.scrollY)
		l := v.shaped.get(ed.o.placeholder, v.style)
		row := rowLayout{pieces: []rowPiece{{to: len(ed.o.placeholder), l: l, ascii: isASCII(ed.o.placeholder), width: l.Width}}, to: len(ed.o.placeholder)}
		cl := &lineLayout{disp: ed.o.placeholder}
		var cc runColors
		muted := ed.theme.LineNumber
		v.paintRow(p, cl, &row, r.X+ed.textX-ed.scrollX, y, &cc, func(int, ui.Color) ui.Color { return muted })
	}
	for line := f.first; line <= f.last; line++ {
		if g := h.geometry(line); !g.hidden {
			ed.paintLine(p, r, f, line, g)
		}
	}
}

// paintBlocks paints the blocks of the lines in view, across the gutter
// and the text.
func (ed *editor) paintBlocks(p *ui.Painter, r ui.Rect, f *frame) {
	if ed.o.decor == nil || len(ed.view.heights.blocks) == 0 {
		return
	}
	ed.eachBlock(f.first, f.last, func(b *block, top float64) { ed.paintBlock(p, r, b, top) })
}

// eachBlock calls fn with the blocks of lines first to last, and their
// tops in the content.
func (ed *editor) eachBlock(first, last int, fn func(b *block, top float64)) {
	h := &ed.view.heights
	for line := first; line <= last; line++ {
		g := h.geometry(line)
		above, below := h.blocksOf(line)
		top := g.top
		for i := range above {
			fn(&above[i], top)
			top += float64(above[i].height)
		}
		top = g.below
		for i := range below {
			fn(&below[i], top)
			top += float64(below[i].height)
		}
	}
}

// blocksInView calls fn with the blocks of the lines in view, as the
// last frame laid them out, and their tops in the content.
func (ed *editor) blocksInView(fn func(b *block, top float64)) {
	h := &ed.view.heights
	if len(h.blocks) == 0 || h.n == 0 {
		return
	}
	y0, y1 := ed.scrollY, ed.scrollY+float64(ed.viewH)
	if ed.o.fitContent {
		y0, y1 = 0, h.total()
	}
	ed.eachBlock(h.lineAt(y0), min(h.lineAt(y1), h.n-1), fn)
}

func (ed *editor) paintBlock(p *ui.Painter, r ui.Rect, b *block, top float64) {
	if ed.o.decor == nil {
		return
	}
	br := ui.Rect{X: r.X, Y: r.Y + float32(top-ed.scrollY), W: r.W, H: b.height}
	if br.Y > p.Visible().Y+p.Visible().H || br.Y+br.H < p.Visible().Y {
		return
	}
	ed.o.decor.paintBlock(p, ed, b, br)
}

// paintLine paints the rows of a line.
func (ed *editor) paintLine(p *ui.Painter, r ui.Rect, f *frame, line int, g lineGeometry) {
	v := &ed.view
	ct := ed.theme
	l := v.line(line)
	lh := v.lineHeight
	ox := r.X + ed.textX - ed.scrollX
	textRect := ui.Rect{X: r.X + ed.gutterW, W: r.W - ed.gutterW}
	var style lineLook
	if ed.o.decor != nil {
		style = ed.o.decor.lineStyle(line)
	}
	active := f.activeLines[line] && ed.o.activeLine && ed.o.decor == nil
	lineSelected := false
	for _, s := range ed.sels {
		if !s.empty() && s.start() <= l.start+len(l.text) && s.end() >= l.start {
			lineSelected = true
		}
	}
	v.colorsOf(l, &f.colors)
	for ri := range l.rows {
		row := &l.rows[ri]
		y := r.Y + float32(g.rows-ed.scrollY) + float32(ri)*lh
		if y > p.Visible().Y+p.Visible().H || y+lh < p.Visible().Y {
			continue
		}
		// Backgrounds: the line's, the app's, then the caret's line.
		if style.bg.A > 0 {
			p.Fill(ui.Rect{X: textRect.X, Y: y, W: textRect.W, H: lh}, style.bg, 0)
		}
		for _, b := range ed.o.backgrounds {
			if line >= b.First && line <= b.Last && b.Color.A > 0 {
				p.Fill(ui.Rect{X: textRect.X, Y: y, W: textRect.W, H: lh}, b.Color, 0)
			}
		}
		if active && !lineSelected {
			p.Fill(ui.Rect{X: textRect.X, Y: y, W: textRect.W, H: lh}, ct.ActiveLine, 0)
		}
		// Words a diff changed.
		if ed.o.decor != nil {
			for _, w := range ed.o.decor.words(line) {
				ed.fillRange(p, f, l, row, ox, y, l.toDisp(w.start, false), l.toDisp(w.end, false), style.word(ct, ed), 3)
			}
		}
		// Matches of the find bar, and occurrences of the selection.
		for _, m := range f.matches {
			if m.end < l.start || m.start > l.start+len(l.text) {
				continue
			}
			c := ct.Match
			if m == f.current {
				c = ct.CurrentMatch
			}
			a, b := max(m.start-l.start, 0), min(m.end-l.start, len(l.text))
			ed.fillRange(p, f, l, row, ox, y, l.toDisp(a, false), l.toDisp(b, false), c, 2)
		}
		if f.occurrence != "" {
			for at := 0; ; {
				i := strings.Index(l.text[at:], f.occurrence)
				if i < 0 {
					break
				}
				a := at + i
				at = a + len(f.occurrence)
				if isWholeWord(l.text, a, at, "") || !isWordRune(firstRune(f.occurrence)) {
					ed.fillRange(p, f, l, row, ox, y, l.toDisp(a, false), l.toDisp(at, false), ct.Occurrence, 2)
				}
			}
		}
		// The selections.
		selColor := ct.Selection
		if !ed.focused {
			selColor = ct.InactiveSelection
		}
		for _, s := range ed.sels {
			if s.empty() || s.end() < l.start || s.start() > l.start+len(l.text) {
				continue
			}
			a, b := max(s.start()-l.start, 0), min(s.end()-l.start, len(l.text))
			da, db := l.toDisp(a, false), l.toDisp(b, true)
			ed.fillRange(p, f, l, row, ox, y, da, db, selColor, 0)
			// Past the end of the line, the newline.
			if s.end() > l.start+len(l.text) && ri == len(l.rows)-1 {
				x := ox + row.indent + row.width
				p.Fill(ui.Rect{X: x, Y: y, W: v.charWidth * 0.6, H: lh}, selColor, 0)
			}
		}
		// Brackets around the caret.
		if f.bracketsOK {
			for _, b := range f.brackets {
				if b.Start >= l.start && b.Start < l.start+len(l.text) {
					f.rects = v.rangeRects(l, row, l.toDisp(b.Start-l.start, false), l.toDisp(min(b.End, l.start+len(l.text))-l.start, false), f.rects[:0])
					for _, rc := range f.rects {
						p.Stroke(ui.Rect{X: ox + rc.X, Y: y + 1, W: rc.W, H: lh - 2}, ct.MatchingBracket, 2, 1)
					}
				}
			}
		}
		if ri == 0 && ed.o.indentGuides {
			ed.paintGuides(p, f, line, ox, y, len(l.rows))
		}
		if ri == 0 && ed.o.signs && style.sign != "" {
			if ds, ok := ed.o.decor.(*diffSide); ok {
				ds.paintSign(p, ed, style.sign, r.X+ed.textX-v.charWidth*1.5, y, style.number)
			}
		}
		v.paintRow(p, l, row, ox, y, &f.colors, nil)
		ed.paintWhitespace(p, f, l, row, ox, y)
		// Diagnostics under the text.
		for _, d := range ed.diags {
			if d.End < l.start || d.Start > l.start+len(l.text) {
				continue
			}
			a, b := max(d.Start-l.start, 0), min(max(d.End, d.Start+1)-l.start, len(l.text))
			f.rects = v.rangeRects(l, row, l.toDisp(a, false), l.toDisp(max(b, a), false), f.rects[:0])
			if a == b && len(l.text) == 0 {
				f.rects = append(f.rects, ui.Rect{X: row.indent, W: v.charWidth})
			}
			for _, rc := range f.rects {
				squiggle(p, ox+rc.X, y+lh-2.5, rc.W, ct.SeverityColor(d.Severity))
			}
		}
		if ri == len(l.rows)-1 && ed.isFolded(line) {
			x := ox + row.indent + row.width + 4
			w := v.charWidth*3 + 8
			p.Fill(ui.Rect{X: x, Y: y + 3, W: w, H: lh - 6}, ct.Occurrence, 4)
			ph := v.shaped.get("⋯", v.style)
			pc := rowLayout{pieces: []rowPiece{{to: 3, l: ph, width: ph.Width}}, to: 3}
			muted := ct.LineNumber
			var cc runColors
			v.paintRow(p, &lineLayout{disp: "⋯"}, &pc, x+(w-ph.Width)/2, y, &cc, func(int, ui.Color) ui.Color { return muted })
		}
		// The carets.
		if f.caretsOn {
			for _, s := range ed.sels {
				if s.head < l.start || s.head > l.start+len(l.text) {
					continue
				}
				d := l.toDisp(s.head-l.start, true)
				if l.rowOf(d, s.head > s.anchor) != ri {
					continue
				}
				x := ox + v.xOf(l, row, d)
				switch ed.o.cursorShape {
				case CursorBlock:
					r := ui.Rect{X: x, Y: y, W: ed.caretWidth(l, row, s.head, x-ox), H: lh}
					if ed.focused {
						p.Fill(r, ct.Cursor.Alpha(0.4), 0)
					} else {
						p.Stroke(r, ct.Cursor, 0, 1)
					}
				case CursorUnderline:
					p.Fill(ui.Rect{X: x, Y: y + lh - 2, W: ed.caretWidth(l, row, s.head, x-ox), H: 2}, ct.Cursor, 0)
				default:
					p.Fill(ui.Rect{X: x - 1, Y: y, W: 2, H: lh}, ct.Cursor, 0)
				}
			}
		}
	}
}

// caretWidth returns the width of a block or underline caret at offset
// at, x in row: that of the character after it, with its combining marks,
// or of a space at the end of the line or of the row.
func (ed *editor) caretWidth(l *lineLayout, row *rowLayout, at int, x float32) float32 {
	v := &ed.view
	end := min(graphemeRight(v.text, at), l.start+len(l.text))
	if end <= at {
		return v.charWidth
	}
	d := l.toDisp(end-l.start, false)
	if d > row.to {
		return v.charWidth
	}
	if w := v.xOf(l, row, d) - x; w > 0 {
		return w
	}
	return v.charWidth
}

func (s lineLook) word(ct *Theme, ed *editor) ui.Color {
	if s.bar.A == 0 {
		return ct.Occurrence
	}
	return s.bar.Alpha(0.22)
}

// fillRange fills displayed bytes a to b of a row.
func (ed *editor) fillRange(p *ui.Painter, f *frame, l *lineLayout, row *rowLayout, ox, y float32, a, b int, c ui.Color, radius float32) {
	if b <= a {
		return
	}
	f.rects = mergeRects(ed.view.rangeRects(l, row, a, b, f.rects[:0]))
	for _, rc := range f.rects {
		p.Fill(ui.Rect{X: ox + rc.X, Y: y, W: rc.W, H: ed.view.lineHeight}, c, radius)
	}
}

// paintGuides draws the indent guides of a line, down its rows.
func (ed *editor) paintGuides(p *ui.Painter, f *frame, line int, ox, y float32, rows int) {
	v := &ed.view
	ct := ed.theme
	tab := ed.tabSize()
	in := ed.lineIndent(line)
	h := v.lineHeight * float32(rows)
	for level := 0; level*tab < in; level++ {
		x := float32(math.Round(float64(ox + float32(level*tab)*v.charWidth)))
		c := ct.IndentGuide
		if level == f.activeGuide.level && line >= f.activeGuide.from && line <= f.activeGuide.to {
			c = ct.ActiveIndentGuide
		}
		p.Fill(ui.Rect{X: x, Y: y, W: 1, H: h}, c, 0)
	}
	for _, col := range ed.o.wrapGuides {
		x := float32(math.Round(float64(ox + float32(col)*v.charWidth)))
		p.Fill(ui.Rect{X: x, Y: y, W: 1, H: h}, ct.WrapGuide, 0)
	}
}

// paintWhitespace draws dots for spaces and arrows for tabs where the
// editor shows whitespace.
func (ed *editor) paintWhitespace(p *ui.Painter, f *frame, l *lineLayout, row *rowLayout, ox, y float32) {
	mode := ed.o.whitespace
	if mode == WhitespaceNone {
		return
	}
	v := &ed.view
	ct := ed.theme
	s := l.text
	// The row's text alone, and in Selection mode only rows a selection
	// meets.
	a, b := l.fromDisp(row.from), l.fromDisp(row.to)
	if mode == WhitespaceSelection && !slices.ContainsFunc(ed.sels, func(sel sel) bool {
		return !sel.empty() && sel.start() < l.start+b && sel.end() > l.start+a
	}) {
		return
	}
	trailFrom := 0
	if mode == WhitespaceTrailing {
		trailFrom = len(strings.TrimRight(s, " \t"))
		a = max(a, trailFrom)
	}
	for i := a; i < b; i++ {
		c := s[i]
		if c != ' ' && c != '\t' {
			continue
		}
		show := false
		switch mode {
		case WhitespaceAll:
			show = true
		case WhitespaceTrailing:
			show = i >= trailFrom
		case WhitespaceBoundary:
			single := c == ' ' && i > 0 && i+1 < len(s) && s[i-1] != ' ' && s[i-1] != '\t' && s[i+1] != ' ' && s[i+1] != '\t'
			show = !single
		case WhitespaceSelection:
			for _, sel := range ed.sels {
				if l.start+i >= sel.start() && l.start+i < sel.end() {
					show = true
					break
				}
			}
		}
		if !show {
			continue
		}
		d := l.toDisp(i, false)
		if d < row.from || d >= row.to {
			continue
		}
		x0 := ox + v.xOf(l, row, d)
		x1 := ox + v.xOf(l, row, l.toDisp(i+1, false))
		cy := y + v.lineHeight/2
		if c == ' ' {
			dot := float32(2)
			p.Fill(ui.Rect{X: (x0+x1)/2 - dot/2, Y: cy - dot/2, W: dot, H: dot}, ct.Whitespace, dot/2)
			continue
		}
		w := min(x1-x0-4, v.charWidth*2)
		var path ui.Path
		path.MoveTo(x0+2, cy).LineTo(x0+2+w, cy)
		path.MoveTo(x0+2+w-3, cy-3).LineTo(x0+2+w, cy).LineTo(x0+2+w-3, cy+3)
		p.StrokePath(&path, 1, ct.Whitespace)
	}
}

// squiggle draws a wavy line under w DIPs from x, y.
func squiggle(p *ui.Painter, x, y, w float32, c ui.Color) {
	if w <= 0 {
		return
	}
	var path ui.Path
	const wave = 4
	amp := float32(1.25)
	path.MoveTo(x, y)
	for i := float32(0); i < w; i += wave / 2 {
		dy := amp
		if int(i/(wave/2))%2 == 1 {
			dy = -amp
		}
		path.LineTo(min(x+i+wave/2, x+w), y+dy)
	}
	p.StrokePath(&path, 1, c)
}

// paintGutter draws the gutter: the numbers of lines, fold markers, the
// marks of changes and the app's.
func (ed *editor) paintGutter(p *ui.Painter, r ui.Rect, f *frame) {
	if ed.gutterW <= 0 {
		return
	}
	v := &ed.view
	ct := ed.theme
	h := &v.heights
	lh := v.lineHeight
	gutter := ui.Rect{X: r.X, Y: r.Y, W: ed.gutterW, H: r.H}
	numbersOn := ed.numbersOn()
	cur := v.text.LineOf(ed.primary().head)
	p.Clip(gutter, 0, func() {
		numRight := r.X + ed.gutterW - 4 - v.charWidth*0.5
		if ed.o.folding && ed.o.decor == nil {
			numRight -= lh * 0.8
		}
		if ed.gitDiff != nil {
			numRight -= 6
		}
		var cc runColors
		for line := f.first; line <= f.last; line++ {
			g := h.geometry(line)
			if g.hidden {
				continue
			}
			y := r.Y + float32(g.rows-ed.scrollY)
			rowsH := float32(g.below - g.rows)
			var style lineLook
			if ed.o.decor != nil {
				style = ed.o.decor.lineStyle(line)
				if style.gutter.A > 0 {
					p.Fill(ui.Rect{X: r.X, Y: y, W: ed.gutterW, H: rowsH}, style.gutter, 0)
				}
			}
			active := f.activeLines[line]
			if numbersOn {
				n := line + 1
				if ed.o.decor != nil {
					n = ed.o.decor.number(line)
				}
				if ed.o.lineNumbers == LineNumbersRelative && line != cur {
					n = max(line-cur, cur-line)
				}
				if n >= 0 {
					s := strconv.Itoa(n)
					lay := v.shaped.get(s, v.style)
					row := rowLayout{pieces: []rowPiece{{to: len(s), l: lay, ascii: true, width: lay.Width}}, to: len(s)}
					col := ct.LineNumber
					if active && ed.o.decor == nil {
						col = ct.ActiveLineNumber
					}
					if style.number.A > 0 {
						col = style.number
					}
					v.paintRow(p, &lineLayout{disp: s}, &row, numRight-lay.Width, y, &cc, func(int, ui.Color) ui.Color { return col })
				}
			}
			for _, m := range ed.o.gutterMarks {
				if m.Line == line {
					d := lh * 0.45
					p.Fill(ui.Rect{X: r.X + 3, Y: y + (lh-d)/2, W: d, H: d}, m.Color, d/2)
				}
			}
			// Change bars.
			barX := r.X + ed.gutterW - 4 - 3
			if ed.o.folding && ed.o.decor == nil {
				barX -= lh * 0.8
			}
			if ed.gitDiff != nil {
				switch ed.gitDiff.status(line) {
				case 1:
					p.Fill(ui.Rect{X: barX, Y: y, W: 3, H: rowsH}, ct.Added, 0)
				case 2:
					p.Fill(ui.Rect{X: barX, Y: y, W: 3, H: rowsH}, ct.Modified, 0)
				case 3:
					var path ui.Path
					path.MoveTo(barX-1, y-3).LineTo(barX+5, y).LineTo(barX-1, y+3).Close()
					p.FillPath(&path, ct.Deleted)
				}
			}
			if style.bar.A > 0 {
				ed.paintBar(p, ui.Rect{X: r.X, Y: y, W: 4, H: rowsH}, style)
			}
			// Fold markers: always on folded lines, on the others while the
			// pointer is over the gutter.
			if ed.o.folding && ed.o.decor == nil {
				folded := ed.isFolded(line)
				hovered := ed.pointerIn && ed.pointer.X < ed.gutterW
				if folded || hovered && ed.foldable(line) {
					cx, cy := r.X+ed.gutterW-4-lh*0.4, y+lh/2
					d := lh * 0.16
					var path ui.Path
					if folded {
						path.MoveTo(cx-d/2, cy-d).LineTo(cx+d/2, cy).LineTo(cx-d/2, cy+d)
					} else {
						path.MoveTo(cx-d, cy-d/2).LineTo(cx, cy+d/2).LineTo(cx+d, cy-d/2)
					}
					col := ct.LineNumber
					if line == ed.hoverLine {
						col = ct.ActiveLineNumber
					}
					p.StrokePath(&path, 1.5, col)
				}
			}
		}
	})
}

// paintBar draws a diff's change bar: solid for additions, striped for
// deletions, as Pierre's, so the two differ without their colors.
func (ed *editor) paintBar(p *ui.Painter, r ui.Rect, s lineLook) {
	if !s.stripes {
		p.Fill(r, s.bar, 0)
		return
	}
	p.Clip(r, 0, func() {
		for y := r.Y; y < r.Y+r.H; y += 4 {
			p.Fill(ui.Rect{X: r.X, Y: y, W: r.W, H: 2}, s.bar, 0)
		}
	})
}

// paintScrollbars draws the scroll bars, the vertical one with markers
// of the matches, diagnostics, changes and carets.
func (ed *editor) paintScrollbars(p *ui.Painter, r ui.Rect, f *frame) {
	v := &ed.view
	ct := ed.theme
	t := p.Theme()
	vertical, horizontal := ed.scrollbars()
	if vertical {
		x := r.X + r.W - scrollbarWidth
		track := ui.Rect{X: x, Y: r.Y, W: scrollbarWidth, H: r.H}
		hovered := ed.pointerIn && ed.region == regionVScroll
		if hovered || ed.scrollbar.active {
			p.Fill(track, t.Scrollbar.Alpha(0.15), 0)
		}
		span := float32(ed.scrollMax()) + ed.viewH
		mark := func(y float64, h float32, x0, w float32, c ui.Color) {
			yy := r.Y + float32(y/float64(span))*r.H
			p.Fill(ui.Rect{X: x + x0, Y: yy, W: w, H: max(h/span*r.H, 2)}, c, 0)
		}
		h := &v.heights
		// Changes against the base, or of the diff, on the left.
		if d, old := ed.diffModel(); d != nil {
			for i, hk := range d.hunks {
				if i > 2000 {
					break
				}
				from, to := hk.B0, hk.B1
				if old {
					from, to = hk.A0, hk.A1
				}
				c := ct.Modified
				switch {
				case hk.A0 == hk.A1:
					c = ct.Added
				case hk.B0 == hk.B1:
					c = ct.Deleted
				}
				if old {
					c = ct.Deleted
					if hk.A0 == hk.A1 {
						continue
					}
				} else if hk.B0 == hk.B1 && ed.o.decor != nil {
					continue
				}
				top := h.top(from)
				mark(top, float32(h.top(max(to, from+1))-top), 0, 3, c)
			}
		}
		// Matches in the middle.
		if ed.find.open {
			last := float32(-10)
			for i, m := range ed.find.matches {
				if i > 20000 {
					break
				}
				y := h.top(v.text.LineOf(m.start))
				yy := float32(y / float64(span) * float64(r.H))
				if yy-last < 2 {
					continue
				}
				last = yy
				mark(y, v.lineHeight, 3, 4, ct.CurrentMatch)
			}
		}
		// Diagnostics on the right.
		for i, d := range ed.diags {
			if i > 2000 {
				break
			}
			mark(h.top(v.text.LineOf(d.Start)), v.lineHeight, 7, 3, ct.SeverityColor(d.Severity))
		}
		// Carets.
		if len(ed.sels) <= 100 {
			for _, s := range ed.sels {
				mark(h.top(v.text.LineOf(s.head)), 2, 0, scrollbarWidth, ct.Text.Alpha(0.5))
			}
		}
		t0, t1 := ed.thumb(false)
		thumb := ui.Rect{X: x + 2, Y: r.Y + t0, W: scrollbarWidth - 4, H: t1 - t0}
		alpha := float32(0.5)
		if hovered || ed.scrollbar.active {
			alpha = 0.9
		}
		p.Fill(thumb, t.Scrollbar.Alpha(alpha), (scrollbarWidth-4)/2)
	}
	if horizontal {
		t0, t1 := ed.thumb(true)
		y := r.Y + r.H - scrollbarWidth
		alpha := float32(0.4)
		if ed.pointerIn && ed.region == regionHScroll || ed.scrollbar.active {
			alpha = 0.9
		}
		p.Fill(ui.Rect{X: r.X + ed.textX + t0, Y: y + 2, W: t1 - t0, H: scrollbarWidth - 4}, t.Scrollbar.Alpha(alpha), (scrollbarWidth-4)/2)
	}
}
