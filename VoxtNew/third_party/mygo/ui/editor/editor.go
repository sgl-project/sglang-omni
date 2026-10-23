package editor

import (
	"math"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/egoist/mygo/internal/l10n"
	"github.com/egoist/mygo/ui"
)

// The sides of a diff viewer, of options.diffSide.
const (
	diffNewSide = 1 + iota
	diffOldSide
)

// options are a code editor's options for a frame, from a
// State or a DiffState.
type options struct {
	// diffSide is the side of a diff viewer the editor shows, which
	// names it.
	diffSide      int8
	readOnly      bool
	font          ui.Font
	lineHeight    float32
	tabSize       int
	indent        IndentStyle
	wrap          bool
	wrapColumn    int
	lineNumbers   LineNumbers
	whitespace    Whitespace
	wrapGuides    []int
	indentGuides  bool
	scrollPastEnd bool
	autoClose     bool
	folding       bool
	activeLine    bool
	cursorShape   CursorShape
	fitContent    bool
	placeholder   string
	label         string
	theme         *Theme
	bindings      []Binding
	gutterMarks   []GutterMark
	backgrounds   []LineBackground
	diffBase      *Buffer
	onCompletion  func(CompletionRequest)
	onHover       func(HoverRequest)
	onDefinition  func(int)
	onGutterClick func(int)
	contextMenu   func(m *ui.Menu)
	onInput       func(ui.InputEvent) bool
	noInputMethod bool
	// The parts' render functions, nil for the editor's own.
	renderFindBar     func(*ui.Context, ui.Element, *FindBar)
	renderGoToLine    func(*ui.Context, ui.Element, *GoToLineBar)
	renderCompletions func(*ui.Context, ui.Element, *CompletionMenu)
	renderHover       func(*ui.Context, ui.Element, *HoverCard)
	// decor shows a diff in the editor.
	decor decor
	// noVScroll leaves vertical scrolling to the owner, as the sides of a
	// split diff share theirs; noScrollbar draws no vertical scroll bar.
	noScrollbar bool
	// minDigits is how many digits the gutter has room for at least.
	minDigits int
	// sharedScroll scrolls the sides of a split diff together.
	sharedScroll *sharedScroll
	// signs leaves room for a diff's + and - before the text.
	signs bool
}

// sharedScroll is the vertical scroll several editors share.
type sharedScroll struct {
	y   float64
	eds []*editor
}

func (s *sharedScroll) setX(x float32) {
	for _, ed := range s.eds {
		ed.scrollX = max(0, min(x, ed.maxScrollX()))
	}
}

func (s *sharedScroll) set(y float64) {
	m := 0.0
	for _, ed := range s.eds {
		m = max(m, ed.scrollMax())
	}
	s.y = max(0, min(y, m))
	for _, ed := range s.eds {
		ed.scrollY = s.y
	}
}

// decor shows a diff in an editor: the colors of lines and their
// gutters, the words changed, the numbers of lines, and blocks.
type decor interface {
	lineStyle(line int) lineLook
	words(line int) []diffRange
	number(line int) int
	paintBlock(p *ui.Painter, ed *editor, b *block, r ui.Rect)
	blockInput(ed *editor, b *block, ev ui.InputEvent, r ui.Rect) bool
}

type lineLook struct {
	bg, gutter, number ui.Color
	bar                ui.Color
	stripes            bool   // the bar is striped, as deletions' are
	sign               string // "+" or "-" in classic indicators
}

// diffRange is a range of a line's bytes a diff highlights.
type diffRange struct{ start, end int }

// fold hides the lines after start up to end.
type fold struct{ start, end int }

// editor is a code editor: a view of a buffer with selections, the input
// that edits it, and its overlays.
type editor struct {
	view  view
	o     options
	theme *Theme
	// texts are MyGo's strings in the window's language, as of the last
	// frame.
	texts l10n.Messages

	sels   []sel
	nextID int
	// gesture is the press dragging a selection.
	gesture gesture
	// addAbove is the stack of selections Add Selection Above and Below
	// made, as Zed's, to take them back going the other way.
	addStack []int
	addDir   int
	// selUndo and selRedo are the selections Undo Selection takes back.
	selUndo, selRedo [][]sel
	folds            []fold
	marked           []ui.TextInputRange // the composition, in bytes per selection
	markedActive     bool
	autoclosed       []autoclose

	scrollY    float64
	scrollX    float32
	anchorLine int
	anchorOff  float64
	reveal     reveal
	r          ui.Rect // the element's box of the last frame, in the window
	gutterW    float32
	textX      float32
	viewH      float32
	shown      [2]int // the first and last lines the last frame showed
	focused    bool   // the editor and its window have the keyboard
	blinkStart time.Time
	services   ui.Services
	chord      *Binding
	pointer    ui.Point
	pointerIn  bool
	hoverLine  int
	// region is the part of the editor under the pointer, which its
	// cursor and the scroll bars' highlight show.
	region     region
	scrollbar  scrollbarDrag
	frameNow   time.Time
	needsBuild bool
	// stale tells that an edit found the buffer edited since the editor
	// followed it, for its command to run again (retry).
	stale bool

	refocus        bool
	wholeWord      bool
	larged         []sel
	toggledWrap    bool
	toggledNumbers bool
	suggestion     string

	find     finder
	goTo     goToLine
	complete completions
	hover    hover
	diags    []Diagnostic
	gitDiff  *gitDiff
	brackets bracketCache
}

// lastClip is what an editor last copied, which pasting its text in any
// editor takes the pieces and lines of, as Zed keeps them with the
// clipboard's text. Editors build on the main thread.
var lastClip clipboard

type clipboard struct {
	text   string
	pieces []string
	lines  []bool // pieces copied as whole lines
}

type autoclose struct {
	sel   int // the selection's id
	at    int // the offset of the closing text
	close string
}

// reveal asks the next frame to scroll the newest caret into view.
type reveal uint8

const (
	revealNone   reveal = iota
	revealFit           // as little as shows it, with a margin
	revealCenter        // centered
	revealTop
)

type gesture struct {
	active   bool
	mode     int // 0 characters, 1 words, 2 lines, 3 columns
	origin   sel
	add      bool // added to the selections, which keep the others
	others   []sel
	colStart ui.Point
}

// init sets up the editor for a frame.
func (ed *editor) init(buf *Buffer, o options, t *ui.Theme) {
	ed.o = o
	ed.theme = o.theme.resolve(t)
	ed.view.theme = ed.theme
	ed.view.readOnly = o.readOnly
	if ed.view.buf != nil && ed.view.buf != buf {
		ed.resetForBuffer()
	}
	ed.view.setBuffer(buf)
	if ed.sels == nil {
		ed.sels = []sel{{goal: -1, id: 0}}
		ed.nextID = 1
		ed.blinkStart = ed.frameNow
		ed.hover.asked = -1
		ed.hoverLine = -1
	}
	if ed.view.lineHeight == 0 {
		ed.view.setStyle(o.font, o.lineHeight, 1, ed.tabSize())
	}
}

func (ed *editor) buf() *Buffer { return ed.view.buf }

// tabSize returns the columns of a tab.
func (ed *editor) tabSize() int {
	if ed.o.tabSize > 0 {
		return ed.o.tabSize
	}
	if l := ed.language(); l != nil && l.TabSize > 0 {
		return l.TabSize
	}
	return 4
}

func (ed *editor) hardTabs() bool {
	switch ed.o.indent {
	case IndentTabs:
		return true
	case IndentSpaces:
		return false
	}
	l := ed.language()
	return l != nil && l.HardTabs
}

func (ed *editor) language() *Language {
	if b := ed.buf(); b != nil {
		return b.Language()
	}
	return nil
}

// wordChars are the language's extra characters of words.
func (ed *editor) wordChars() string {
	if l := ed.language(); l != nil {
		return l.WordChars
	}
	return ""
}

// sync follows the buffer's edits: selections, folds, diagnostics and
// the overlays move with the text.
func (ed *editor) sync() {
	f := &ed.find
	searched := !f.stale && f.version == ed.view.version
	ed.view.sync(func(cs []change) {
		for i := range ed.sels {
			ed.sels[i] = mapSel(ed.sels[i], cs)
		}
		for _, h := range [][][]sel{ed.selUndo, ed.selRedo} {
			for _, sels := range h {
				for i := range sels {
					sels[i] = mapSel(sels[i], cs)
				}
			}
		}
		for i := range ed.find.scope {
			ed.find.scope[i] = mapSel(ed.find.scope[i], cs)
		}
		for i := range ed.folds {
			f := &ed.folds[i]
			f.start, f.end = mapLine(f.start, cs), mapLine(f.end, cs)
		}
		ed.folds = slices.DeleteFunc(ed.folds, func(f fold) bool { return f.end <= f.start })
		for i := range ed.autoclosed {
			ed.autoclosed[i].at = mapOffset(ed.autoclosed[i].at, cs, biasRight)
		}
		for i := range ed.diags {
			d := &ed.diags[i]
			d.Start, d.End = mapOffset(d.Start, cs, biasRight), mapOffset(d.End, cs, biasLeft)
			d.End = max(d.End, d.Start)
		}
		for i := range ed.marked {
			m := &ed.marked[i]
			m.Start, m.End = mapOffset(m.Start, cs, biasLeft), mapOffset(m.End, cs, biasRight)
		}
		// The view stays on the line it shows.
		ed.anchorLine = mapLine(ed.anchorLine, cs)
		ed.complete.moved(cs)
		ed.hover.close()
		if searched {
			f.moved(cs)
		} else {
			f.stale = true
		}
	})
	if searched && !f.stale {
		f.version = ed.view.version
	}
	ed.sels = mergeSels(ed.sels)
}

// mapSel moves a selection through changes: a caret with the text
// inserted at it, and a selection keeping what it holds, the text
// inserted at its edges left out, as Zed's anchors.
func mapSel(s sel, cs []change) sel {
	if s.empty() {
		s.anchor = mapOffset(s.anchor, cs, biasRight)
		s.head = s.anchor
		return s
	}
	start, end := mapOffset(s.start(), cs, biasRight), mapOffset(s.end(), cs, biasLeft)
	end = max(start, end)
	if s.head < s.anchor {
		s.head, s.anchor = start, end
	} else {
		s.anchor, s.head = start, end
	}
	return s
}

// resetForBuffer forgets what the editor knew of the buffer it showed,
// as it shows another: its carets, folds, marks and what the app gave
// it of the old text.
func (ed *editor) resetForBuffer() {
	ed.sels = []sel{{goal: -1, id: ed.nextID}}
	ed.nextID++
	ed.selUndo, ed.selRedo = nil, nil
	ed.folds, ed.autoclosed, ed.marked, ed.diags = nil, nil, nil, nil
	ed.view.inlays, ed.suggestion = nil, ""
	ed.complete.close()
	ed.hover.close()
	ed.find.scope, ed.find.stale, ed.find.lowerVer, ed.find.lower = nil, true, 0, ""
	ed.goTo.open = false
	ed.gesture.active = false
	ed.brackets = bracketCache{}
	ed.scrollY, ed.scrollX, ed.anchorLine, ed.anchorOff = 0, 0, 0, 0
}

// setSels sets the selections, noting the old ones for Undo Selection.
func (ed *editor) setSels(sels []sel, reveal reveal) {
	if !slices.Equal(sels, ed.sels) {
		ed.dropSuggestion()
		ed.selUndo = append(ed.selUndo, slices.Clone(ed.sels))
		if len(ed.selUndo) > 1024 {
			ed.selUndo = ed.selUndo[1:]
		}
		ed.selRedo = nil
	}
	ed.sels = mergeSels(sels)
	if reveal != revealNone {
		ed.reveal = max(ed.reveal, reveal)
	}
	ed.blinkStart = ed.frameNow
	ed.autoclosed = ed.autoclosed[:0]
	// What select next, add cursor and select larger go on from.
	ed.wholeWord = false
	ed.addStack = ed.addStack[:0]
	ed.larged = ed.larged[:0]
}

// primary returns the newest selection.
func (ed *editor) primary() sel { return ed.sels[newest(ed.sels)] }

func (ed *editor) newSel(anchor, head int) sel {
	ed.nextID++
	return sel{anchor: anchor, head: head, goal: -1, id: ed.nextID}
}

// hiddenSpans returns the lines folds hide, merged.
func (ed *editor) hiddenSpans() []lineSpan {
	var spans []lineSpan
	for _, f := range ed.folds {
		if f.end > f.start {
			spans = append(spans, lineSpan{f.start + 1, f.end})
		}
	}
	slices.SortFunc(spans, func(a, b lineSpan) int { return a.from - b.from })
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

// rowTop returns the top of line's rows in the content.
func (ed *editor) rowTop(line int) float64 { return ed.view.heights.geometry(line).rows }

// caretRect returns the rectangle of the caret at off, in the content,
// and whether its line shows.
func (ed *editor) caretRect(off int) (ui.Rect, bool) {
	v := &ed.view
	line := v.text.LineOf(off)
	if v.heights.isHidden(line) {
		return ui.Rect{}, false
	}
	l := v.line(line)
	d := l.toDisp(off-l.start, true)
	ri := l.rowOf(d, false)
	r := &l.rows[ri]
	x := v.xOf(l, r, d)
	y := ed.rowTop(line) + float64(ri)*float64(v.lineHeight)
	return ui.Rect{X: x, Y: float32(y), W: 2, H: v.lineHeight}, true
}

// offsetAt returns the offset nearest to x, y in the content.
func (ed *editor) offsetAt(x float32, y float64) int {
	v := &ed.view
	if v.text.LineCount() == 0 {
		return 0
	}
	line := v.heights.lineAt(y)
	g := v.heights.geometry(line)
	if g.hidden {
		return v.text.LineEnd(line)
	}
	l := v.line(line)
	ri := 0
	switch {
	case y < g.rows:
		ri = 0
		if line > 0 && y < g.top {
			ri = 0
		}
	case y >= g.below:
		ri = len(l.rows) - 1
	default:
		ri = min(int((y-g.rows)/float64(v.lineHeight)), len(l.rows)-1)
	}
	d := v.dispAt(l, &l.rows[ri], x)
	return l.start + l.fromDisp(d)
}

// verticalMove returns where a caret at off with goal x goes moving rows
// down (positive) or up, and the goal it keeps.
func (ed *editor) verticalMove(off int, goal float32, rows int) (int, float32) {
	v := &ed.view
	line := v.text.LineOf(off)
	l := v.line(line)
	d := l.toDisp(off-l.start, true)
	ri := l.rowOf(d, false)
	if goal < 0 {
		goal = v.xOf(l, &l.rows[ri], d)
	}
	n := v.text.LineCount()
	for rows != 0 {
		if rows > 0 {
			if ri+1 < len(l.rows) {
				ri++
			} else {
				next := line + 1
				for next < n && v.heights.isHidden(next) {
					next++
				}
				if next >= n {
					return v.text.Len(), goal
				}
				line, l, ri = next, v.line(next), 0
			}
			rows--
		} else {
			if ri > 0 {
				ri--
			} else {
				prev := line - 1
				for prev >= 0 && v.heights.isHidden(prev) {
					prev--
				}
				if prev < 0 {
					return 0, goal
				}
				line, l = prev, v.line(prev)
				ri = len(l.rows) - 1
			}
			rows++
		}
	}
	dd := v.dispAt(l, &l.rows[ri], goal)
	return l.start + l.fromDisp(dd), goal
}

// scrollMax returns how far the content scrolls down.
func (ed *editor) scrollMax() float64 {
	total := ed.view.heights.total()
	if ed.o.scrollPastEnd {
		return max(0, total-float64(ed.view.lineHeight))
	}
	return max(0, total-float64(ed.viewH))
}

// setScroll scrolls to y, keeping the place by its line.
func (ed *editor) setScroll(y float64) {
	y = max(0, min(y, ed.scrollMax()))
	ed.scrollY = y
	ed.anchorLine = ed.view.heights.lineAt(y)
	ed.anchorOff = y - ed.view.heights.top(ed.anchorLine)
}

// restoreScroll puts the scroll back where its line is, as the heights
// above it change.
func (ed *editor) restoreScroll() {
	h := &ed.view.heights
	line := min(ed.anchorLine, max(h.n-1, 0))
	ed.scrollY = max(0, min(h.top(line)+ed.anchorOff, ed.scrollMax()))
}

// doReveal scrolls the newest caret into view.
func (ed *editor) doReveal() {
	if ed.reveal == revealNone || ed.viewH <= 0 {
		return
	}
	mode := ed.reveal
	ed.reveal = revealNone
	s := ed.primary()
	v := &ed.view
	line := v.text.LineOf(s.head)
	// Unfold the line the caret is on.
	if v.heights.isHidden(line) {
		ed.unfoldLine(line)
		v.heights.setHidden(ed.hiddenSpans())
	}
	cr, ok := ed.caretRect(s.head)
	if !ok {
		return
	}
	lh := float64(v.lineHeight)
	top, bottom := float64(cr.Y), float64(cr.Y)+lh
	h := float64(ed.viewH)
	margin := min(3*lh, max(0, (h-lh)/2))
	switch mode {
	case revealCenter:
		if top < ed.scrollY || bottom > ed.scrollY+h {
			ed.setScroll(top - (h-lh)/2)
		}
	case revealTop:
		ed.setScroll(top - margin)
	default:
		if top-margin < ed.scrollY {
			ed.setScroll(top - margin)
		} else if bottom+margin > ed.scrollY+h {
			ed.setScroll(bottom + margin - h)
		}
	}
	// Sideways.
	w := ed.r.W - ed.textX - 8
	if !v.wrap && w > 0 {
		x := cr.X
		em := v.charWidth * 2
		if x-em < ed.scrollX {
			ed.scrollX = max(0, x-em)
		} else if x+em > ed.scrollX+w {
			ed.scrollX = x + em - w
		}
	}
}

func (ed *editor) unfoldLine(line int) {
	ed.folds = slices.DeleteFunc(ed.folds, func(f fold) bool { return line > f.start && line <= f.end })
}

// gutterWidth returns the width of the gutter: the numbers, the fold
// markers and the bars of changes.
// numbersOn reports whether the gutter numbers the lines, as the options
// say, or the opposite once Toggle Line Numbers ran.
func (ed *editor) numbersOn() bool {
	return (ed.o.lineNumbers != LineNumbersOff) != ed.toggledNumbers
}

func (ed *editor) gutterWidth() float32 {
	v := &ed.view
	w := float32(0)
	if ed.numbersOn() {
		digits := max(ed.o.minDigits, len(strconv.Itoa(v.text.LineCount())))
		w += float32(digits)*v.charWidth + v.charWidth*1.5
	}
	if ed.o.folding && ed.o.decor == nil {
		w += v.lineHeight * 0.8
	}
	if ed.o.diffBase != nil {
		w += 6
	}
	if len(ed.o.gutterMarks) > 0 && !ed.numbersOn() {
		w += v.lineHeight * 0.7
	}
	if w > 0 {
		w += 4
	}
	if ed.o.decor != nil {
		// Room for the change bars at the gutter's left edge.
		w += 6
	}
	return float32(math.Ceil(float64(w)))
}

// visibleLines returns the lines that show in the frame's view, through
// a clip in the content's coordinates.
func (ed *editor) visibleLines(y0, y1 float64) (int, int) {
	h := &ed.view.heights
	first := h.lineAt(y0)
	last := h.lineAt(y1)
	return first, min(last, h.n-1)
}

// slice returns the bytes from a to b.
func (ed *editor) slice(a, b int) string { return ed.view.text.Slice(a, b) }

// selectedText returns the text of the selections, joined by newlines.
func (ed *editor) selectedText() string {
	var parts []string
	for _, s := range ed.sels {
		if !s.empty() {
			parts = append(parts, ed.slice(s.start(), s.end()))
		}
	}
	return strings.Join(parts, "\n")
}
