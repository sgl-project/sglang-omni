package editor

import (
	"strconv"

	"github.com/egoist/mygo/ui"
)

// DiffState is a diff viewer's state, kept by the app across
// frames: the texts it compares, its options, and what it keeps of its
// own, as the lines expanded and its scroll position. Zero options are
// the defaults, those of Pierre's diffs.
type DiffState struct {
	// Old and New are the texts compared; empty buffers are made when
	// nil. New is the one edited when Editable, and can be the buffer of
	// an editor's too: an edit in either shows in both.
	Old, New *Buffer
	// Layout shows the texts side by side, or one over the other.
	Layout DiffLayout
	// Editable lets the new text be edited where it shows, with all an
	// editor does; the diff follows as it is typed.
	Editable bool
	// Inline highlights what changed within changed lines.
	Inline DiffInline
	// Indicators mark changed lines: with bars, + and -, or nothing.
	Indicators DiffIndicators
	// ContextLines is how many unchanged lines show around changes, 3
	// when 0, none when negative; the others collapse into separators
	// that expand them, ExpansionLines (100 when 0) a click, all of them
	// with Shift. ExpandUnchanged shows them all.
	ContextLines    int
	ExpansionLines  int
	ExpandUnchanged bool
	// NoBackground leaves changed lines without their tint, and
	// NoLineNumbers the gutter without numbers.
	NoBackground  bool
	NoLineNumbers bool
	// Wrap wraps long lines rather than scrolling them sideways.
	Wrap bool
	// FitContent makes the viewer as high as its lines, as a file of a
	// list of them, rather than scrolling them itself.
	FitContent bool
	// Font, LineHeight, TabSize and Theme are those of State.
	Font       ui.Font
	LineHeight float32
	TabSize    int
	Theme      *Theme
	// Bindings add key bindings to the editors of the sides.
	Bindings []Binding
	// HunkActions shows Revert and Keep on the change under the pointer,
	// or the caret's, which put the old lines back in New, or make the
	// new lines Old's, as an agent's changes are reviewed.
	HunkActions bool
	// LineSelection lets the numbers of lines be clicked and dragged to
	// select lines, which OnLineSelection reports.
	LineSelection   bool
	OnLineSelection func(DiffLineRange)
	// OnGutterButton shows a + on the gutter of the line under the
	// pointer, as to comment on it, and is called with the line when it
	// is clicked.
	OnGutterButton func(DiffLineRange)
	// Annotations show what RenderAnnotation builds under lines, as
	// comments of a review, as high as each asks; the other side of a
	// split view leaves the room empty.
	Annotations      []DiffAnnotation
	RenderAnnotation func(c *ui.Context, a DiffAnnotation)

	// RenderSeparator, RenderHunkActions and RenderGutterButton build the
	// viewer's separators, the actions on a change and the gutter's
	// button in place of its own (DefaultSeparator, DefaultHunkActions
	// and DefaultGutterButton, which they can call to wrap them): each is
	// given the element the viewer places, to style and fill, and what
	// the part shows and does. A separator is as wide as the viewer, and
	// 32 DIPs high.
	RenderSeparator    func(c *ui.Context, sep ui.Element, s *DiffSeparator)
	RenderHunkActions  func(c *ui.Context, bar ui.Element, a *DiffHunkActions)
	RenderGutterButton func(c *ui.Context, btn ui.Element, line DiffLineRange)
	// RenderFindBar and RenderGoToLine build the bars of the sides'
	// editors, as State's.
	RenderFindBar  func(c *ui.Context, bar ui.Element, f *FindBar)
	RenderGoToLine func(c *ui.Context, bar ui.Element, g *GoToLineBar)

	model   diffModel
	eds     [2]editor // the old side's and the new side's
	sides   [2]diffSide
	oldView view // the old text, for its lines in a unified view
	scroll  sharedScroll
	// revealed are the unchanged lines expanded, in the old text, which
	// edits of the new text leave where they are; revealedVer is the old
	// text's version they are of.
	revealed    []lineSpan
	revealedVer uint64
	revealedOf  *Buffer
	expandV     int
	selected    DiffLineRange
	hasSel      bool
	// oldActive is set while the old side of a split view has the
	// keyboard, and stays after it goes, until the new side has it.
	oldActive bool
	prepared  diffPrepared
	// blocksV counts the blocks prepare made, and built is the geometry
	// the separators and buttons were built for, which a frame changing
	// it builds again (diffGeom).
	blocksV int
	built   diffGeom
	gapList []diffGap
	partial *diffPartial
}

// DiffLayout is how a diff viewer lays out its two texts.
type DiffLayout uint8

const (
	// DiffSplit shows the old text on the left and the new on the right,
	// their lines aligned.
	DiffSplit DiffLayout = iota
	// DiffUnified shows the new text, with the lines deleted from the
	// old above where they were.
	DiffUnified
)

// DiffInline is how a diff viewer highlights what changed within lines.
type DiffInline uint8

const (
	// DiffInlineWordAlt highlights words, joining those one character
	// apart, as "a b c" changed reads as one highlight.
	DiffInlineWordAlt DiffInline = iota
	// DiffInlineWord highlights each word changed.
	DiffInlineWord
	// DiffInlineWordLine highlights from the first word changed to the
	// last.
	DiffInlineWordLine
	// DiffInlineChar highlights characters.
	DiffInlineChar
	// DiffInlineNone highlights nothing within lines.
	DiffInlineNone
)

// DiffIndicators is how a diff viewer marks changed lines.
type DiffIndicators uint8

const (
	// DiffBars draws a bar at the gutter's edge, striped for deletions.
	DiffBars DiffIndicators = iota
	// DiffClassic shows + and - before the lines.
	DiffClassic
	// DiffNoIndicators marks lines by their background alone.
	DiffNoIndicators
)

// DiffSide is a side of a diff: the old text or the new.
type DiffSide uint8

const (
	DiffOld DiffSide = iota
	DiffNew
)

// DiffLineRange is lines Start to End, inclusive, from 0, of a side.
type DiffLineRange struct {
	Side       DiffSide
	Start, End int
}

// DiffAnnotation is room under a line of a side for what
// RenderAnnotation builds there, Height DIPs high. Key identifies it
// among the others.
type DiffAnnotation struct {
	Side   DiffSide
	Line   int
	Height float32
	Key    any
}

// DiffHunk is a change: the old lines OldStart to OldEnd, exclusive,
// replaced by the new lines NewStart to NewEnd.
type DiffHunk struct {
	OldStart, OldEnd, NewStart, NewEnd int
}

func (d *DiffState) context() int {
	switch {
	case d.ContextLines < 0:
		return 0
	case d.ContextLines == 0:
		return 3
	}
	return d.ContextLines
}

func (d *DiffState) expansionLines() int {
	if d.ExpansionLines <= 0 {
		return 100
	}
	return d.ExpansionLines
}

func (d *DiffState) ensure() {
	if d.Old == nil {
		d.Old = NewBuffer("")
	}
	if d.New == nil {
		d.New = NewBuffer("")
	}
	if pt := d.partial; pt != nil && (pt.old != d.Old || pt.new != d.New) {
		// Texts of the app's in place of a patch's.
		d.partial = nil
		d.prepared = diffPrepared{}
	}
	d.model.set(d.Old, d.New)
	d.model.update()
	// The lines revealed move with edits of the old text, as Keep makes,
	// and go with it.
	if d.revealedOf != d.Old {
		d.revealed, d.revealedOf, d.revealedVer = nil, d.Old, d.Old.Version()
		d.expandV++
	}
	if ver := d.Old.Version(); ver != d.revealedVer {
		if changes, _, ok := d.Old.changesSince(d.revealedVer); ok && len(d.revealed) > 0 {
			for _, cs := range changes {
				for i := range d.revealed {
					d.revealed[i] = lineSpan{mapLine(d.revealed[i].from, cs), mapLine(d.revealed[i].to, cs)}
				}
			}
		} else if !ok {
			d.revealed = nil
		}
		d.revealedVer = ver
		d.expandV++
	}
}

// Hunks returns the changes between the texts.
func (d *DiffState) Hunks() []DiffHunk {
	d.ensure()
	out := make([]DiffHunk, len(d.model.hunks))
	for i, h := range d.model.hunks {
		out[i] = DiffHunk{h.A0, h.A1, h.B0, h.B1}
	}
	return out
}

// Stats returns how many lines the changes add and delete.
func (d *DiffState) Stats() (added, deleted int) {
	d.ensure()
	return d.model.stats()
}

// Revert puts the old lines of hunk i back in New.
func (d *DiffState) Revert(i int) {
	d.ensure()
	d.model.revert(i)
}

// Accept makes the new lines of hunk i Old's, so it is no longer a
// change.
func (d *DiffState) Accept(i int) {
	d.ensure()
	d.model.accept(i)
}

// ExpandAll shows every unchanged line; CollapseAll collapses them again
// into separators.
func (d *DiffState) ExpandAll() {
	d.ensure()
	d.ExpandUnchanged = true
}

func (d *DiffState) CollapseAll() {
	d.ExpandUnchanged = false
	d.revealed = nil
	d.expandV++
}

// SelectedLines returns the lines selected by their numbers, and whether
// there are.
func (d *DiffState) SelectedLines() (DiffLineRange, bool) { return d.selected, d.hasSel }

// SetSelectedLines selects lines, as clicking their numbers does; ok
// false clears the selection.
func (d *DiffState) SetSelectedLines(r DiffLineRange, ok bool) {
	if r.End < r.Start {
		r.Start, r.End = r.End, r.Start
	}
	d.selected, d.hasSel = r, ok
}

// GoToHunk puts the caret of the active side (ActiveSide) at the start
// of hunk i and scrolls it to the middle of the view.
func (d *DiffState) GoToHunk(i int) {
	d.ensure()
	if i < 0 || i >= len(d.model.hunks) {
		return
	}
	h := d.model.hunks[i]
	side := d.ActiveSide()
	ed := d.editorOf(side)
	line := h.B0
	if side == DiffOld {
		line = h.A0
	}
	r := &ed.view.text
	off := r.LineStart(min(line, r.LineCount()-1))
	ed.setSels([]sel{ed.newSel(off, off)}, revealCenter)
}

// Run runs a command in the editor of the active side (ActiveSide), as
// its key would, and reports whether it did something: Find opens its
// find bar, as an app's menu item does.
func (d *DiffState) Run(cmd Command) bool {
	ed := d.editorOf(d.ActiveSide())
	return ed.retry(func() bool { return ed.run(cmd) })
}

// Find opens the find bar of the active side's editor (ActiveSide) with
// query, and selects its first match after the caret.
func (d *DiffState) Find(query string) {
	ed := d.editorOf(d.ActiveSide())
	f := &ed.find
	f.open, f.query, f.stale = true, query, true
	f.step(ed, 1)
}

// ActiveSide returns the side that has the keyboard, or had it last: the
// new side before either had it, and in a unified view. Cursor,
// CursorPos, CurrentHunk, GoToHunk, Run and Find act in its editor.
func (d *DiffState) ActiveSide() DiffSide {
	if d.oldActive && d.Layout == DiffSplit {
		return DiffOld
	}
	return DiffNew
}

// Cursor returns the offset of the newest caret of the active side, in
// its text.
func (d *DiffState) Cursor() int {
	return d.editorOf(d.ActiveSide()).primary().head
}

// CursorPos returns the line and column of the newest caret of the
// active side, in its text.
func (d *DiffState) CursorPos() Pos {
	ed := d.editorOf(d.ActiveSide())
	return posOf(ed.view.text, ed.primary().head)
}

// CurrentHunk returns the hunk the caret of the active side is in, with
// inside true, as for Revert; else the last hunk before the caret, as
// "2 of 5" counts changes and Next Change steps from (GoToHunk(i+1)), or
// -1 before the first. On a side the hunk has no lines of, as the new
// side of a deletion, the caret is in it on the line where its lines
// were.
func (d *DiffState) CurrentHunk() (i int, inside bool) {
	side := d.ActiveSide()
	ed := d.editorOf(side)
	line := ed.view.text.LineOf(ed.primary().head)
	if j := d.model.hunkAt(line, side == DiffOld); j >= 0 {
		return j, true
	}
	i = -1
	for j, h := range d.model.hunks {
		start := h.B0
		if side == DiffOld {
			start = h.A0
		}
		if start > line {
			break
		}
		i = j
	}
	return i, false
}

// editorOf returns the editor of a side, made ready to act in before the
// viewer is first built.
func (d *DiffState) editorOf(side DiffSide) *editor {
	d.ensure()
	ed, buf := &d.eds[side], d.New
	if side == DiffOld {
		buf = d.Old
	}
	if ed.view.buf != buf || ed.sels == nil {
		ed.init(buf, d.sideOptions(side == DiffOld), ui.LightTheme())
	}
	ed.sync()
	return ed
}

// sideOptions are the options of the editor of the old side, or the new.
func (d *DiffState) sideOptions(old bool) options {
	// Both gutters as wide, so that both sides wrap alike.
	digits := max(3, len(strconv.Itoa(max(d.Old.LineCount(), d.New.LineCount()))))
	o := options{
		readOnly: old || !d.Editable, font: d.Font, lineHeight: d.LineHeight, tabSize: d.TabSize,
		wrap: d.Wrap, lineNumbers: LineNumbersOn, whitespace: WhitespaceNone, autoClose: true,
		theme: d.Theme, bindings: d.Bindings, fitContent: d.FitContent, minDigits: digits,
		renderFindBar: d.RenderFindBar, renderGoToLine: d.RenderGoToLine,
	}
	if d.NoLineNumbers {
		o.lineNumbers = LineNumbersOff
	}
	o.diffSide = diffNewSide
	if old {
		o.diffSide = diffOldSide
	}
	return o
}

// DiffView builds the diff viewer.
func DiffView(c *ui.Context, d *DiffState) ui.Element {
	d.ensure()
	t := c.Theme()
	ct := d.Theme.resolve(t)
	opts := d.sideOptions
	d.sides[0] = diffSide{d: d, old: true, hide: d.sides[0].hide, blks: d.sides[0].blks}
	d.sides[1] = diffSide{d: d, unified: d.Layout == DiffUnified, hide: d.sides[1].hide, blks: d.sides[1].blks}
	if d.Layout == DiffUnified {
		d.oldView.theme = ct
		d.oldView.setBuffer(d.Old)
		d.oldView.sync(nil)
		o := opts(false)
		o.decor = &d.sides[1]
		if d.Indicators == DiffClassic {
			o.signs = true
		}
		root := ui.Box(c).Direction(ui.LeftToRight).Clip()
		root.Draw(func(p *ui.Painter, r ui.Rect) {
			ed := &d.eds[1]
			ed.layout(r.W, r.H, p.Scale())
			d.oldView.setStyle(ed.view.font, ed.o.lineHeight, p.Scale(), ed.tabSize())
			d.oldView.setWrap(ed.view.wrap, ed.view.wrapWidth)
			d.prepare()
			ed.layout(r.W, r.H, p.Scale())
			d.rebuildIfMoved()
		})
		root.Children(func() {
			ed := d.eds[1].build(c, d.New, o)
			ed.Fill()
			if d.FitContent {
				ed.FillWidth()
			}
			d.buildSeparators(c, ct)
		})
		if d.FitContent {
			root.Height(float32(d.eds[1].view.heights.total()))
		}
		return root
	}
	// Side by side: two editors sharing their scroll.
	d.scroll.eds = []*editor{&d.eds[0], &d.eds[1]}
	// The old text on the left, as code reads left to right.
	row := ui.Row(c).Direction(ui.LeftToRight).Clip()
	row.Draw(func(p *ui.Painter, r ui.Rect) {
		w := (r.W - 1) / 2
		for i := range d.eds {
			d.eds[i].layout(w, r.H, p.Scale())
		}
		d.prepare()
		for i := range d.eds {
			d.eds[i].layout(w, r.H, p.Scale())
		}
		if !d.FitContent {
			d.scroll.set(d.scroll.y)
			// The new side reveals its carets for both.
			ed := &d.eds[1]
			if ed.reveal != revealNone {
				ed.doReveal()
				d.scroll.set(ed.scrollY)
			}
			if ed0 := &d.eds[0]; ed0.reveal != revealNone {
				ed0.doReveal()
				d.scroll.set(ed0.scrollY)
			}
		}
		d.rebuildIfMoved()
	})
	row.Children(func() {
		for i, b := range []*Buffer{d.Old, d.New} {
			o := opts(i == 0)
			o.decor = &d.sides[i]
			o.sharedScroll = &d.scroll
			if d.FitContent {
				o.sharedScroll = nil
			}
			o.noScrollbar = i == 0
			if d.Indicators == DiffClassic {
				o.signs = true
			}
			ed := d.eds[i].build(c, b, o)
			ed.Grow(1).Basis(0).FillHeight()
			if i == 0 {
				ui.Box(c).Width(1).FillHeight().Background(ct.Background.Mix(t.Border, 0.6))
			}
			if d.eds[i].focused {
				d.oldActive = i == 0
			}
		}
		d.buildSeparators(c, ct)
	})
	if d.FitContent {
		row.Height(float32(max(d.eds[0].view.heights.total(), d.eds[1].view.heights.total())))
	}
	return row
}

// expand reveals lines of the gap starting at old line a: at its top,
// below the change before it (dir 1), at its bottom (dir -1), or all.
func (d *DiffState) expand(a int, dir int, all bool) {
	for _, g := range d.gapList {
		if g.a != a || g.hidden == 0 {
			continue
		}
		from, to := g.hideA, g.hideA+g.hidden-1
		n := d.expansionLines()
		switch {
		case all:
		case dir > 0:
			to = min(to, from+n-1)
		default:
			from = max(from, to-n+1)
		}
		d.revealed = mergeSpans(append(d.revealed, lineSpan{from, to}))
		d.expandV++
		return
	}
}

// revealedAt returns the last line of the lines revealed holding old line
// a, or -1.
func (d *DiffState) revealedAt(a int) int {
	for _, sp := range d.revealed {
		if a >= sp.from && a <= sp.to {
			return sp.to
		}
	}
	return -1
}

// revealedBefore returns the first line of the lines revealed holding old
// line a, or -1.
func (d *DiffState) revealedBefore(a int) int {
	for _, sp := range d.revealed {
		if a >= sp.from && a <= sp.to {
			return sp.from
		}
	}
	return -1
}

// selectLine selects a line by its number, or extends the selection to
// it from its nearer end.
func (d *DiffState) selectLine(side DiffSide, line int, extend bool) {
	switch {
	case extend && d.hasSel && d.selected.Side == side:
		if line < d.selected.Start {
			d.selected.Start = line
		} else {
			d.selected.End = line
		}
	case d.hasSel && d.selected.Side == side && d.selected.Start == line && d.selected.End == line:
		d.hasSel = false
	default:
		d.selected, d.hasSel = DiffLineRange{Side: side, Start: line, End: line}, true
	}
	if d.OnLineSelection != nil && d.hasSel {
		d.OnLineSelection(d.selected)
	}
}

// activeHunk returns the hunk under the pointer on either side, or else
// the one holding the new side's newest caret, or -1.
func (d *DiffState) activeHunk() int {
	for i := range d.eds {
		ed := &d.eds[i]
		if !ed.pointerIn || ed.hoverLine < 0 {
			continue
		}
		side := &d.sides[i]
		if i == 1 && side.unified {
			if j := d.model.hunkAtNew(ed.hoverLine); j >= 0 {
				return j
			}
			continue
		}
		if j := side.hunkOf(ed.hoverLine); j >= 0 {
			return j
		}
		if i == 1 {
			if j := d.model.hunkAtNew(ed.hoverLine); j >= 0 {
				return j
			}
		}
	}
	ed := &d.eds[1]
	if ed.focused && len(ed.sels) > 0 {
		line := ed.view.text.LineOf(ed.primary().head)
		return d.model.hunkAtNew(line)
	}
	return -1
}

// isDark reports whether a background is dark.
func isDark(c ui.Color) bool {
	l := 0.2126*float64(c.R) + 0.7152*float64(c.G) + 0.0722*float64(c.B)
	return l < 128
}

// diffGeom is what places the elements a diff view builds over its lines,
// as its separators, which the view builds from the last frame's layout:
// the blocks, the wraps of the lines, the height and the scroll.
type diffGeom struct {
	blocks         int
	wrapsA, wrapsB uint64
	total, scroll  float64
}

func (d *DiffState) geom() diffGeom {
	a := &d.eds[0].view
	if d.Layout == DiffUnified {
		a = &d.oldView
	}
	b := &d.eds[1]
	return diffGeom{d.blocksV, a.heights.wraps.version, b.view.heights.wraps.version, b.view.heights.total(), b.scrollY}
}

// rebuildIfMoved asks for the view to be built again when laying it out
// for painting moved what its elements were built for, as expanding
// lines from a separator's button does: else the separator stays until
// something else builds the view, as the pointer moving.
func (d *DiffState) rebuildIfMoved() {
	if d.geom() != d.built {
		d.eds[1].services.Invalidate()
	}
}
