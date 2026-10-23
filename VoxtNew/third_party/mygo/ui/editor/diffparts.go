package editor

import (
	"fmt"

	"github.com/egoist/mygo/internal/l10n"
	"github.com/egoist/mygo/ui"
)

// The parts of a diff viewer around its lines, as Pierre's diffs render
// them: the separators of unchanged lines and of a patch's hunks, the
// actions on a change, and the button on the gutter. The viewer places
// each, and a render function of the DiffState styles and fills it
// (RenderSeparator and the others), else the viewer's own look
// (DefaultSeparator and the others).

// DiffSeparator is a separator of a diff, as RenderSeparator builds it:
// unchanged lines collapsed between changes, which it expands, or the
// header of a patch's hunk.
type DiffSeparator struct {
	// Lines is how many unchanged lines it collapses, 0 for a header.
	Lines int
	// Step is how many lines ExpandUp and ExpandDown show, the
	// DiffState's ExpansionLines.
	Step int
	// First and Last report that it is before the first change, or after
	// the last, which expand one way only: up toward the change, or down
	// from it.
	First, Last bool
	// Header is the header of a patch's hunk (SetPatch), as
	// "@@ -12,7 +12,8 @@ func main", for a separator of no lines.
	Header string

	d     *DiffState
	a     int // the old line its gap starts at
	theme *Theme
	font  ui.Font
}

// ExpandUp shows Step more of its lines, those right above the change
// below it; ExpandDown those right below the change above it; ExpandAll
// all of them.
func (s *DiffSeparator) ExpandUp()   { s.d.expand(s.a, -1, false) }
func (s *DiffSeparator) ExpandDown() { s.d.expand(s.a, 1, false) }
func (s *DiffSeparator) ExpandAll()  { s.d.expand(s.a, 1, true) }

// Font is the viewer's font, and Theme its colors.
func (s *DiffSeparator) Font() ui.Font { return s.font }
func (s *DiffSeparator) Theme() *Theme { return s.theme }

// DefaultSeparator builds the viewer's own separator into sep, as
// Pierre's: a bar with chevrons expanding the lines up and down, or all
// of them when they are few, and "N unmodified lines", which expands them
// all; or a hunk's header. Shift expands all the lines from any button.
func DefaultSeparator(c *ui.Context, sep ui.Element, s *DiffSeparator) {
	t := c.Theme()
	tx := l10n.In(c.Locale())
	ct := s.Theme()
	sep.Padding(4)
	sep.Children(func() {
		bar := ui.Row(c).Grow(1).AlignSelf(ui.Stretch).AlignItems(ui.Center).Gap(t.Space(0.5)).Padding(0, t.Space(1.5)).
			Radius(6).Background(mixLab(ct.Background, ct.Text, 0.06))
		bar.Children(func() {
			if s.Lines == 0 {
				ui.Text(c, s.Header).Font(s.Font().Family).FontSize(s.Font().Size - 1).TextColor(t.TextMuted).SingleLine().Shrink(1)
				return
			}
			expand := func(fn func()) {
				if c.Modifiers()&ui.Shift != 0 {
					fn = s.ExpandAll
				}
				fn()
			}
			chevron := func(dir int, label string, fn func()) {
				b := ui.ButtonBase(c).Key(label).Size(20, 20).Radius(4).Label(label).Tooltip(label).Cursor(ui.CursorPointer)
				if b.Hovered() {
					b.Background(ct.Text.Alpha(0.08))
				}
				b.Draw(func(p *ui.Painter, r ui.Rect) {
					cx, cy, d := r.X+r.W/2, r.Y+r.H/2, float32(3.5)
					var path ui.Path
					switch dir {
					case 1:
						path.MoveTo(cx-d, cy-d/2).LineTo(cx, cy+d/2).LineTo(cx+d, cy-d/2)
					case -1:
						path.MoveTo(cx-d, cy+d/2).LineTo(cx, cy-d/2).LineTo(cx+d, cy+d/2)
					default:
						path.MoveTo(cx-d, cy-2).LineTo(cx, cy-2-d).LineTo(cx+d, cy-2)
						path.MoveTo(cx-d, cy+2).LineTo(cx, cy+2+d).LineTo(cx+d, cy+2)
					}
					p.StrokePath(&path, 1.5, t.TextMuted)
				})
				if b.Clicked() {
					expand(fn)
				}
			}
			switch {
			case s.First:
				chevron(-1, tx.ExpandUp(), s.ExpandUp)
			case s.Last:
				chevron(1, tx.ExpandDown(), s.ExpandDown)
			case s.Lines <= s.Step:
				chevron(0, tx.ExpandAll(), s.ExpandAll)
			default:
				chevron(1, tx.ExpandDown(), s.ExpandDown)
				chevron(-1, tx.ExpandUp(), s.ExpandUp)
			}
			label := tx.UnmodifiedLines(s.Lines)
			b := ui.ButtonBase(c).Key("all").Padding(t.Space(0.5), t.Space(1)).Radius(4).Label(tx.Expand(label)).Cursor(ui.CursorPointer)
			b.Children(func() { ui.Text(c, label).FontSize(max(t.FontSize-1, 11)).TextColor(t.TextMuted) })
			if b.Clicked() {
				s.ExpandAll()
			}
		})
	})
}

// buildSeparators places the separators in view across the viewer, both
// sides of a split one, over the room the lines leave them, and builds
// them.
func (d *DiffState) buildSeparators(c *ui.Context, ct *Theme) {
	ed := &d.eds[1]
	d.built = d.geom()
	render := d.RenderSeparator
	if render == nil {
		render = DefaultSeparator
	}
	ed.blocksInView(func(b *block, top float64) {
		s := &DiffSeparator{d: d, Step: d.expansionLines(), theme: ct, font: ed.view.font}
		var key string
		switch b.kind {
		case blockSeparator:
			g := d.gapList[b.hunk]
			s.Lines, s.First, s.Last, s.a = b.to-b.from, g.before, g.after, g.a
			key = fmt.Sprint("separator ", g.a)
		case blockHeader:
			s.Header, _ = b.id.(string)
			key = fmt.Sprint("header ", b.hunk)
		default:
			return
		}
		label := s.Header
		if s.Lines > 0 {
			label = l10n.In(c.Locale()).UnmodifiedLines(s.Lines)
		}
		sep := ui.Box(c).Key(key).Absolute().Left(0).Right(scrollbarWidth).Top(float32(top - ed.scrollY)).Height(b.height).
			Role(ui.RoleGroup).Label(label).Cursor(ui.CursorDefault).KeepFocus().HandleInput(shield)
		render(c, sep, s)
	})
}

// DiffHunkActions are the actions on a change of a diff, as
// RenderHunkActions builds them: reverting it, or keeping it.
type DiffHunkActions struct {
	// Index is the change's, in Hunks, and Hunk the change.
	Index int
	Hunk  DiffHunk

	d *DiffState
}

// Revert puts the change's old lines back in New; Keep makes its new
// lines Old's, so that it is no longer a change.
func (a *DiffHunkActions) Revert() { a.d.Revert(a.Index) }
func (a *DiffHunkActions) Keep()   { a.d.Accept(a.Index) }

// DefaultHunkActions builds the viewer's own actions into bar: Revert and
// Keep buttons.
func DefaultHunkActions(c *ui.Context, bar ui.Element, a *DiffHunkActions) {
	t := c.Theme()
	lh := a.d.eds[1].view.lineHeight
	bar.Row().Gap(t.Space(1))
	tx := l10n.In(c.Locale())
	bar.Children(func() {
		for _, act := range []struct {
			key, label string
			fn         func()
		}{{"revert", tx.Revert(), a.Revert}, {"keep", tx.Keep(), a.Keep}} {
			btn := ui.ButtonBase(c).Key(act.key).Padding(0, t.Space(2)).Height(lh-4).Radius(t.Radius-2).
				Background(t.Surface).Border(1, t.Border).Label(tx.HunkAction(act.label))
			btn.Children(func() { ui.Text(c, act.label).FontSize(max(11, t.FontSize-2)) })
			if btn.Clicked() {
				act.fn()
			}
		}
	})
}

// DefaultGutterButton builds the viewer's own gutter button into btn: a
// + on the accent color.
func DefaultGutterButton(c *ui.Context, btn ui.Element, r DiffLineRange) {
	t := c.Theme()
	btn.Radius(4).Background(t.Accent).Label(l10n.In(c.Locale()).Line(r.Start + 1))
	btn.Draw(func(p *ui.Painter, r ui.Rect) {
		cx, cy := r.X+r.W/2, r.Y+r.H/2
		p.Line(cx-4, cy, cx+4, cy, 1.5, t.AccentText)
		p.Line(cx, cy-4, cx, cy+4, 1.5, t.AccentText)
	})
}
