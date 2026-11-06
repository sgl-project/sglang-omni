package ui

import (
	"math"

	"github.com/egoist/mygo/internal/scene"
	"github.com/egoist/mygo/internal/text"
)

// Font is a font for text a Painter draws itself (Shape): a family list
// as Element.Font takes it, a size in DIPs (14 when 0), a weight from 100
// to 900 (400 when 0), italics, and OpenType features as
// Element.FontFeatures takes them, comma separated.
type Font struct {
	Family   string
	Size     float32
	Weight   int
	Italic   bool
	Features string
	// Lang is the language of the text, as for Element.Lang; "" is the
	// language the app shows.
	Lang string
	// Thicken draws the glyphs with a thicker stroke, as Ghostty's
	// font-thicken: on macOS, with Core Text's font smoothing at its
	// strongest, whatever the color and the user's setting. It changes
	// nothing elsewhere.
	Thicken bool
	// Antialiased draws the glyphs without font smoothing, thinner, as
	// browsers draw text styled -webkit-font-smoothing: antialiased on
	// macOS, for drawings that match a web page's. It changes nothing
	// elsewhere.
	Antialiased bool
}

func (f Font) style() text.Style {
	return text.Style{Family: f.Family, Size: f.Size, Weight: f.Weight, Italic: f.Italic, Features: f.Features, Lang: f.Lang}
}

// FontMetrics are the vertical metrics of a font, in DIPs: how far its
// glyphs reach above and below the baseline, and the gap it leaves
// between lines.
type FontMetrics struct {
	Ascent, Descent, LineGap float32
}

// Metrics returns the font's metrics.
func (f Font) Metrics() FontMetrics {
	a, d, h := text.Shared().Metrics(f.style())
	return FontMetrics{Ascent: a, Descent: d, LineGap: max(h-a-d, 0)}
}

// Glyph is a glyph of text that Shape placed, which Painter.Glyphs draws.
type Glyph struct {
	// Cluster is the rune of the text the glyph's cluster starts at, and
	// Runes how many runes the cluster holds: a ligature holds several,
	// and so does a character with combining marks, whose glyphs share
	// the cluster.
	Cluster, Runes int
	// X is the left of the glyph's advance from the start of the text,
	// and Advance its width. Move X to draw the glyph elsewhere, as in the
	// cell of a grid.
	X, Advance float32

	font  *text.Font
	id    uint32
	thick bool
	flat  bool
}

// Shape shapes s in font f, on one line, for Painter.Glyphs: the
// system's text engine finds the font, falls back to others for what it
// lacks, and applies ligatures and kerning. Unlike the text of elements,
// which ui caches by content, it caches nothing: keep the glyphs of text
// drawn in several frames.
func Shape(s string, f Font) []Glyph {
	l := text.Shared().Shape(text.Params{Text: s, Style: f.style(), NoWrap: true, KeepSpaces: true})
	var out []Glyph
	for _, line := range l.Lines {
		for _, g := range line.Glyphs {
			out = append(out, Glyph{Cluster: g.Cluster, Runes: g.Runes, X: line.X + g.X, Advance: g.Advance, font: g.Font, id: g.ID, thick: f.Thicken, flat: f.Antialiased})
		}
	}
	return out
}

// Glyphs draws glyphs that Shape made, from x, on the baseline at y, in
// color c.
func (p *Painter) Glyphs(glyphs []Glyph, x, y float32, c Color) {
	sys := p.rt.text
	s := p.scale
	shade := p.shadeOf(c)
	c = c.Alpha(p.opacity)
	color := c.scene()
	baseline := sys.Baseline(y * s)
	start := int32(len(p.s.Glyphs))
	left, right := p.clip.X*s, (p.clip.X+p.clip.W)*s
	var run *glyphRun
	if sys.JoinsGlyphs() {
		run = &glyphRun{}
	}
	for _, g := range glyphs {
		if g.font == nil {
			continue
		}
		pen := (x + g.X) * s
		if !p.transform.Set && (pen > right || pen+(g.Advance+g.font.Size)*s < left) {
			continue
		}
		ix := float32(math.Floor(float64(pen)))
		shade := shade
		if g.thick {
			shade = text.Thick
		} else if g.flat {
			shade = text.Flat
		}
		gi := sys.Glyph(g.font, g.id, s, pen, shade, p.opaque)
		if !gi.OK {
			continue
		}
		sg := scene.Glyph{
			X: ix + gi.Left, Y: baseline + gi.Top, W: float32(gi.W), H: float32(gi.H),
			U: gi.X, V: gi.Y, UW: gi.W, VH: gi.H,
			Color: color, Wide: p.glyphWide(c), Colored: gi.Colored, Subpixel: gi.Subpixel, Thin: gi.Thin,
		}
		if run != nil {
			run.add(p, text.Glyph{Font: g.font, ID: g.id}, gi, pen, sg, shade, baseline)
			continue
		}
		p.s.Glyphs = append(p.s.Glyphs, sg)
	}
	if run != nil {
		run.flush(p, baseline)
	}
	if end := int32(len(p.s.Glyphs)); end > start {
		p.add(scene.Op{Kind: scene.OpGlyphs, Start: start, End: end})
	}
}

// TextLine paints the glyphs of the first line of l, a layout of this
// module's text system that only its packages make, as package ui/editor
// does, from x, on the baseline at y, each in the color color returns for
// the rune its cluster starts at. Glyphs outside the painter's clip are
// left out.
func (p *Painter) TextLine(l *text.Layout, x, y float32, color func(cluster int) Color) {
	if l == nil || len(l.Lines) == 0 {
		return
	}
	sys := p.rt.text
	s := p.scale
	baseline := sys.Baseline(y * s)
	left, right := p.clip.X*s, (p.clip.X+p.clip.W)*s
	start := int32(len(p.s.Glyphs))
	var run *glyphRun
	if sys.JoinsGlyphs() {
		run = &p.rt.glyphRun
		run.ids, run.pens, run.glyphs = run.ids[:0], run.pens[:0], run.glyphs[:0]
	}
	for _, g := range l.Lines[0].Glyphs {
		pen := (x + g.X) * s
		if !p.transform.Set && (pen > right || pen+(g.Advance+g.Size)*s < left) {
			continue
		}
		c := color(g.Cluster)
		shade := p.shadeOf(c)
		gi := sys.Glyph(g.Font, g.ID, s, pen, shade, p.opaque)
		if !gi.OK {
			continue
		}
		c = c.Alpha(p.opacity)
		sg := scene.Glyph{
			X: float32(math.Floor(float64(pen))) + gi.Left, Y: baseline + gi.Top, W: float32(gi.W), H: float32(gi.H),
			U: gi.X, V: gi.Y, UW: gi.W, VH: gi.H,
			Color: c.scene(), Wide: p.glyphWide(c), Colored: gi.Colored, Subpixel: gi.Subpixel, Thin: gi.Thin,
		}
		if run != nil {
			run.add(p, text.Glyph{Font: g.Font, ID: g.ID}, gi, pen, sg, shade, baseline)
			continue
		}
		p.s.Glyphs = append(p.s.Glyphs, sg)
	}
	if run != nil {
		run.flush(p, baseline)
	}
	if end := int32(len(p.s.Glyphs)); end > start {
		p.add(scene.Op{Kind: scene.OpGlyphs, Start: start, End: end})
	}
}

// Visible returns what the painter's clip leaves of the window, in DIPs:
// the part of an element's box that shows inside the scroll containers
// and clips around it, as an element taller than the window, showing a
// long document, paints only what falls in it.
func (p *Painter) Visible() Rect { return p.clip }

// Scale returns how many device pixels a DIP is, for drawing that lines
// up with the pixels of the display.
func (p *Painter) Scale() float32 { return p.scale }
