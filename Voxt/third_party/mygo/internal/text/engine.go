// Package text lays out text and rasterizes glyphs with the system's own
// text stack: DirectWrite on Windows, Core Text on macOS and Pango with
// cairo on Linux. They find the fonts, fall back to other fonts for what
// one lacks, shape and break lines; the package assembles their lines into
// layouts, with carets, selection and truncation of its own, and keeps the
// glyphs in the atlases scenes draw from. All methods of System are safe
// from any goroutine; the ui package uses them from the main thread.
package text

import (
	"strconv"
	"strings"
	"unicode/utf16"
	"unicode/utf8"

	"github.com/egoist/mygo/internal/scene"
)

// An engine is the system's own text stack: DirectWrite on Windows, Core
// Text on macOS and Pango on Linux. It finds fonts, falling back to the
// system's choices for what a font lacks, shapes and breaks paragraphs
// into lines, and rasterizes glyphs. The System calls it with its lock
// held, so engines need no locking of their own.
type engine interface {
	// font returns the font a style asks for, at its size, which sets the
	// default line height: the first family of the style the system has,
	// or its user interface font.
	font(style Style) *Font
	// shape breaks text, a paragraph without newlines, into lines of at
	// most width DIPs, or into one line when width is 0, and shapes them.
	// rtl sets the paragraph's direction; wholeWords lets a word longer
	// than a line overflow it instead of breaking it. spans style runs of
	// the text apart from style, with their ends relative to it.
	shape(text []rune, style Style, spans []Span, width float32, rtl, wholeWords bool) []shapedLine
	// glyph rasterizes glyph id of f at scale pixels per DIP, its origin
	// dx pixels (0 ≤ dx < 1) right of the left edge of a pixel, for text
	// of a shade when f is shaded or the shade is Thick, with subpixel
	// antialiasing if subpixel.
	glyph(f *Font, id uint32, scale, dx float32, shade Shade, subpixel bool) bitmap
	// positions returns how many horizontal positions within a pixel
	// glyphs of f at scale pixels per DIP are drawn at, with subpixel
	// antialiasing if subpixel, and whether a pen goes to the nearest
	// (round) or to the one left of it.
	positions(f *Font, scale float32, subpixel bool) (n int, round bool)
	// baseline returns the whole pixel a baseline y device pixels from the
	// top is drawn at.
	baseline(y float32) float32
	// decorate returns the strokes of a decoration (see System.Decorate).
	decorate(r decoRange) []Stroke
	// join reports whether the engine's text stack adds the coverage of
	// the glyphs of a run whose edges share pixels before blending it.
	join() bool
	// textParams returns how renderers correct the coverage of glyphs,
	// and whether glyphs take subpixel antialiasing, as the system's
	// settings say.
	textParams() (p scene.TextParams, subpixel bool)
	// register adds the fonts of a font file under family, or under their
	// own family names when family is "".
	register(data []byte, family string) error
	// appFamilies lists the families of the app's fonts, in no order,
	// under the names it registered them with, or their own when it gave
	// none.
	appFamilies() []FontFamily
	// systemFamilies returns a function listing the system's families, in
	// no order and without the app's, which runs without the System's lock
	// on any goroutine, or nil when the engine knows none.
	systemFamilies() func() []FontFamily
	// resetScratch frees the room shape took from the engine's scratch,
	// as a layout starts.
	resetScratch()
}

// shapeScratch is room engines take the glyphs, runs and lines shape returns
// from, which live until the System lays out the next text: a layout
// copies them into its own lines. Engines embed it.
type shapeScratch struct {
	glyphs []Glyph
	runs   []shapedRun
	lines  []shapedLine
}

// maxScratch is how many glyphs shapeScratch keeps room for between layouts:
// a long text's room goes with it.
const maxScratch = 4096

func (s *shapeScratch) resetScratch() {
	if cap(s.glyphs) > maxScratch {
		s.glyphs, s.runs, s.lines = nil, nil, nil
	}
	s.glyphs, s.runs, s.lines = s.glyphs[:0], s.runs[:0], s.lines[:0]
}

// glyphRoom returns room for n glyphs until the next layout.
func (s *shapeScratch) glyphRoom(n int) []Glyph {
	if cap(s.glyphs)-len(s.glyphs) < n {
		// Slices taken before keep the old room.
		s.glyphs = make([]Glyph, 0, max(2*cap(s.glyphs), n, 256))
	}
	k := len(s.glyphs)
	s.glyphs = s.glyphs[:k+n]
	return s.glyphs[k : k+n : k+n]
}

// addRun adds a run after those taken since mark, the length of runs
// then, and returns them.
func (s *shapeScratch) addRun(mark int, r shapedRun) []shapedRun {
	s.runs = append(s.runs, r)
	return s.runs[mark:len(s.runs):len(s.runs)]
}

// addLine adds a line after those taken since mark, the length of lines
// then, and returns them.
func (s *shapeScratch) addLine(mark int, l shapedLine) []shapedLine {
	s.lines = append(s.lines, l)
	return s.lines[mark:len(s.lines):len(s.lines)]
}

// shapedLine is a line of a paragraph as an engine laid it out.
type shapedLine struct {
	// start and end are the runes of the paragraph the line holds,
	// whitespace ending it included.
	start, end int
	runs       []shapedRun
}

// shapedRun is a run of glyphs of one font and direction.
type shapedRun struct {
	font *Font
	// start and end are the runes of the paragraph the run holds.
	start, end int
	// glyphs are positioned relative to the start of the line's pen, Y
	// down from the baseline, with Cluster the paragraph's rune that
	// starts the glyph's cluster; Runes is set later.
	glyphs []Glyph
}

// bitmap is a rasterized glyph.
type bitmap struct {
	// left, top, w and h are the image's box in pixels relative to the
	// glyph's origin, y down.
	left, top, w, h int
	// pix holds a byte of coverage per pixel, or for color glyphs four
	// bytes of premultiplied RGBA, or for subpixel ones the coverage of
	// the red, green and blue subpixels and a fourth byte.
	pix             []byte
	color, subpixel bool
}

// Font is a font of the system, or of the app, at a size. Layouts and
// glyphs refer to fonts by pointer: an engine returns the same Font for
// the same font and size.
type Font struct {
	// Size is the em size in DIPs.
	Size float32
	// Ascent, Descent and LineGap are the font's vertical metrics in
	// DIPs.
	Ascent, Descent, LineGap float32

	native uintptr // the engine's font
	// shaded fonts' glyphs differ by the shade of the text; thin ones are
	// too thin for antialiasing (GlyphImage.Thin). The engine draws the
	// glyphs of fonts that thicken thicker for Thick.
	shaded, thin, thickens bool
	// The tops of the font's underline and strikethrough, in DIPs above
	// the baseline, and their thickness, for engines that decorate with
	// them (DirectWrite and Pango).
	underlineTop, underlineThick, strikeTop, strikeThick float32
}

// generic names the families "system-ui", "sans-serif", "serif" and
// "monospace" stand for, and their aliases: "" when family is none.
func generic(family string) string {
	switch strings.ToLower(family) {
	case "", "system-ui", "ui-sans-serif", "-apple-system", "blinkmacsystemfont":
		return "system-ui"
	case "sans-serif":
		return "sans-serif"
	case "serif", "ui-serif":
		return "serif"
	case "monospace", "ui-monospace":
		return "monospace"
	}
	return ""
}

// feature is an OpenType feature of Style.Features: its tag and value, 0
// to turn it off, 1 on, or the alternate to pick.
type feature struct {
	tag   [4]byte
	value uint32
}

// features parses Style.Features, leaving out what is not a tag of four
// printable characters with an optional =value.
func features(list string) []feature {
	if list == "" {
		return nil
	}
	var out []feature
	for _, item := range strings.Split(list, ",") {
		tag, value, set := strings.Cut(strings.TrimSpace(item), "=")
		tag = strings.TrimSpace(tag)
		f := feature{value: 1}
		if set {
			v, err := strconv.ParseUint(strings.TrimSpace(value), 10, 32)
			if err != nil {
				continue
			}
			f.value = uint32(v)
		}
		if len(tag) != 4 || strings.ContainsFunc(tag, func(r rune) bool { return r < 0x20 || r > 0x7e }) {
			continue
		}
		copy(f.tag[:], tag)
		out = append(out, f)
	}
	return out
}

// familyList splits a comma-separated list of families, without quotes.
func familyList(family string) []string {
	var out []string
	for _, f := range strings.Split(family, ",") {
		if f = strings.Trim(strings.TrimSpace(f), `"'`); f != "" {
			out = append(out, f)
		}
	}
	return out
}

// utf16Text encodes runes as UTF-16, with the rune index of every code
// unit and of the end.
func utf16Text(runes []rune) (text []uint16, index []int) {
	return appendUTF16(make([]uint16, 0, len(runes)+1), make([]int, 0, len(runes)+1), runes)
}

// appendUTF16 appends the UTF-16 encoding of runes to text, and the rune
// index of every code unit and of the end to index.
func appendUTF16(text []uint16, index []int, runes []rune) ([]uint16, []int) {
	for i, r := range runes {
		if r >= 0x10000 && r <= utf8.MaxRune {
			a, b := utf16.EncodeRune(r)
			text = append(text, uint16(a), uint16(b))
			index = append(index, i, i)
			continue
		}
		if r > 0xFFFF || (r >= 0xD800 && r < 0xE000) {
			r = utf8.RuneError
		}
		text = append(text, uint16(r))
		index = append(index, i)
	}
	index = append(index, len(runes))
	return text, index
}

// utf8Text encodes runes as UTF-8, with the rune index of every byte and
// of the end.
func utf8Text(runes []rune) (text []byte, index []int) {
	text = make([]byte, 0, len(runes)+1)
	index = make([]int, 0, len(runes)+1)
	for i, r := range runes {
		n := len(text)
		text = utf8.AppendRune(text, r)
		for range len(text) - n {
			index = append(index, i)
		}
	}
	index = append(index, len(runes))
	return text, index
}
