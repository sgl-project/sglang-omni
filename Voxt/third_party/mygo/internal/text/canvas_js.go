//go:build js

package text

import (
	"errors"
	"math"
	"strconv"
	"strings"
	"syscall/js"
	"unicode"

	"github.com/go-text/typesetting/bidi"
	"github.com/go-text/typesetting/segmenter"

	"github.com/egoist/mygo/internal/scene"
)

// newEngine returns the canvas engine, or where there is no canvas to
// measure and draw text with (Node, workers of browsers without
// OffscreenCanvas), the stub engine.
func newEngine() engine {
	if js.Global().Get("document").IsUndefined() && js.Global().Get("OffscreenCanvas").IsUndefined() {
		return &stubEngine{}
	}
	return newCanvas()
}

// canvasEngine lays out and draws text with a canvas's 2D context, in
// browsers: the browser finds the fonts, falls back for what one lacks and
// shapes, as for its pages. Canvases tell how wide text is and draw it,
// not its glyphs: the engine's glyphs are grapheme clusters, measured
// within their words, which keeps the fonts' kerning, and drawn alone; a
// word of a script that joins or reorders its letters (Arabic, the scripts
// of India, …) is one glyph, drawn whole. go-text's segmenter finds the
// clusters and where lines may break, and its bidi package the direction
// of runs.
type canvasEngine struct {
	shapeScratch
	canvas, ctx js.Value
	// current is the context's font now, kerning its fontKerning and fill
	// its fillStyle.
	current, kerning, fill string
	fonts                  map[string]*Font
	css                    []fontCSS // the CSS font of each Font, by native
	// emoji are the twins of fonts that draw emoji, unshaded: emoji
	// keep their colors whatever the text's.
	emoji map[*Font]*Font
	// The glyphs' texts, by ID.
	ids   map[string]uint32
	texts []string
	// words caches the pen positions of the clusters of words.
	words map[wordKey][]float32
	// names are the families the app registered fonts under.
	names []string
	seg   segmenter.Segmenter
	para  bidi.Paragraph
	// image is the canvas's pixels read back, reused.
	image []byte
	w, h  int
}

// fontCSS is a CSS font without its size: the style and weight before it,
// the families after.
type fontCSS struct{ pre, family string }

type wordKey struct {
	font   uintptr
	word   string
	noKern bool
}

func newCanvas() *canvasEngine {
	e := &canvasEngine{fonts: map[string]*Font{}, ids: map[string]uint32{"": 0}, texts: []string{""}, words: map[wordKey][]float32{}}
	doc := js.Global().Get("document")
	if doc.IsUndefined() {
		e.canvas = js.Global().Get("OffscreenCanvas").New(64, 64)
	} else {
		e.canvas = doc.Call("createElement", "canvas")
		e.canvas.Set("width", 64)
		e.canvas.Set("height", 64)
	}
	e.w, e.h = 64, 64
	e.ctx = e.canvas.Call("getContext", "2d", map[string]any{"willReadFrequently": true})
	e.ctx.Set("textBaseline", "alphabetic")
	e.ctx.Set("textAlign", "left")
	e.ctx.Set("direction", "ltr")
	e.ctx.Set("fillStyle", "#fff")
	return e
}

// cssFamily turns a list of families into CSS's, with the generic ones
// the system's own.
func cssFamily(list string) string {
	var out []string
	for _, f := range familyList(list) {
		switch generic(f) {
		case "system-ui":
			out = append(out, `system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji"`)
		case "sans-serif":
			out = append(out, "sans-serif")
		case "serif":
			out = append(out, "serif")
		case "monospace":
			out = append(out, `ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace`)
		default:
			out = append(out, strconv.Quote(f))
		}
	}
	if len(out) == 0 {
		return cssFamily("system-ui")
	}
	return strings.Join(out, ", ")
}

// cssOf returns the CSS font of a style.
func cssOf(style Style) fontCSS {
	pre := strconv.Itoa(style.weight()) + " "
	if style.Italic {
		pre = "italic " + pre
	}
	return fontCSS{pre, cssFamily(style.Family)}
}

// at returns the CSS font at size px.
func (c fontCSS) at(size float32) string {
	return c.pre + strconv.FormatFloat(float64(size), 'f', -1, 32) + "px " + c.family
}

func (e *canvasEngine) setFont(css string) {
	if css != e.current {
		e.current = css
		e.ctx.Set("font", css)
	}
}

func (e *canvasEngine) font(style Style) *Font {
	size := style.FontSize()
	fc := cssOf(style)
	css := fc.at(size)
	if f, ok := e.fonts[css]; ok {
		return f
	}
	e.setFont(css)
	m := e.ctx.Call("measureText", "Hg")
	// Shaded: browsers draw text of each color with its own coverage, as
	// Core Text's font smoothing emboldens lighter text more, and Skia
	// corrects coverage by the text's luminance. Thick text is drawn in
	// white, the boldest.
	f := &Font{Size: size, native: uintptr(len(e.css)), shaded: true, thickens: true}
	if a := m.Get("fontBoundingBoxAscent"); a.Type() == js.TypeNumber {
		f.Ascent, f.Descent = float32(a.Float()), float32(m.Get("fontBoundingBoxDescent").Float())
	} else {
		f.Ascent, f.Descent = size*0.9, size*0.25
	}
	// Underlines a tenth of an em below the baseline, strikethroughs
	// through the middle of an x.
	thick := max(size/14, 1)
	f.underlineTop, f.underlineThick = -size/10, thick
	x := e.ctx.Call("measureText", "x").Get("actualBoundingBoxAscent")
	if x.Type() == js.TypeNumber {
		f.strikeTop = float32(x.Float())/2 + thick/2
	} else {
		f.strikeTop = size * 0.3
	}
	f.strikeThick = thick
	e.css = append(e.css, fc)
	e.fonts[css] = f
	return f
}

// glyphID returns the ID of a glyph drawing text.
func (e *canvasEngine) glyphID(text string) uint32 {
	if id, ok := e.ids[text]; ok {
		return id
	}
	id := uint32(len(e.texts))
	e.texts = append(e.texts, text)
	e.ids[text] = id
	return id
}

// measure returns how wide text is in f, kerned unless noKern.
func (e *canvasEngine) measure(f *Font, text string, noKern bool) float32 {
	e.setFont(e.sized(f, f.Size))
	kerning := "normal"
	if noKern {
		kerning = "none"
	}
	if kerning != e.kerning {
		e.kerning = kerning
		e.ctx.Set("fontKerning", kerning)
	}
	return float32(e.ctx.Call("measureText", text).Get("width").Float())
}

// noKern reports whether a style's OpenType features turn kerning off,
// the one feature canvases have (fontKerning): the others they leave as
// the font has them.
func noKern(style Style) bool {
	for _, f := range features(style.Features) {
		if f.tag == [4]byte{'k', 'e', 'r', 'n'} {
			return f.value == 0
		}
	}
	return false
}

// sized returns the CSS font of f at size px.
func (e *canvasEngine) sized(f *Font, size float32) string { return e.css[f.native].at(size) }

// cluster is a glyph being laid out: a grapheme cluster, or a word of a
// script whose letters join, of runes [start, end).
type cluster struct {
	start, end int
	font       *Font
	x, advance float32
	level      bidi.Level
	text       string
}

// complexScript reports whether a rune is of a script whose letters
// change with their neighbors, or reorder: its words are drawn whole.
func complexScript(r rune) bool {
	return unicode.In(r, unicode.Arabic, unicode.Syriac, unicode.Nko, unicode.Mongolian, unicode.Devanagari,
		unicode.Bengali, unicode.Gurmukhi, unicode.Gujarati, unicode.Oriya, unicode.Tamil, unicode.Telugu,
		unicode.Kannada, unicode.Malayalam, unicode.Sinhala, unicode.Thai, unicode.Lao, unicode.Tibetan,
		unicode.Myanmar, unicode.Khmer, unicode.Javanese, unicode.Balinese)
}

func (e *canvasEngine) shape(text []rune, style Style, spans []Span, width float32, rtl, wholeWords bool) []shapedLine {
	if len(text) == 0 {
		return nil
	}
	// The style of each rune: spans', then the text's.
	styles := make([]Style, 0, len(spans)+1)
	fontAt := make([]uint8, len(text))
	{
		start := 0
		for _, sp := range spans {
			end := min(sp.End, len(text))
			if end <= start {
				continue
			}
			for i := start; i < end; i++ {
				fontAt[i] = uint8(len(styles))
			}
			styles = append(styles, sp.style(style))
			start = end
			if len(styles) == 255 {
				break
			}
		}
		for i := start; i < len(text); i++ {
			fontAt[i] = uint8(len(styles))
		}
		styles = append(styles, style)
	}
	fonts := make([]*Font, len(styles))
	for i, s := range styles {
		fonts[i] = e.font(s)
	}

	// The direction of each rune.
	levels := make([]bidi.Level, len(text))
	mixed := rtl
	for _, r := range text {
		if r >= 0x0590 {
			mixed = true
			break
		}
	}
	if mixed {
		dir := bidi.LeftToRight
		if rtl {
			dir = bidi.RightToLeft
		}
		runs := e.para.Segment(text, dir)
		for i := range runs.NumRuns() {
			r := runs.Run(i)
			for k := r.Start; k < min(r.End, len(text)); k++ {
				levels[k] = r.Level
			}
		}
	}

	// Where lines may break, and the clusters between.
	e.seg.Init(text)
	var breaks []int
	for it := e.seg.LineIterator(); it.Next(); {
		l := it.Line()
		breaks = append(breaks, l.Offset+len(l.Text))
	}
	var bounds []int
	for it := e.seg.GraphemeIterator(); it.Next(); {
		g := it.Grapheme()
		bounds = append(bounds, g.Offset)
	}
	bounds = append(bounds, len(text))

	// The clusters of each piece of text between breaks, of one style and
	// direction, measured as a word.
	var clusters []cluster
	b := 0 // the cluster boundary
	start := 0
	for _, end := range breaks {
		for start < end {
			// The piece: same style, level and script complexity.
			pe := start + 1
			for pe < end && fontAt[pe] == fontAt[start] && levels[pe] == levels[start] {
				pe++
			}
			// Whole clusters only.
			for b < len(bounds)-1 && bounds[b] < start {
				b++
			}
			ce := b
			for ce < len(bounds)-1 && bounds[ce+1] <= pe {
				ce++
			}
			if ce == b {
				ce = b + 1
			}
			pe = bounds[ce]
			st := styles[fontAt[start]]
			f := fonts[fontAt[start]]
			clusters = e.measureWord(clusters, text, bounds[b:ce+1], f, st.LetterSpacing, noKern(st), levels[start])
			b, start = ce, pe
		}
	}

	// Inline objects take their width, whatever the canvas measured of
	// their character.
	{
		start := 0
		for _, sp := range spans {
			end := min(sp.End, len(text))
			if sp.ObjectWidth > 0 {
				for k := range clusters {
					if c := &clusters[k]; c.start >= start && c.start < end {
						c.advance = sp.ObjectWidth
					}
				}
			}
			start = max(start, end)
		}
	}

	// Lines: whole pieces between breaks that fit, or clusters of those
	// longer than a line unless wholeWords.
	var lines [][2]int // clusters [i, j)
	if width <= 0 {
		lines = append(lines, [2]int{0, len(clusters)})
	} else {
		lineStart, lineW := 0, float32(0)
		ci := 0
		for _, end := range breaks {
			cj := ci
			for cj < len(clusters) && clusters[cj].start < end {
				cj++
			}
			// The piece's width, less the whitespace ending it.
			var w, trail float32
			for k := ci; k < cj; k++ {
				w += clusters[k].advance
			}
			for k := cj - 1; k >= ci && isSpace(text, clusters[k]); k-- {
				trail += clusters[k].advance
			}
			if lineW+w-trail > width+1e-3 && ci > lineStart {
				lines = append(lines, [2]int{lineStart, ci})
				lineStart, lineW = ci, 0
			}
			if w-trail > width+1e-3 && !wholeWords {
				// Break the piece within itself.
				for k := ci; k < cj; k++ {
					a := clusters[k].advance
					if lineW+a > width+1e-3 && k > lineStart && !isSpace(text, clusters[k]) {
						lines = append(lines, [2]int{lineStart, k})
						lineStart, lineW = k, 0
					}
					lineW += a
				}
			} else {
				lineW += w
			}
			ci = cj
		}
		lines = append(lines, [2]int{lineStart, len(clusters)})
	}

	out := make([]shapedLine, 0, len(lines))
	for _, ln := range lines {
		cs := clusters[ln[0]:ln[1]]
		if len(cs) == 0 {
			continue
		}
		sl := shapedLine{start: cs[0].start, end: cs[len(cs)-1].end}
		visual := reorder(cs, rtl)
		x := float32(0)
		for k := 0; k < len(visual); {
			c := visual[k]
			r := shapedRun{font: c.font, start: c.start, end: c.end}
			for k < len(visual) && visual[k].font == c.font && visual[k].level%2 == c.level%2 {
				v := visual[k]
				r.start, r.end = min(r.start, v.start), max(r.end, v.end)
				r.glyphs = append(r.glyphs, Glyph{Font: v.font, ID: e.glyphID(v.text), X: x, Advance: v.advance, Cluster: v.start, RTL: v.level%2 == 1})
				x += v.advance
				k++
			}
			sl.runs = append(sl.runs, r)
		}
		out = append(out, sl)
	}
	return out
}

func isSpace(text []rune, c cluster) bool {
	return c.end-c.start == 1 && unicode.IsSpace(text[c.start])
}

// measureWord appends the clusters of a word of runes [bounds[0],
// bounds[len-1]) in f, at level, their boundaries bounds, positioned as
// the browser lays the word out. Clusters of scripts whose letters join
// make one, which the browser shapes whole.
func (e *canvasEngine) measureWord(out []cluster, text []rune, bounds []int, f *Font, spacing float32, noKern bool, level bidi.Level) []cluster {
	for k := 0; k < len(bounds)-1; {
		j := k + 1
		joins := complexScript(text[bounds[k]])
		for j < len(bounds)-1 && complexScript(text[bounds[j]]) == joins {
			j++
		}
		if joins {
			word := string(text[bounds[k]:bounds[j]])
			w := e.measure(f, word, noKern)
			out = append(out, cluster{start: bounds[k], end: bounds[j], font: f, advance: w + spacing*float32(j-k), level: level, text: word})
		} else {
			out = e.measureClusters(out, text, bounds[k:j+1], f, spacing, noKern, level)
		}
		k = j
	}
	return out
}

// measureClusters appends clusters of runes [bounds[0], bounds[len-1]),
// each drawn alone, at the positions the browser gives them in the whole.
func (e *canvasEngine) measureClusters(out []cluster, text []rune, bounds []int, f *Font, spacing float32, noKern bool, level bidi.Level) []cluster {
	start, end := bounds[0], bounds[len(bounds)-1]
	key := wordKey{f.native, string(text[start:end]), noKern}
	pens, ok := e.words[key]
	if !ok {
		pens = make([]float32, len(bounds))
		if len(bounds) > 33 {
			// Long words: each cluster alone.
			for k := 1; k < len(bounds); k++ {
				pens[k] = pens[k-1] + e.measure(f, string(text[bounds[k-1]:bounds[k]]), noKern)
			}
		} else {
			// A cluster ends where the text through it ends, and starts
			// its own width before: kerning with the cluster before it
			// moves it, as the browser draws the word, not the next one.
			n := len(bounds) - 1
			pens[n] = e.measure(f, string(text[start:end]), noKern)
			for k := 1; k < n; k++ {
				through := e.measure(f, string(text[start:bounds[k+1]]), noKern)
				alone := e.measure(f, string(text[bounds[k]:bounds[k+1]]), noKern)
				pens[k] = max(through-alone, pens[k-1])
			}
		}
		if len(e.words) > 1<<14 {
			clear(e.words)
		}
		e.words[key] = pens
	}
	for k := 1; k < len(bounds); k++ {
		c := cluster{
			start: bounds[k-1], end: bounds[k], font: f,
			advance: max(pens[k]-pens[k-1], 0) + spacing, level: level,
			text: string(text[bounds[k-1]:bounds[k]]),
		}
		if isEmoji(text[bounds[k-1]:bounds[k]]) {
			c.font = e.emojiFont(f)
		}
		out = append(out, c)
	}
	return out
}

// isEmoji reports whether a cluster is an emoji: a pictograph, a flag, or
// a character with the emoji presentation selector.
func isEmoji(cluster []rune) bool {
	r := cluster[0]
	if r >= 0x1F000 && r <= 0x1FAFF || r >= 0x2600 && r <= 0x27BF {
		return true
	}
	for _, r := range cluster[1:] {
		if r == 0xFE0F {
			return true
		}
	}
	return false
}

// emojiFont returns f's twin for emoji.
func (e *canvasEngine) emojiFont(f *Font) *Font {
	if t, ok := e.emoji[f]; ok {
		return t
	}
	if e.emoji == nil {
		e.emoji = map[*Font]*Font{}
	}
	t := *f
	t.shaded, t.thickens = false, false
	e.emoji[f] = &t
	return &t
}

// reorder returns the clusters of a line in visual order (UAX #9's L2):
// from the highest level down to the lowest odd one, runs at that level or
// higher reverse.
func reorder(cs []cluster, rtl bool) []cluster {
	var hi bidi.Level
	lo := bidi.Level(127)
	for _, c := range cs {
		hi = max(hi, c.level)
		if c.level%2 == 1 {
			lo = min(lo, c.level)
		}
	}
	if hi == 0 {
		return cs
	}
	v := append([]cluster(nil), cs...)
	// Whitespace ending the line takes the paragraph's level (L1).
	base := bidi.Level(0)
	if rtl {
		base = 1
	}
	for k := len(v) - 1; k >= 0 && v[k].end-v[k].start == 1 && v[k].text == " "; k-- {
		v[k].level = base
	}
	for l := hi; l >= lo && l > 0; l-- {
		for i := 0; i < len(v); {
			if v[i].level < l {
				i++
				continue
			}
			j := i
			for j < len(v) && v[j].level >= l {
				j++
			}
			for a, b := i, j-1; a < b; a, b = a+1, b-1 {
				v[a], v[b] = v[b], v[a]
			}
			i = j
		}
	}
	return v
}

// shadeGray is the gray glyphs of text of each shade are drawn in: of
// relative luminance 0, 1/4, 1/2 and 3/4 (ShadeOf), and white for Thick.
// Canvases draw no text unsmoothed: Flat is drawn as the darkest shade.
var shadeGray = [...]uint8{0, 137, 188, 225, 255, 0}

// glyph draws a glyph's text in the gray of its shade and reads its
// coverage back: gray, or color for color fonts (emoji).
func (e *canvasEngine) glyph(f *Font, id uint32, scale, dx float32, shade Shade, _ bool) bitmap {
	if int(id) >= len(e.texts) || id == 0 {
		return bitmap{}
	}
	text := e.texts[id]
	if strings.TrimSpace(text) == "" {
		return bitmap{}
	}
	// The font at the size text was measured at, the canvas scaled: at
	// the pixels' size, the system's font may take another optical size
	// (SF Pro Display rather than Text), whose glyphs the advances measured
	// do not fit, as pages draw at the CSS size, scaled.
	e.setFont(e.sized(f, f.Size))
	m := e.ctx.Call("measureText", text)
	s := float64(scale)
	l, r := s*m.Get("actualBoundingBoxLeft").Float(), s*m.Get("actualBoundingBoxRight").Float()
	a, d := s*m.Get("actualBoundingBoxAscent").Float(), s*m.Get("actualBoundingBoxDescent").Float()
	const pad = 1
	left := int(math.Floor(-l+float64(dx))) - pad
	right := int(math.Ceil(r+float64(dx))) + pad
	top := int(math.Floor(-a)) - pad
	bottom := int(math.Ceil(d)) + pad
	w, h := right-left, bottom-top
	if w <= 2*pad || h <= 2*pad || w > 2048 || h > 2048 {
		return bitmap{}
	}
	if w > e.w || h > e.h {
		e.w, e.h = max(w, e.w), max(h, e.h)
		e.canvas.Set("width", e.w)
		e.canvas.Set("height", e.h)
		// A canvas resized forgets its state.
		e.current = ""
		e.setFont(e.sized(f, f.Size))
		e.ctx.Set("textBaseline", "alphabetic")
		e.ctx.Set("textAlign", "left")
		e.fill = ""
	}
	gray := shadeGray[0]
	if int(shade) < len(shadeGray) {
		gray = shadeGray[shade]
	}
	if fill := "rgb(" + strconv.Itoa(int(gray)) + "," + strconv.Itoa(int(gray)) + "," + strconv.Itoa(int(gray)) + ")"; fill != e.fill {
		e.fill = fill
		e.ctx.Set("fillStyle", fill)
	}
	e.ctx.Call("clearRect", 0, 0, w, h)
	e.ctx.Call("setTransform", s, 0, 0, s, 0, 0)
	e.ctx.Call("fillText", text, (float64(dx)-float64(left))/s, -float64(top)/s)
	e.ctx.Call("setTransform", 1, 0, 0, 1, 0, 0)
	data := e.ctx.Call("getImageData", 0, 0, w, h).Get("data")
	n := 4 * w * h
	if cap(e.image) < n {
		e.image = make([]byte, n)
	}
	px := e.image[:n]
	js.CopyBytesToGo(px, data)
	// The ink's box: measurements leave room around it.
	x0, y0, x1, y1 := w, h, 0, 0
	for y := range h {
		for x := range w {
			if px[4*(y*w+x)+3] != 0 {
				x0, y0, x1, y1 = min(x0, x), min(y0, y), max(x1, x+1), max(y1, y+1)
			}
		}
	}
	if x0 >= x1 {
		return bitmap{}
	}
	if x0 > 0 || y0 > 0 || x1 < w || y1 < h {
		cw, ch := x1-x0, y1-y0
		for y := range ch {
			copy(px[4*y*cw:4*(y+1)*cw], px[4*((y+y0)*w+x0):4*((y+y0)*w+x1)])
		}
		left, top, w, h = left+x0, top+y0, cw, ch
		n = 4 * w * h
		px = px[:n]
	}
	// Gray text is its gray where it covers: the alpha is the coverage.
	// Others are colored.
	color := false
	far := func(c uint8) bool { return int(c) > int(gray)+32 || int(gray) > int(c)+32 }
	for i := 0; i < n; i += 4 {
		if px[i+3] > 32 && (far(px[i]) || far(px[i+1]) || far(px[i+2])) {
			color = true
			break
		}
	}
	bm := bitmap{left: left, top: top, w: w, h: h, color: color}
	if color {
		bm.pix = make([]byte, n)
		for i := 0; i < n; i += 4 {
			al := uint16(px[i+3])
			bm.pix[i] = uint8((uint16(px[i])*al + 127) / 255)
			bm.pix[i+1] = uint8((uint16(px[i+1])*al + 127) / 255)
			bm.pix[i+2] = uint8((uint16(px[i+2])*al + 127) / 255)
			bm.pix[i+3] = px[i+3]
		}
		return bm
	}
	bm.pix = make([]byte, w*h)
	for i := range w * h {
		bm.pix[i] = px[4*i+3]
	}
	return bm
}

// maxTexts is how many glyphs, texts, the engine keeps before it has the
// System let go of its fonts with them: a glyph is a cluster, or a whole
// word of some scripts, which pile up with what a long session shows.
const maxTexts = 1 << 16

func (e *canvasEngine) fontCount() int {
	if len(e.texts) > maxTexts {
		return maxFonts + 1
	}
	return len(e.fonts)
}

// forgetFonts lets go of the fonts, and of the glyphs and measurements,
// which refer to them by number.
func (e *canvasEngine) forgetFonts() {
	clear(e.fonts)
	clear(e.emoji)
	e.css = e.css[:0]
	clear(e.words)
	clear(e.ids)
	e.ids[""] = 0
	e.texts = e.texts[:1]
}

// textParams leaves coverage as the browser drew it, without subpixel
// antialiasing, which canvases do not read back.
func (e *canvasEngine) textParams() (scene.TextParams, bool) { return scene.TextParams{}, false }

// positions are a quarter of a pixel apart, the nearest to the pen, as
// Skia, which draws the text of Chrome's canvases, places glyphs.
func (e *canvasEngine) positions(*Font, float32, bool) (int, bool) { return 4, true }

func (e *canvasEngine) baseline(y float32) float32 { return float32(math.Round(float64(y))) }

// join: browsers' canvases blend each glyph of a run in turn (Skia).
func (e *canvasEngine) join() bool { return false }

// decorate places underlines and strikethroughs along the advances of the
// runs, at the positions of their fonts, whole pixels from the baseline.
func (e *canvasEngine) decorate(r decoRange) []Stroke {
	s := r.scale
	base := e.baseline(r.baseline() * s)
	var out []Stroke
	r.runs(func(f *Font, i, j int) {
		x0, x1 := r.span(i, j)
		top, thick := f.strikeTop, f.strikeThick
		if r.d == Underline {
			top, thick = f.underlineTop, f.underlineThick
		}
		t := float32(math.Round(float64(base - top*s)))
		st := Stroke{X0: (r.x + x0) * s, X1: (r.x + x1) * s, Top: t, Bottom: t + max(float32(math.Round(float64(thick*s))), 1)}
		if n := len(out); n > 0 && out[n-1].Top == st.Top && out[n-1].Bottom == st.Bottom {
			out[n-1].X1 = max(out[n-1].X1, st.X1)
			return
		}
		out = append(out, st)
	})
	return out
}

// appFamilies lists the app's families. A family is monospaced when an
// "i" is as wide as a "W" in it, which tells only once its face has
// loaded.
func (e *canvasEngine) appFamilies() []FontFamily {
	out := make([]FontFamily, len(e.names))
	for i, name := range e.names {
		e.setFont("100px " + strconv.Quote(name))
		mono := e.ctx.Call("measureText", "i").Get("width").Float() == e.ctx.Call("measureText", "W").Get("width").Float()
		out[i] = FontFamily{Name: name, Monospaced: mono}
	}
	return out
}

// systemFamilies lists none: a browser does not tell what fonts it has.
func (e *canvasEngine) systemFamilies() func() []FontFamily { return nil }

// register adds a font file to the document's fonts, as a FontFace of
// family, or of its own family when family is "" (which the browser does
// not tell: the face needs a family). Browsers load faces in the
// background: text laid out before draws with others.
func (e *canvasEngine) register(data []byte, family string) error {
	if family == "" {
		return errNoFamily
	}
	buf := js.Global().Get("Uint8Array").New(len(data))
	js.CopyBytesToJS(buf, data)
	face := js.Global().Get("FontFace").New(family, buf.Get("buffer"))
	fonts := js.Global().Get("document").Get("fonts")
	fonts.Call("add", face)
	face.Call("load").Call("catch", js.Global().Get("Function").New(""))
	e.names = append(e.names, family)
	clear(e.words)
	return nil
}

var errNoFamily = errors.New("mygo: browsers need the family of a font to add")
