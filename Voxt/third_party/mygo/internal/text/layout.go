package text

import (
	"cmp"
	"encoding/binary"
	"image"
	"math"
	"slices"
	"strings"
	"sync"
	"unicode"
	"unsafe"

	"github.com/egoist/mygo/internal/scene"
)

// Style selects the font of a text.
type Style struct {
	// Family is a comma-separated list of font families, where "system-ui"
	// and "monospace" are the system's own; "" is the system UI font.
	Family string
	// Size in DIPs; 0 is 14.
	Size float32
	// Weight from 100 (thin) to 900 (black); 0 is 400.
	Weight int
	Italic bool
	// LineHeight is the height of a line as a multiple of Size; 0 takes
	// the font's own line spacing.
	LineHeight float32
	// LetterSpacing adds DIPs after every character; negative tightens.
	LetterSpacing float32
	// Features are OpenType features, comma separated: a tag turns one
	// on, and tag=value sets it, as "tnum, liga=0, salt=2".
	Features string
	// Lang is the language of the text, a BCP 47 tag as "ja" or
	// "zh-Hant": the system's text stack picks the glyphs of that
	// language, as the Han characters Chinese and Japanese write apart,
	// and falls back to fonts for it. "" is the language of the System
	// (SetLanguage).
	Lang string
}

func (s Style) weight() int {
	if s.Weight <= 0 {
		return 400
	}
	return min(s.Weight, 999)
}

// FontSize returns the size in DIPs.
func (s Style) FontSize() float32 {
	if s.Size <= 0 {
		return 14
	}
	return s.Size
}

// Align is the horizontal alignment of the lines of a layout.
type Align uint8

const (
	// Start aligns lines to the left, or to the right in right-to-left
	// paragraphs.
	Start Align = iota
	Center
	End
	// Left and Right align lines so whatever their direction, as the
	// start and the end of an interface laid out right to left do.
	Left
	Right
)

// Params describe a text to lay out.
type Params struct {
	Text  string
	Style Style
	// Width wraps lines to this many DIPs; 0 or less only breaks lines at
	// newlines.
	Width float32
	// MaxLines truncates the text to that many lines, ending it with
	// Ellipsis ("…" when empty); 0 is unlimited.
	MaxLines int
	Ellipsis string
	// NoWrap breaks lines only at newlines, aligning them within Width
	// all the same.
	NoWrap bool
	Align  Align
	// RTL makes paragraphs without a strongly directional character, as
	// numbers or an empty line, right to left, as in an interface laid out
	// right to left; the others read in the direction of their first
	// strongly directional character, as Unicode's bidirectional
	// algorithm finds it by default (rules P2 and P3), whichever this is.
	RTL bool
	// KeepSpaces keeps the advance of whitespace at the end of wrapped
	// lines, as text editors do.
	KeepSpaces bool
	// NoBreakWords only breaks lines between words, letting a long word
	// overflow the width instead of breaking it.
	NoBreakWords bool
	// Spans styles runs of the text apart from Style, as EncodeSpans
	// writes them.
	Spans string
}

// Span styles the runes of a text before End, after those of the spans
// before it. What it sets replaces the Style of the layout; Italic only
// turns italics on.
type Span struct {
	End           int
	Family        string
	Size          float32
	Weight        int
	Italic        bool
	LetterSpacing float32
	Features      string
	// Lang is the language of the runes, as Style.Lang; "" keeps the
	// text's.
	Lang string
	// ObjectWidth and ObjectHeight, when ObjectWidth is set, make each
	// rune of the span, an object replacement character (U+FFFC), an
	// inline object: a box that wide, standing that high on the baseline,
	// which lines break around and which draws no glyph, as an image in a
	// rich text's line (Glyph.Object).
	ObjectWidth, ObjectHeight float32
}

// style returns the style of the span's runes in a text of style base.
func (sp Span) style(base Style) Style {
	s := base
	if sp.Family != "" {
		s.Family = sp.Family
	}
	if sp.Size > 0 {
		s.Size = sp.Size
	}
	if sp.Weight > 0 {
		s.Weight = sp.Weight
	}
	if sp.Italic {
		s.Italic = true
	}
	if sp.LetterSpacing != 0 {
		s.LetterSpacing = sp.LetterSpacing
	}
	if sp.Features != "" {
		s.Features = sp.Features
	}
	if sp.Lang != "" {
		s.Lang = sp.Lang
	}
	return s
}

// EncodeSpans writes spans for Params.Spans, which, as a string, keeps
// Params a value layouts are cached by.
func EncodeSpans(spans []Span) string {
	var b []byte
	str := func(s string) {
		b = binary.AppendUvarint(b, uint64(len(s)))
		b = append(b, s...)
	}
	for _, sp := range spans {
		b = binary.AppendUvarint(b, uint64(max(sp.End, 0)))
		str(sp.Family)
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(sp.Size))
		b = binary.AppendUvarint(b, uint64(max(sp.Weight, 0)))
		if sp.Italic {
			b = append(b, 1)
		} else {
			b = append(b, 0)
		}
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(sp.LetterSpacing))
		str(sp.Features)
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(sp.ObjectWidth))
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(sp.ObjectHeight))
		str(sp.Lang)
	}
	return string(b)
}

// DecodeSpans reads what EncodeSpans wrote.
func DecodeSpans(s string) []Span {
	b := []byte(s)
	var out []Span
	uvarint := func() int {
		v, n := binary.Uvarint(b)
		if n <= 0 {
			b = nil
			return 0
		}
		b = b[n:]
		return int(v)
	}
	str := func() string {
		n := uvarint()
		if n > len(b) {
			b = nil
			return ""
		}
		s := string(b[:n])
		b = b[n:]
		return s
	}
	f32 := func() float32 {
		if len(b) < 4 {
			b = nil
			return 0
		}
		v := math.Float32frombits(binary.LittleEndian.Uint32(b))
		b = b[4:]
		return v
	}
	for len(b) > 0 {
		var sp Span
		sp.End = uvarint()
		sp.Family = str()
		sp.Size = f32()
		sp.Weight = uvarint()
		if len(b) > 0 {
			sp.Italic = b[0] != 0
			b = b[1:]
		}
		sp.LetterSpacing = f32()
		sp.Features = str()
		sp.ObjectWidth = f32()
		sp.ObjectHeight = f32()
		sp.Lang = str()
		if b == nil {
			break
		}
		out = append(out, sp)
	}
	return out
}

// spansIn returns the spans of runes start to end of a text, with their
// ends relative to start.
func spansIn(spans []Span, start, end int) []Span {
	var out []Span
	from := 0
	for _, sp := range spans {
		s0 := from
		from = max(sp.End, from)
		if from <= start {
			continue
		}
		if s0 >= end {
			break
		}
		sp.End = min(from, end) - start
		out = append(out, sp)
	}
	return out
}

// Layout is shaped and wrapped text. Positions are in DIPs relative to the
// top-left corner of the text.
type Layout struct {
	Params Params
	Runes  []rune
	Lines  []Line
	// Width is the width of the longest line, Height the sum of the line
	// heights.
	Width, Height float32
	boundaries    []int // retained whole-grapheme offsets for visual navigation
	// Truncated reports that MaxLines cut the text.
	Truncated bool
}

// Line is a line of a Layout.
type Line struct {
	// X, Y, Width and Height are the line's box; X includes the alignment.
	X, Y, Width, Height float32
	// Baseline is the y of the baseline; Ascent and Descent the extent of
	// the fonts above and below it.
	Baseline, Ascent, Descent float32
	// Start and End are the line's runes; a newline ending it is not
	// included.
	Start, End int
	// RTL reports a right-to-left paragraph.
	RTL    bool
	Glyphs []Glyph

	carets   []float32
	stops    []caretStop
	upstream []float32
}

// Glyph is a positioned glyph.
type Glyph struct {
	Font *Font
	ID   uint32
	// Size is the font size in DIPs.
	Size float32
	// X is the glyph's drawing origin, including shaping offsets, Y its
	// baseline. Advance moves the pen independently of those offsets.
	X, Y    float32
	Advance float32
	// Cluster is the first rune of the glyph's cluster, Runes how many
	// runes the cluster holds; glyphs of the ellipsis have Runes 0.
	Cluster, Runes int
	RTL            bool
	// Object marks the box of an inline object (Span.ObjectWidth), which
	// draws no glyph: what shows there is the caller's.
	Object bool
}

// System lays out text and rasterizes glyphs with the system's own text
// engine. Use Shared.
type System struct {
	mu  sync.Mutex
	eng engine
	// uiFamily is the family system-ui stands for where the engine does
	// not know it (SetUIFamily), and rendering the desktop's settings for
	// rasterizing text, if the app gave them (SetFontRendering).
	uiFamily  string
	rendering *[3]string
	// families are the system's font families as FontFamilies last listed
	// them, while listed; RegisterFont forgets them, counting familiesGen.
	families    []FontFamily
	listed      bool
	familiesGen uint64
	// lang is the language of texts whose style names none (SetLanguage).
	lang string
	// fonts caches the font of each style.
	fonts map[Style]*Font

	// Layouts are kept by recency within a memory budget, not by how
	// many frames a scrolling window has drawn.
	layouts        map[Params]*cached
	oldest, newest *cached
	layoutBytes    int
	frame          uint64
	// made counts the layouts made (LayoutsMade), and gen the times the
	// system forgot its layouts (Generation).
	made, gen uint64

	glyphs map[glyphKey]*atlasEntry
	places map[placeKey]placement
	// runs holds joined glyphs (GlyphRun).
	runs  map[runKey]*atlasEntry
	masks map[uint64]*atlasEntry
	// transient holds the masks drawn for the frame being painted alone,
	// recent the frame each of them was last drawn in.
	transient map[uint64]GlyphImage
	recent    map[uint64]uint64
	// MaskAtlas holds coverage masks, ColorAtlas color glyphs.
	MaskAtlas, ColorAtlas *scene.Atlas
	// full tells which atlases (mask, color) left out something the frame
	// draws, and want how many pixels that needed; failed counts failed
	// allocations.
	full [2]bool
	want [2]int
	// wantSize is the largest bitmap left out of each atlas, including
	// padding: area alone does not tell whether a long, thin mask fits.
	wantSize [2]image.Point
	failed   int
	// subpixel tells that the system's settings ask for subpixel
	// antialiasing (TextParams).
	subpixel bool
	// rooms counts MakeRoom calls during the frame.
	rooms int
	// Buffers layouts reuse: the first runes of clusters (line), and the
	// advances and text of a line ending with an ellipsis (ellipsize).
	starts     []int
	advances   []float32
	ellipsized []rune
	// marks caches the width of the ellipsis of each style.
	marks map[markKey]float32
}

type markKey struct {
	style Style
	mark  string // Params.Ellipsis
	rtl   bool
}

// cached is a layout, with room for one line, as most layouts have, and
// the frame that used it last.
type cached struct {
	layout     Layout
	line       [1]Line
	used       uint64
	bytes      int
	prev, next *cached
}

var shared = sync.OnceValue(newSystem)

func newSystem() *System {
	return &System{
		fonts:     map[Style]*Font{},
		layouts:   map[Params]*cached{},
		marks:     map[markKey]float32{},
		glyphs:    map[glyphKey]*atlasEntry{},
		places:    map[placeKey]placement{},
		runs:      map[runKey]*atlasEntry{},
		masks:     map[uint64]*atlasEntry{},
		transient: map[uint64]GlyphImage{},
		recent:    map[uint64]uint64{},
		MaskAtlas: scene.NewAtlas(1, 256, 256),
		// Most interfaces have no color or subpixel glyphs. Keep just a
		// transparent texel until one asks for room.
		ColorAtlas: scene.NewAtlas(4, 1, 1),
	}
}

// Shared returns the process-wide text system.
func Shared() *System { return shared() }

// engine starts the system's text engine on first use.
func (s *System) engine() engine {
	if s.eng == nil {
		s.eng = newEngine()
		if u, ok := s.eng.(uiFamilySetter); ok {
			u.setUIFamily(s.uiFamily)
		}
		if f, ok := s.eng.(fontRenderer); ok && s.rendering != nil {
			r := s.rendering
			f.setFontRendering(r[0], r[1], r[2])
		}
	}
	return s.eng
}

// maxFonts is how many Fonts an engine keeps before the System lets them
// go: a Font holds a font of the system at one size.
const maxFonts = 512

// fontForgetter is an engine whose Fonts the System lets go of when they
// are many.
type fontForgetter interface {
	fontCount() int
	forgetFonts()
}

// uiFamilySetter is an engine that takes the family of the desktop's
// interface font from the app.
type uiFamilySetter interface{ setUIFamily(family string) }

// fontRenderer is an engine that takes the desktop's settings for
// rasterizing text from the app (platform.FontRendering).
type fontRenderer interface {
	setFontRendering(antialias, hinting, subpixels string)
}

// SetFontRendering sets how text is rasterized where the system's text
// stack does not know the desktop's settings, as Pango on Linux:
// antialiasing ("none", "gray" or "subpixel"), hinting ("none", "slight",
// "medium" or "full") and the order of the screen's subpixels ("rgb",
// "bgr", "vrgb" or "vbgr"), "" for the default. Other engines ignore it.
func (s *System) SetFontRendering(antialias, hinting, subpixels string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	r := [3]string{antialias, hinting, subpixels}
	if s.rendering != nil && *s.rendering == r {
		return
	}
	s.rendering = &r
	// An engine started already takes them now, and forgets what it drew.
	if f, ok := s.eng.(fontRenderer); ok {
		f.setFontRendering(antialias, hinting, subpixels)
		s.gen++
		clear(s.fonts)
		s.clearLayouts()
		clear(s.marks)
		clear(s.glyphs)
		clear(s.places)
		clear(s.runs)
	}
}

// SetUIFamily sets the family "system-ui" stands for where the system's
// text stack does not know the desktop's interface font, as Pango on
// Linux, which finds fontconfig's default sans-serif. Other engines ignore
// it.
func (s *System) SetUIFamily(family string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if family == s.uiFamily {
		return
	}
	s.uiFamily = family
	// An engine started already takes it now, and forgets what it found.
	if u, ok := s.eng.(uiFamilySetter); ok {
		u.setUIFamily(family)
		s.gen++
		clear(s.fonts)
		s.clearLayouts()
		clear(s.marks)
	}
}

// SetLanguage sets the language of the texts whose Style names none, a
// BCP 47 tag, or "" for none: the system's text stack then picks glyphs
// and fallback fonts as it does by default, by the user's settings. The
// ui package sets the language the app shows. The layouts made before are
// made anew (Generation).
func (s *System) SetLanguage(tag string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if tag == s.lang {
		return
	}
	s.lang = tag
	s.gen++
	s.clearLayouts()
	clear(s.marks)
}

// Language returns the language SetLanguage set.
func (s *System) Language() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.lang
}

// RegisterFont adds a TrueType or OpenType font (or collection) to the
// fonts text can use, under family, or the font's own family name when
// family is empty.
func (s *System) RegisterFont(data []byte, family string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.engine().register(data, family); err != nil {
		return err
	}
	// The system's families may hold the app's under their own names.
	s.families, s.listed = nil, false
	s.familiesGen++
	s.gen++
	clear(s.fonts)
	s.clearLayouts()
	clear(s.marks)
	return nil
}

// A FontFamily is a font family text can use.
type FontFamily struct {
	Name string
	// Monospaced is whether its regular font gives every character the
	// same advance.
	Monospaced bool
	// Registered is whether the app registered it (RegisterFont).
	Registered bool
}

// FontFamilies lists the families text can use: the app's registered ones
// first, under the names it gave them, or their own when it gave none,
// then the system's, each once and sorted without regard to case. The
// system's are listed once, outside the lock, and kept until RegisterFont.
func (s *System) FontFamilies() []FontFamily {
	s.mu.Lock()
	app := s.engine().appFamilies()
	system, listed, gen := s.families, s.listed, s.familiesGen
	var list func() []FontFamily
	if !listed {
		list = s.eng.systemFamilies()
	}
	s.mu.Unlock()
	if !listed {
		if list != nil {
			system = list()
		}
		s.mu.Lock()
		if s.familiesGen == gen {
			s.families, s.listed = system, true
		}
		s.mu.Unlock()
	}
	seen := map[string]int{} // the index of each family, by lowercased name
	var out []FontFamily
	add := func(families []FontFamily, registered bool) {
		from := len(out)
		for _, f := range families {
			key := strings.ToLower(f.Name)
			i, ok := seen[key]
			switch {
			case f.Name == "":
			case !ok:
				f.Registered = registered
				seen[key] = len(out)
				out = append(out, f)
			case i >= from:
				// Fonts of one family, as its bold after its regular: it is
				// monospaced if one of them is.
				out[i].Monospaced = out[i].Monospaced || f.Monospaced
			}
		}
		slices.SortFunc(out[from:], func(a, b FontFamily) int {
			return cmp.Or(strings.Compare(strings.ToLower(a.Name), strings.ToLower(b.Name)), strings.Compare(a.Name, b.Name))
		})
	}
	add(app, true)
	add(system, false)
	return out
}

// Preload starts the text engine and finds the system UI font, which the
// first layout otherwise waits for.
func (s *System) Preload() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.font(Style{})
}

// EndFrame forgets layouts no frame used for a while. The ui package calls
// it after each frame.
func (s *System) EndFrame() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for k, f := range s.recent {
		// Windows take turns: a mask drawn again within a few frames is
		// probably drawn by every frame of its window.
		if s.frame-f >= 8 {
			delete(s.recent, k)
		}
	}
	s.frame++
	if f, ok := s.eng.(fontForgetter); ok && f.fontCount() > maxFonts {
		// Fonts pile up with sizes, as with animated ones: let them all go,
		// with what refers to them. The atlas takes back the room of their
		// glyphs as it makes room; the next frame lays out and draws its
		// text anew.
		s.gen++
		s.clearLayouts()
		clear(s.marks)
		clear(s.fonts)
		clear(s.glyphs)
		clear(s.places)
		clear(s.runs)
		f.forgetFonts()
		return
	}
	for s.oldest != nil && s.overLayoutBudget(0) {
		s.removeLayout(s.oldest)
	}
	for s.oldest != nil && s.frame-s.oldest.used > 240 {
		s.removeLayout(s.oldest)
	}
}

// Layout shapes and wraps p.Text. The result is shared and must not be
// modified.
func (s *System) Layout(p Params) *Layout {
	s.mu.Lock()
	defer s.mu.Unlock()
	if c, ok := s.layouts[p]; ok {
		s.useLayout(c)
		return &c.layout
	}
	// A short label may be a slice of a large document. Cache owned strings
	// so the memory estimate reflects what those parameters keep alive.
	p.Text = strings.Clone(p.Text)
	p.Spans = strings.Clone(p.Spans)
	p.Ellipsis = strings.Clone(p.Ellipsis)
	p.Style.Family = strings.Clone(p.Style.Family)
	p.Style.Features = strings.Clone(p.Style.Features)
	p.Style.Lang = strings.Clone(p.Style.Lang)
	c := s.layout(p)
	c.bytes = layoutBytes(c)
	// A large paragraph still lays out correctly; keeping it would evict
	// all the ordinary labels and lines of other windows.
	if c.bytes <= maxLayoutBytes {
		for s.oldest != nil && (s.overLayoutBudget(c.bytes) || len(s.layouts) >= maxLayouts) {
			s.removeLayout(s.oldest)
		}
		s.layouts[p] = c
		s.layoutBytes += c.bytes
		s.useLayout(c)
	}
	return &c.layout
}

// The memory estimate includes text, shaped glyphs and room for lazy caret
// geometry. A count limit also bounds the map for many very short labels.
const (
	liveLayoutFrames   = 8
	maxLiveLayoutBytes = 64 << 20
	maxLayoutBytes     = 8 << 20
	maxLayouts         = 4096
)

// Above the ordinary budget, keep layouts used within a few global frames:
// otherwise sequential reuse of a larger working set evicts each layout just
// before its next use. The hard cap and entry limit still apply to live text.
// EndFrame also trims excess once later frames stop using it; no timer runs
// while all windows are idle.
func (s *System) overLayoutBudget(n int) bool {
	return s.layoutBytes+n > maxLiveLayoutBytes ||
		s.layoutBytes+n > maxLayoutBytes && s.oldest != nil && s.frame-s.oldest.used >= liveLayoutFrames
}

func layoutBytes(c *cached) int {
	l, p := &c.layout, c.layout.Params
	n := int(unsafe.Sizeof(*c)+unsafe.Sizeof(p)) + len(p.Text) + len(p.Spans) + len(p.Ellipsis) + len(p.Style.Family) + len(p.Style.Features) + len(p.Style.Lang)
	n += cap(l.Runes) * int(unsafe.Sizeof(rune(0)))
	if cap(l.Lines) > 1 {
		n += cap(l.Lines) * int(unsafe.Sizeof(Line{}))
	}
	// Selection can create boundaries, caret positions and visual stops
	// after insertion; reserve their room before caching the layout.
	n += (len(l.Runes) + 1) * int(unsafe.Sizeof(int(0)))
	for _, line := range l.Lines {
		n += cap(line.Glyphs) * int(unsafe.Sizeof(Glyph{}))
		n += (line.End - line.Start + 1) * 2 * int(unsafe.Sizeof(float32(0))+unsafe.Sizeof(caretStop{}))
	}
	return n
}

func (s *System) unlinkLayout(c *cached) {
	if c.prev != nil {
		c.prev.next = c.next
	} else {
		s.oldest = c.next
	}
	if c.next != nil {
		c.next.prev = c.prev
	} else {
		s.newest = c.prev
	}
	// Returned layouts may outlive the cache. Do not let them retain it.
	c.prev, c.next = nil, nil
}

func (s *System) useLayout(c *cached) {
	c.used = s.frame
	if s.newest == c {
		return
	}
	if c.prev != nil || c.next != nil || s.oldest == c {
		s.unlinkLayout(c)
	}
	c.prev = s.newest
	if s.newest != nil {
		s.newest.next = c
	} else {
		s.oldest = c
	}
	s.newest = c
}

func (s *System) removeLayout(c *cached) {
	s.unlinkLayout(c)
	delete(s.layouts, c.layout.Params)
	s.layoutBytes -= c.bytes
}

func (s *System) clearLayouts() {
	for s.oldest != nil {
		s.unlinkLayout(s.oldest)
	}
	clear(s.layouts)
	s.layoutBytes = 0
}

// LayoutsMade returns how many layouts the system has made, those Layout
// found in its cache left out.
func (s *System) LayoutsMade() uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.made
}

// Generation returns a number that changes whenever the system forgets
// its layouts, as it lets go of its fonts (EndFrame) or takes other ones:
// the layouts it returned before are then to be made anew, and those of
// fonts it let go of must not be drawn.
func (s *System) Generation() uint64 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.gen
}

// Shape lays out p.Text as Layout does, but caches nothing, for text that
// its caller keeps or rarely draws twice. The result belongs to the
// caller.
func (s *System) Shape(p Params) *Layout {
	s.mu.Lock()
	defer s.mu.Unlock()
	return &s.layout(p).layout
}

// Metrics returns the ascent, descent and default line height of a style.
func (s *System) Metrics(style Style) (ascent, descent, lineHeight float32) {
	s.mu.Lock()
	defer s.mu.Unlock()
	m := metricsOf(s.font(style), style)
	return m.ascent, m.descent, m.lineHeight
}

// font returns the font of a style.
func (s *System) font(style Style) *Font {
	key := Style{Family: style.Family, Size: style.FontSize(), Weight: style.weight(), Italic: style.Italic}
	if f, ok := s.fonts[key]; ok {
		return f
	}
	f := s.engine().font(key)
	if len(s.fonts) >= 1024 {
		clear(s.fonts)
	}
	s.fonts[key] = f
	return f
}

// lineMetrics are the vertical metrics of a style's lines.
type lineMetrics struct {
	ascent, descent, lineHeight float32
}

func metricsOf(f *Font, style Style) lineMetrics {
	size := style.FontSize()
	m := lineMetrics{ascent: size * 0.8, descent: size * 0.2}
	gap := float32(0)
	if f != nil && f.Ascent > 0 {
		m.ascent, m.descent, gap = f.Ascent, f.Descent, max(f.LineGap, 0)
	}
	m.lineHeight = m.ascent + m.descent + gap
	if style.LineHeight > 0 {
		m.lineHeight = style.LineHeight * size
	}
	return m
}

func (s *System) layout(p Params) *cached {
	s.made++
	s.engine().resetScratch()
	m := metricsOf(s.font(p.Style), p.Style)
	runes := []rune(p.Text)
	spans := DecodeSpans(p.Spans)
	c := &cached{layout: Layout{Params: p, Runes: runes}}
	l := &c.layout
	// The layout keeps its parameters as they came; the engine shapes in
	// the system's language where they name none.
	if p.Style.Lang == "" {
		p.Style.Lang = s.lang
	}
	l.Lines = c.line[:0]
	y := float32(0)
	for start := 0; start <= len(runes); {
		end := start
		for end < len(runes) && runes[end] != '\n' {
			end++
		}
		maxLines := 0
		if p.MaxLines > 0 {
			maxLines = p.MaxLines - len(l.Lines)
		}
		var truncated bool
		l.Lines, truncated = s.paragraph(l.Lines, p, spans, runes, start, end, maxLines, end < len(runes), m, &y)
		if truncated {
			l.Truncated = true
			break
		}
		start = end + 1
	}
	for _, line := range l.Lines {
		l.Width = max(l.Width, line.Width)
	}
	l.Height = y
	// Align each line within the wrap width, or the longest line.
	box := l.Width
	if p.Width > 0 {
		box = p.Width
	}
	for i := range l.Lines {
		line := &l.Lines[i]
		var dx float32
		align := p.Align
		if line.RTL {
			switch align {
			case Start:
				align = End
			case End:
				align = Start
			}
		}
		switch align {
		case Center:
			dx = (box - line.Width) / 2
		case End, Right:
			dx = box - line.Width
		}
		if dx != 0 {
			line.X += dx
			for j := range line.Glyphs {
				line.Glyphs[j].X += dx
			}
		}
	}
	return c
}

// paragraph lays out runes[start:end], a paragraph without newlines, in at
// most maxLines lines (0 is unlimited) from *y down, appends them to out,
// and moves *y past it. It reports whether it cut the text, which
// continues after the paragraph when continues is set.
func (s *System) paragraph(out []Line, p Params, spans []Span, runes []rune, start, end, maxLines int, continues bool, m lineMetrics, y *float32) ([]Line, bool) {
	text := runes[start:end]
	// A carriage return before the newline is part of it.
	if n := len(text); n > 0 && text[n-1] == '\r' {
		text = text[:n-1]
	}
	rtl := isRTL(text, p.RTL)
	spans = spansIn(spans, start, start+len(text))
	var shaped []shapedLine
	if len(text) > 0 {
		width := max(p.Width, 0)
		if p.NoWrap {
			width = 0
		}
		shaped = s.engine().shape(text, p.Style, spans, width, rtl, p.NoBreakWords)
	}
	if len(shaped) == 0 {
		shaped = []shapedLine{{end: len(text)}}
	}
	truncated := false
	if maxLines > 0 && (len(shaped) > maxLines || len(shaped) == maxLines && continues) {
		last := s.ellipsize(text, spans, shaped[maxLines-1].start, p, rtl)
		shaped = append(shaped[:maxLines-1], last)
		truncated = true
	}
	// The glyphs of the lines share one allocation.
	n := 0
	for _, sl := range shaped {
		n += glyphCount(sl)
	}
	var room []Glyph
	if n > 0 {
		room = make([]Glyph, n)
	}
	first := len(out)
	for _, sl := range shaped {
		k := glyphCount(sl)
		out = append(out, line(p, text, sl, start, rtl, m, y, room[:0:k], &s.starts, spans))
		room = room[k:]
	}
	lines := out[first:]
	// Runes between lines, if an engine left any out, belong to the line
	// before.
	for i := 0; i < len(lines)-1; i++ {
		lines[i].End = max(lines[i].End, lines[i+1].Start)
	}
	if !truncated {
		lines[len(lines)-1].End = end
	}
	return out, truncated
}

// glyphCount returns how many glyphs a shaped line has.
func glyphCount(sl shapedLine) int {
	n := 0
	for _, run := range sl.runs {
		n += len(run.glyphs)
	}
	return n
}

// line positions a line of the paragraph text, which starts at rune
// offset of the layout, with its top at *y, and moves *y past it. Its
// glyphs go to room, which has room for those of sl; buf is a buffer.
func line(p Params, text []rune, sl shapedLine, offset int, rtl bool, m lineMetrics, y *float32, room []Glyph, buf *[]int, spans []Span) Line {
	line := Line{Start: offset + sl.start, End: offset + sl.end, RTL: rtl, Ascent: m.ascent, Descent: m.descent}
	if cap(room) > 0 {
		line.Glyphs = room
	}
	// Whitespace ending a line takes no room, unless kept.
	trim := sl.end
	if !p.KeepSpaces {
		for trim > sl.start && unicode.IsSpace(text[trim-1]) {
			trim--
		}
	}
	size := p.Style.FontSize()
	x0 := float32(math.MaxFloat32)
	starts := *buf
	// The width adds up in float64, as engines move their pens: a float32
	// sum of a long line's advances falls short of the engine's own width
	// by more than Core Text forgives as it breaks the text at it again.
	var width float64
	for _, run := range sl.runs {
		runSize := size
		if f := run.font; f != nil {
			line.Ascent = max(line.Ascent, f.Ascent)
			line.Descent = max(line.Descent, f.Descent)
			if f.Size > 0 {
				runSize = f.Size // a span of another size
			}
		}
		// A cluster holds the runes up to the next one of its run.
		starts = starts[:0]
		for _, g := range run.glyphs {
			if g.Cluster >= 0 {
				starts = append(starts, g.Cluster)
			}
		}
		slices.Sort(starts)
		starts = slices.Compact(starts)
		for _, g := range run.glyphs {
			if g.Cluster < 0 { // the ellipsis
				g.Cluster = line.End
			} else {
				if g.Cluster >= trim && g.Cluster < sl.end {
					continue
				}
				next := run.end
				if i, _ := slices.BinarySearch(starts, g.Cluster); i+1 < len(starts) {
					next = starts[i+1]
				}
				g.Runes = max(next-g.Cluster, 1)
				if o := objectAt(spans, g.Cluster); o != nil {
					// An inline object stands on the baseline, as high as
					// it is, whatever font its run took.
					g.Object, g.Advance, g.ID = true, o.ObjectWidth, 0
					line.Ascent = max(line.Ascent, o.ObjectHeight)
				}
				g.Cluster += offset
			}
			g.Size = runSize
			x0 = min(x0, g.X)
			// Logical width follows the pen, not the glyphs' drawing
			// offsets. Pango may kern "11" by offsetting its second glyph
			// while keeping whole-pixel advances for line breaking.
			width += float64(g.Advance)
			line.Glyphs = append(line.Glyphs, g)
		}
	}
	if len(line.Glyphs) == 0 {
		x0 = 0
	}
	line.Width = float32(max(width, 0))
	line.Height = max(m.lineHeight, line.Ascent+line.Descent)
	line.Y = *y
	line.Baseline = *y + (line.Height-line.Ascent-line.Descent)/2 + line.Ascent
	for i := range line.Glyphs {
		line.Glyphs[i].X -= x0
		line.Glyphs[i].Y += line.Baseline
	}
	*y += line.Height
	*buf = starts
	return line
}

// objectAt returns the span of rune i if it makes it an inline object,
// else nil.
func objectAt(spans []Span, i int) *Span {
	k, _ := slices.BinarySearchFunc(spans, i, func(sp Span, i int) int {
		if sp.End <= i {
			return -1
		}
		return 1
	})
	if k < len(spans) && spans[k].ObjectWidth > 0 {
		return &spans[k]
	}
	return nil
}

// ellipsis ends text cut short, unless Params.Ellipsis gives another.
var ellipsis = []rune{'…'}

// ellipsize lays out the paragraph text from rune start on one line ending
// with an ellipsis: as many of its graphemes as fit the width with it.
func (s *System) ellipsize(text []rune, spans []Span, start int, p Params, rtl bool) shapedLine {
	e := s.engine()
	rest := text[start:]
	restSpans := spansIn(spans, start, len(text))
	mark := ellipsis
	if p.Ellipsis != "" {
		mark = []rune(p.Ellipsis)
	}
	cut := len(rest)
	if p.Width > 0 {
		key := markKey{p.Style, p.Ellipsis, rtl}
		ellipsis, ok := s.marks[key]
		if !ok {
			for _, l := range e.shape(mark, p.Style, nil, 0, rtl, false) {
				ellipsis = max(ellipsis, advance(l))
			}
			if len(s.marks) >= 256 {
				clear(s.marks)
			}
			s.marks[key] = ellipsis
		}
		// The advance of each cluster, at its first rune.
		s.advances = slices.Grow(s.advances[:0], len(rest))[:len(rest)]
		advances := s.advances
		clear(advances)
		for _, l := range e.shape(rest, p.Style, restSpans, 0, rtl, false) {
			for _, run := range l.runs {
				for _, g := range run.glyphs {
					if g.Cluster >= 0 && g.Cluster < len(rest) {
						advances[g.Cluster] += g.Advance
					}
				}
			}
		}
		var b Boundaries
		b.Reset(rest)
		w := float32(0)
		cut = 0
		for cut < len(rest) {
			next := b.NextGrapheme(cut)
			gw := float32(0)
			for _, a := range advances[cut:next] {
				gw += a
			}
			if w+gw+ellipsis > p.Width {
				break
			}
			w += gw
			cut = next
		}
	}
	for cut > 0 && unicode.IsSpace(rest[cut-1]) {
		cut--
	}
	t := append(append(s.ellipsized[:0], rest[:cut]...), mark...)
	s.ellipsized = t
	out := shapedLine{start: start, end: start + cut}
	// The ellipsis takes the style of the text it ends.
	tSpans := spansIn(restSpans, 0, cut)
	if n := len(tSpans); n > 0 && tSpans[n-1].End == cut {
		tSpans[n-1].End += len(mark)
	}
	for _, l := range e.shape(t, p.Style, tSpans, 0, rtl, false) {
		for _, run := range l.runs {
			run.start, run.end = start+min(run.start, cut), start+min(run.end, cut)
			for i := range run.glyphs {
				if g := &run.glyphs[i]; g.Cluster >= cut {
					g.Cluster = -1
				} else {
					g.Cluster += start
				}
			}
			out.runs = append(out.runs, run)
		}
	}
	return out
}

// advance returns the width a shaped line's glyphs take.
func advance(l shapedLine) float32 {
	var width float32
	for _, run := range l.runs {
		for _, g := range run.glyphs {
			width += g.Advance
		}
	}
	return max(width, 0)
}

// isRTL reports whether a paragraph is right-to-left: whether its first
// strongly directional rune is, else neutral.
func isRTL(para []rune, neutral bool) bool {
	for _, r := range para {
		switch {
		case unicode.In(r, unicode.Hebrew, unicode.Arabic, unicode.Syriac, unicode.Thaana, unicode.Nko, unicode.Samaritan, unicode.Mandaic, unicode.Adlam):
			return true
		case unicode.IsLetter(r):
			return false
		}
	}
	return neutral
}
