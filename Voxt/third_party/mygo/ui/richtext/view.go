package richtext

import (
	"slices"
	"time"
	"unicode/utf16"
	"unicode/utf8"
	"unsafe"

	"github.com/egoist/mygo/internal/l10n"
	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/internal/rope"
	"github.com/egoist/mygo/ui"
)

// EditorOptions configure an Editor.
type EditorOptions struct {
	// Label names the editor to assistive technology; "Rich text" when
	// empty.
	Label string
	// Placeholder shows while the text is empty.
	Placeholder string
	// Font is the text's: its zero fields take the theme's font and size.
	Font ui.Font
	// ReadOnly shows the text without letting the user change it: it is
	// selected and copied, and its links and tasks are clicked.
	ReadOnly bool
	// NoMarkdown turns off the Markdown shortcuts: "# " at the start of a
	// paragraph making a heading, **bold** making bold, and the others.
	NoMarkdown bool
	// OnLink opens the URL of a link clicked, instead of the system,
	// which opens only those ui.SafeToOpen allows.
	OnLink func(url string)
	// OnLinkCommand runs for ⌘K (Ctrl+K), as an app asks for the URL of a
	// link in a popover and sets it with Document.SetLink. Without it,
	// ⌘K links a selected URL to itself.
	OnLinkCommand func()
	// TagStyle styles the runs that carry a Tag, as an app shows mentions
	// or comments.
	TagStyle func(tag string, s *ui.Span)
	// Theme sets how the text looks: colors, code, headings, lists and
	// quotations. What it leaves zero, and all of it when nil, takes the
	// window theme's DefaultTheme.
	Theme *Theme
}

// Editor creates an editor of doc's attributed text, as NSTextView edits
// attributed text: bold, italic, underline, strikethrough, code, fonts,
// sizes and colors, links, headings, lists, tasks, quotations, blocks of
// code, alignment and attachments, with the shortcuts of word processors
// (⌘B, ⌘I, ⌘U, ⌘K, ⇧⌘7, ⇧⌘8, ⌘], …), undo of formats with the text, rich
// copy and paste, and input methods composing in the typing attributes.
// The app keeps doc from frame to frame:
//
//	doc := richtext.NewDocument(richtext.NewText("Hello"))
//	// In the view:
//	if richtext.Editor(c, doc).Fill().Changed() {
//		app.save(doc.Text())
//	}
//
// A toolbar formats the selection through doc (doc.ToggleBold,
// doc.SetHeading, …) and shows its formats with doc.Formatting. Changed
// reports the edits the user made in the editor, not those the app made
// through doc. It lays out and draws only the paragraphs in view, so a
// long document edits as a short one.
func Editor(c *ui.Context, doc *Document, options ...EditorOptions) ui.Element {
	t := c.Theme()
	e := build(c, doc, options, true)
	e.Radius(t.Radius).Background(t.Surface).Border(1, t.Border).MinHeight(t.Space(30))
	if e.Focused() {
		e.Border(1, t.Accent)
	}
	return e
}

// EditorBase creates an Editor without padding, background, border or
// corners, for an editor of the app's own look.
func EditorBase(c *ui.Context, doc *Document, options ...EditorOptions) ui.Element {
	return build(c, doc, options, false)
}

// view is the state of an editor: the text it shows, which follows its
// document, the layouts of its paragraphs, where it is scrolled to, the
// selection and an input method's composition.
type view struct {
	doc  *Document
	opts EditorOptions
	// shown is the text laid out, of the document's version and
	// selVersion.
	shown      Text
	version    uint64
	selVersion uint64
	paras      []paraView
	hs         heights
	laid       int
	look       look
	base       ui.Font
	// width is the content's, which the layouts wrap at; padX and padY
	// the room around the text in the element.
	width, viewH float32
	// laidLook is the look the layouts were made in.
	laidLook    lookKey
	padX, padY  float32
	rect        ui.Rect
	scroll      float64
	anchor      int
	anchorOff   float64
	reveal      bool
	first, last int
	// The selection, in runes: anchor and caret, and the caret's visual
	// edge.
	selAnchor, caret int
	affinity         ui.TextAffinity
	desiredX         float32
	hasDesired       bool
	// An input method's composition, shown at the caret.
	compose         string
	composeCaret    int
	composeSelected ui.TextInputRange
	composing       bool
	// mode is how the next edit joins the steps of undo, coalesce whether
	// typing goes on.
	mode     editMode
	coalesce bool
	// The pointer: a drag selecting by runes, words or paragraphs, from
	// where it started; a scroll bar's thumb held; a link pressed.
	dragging  bool
	dragUnit  int
	dragStart [2]int
	pressMods ui.Modifiers
	pressLink string
	// cursor is the pointer's shape where it hovers: an arrow over task
	// boxes and the scroll bar, a hand over links, an I-beam over text.
	cursor     ui.Cursor
	thumb      bool
	thumbGrab  float32
	focused    bool
	blinkStart time.Time
	services   ui.Services
	// changed is set by an edit of the user, which the next build reports.
	changed bool
	// numbers caches the numbers of the items of numbered lists, for the
	// text of numbersOf.
	numbers   map[int]int
	numbersOf Text
}

type editMode uint8

const (
	newStep editMode = iota
	typingStep
	sameStep
)

// paraView is what an editor made of a paragraph: its layout, for the
// runs, style and composition it was made of, the UTF-16 offset of each of
// its runes, and its height with the room around it, 0 until laid out.
type paraView struct {
	layout *ui.TextLayout
	units  []int
	// marker is the list marker painted beside the paragraph, laid out.
	marker       string
	markerLayout *ui.TextLayout
	id           uint64
	runs         *attrRun
	nruns        int
	style        ParagraphStyle
	compose      string
	at           int
	h            float32
	// space is the space above and below it, in h.
	space float32
}

// look is how the text of an editor looks: its font and the theme's
// colors, set as it builds.
type look struct {
	Theme
	size   float32
	indent float32  // DIPs a level of indentation takes
	border ui.Color // around an attachment not loaded yet
	// width is the room of the paragraph being laid out, which
	// attachments shrink to.
	width    float32
	tagStyle func(tag string, s *ui.Span)
}

func build(c *ui.Context, doc *Document, options []EditorOptions, styled bool) ui.Element {
	if doc == nil {
		panic("richtext: Editor requires a non-nil Document")
	}
	t := c.Theme()
	e := ui.Box(c).Focusable().Clip()
	v := ui.Local(e, "richtext", func() view { return view{cursor: ui.CursorText} })
	if v.doc != doc {
		*v = view{doc: doc, reveal: true}
		doc.lock()
		v.selAnchor, v.caret = doc.anchor, doc.caret
		v.selVersion = doc.selVersion
		doc.mu.Unlock()
	}
	v.opts = EditorOptions{}
	if len(options) > 0 {
		v.opts = options[0]
	}
	v.services = c.Services()
	v.setLook(t, styled)
	v.sync()
	focused := e.Focused()
	if focused != v.focused {
		v.focused = focused
		v.blinkStart = c.Now()
		if !focused {
			v.dragging, v.thumb = false, false
		}
	}
	if v.changed {
		v.changed = false
		e.SetChanged()
	}
	label := v.opts.Label
	if label == "" {
		label = l10n.In(c.Locale()).RichText()
	}
	e.Role(ui.RoleTextField).Label(label).Access(v.access)
	if !v.opts.ReadOnly {
		e.HandleTextInput(v)
	}
	e.HandleInput(v.input)
	e.Cursor(v.cursor)
	e.Draw(v.paint)
	e.ContextMenu(v.menu)
	if styled {
		e.FocusRing(false)
	}
	if focused && !v.opts.ReadOnly {
		// The caret blinks for half a minute after the last change.
		const blink = 530 * time.Millisecond
		if phase := c.Now().Sub(v.blinkStart); phase < 30*time.Second {
			c.After(blink - phase%blink)
		}
	}
	return e
}

// setLook sets how the text looks, from the editor's font and the theme.
func (v *view) setLook(t *ui.Theme, styled bool) {
	f := v.opts.Font
	if f.Size <= 0 {
		f.Size = t.FontSize
	}
	if f.Size <= 0 {
		f.Size = 14
	}
	if f.Family == "" {
		f.Family = t.Font
	}
	v.base = f
	def := defaultTheme(t)
	th := resolve(v.opts.Theme, &def)
	v.look = look{
		Theme:    th,
		size:     f.Size,
		indent:   f.Size * th.Indent,
		border:   t.Border,
		tagStyle: v.opts.TagStyle,
	}
	if styled {
		v.padX, v.padY = th.Padding[0], th.Padding[1]
	} else {
		v.padX, v.padY = 0, 0
	}
}

// sync gets the editor in step with its document: the paragraphs that
// changed give up their layouts, and a selection the app set is taken.
func (v *view) sync() {
	v.doc.lock()
	defer v.doc.mu.Unlock()
	v.lockedSync()
}

// lockedSync is sync, with the document's lock held. It returns the change
// it took, and whether there was one the log described.
func (v *view) lockedSync() (change, bool) {
	d := v.doc
	var c change
	moved := false
	if d.version != v.version {
		c, moved = d.since(v.version)
		if moved && v.version != 0 && len(v.paras) == v.shown.ParagraphCount() {
			v.replaced(c, d.text)
		} else {
			v.shown = d.text
			v.paras = make([]paraView, d.text.ParagraphCount())
			v.laid = 0
			v.resetHeights()
			moved = false
		}
		v.shown, v.version = d.text, d.version
		n := v.shown.Len()
		if moved {
			v.selAnchor = mapPosition(v.selAnchor, c.start, c.oldEnd-c.start, c.newEnd-c.start, v.selAnchor > v.caret)
			v.caret = mapPosition(v.caret, c.start, c.oldEnd-c.start, c.newEnd-c.start, v.caret > v.selAnchor)
		}
		v.selAnchor, v.caret = min(v.selAnchor, n), min(v.caret, n)
	}
	if d.selVersion != v.selVersion {
		v.selVersion = d.selVersion
		v.selAnchor, v.caret = d.anchor, d.caret
		v.affinity = ui.TextDownstream
		v.hasDesired = false
		v.compose, v.composing = "", false
		v.composeCaret, v.composeSelected = 0, ui.TextInputRange{}
		v.reveal = true
	}
	return c, moved
}

// replaced takes a change of the text: the paragraphs from the one holding
// c.start to the one holding c.oldEnd of the text shown give way to those
// of now, without layouts.
func (v *view) replaced(c change, now Text) {
	pa, pz := v.shown.paraAt(c.start), v.shown.paraAt(c.oldEnd)
	npz := now.paraAt(c.newEnd)
	for _, p := range v.paras[pa : pz+1] {
		if p.layout != nil {
			v.laid--
		}
	}
	if npz == pz {
		// As many paragraphs: they give up their heights in place, as a
		// keystroke's does, rather than summing all the others again.
		for p := pa; p <= pz; p++ {
			if h := v.paras[p].h; h > 0 {
				v.hs.add(p, -float64(h), 1)
			}
			v.paras[p] = paraView{}
		}
	} else {
		added := make([]paraView, npz-pa+1)
		v.paras = slices.Replace(v.paras, pa, pz+1, added...)
		v.resetHeights()
	}
	v.shown = now
	switch {
	case v.anchor > pz:
		v.anchor += npz - pz
	case v.anchor > pa:
		v.anchor, v.anchorOff = pa, 0
	}
	v.numbersOf = Text{}
}

// resetHeights sums the heights of the paragraphs laid out.
func (v *view) resetHeights() {
	hs := make([]float32, len(v.paras))
	for i := range v.paras {
		hs[i] = v.paras[i].h
	}
	v.hs.reset(hs)
	v.estimate()
}

// estimate sets the height of paragraphs not laid out yet to the average
// of those that are, or a line of text.
func (v *view) estimate() {
	m, unknown := v.hs.totals()
	if n := len(v.paras) - unknown; n > 0 {
		v.hs.est = m / float64(n)
	} else {
		v.hs.est = float64(v.look.size) * 1.3
	}
}

// para returns paragraph p of the text shown.
func (v *view) para(p int) *richPara { return seqAt(v.shown.seq(), p) }

// paraFont returns the font of a paragraph of style st.
func (v *view) paraFont(st ParagraphStyle) ui.Font {
	f := v.base
	switch {
	case st.Heading > 0:
		h := v.look.heading(st.Heading)
		f.Size *= h.Size
		f.Weight = h.Weight
	case st.CodeBlock:
		f.Family = v.look.CodeFont
		f.Size *= v.look.CodeSize
	}
	return f
}

// lookKey is what of the look the layouts of paragraphs are made of.
type lookKey struct {
	base  ui.Font
	theme Theme
	tags  bool
}

func (v *view) lookKey() lookKey {
	return lookKey{base: v.base, theme: v.look.Theme, tags: v.look.tagStyle != nil}
}

// heading returns the style of headings of level, from 1.
func (look *look) heading(level int) *HeadingStyle {
	return &look.Headings[min(max(level, 1), 6)-1]
}

// insets returns the room left and right of paragraph p: its indentation,
// its list's marker, a quotation's bar, a block of code's padding.
func (v *view) insets(st ParagraphStyle) (left, right float32) {
	look := &v.look
	left = float32(st.Indent) * look.indent
	if st.List != NoList {
		left += look.indent
	}
	if st.Quote {
		left += look.indent * 0.75
	}
	if st.CodeBlock {
		left += look.size * look.CodeBlockPadding[0]
		right += look.size * look.CodeBlockPadding[0]
	}
	return left, right
}

// space returns the room above and below paragraph p: a heading's
// margins, a block of code's padding at its ends.
func (v *view) space(p int, st ParagraphStyle) (above, below float32) {
	look := &v.look
	switch {
	case st.Heading > 0:
		h := look.heading(st.Heading)
		if p > 0 {
			above = look.size * h.Size * h.Top
		}
		below = look.size * h.Bottom
	case st.CodeBlock:
		if p == 0 || !v.para(p-1).style.CodeBlock {
			above = look.size * look.CodeBlockPadding[1]
		}
		if p+1 >= len(v.paras) || !v.para(p+1).style.CodeBlock {
			below = look.size * look.CodeBlockPadding[1]
		}
	}
	return above, below
}

// span returns how a run with attributes a looks in a paragraph of style
// st.
func (look *look) span(text string, a Attributes, st ParagraphStyle) ui.Span {
	s := ui.Span{Text: text, Font: a.Font, Size: a.Size, Weight: a.Weight, Italic: a.Italic, Color: a.Color,
		Underline: a.Underline, Strikethrough: a.Strikethrough, Background: a.Background}
	if a.Code {
		s.Font = look.CodeFont
		if s.Size == 0 {
			s.Size = look.size * look.CodeSize
		}
		if s.Background.A == 0 {
			s.Background = look.CodeBackground
		}
		// A pill hugging the code, as Markdown views show it.
		s.Radius = max(3, s.Size/4)
	}
	if a.Link != "" {
		if s.Color.A == 0 {
			s.Color = look.Link
		}
		s.Underline = true
	}
	if st.List == Tasks && st.Checked {
		s.Strikethrough = true
		if s.Color.A == 0 {
			s.Color = look.Muted
		}
	}
	if st.Quote && s.Color.A == 0 {
		s.Color = look.QuoteText
	}
	if (a.Code || st.CodeBlock) && s.Color.A == 0 && look.Code != look.Text {
		s.Color = look.Code
	}
	if st.Heading > 0 && s.Color.A == 0 {
		if c := look.heading(st.Heading).Color; c != look.Text {
			s.Color = c
		}
	}
	if a.Tag != "" && look.tagStyle != nil {
		look.tagStyle(a.Tag, &s)
	}
	if at := a.Attachment; at != nil {
		// A box of the attachment's size, scaled down to the width.
		w, h := at.size()
		if look.width > 0 && w > look.width {
			w, h = look.width, h*look.width/w
		}
		s.ObjectWidth, s.ObjectHeight = w, h
	}
	return s
}

// lineOf returns the text of paragraph p, without its newline.
func (v *view) lineOf(p int) string { return rope.TextOf(v.shown.text).Line(p) }

// paraLayout returns paragraph p laid out, with the composition at the
// caret: it keeps the layout until the paragraph's runs or style change.
func (v *view) paraLayout(p int) *paraView {
	pv := &v.paras[p]
	para := v.para(p)
	compose, at := "", 0
	if v.compose != "" {
		if cp := v.shown.paraAt(v.caret); cp == p {
			compose, at = v.compose, v.caret-v.shown.paraStart(p)
		}
	}
	var first *attrRun
	if len(para.runs) > 0 {
		first = unsafe.SliceData(para.runs)
	}
	if pv.layout != nil && pv.id == para.id && pv.runs == first && pv.nruns == len(para.runs) && pv.style == para.style && pv.compose == compose && pv.at == at {
		// The space around it follows its neighbours, which may have
		// changed without it: a code block's last line, the first heading.
		if above, below := v.space(p, para.style); above+below != pv.space {
			h := pv.h - pv.space + above + below
			v.hs.add(p, float64(h-pv.h), 0)
			pv.h, pv.space = h, above+below
		}
		return pv
	}
	composed := compose
	left, right := v.insets(para.style)
	v.look.width = max(v.width-left-right, 1)
	line := v.lineOf(p)
	var spans []ui.Span
	var typing Attributes
	if compose != "" {
		v.doc.lock()
		typing = v.doc.typingAttributes()
		v.doc.mu.Unlock()
	}
	off, done := 0, 0
	add := func(n int, a Attributes) {
		end := off
		for k := 0; k < n && end < len(line); k++ {
			_, size := utf8.DecodeRuneInString(line[end:])
			end += size
		}
		if end > off {
			spans = append(spans, v.look.span(line[off:end], a, para.style))
		}
		off = end
	}
	for _, run := range para.runs {
		n := min(run.n, para.n-done)
		if compose != "" && done <= at && at < done+n {
			add(at-done, run.a)
			spans = append(spans, v.look.span(compose, typing, para.style))
			add(done+n-at, run.a)
			compose = ""
		} else {
			add(n, run.a)
		}
		done += n
		if done >= para.n {
			break
		}
	}
	if compose != "" {
		spans = append(spans, v.look.span(compose, typing, para.style))
	}
	shape := ui.TextShape{Width: v.look.width, Align: para.style.Align}
	l := ui.ShapeRichTextWith(spans, v.paraFont(para.style), shape)
	units := pv.units[:0]
	units = append(units, 0)
	for _, s := range spans {
		for _, r := range s.Text {
			units = append(units, units[len(units)-1]+utf16.RuneLen(r))
		}
	}
	above, below := v.space(p, para.style)
	_, lh := l.Size()
	h := above + lh + below
	if pv.layout == nil {
		v.laid++
	}
	if pv.h > 0 {
		v.hs.add(p, float64(h-pv.h), 0)
	} else {
		v.hs.add(p, float64(h), -1)
	}
	*pv = paraView{layout: l, units: units, id: para.id, runs: first, nruns: len(para.runs), style: para.style,
		compose: composed, at: at, h: h, space: above + below}
	return pv
}

// maxLaid is how many paragraphs keep their layouts out of view.
const maxLaid = 512

// origin returns where the layout of paragraph p starts in the content.
func (v *view) origin(p int) (x float32, y float64) {
	st := v.para(p).style
	left, _ := v.insets(st)
	above, _ := v.space(p, st)
	return left, v.hs.top(p) + float64(above)
}

// local returns where rune i of paragraph p is in its layout, which holds
// the composition at the caret.
func (v *view) local(p, i int) int {
	l := i - v.shown.paraStart(p)
	if v.compose != "" && i > v.caret && v.shown.paraAt(v.caret) == p {
		l += utf8.RuneCountInString(v.compose)
	}
	return l
}

// global returns the rune of the text at rune l of the layout of
// paragraph p.
func (v *view) global(p, l int) int {
	start := v.shown.paraStart(p)
	if v.compose != "" && v.shown.paraAt(v.caret) == p {
		n, c := utf8.RuneCountInString(v.compose), v.caret-start
		switch {
		case l > c+n:
			l -= n
		case l > c:
			l = c
		}
	}
	return min(start+l, start+v.para(p).n)
}

// runeOfUnit returns the rune of a paragraph's layout at UTF-16 unit u.
func (pv *paraView) runeOfUnit(u int) int {
	i, found := slices.BinarySearch(pv.units, u)
	if !found && i > 0 {
		i--
	}
	return min(i, len(pv.units)-1)
}

// caretAt returns the caret before rune i, extra runes into a composition
// there, in the content: its x, top and height.
func (v *view) caretAt(i, extra int, affinity ui.TextAffinity) (x float32, y float64, h float32) {
	p := v.shown.paraAt(i)
	pv := v.paraLayout(p)
	l := min(v.local(p, i)+extra, len(pv.units)-1)
	r := pv.layout.CaretAt(ui.TextCaretPosition{Index: pv.units[l], Affinity: affinity})
	ox, oy := v.origin(p)
	return ox + r.X, oy + float64(r.Y), r.H
}

// positionAt returns the caret position closest to (x, y) in the content.
func (v *view) positionAt(x float32, y float64) (int, ui.TextAffinity) {
	p, l, aff := v.layoutPositionAt(x, y)
	if p < 0 {
		return v.shown.Len(), ui.TextUpstream
	}
	return v.global(p, l), aff
}

// layoutPositionAt returns the paragraph closest to (x, y) in the content
// and the rune of its layout there, which holds the composition; p is -1
// past the paragraphs.
func (v *view) layoutPositionAt(x float32, y float64) (p, l int, aff ui.TextAffinity) {
	n := len(v.paras)
	y = max(y, 0)
	p = v.hs.at(y)
	for range n {
		pv := v.paraLayout(p)
		top := v.hs.top(p)
		switch {
		case y < top && p > 0:
			p--
		case y >= top+float64(pv.h) && p < n-1:
			p++
		default:
			ox, oy := v.origin(p)
			pos := pv.layout.PositionAt(ui.Point{X: x - ox, Y: float32(y - oy)})
			return p, pv.runeOfUnit(pos.Index), pos.Affinity
		}
	}
	return -1, 0, ui.TextUpstream
}

// lineEdges returns the edges of the visual line holding rune i.
func (v *view) lineEdges(i int) (int, int) {
	p := v.shown.paraAt(i)
	pv := v.paraLayout(p)
	r := pv.layout.CaretAt(ui.TextCaretPosition{Index: pv.units[min(v.local(p, i), len(pv.units)-1)], Affinity: v.affinity})
	mid := r.Y + r.H/2
	a := pv.layout.PositionAt(ui.Point{X: -1e6, Y: mid})
	z := pv.layout.PositionAt(ui.Point{X: 1e6, Y: mid})
	s, e := v.global(p, pv.runeOfUnit(a.Index)), v.global(p, pv.runeOfUnit(z.Index))
	if s > e {
		s, e = e, s
	}
	return s, e
}

// setScroll scrolls the content to y, keeping the place by the paragraph
// there.
func (v *view) setScroll(y float64) {
	v.scroll = y
	v.anchor = v.hs.at(y)
	v.anchorOff = y - v.hs.top(v.anchor)
}

// layout lays out the paragraphs in view of a content box w×h, and those
// the caret is in when it is to be revealed.
func (v *view) layout(w, h float32) {
	if key := v.lookKey(); w != v.width || key != v.laidLook {
		// Another width, font or theme: every paragraph anew.
		v.laidLook = key
		for i := range v.paras {
			v.paras[i] = paraView{}
		}
		v.laid = 0
		v.width = w
		v.resetHeights()
	}
	v.viewH = h
	v.estimate()
	v.place(h)
	if v.reveal {
		v.reveal = false
		_, y, ch := v.caretAt(v.caret, v.composeCaret, v.affinity)
		switch {
		case y < v.scroll:
			v.setScroll(y)
		case y+float64(ch) > v.scroll+float64(h):
			p := v.shown.paraAt(v.caret)
			for q, room := p-1, y+float64(ch)-v.hs.top(p); q >= 0 && room < float64(h); q-- {
				room += float64(v.paraLayout(q).h)
			}
			_, y, ch = v.caretAt(v.caret, v.composeCaret, v.affinity)
			v.setScroll(y + float64(ch) - float64(h))
		}
		v.place(h)
	}
	total := v.hs.top(len(v.paras))
	if s := max(0, min(v.scroll, total-float64(h))); s != v.scroll {
		v.setScroll(s)
		v.place(h)
	}
	if v.laid > maxLaid+v.last-v.first {
		v.forget()
	}
}

// place lays out the paragraphs in view, from the anchor down, and keeps
// the view where the anchor is.
func (v *view) place(h float32) {
	n := len(v.paras)
	v.anchor = max(0, min(v.anchor, n-1))
	v.paraLayout(v.anchor)
	v.first = v.anchor
	for v.first > 0 && v.hs.top(v.anchor)+v.anchorOff-v.hs.top(v.first) < 0 {
		v.first--
		v.paraLayout(v.first)
	}
	v.setScroll(max(0, v.hs.top(v.anchor)+v.anchorOff))
	y := v.hs.top(v.first)
	p := v.first
	for ; p < n; p++ {
		y += float64(v.paraLayout(p).h)
		if y >= v.scroll+float64(h) {
			break
		}
	}
	v.last = min(p, n-1)
}

// forget drops the layouts of the paragraphs far from view, keeping their
// heights.
func (v *view) forget() {
	lo, hi := v.first-maxLaid/4, v.last+maxLaid/4
	for i := range v.paras {
		if (i < lo || i > hi) && v.paras[i].layout != nil {
			v.paras[i].layout, v.paras[i].units = nil, nil
			v.paras[i].runs = nil
			v.laid--
		}
	}
}

// contentPoint returns the point of the content at (x, y) of the element.
func (v *view) contentPoint(x, y float32) (float32, float64) {
	return x - v.padX, float64(y-v.padY) + v.scroll
}

// selection returns the selection, in order.
func (v *view) selection() (int, int) { return min(v.selAnchor, v.caret), max(v.selAnchor, v.caret) }

// access describes the editor to assistive technology: its text, and the
// selection.
func (v *view) access(n *platform.AccessNode) {
	n.Value = v.shown.String()
	n.SelStart, n.SelEnd = v.selection()
	n.States |= platform.AccessMultiline
	if v.opts.ReadOnly {
		n.States |= platform.AccessReadOnly
	}
	if v.shown.Len() == 0 {
		n.Placeholder = v.opts.Placeholder
	}
	n.TextRuns = v.accessRuns(n.TextRuns[:0])
}

// accessRuns appends the runs of the text that have formats, as screen
// readers read them: the attributes as they show, and the headings.
func (v *view) accessRuns(out []platform.AccessTextRun) []platform.AccessTextRun {
	for start, p := range v.shown.Paragraphs() {
		font := v.paraFont(p.Style)
		at := start
		for _, r := range p.Runs {
			n := utf8.RuneCountInString(r.Text)
			s := v.look.span(r.Text, r.Attributes, p.Style)
			run := platform.AccessTextRun{Start: at, End: at + n, Family: s.Font, Size: s.Size, Weight: s.Weight,
				Italic: s.Italic, Underline: s.Underline, Strikethrough: s.Strikethrough, Link: r.Attributes.Link, Heading: p.Style.Heading}
			if run.Family == "" && font.Family != v.base.Family {
				run.Family = font.Family
			}
			if run.Size == 0 && font.Size != v.base.Size {
				run.Size = font.Size
			}
			if run.Weight == 0 && font.Weight > 0 {
				run.Weight = font.Weight
			}
			if s.Color.A > 0 {
				run.Color = [4]uint8{s.Color.R, s.Color.G, s.Color.B, s.Color.A}
			}
			if s.Background.A > 0 {
				run.Background = [4]uint8{s.Background.R, s.Background.G, s.Background.B, s.Background.A}
			}
			if run != (platform.AccessTextRun{Start: run.Start, End: run.End}) {
				// Neighbors alike are one run, or screen readers announce a
				// change of format that is not.
				if k := len(out) - 1; k >= 0 && out[k].End == run.Start && sameFormat(out[k], run) {
					out[k].End = run.End
				} else {
					out = append(out, run)
				}
			}
			at += n
		}
	}
	return out
}

// sameFormat reports whether two runs have the same formats.
func sameFormat(a, b platform.AccessTextRun) bool {
	a.Start, a.End, b.Start, b.End = 0, 0, 0, 0
	return a == b
}
