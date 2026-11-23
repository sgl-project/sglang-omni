package markdown

import (
	"strings"
	"sync"

	"github.com/egoist/mygo/internal/l10n"
	"github.com/egoist/mygo/ui"
	"github.com/egoist/mygo/ui/markdown/parser"
)

// Options are how views show Markdown: their theme, what links and images
// do, how blocks render, and how text being written comes in. Keep one in
// the app's state and give it to every view of a kind, as the messages of
// a chat.
type Options struct {
	// Theme is the look: DefaultTheme of the window's theme when nil,
	// whose values those the theme leaves zero take.
	Theme *Theme
	// Extensions is the syntax read beyond CommonMark: Default when zero.
	Extensions parser.Extensions

	// OnLink, when set, handles a click on a link, given its destination.
	// Without it, links to http, https and mailto addresses open as a
	// ui.Link opens them, a path without a scheme goes there in the
	// router around the view, a link to #a-heading scrolls to that
	// heading, and links of other schemes do nothing.
	OnLink func(url string)
	// LoadImage loads the image at url, on a goroutine of its own, for the
	// views to show; until it has, they show its description. Without
	// it, images show as their descriptions. FileImages loads files.
	LoadImage func(url string) (ui.ImageSource, error)

	// CodeBlock, when set, builds code blocks in place of DefaultCodeBlock,
	// which it can call to wrap it; Code builds the code blocks of the
	// languages it lists, as "mermaid", in place of either.
	CodeBlock func(c *ui.Context, b *CodeBlockView)
	Code      map[string]func(c *ui.Context, b *CodeBlockView)
	// WrapCode wraps the lines of code blocks rather than scrolling them
	// sideways.
	WrapCode bool
	// Math, when set, builds blocks of display math from their TeX, which
	// show as their TeX without it; math within text shows as its TeX.
	Math func(c *ui.Context, tex string)
	// HTML, when set, builds HTML blocks from their HTML, which show as
	// their HTML without it; within text, the tags of formatting (<b>,
	// <i>, <code>, <br>, <mark>, <sub>, …) apply, and other tags show as
	// text.
	HTML func(c *ui.Context, html string)
	// Block, when set, is asked first for each block, and builds it in
	// its place when it returns true.
	Block func(c *ui.Context, d *parser.Document, i int) bool

	// Reveal is how Stream shows text as it comes: DefaultReveal when nil.
	Reveal *Reveal
	// ShareSelection leaves the selection to a Selectable container around
	// the view, as one across a chat's messages, rather than giving the
	// view one of its own.
	ShareSelection bool

	images sync.Map // url → *imageEntry
}

// View shows src, finished Markdown, as it reads: CommonMark and what
// Options.Extensions add, parsed again only where src changed since the
// last frame. Its text is selectable, copied as text, as HTML, which
// keeps its formatting where it is pasted (a rich text editor, mail, a
// word processor), and as Markdown (transfer.Markdown), or, with the
// context menu's Copy as Markdown (Shift+Cmd+C, Shift+Ctrl+C on Linux and
// Windows), as Markdown's text. o may be nil.
//
//	markdown.View(c, app.readme, &app.md)
func View(c *ui.Context, src string, o *Options) ui.Element {
	return build(c, src, o, false)
}

// Stream shows src, Markdown still being written, as a language model's
// reply: text appended to src since the last frame is parsed alone, the
// end shows what its half-written syntax will most likely become (see
// Document.SetStreaming), and text comes in as Options.Reveal says,
// smoothly, fading in. Show the message with View once it is complete,
// in the same place, so that its parse carries on; what is still being
// revealed then still comes in.
//
//	if msg.done {
//		markdown.View(c, msg.text, &app.md)
//	} else {
//		markdown.Stream(c, msg.text, &app.md)
//	}
func Stream(c *ui.Context, src string, o *Options) ui.Element {
	return build(c, src, o, true)
}

// stateKey keys a view's state among its root's.
type stateKey struct{}

// viewState is what a view keeps from frame to frame.
type viewState struct {
	o     *Options
	doc   *parser.Document
	ext   parser.Extensions
	uiTh  ui.Theme
	def   *Theme
	th    Theme
	paras []paraCache
	// elems notes, for copying, the block each paragraph's element shows,
	// and, for a code block's, its first line.
	elems  []elemInfo
	reveal revealState
	codes  map[int]*codeState
	tables map[int]*tableState
	// scrollTo is a block to scroll into view, -1 for none.
	scrollTo int
	notes    map[string]int
	sb       spanBuilder
	fade     func(src int) float32
}

type paraCache struct {
	start, end int
	kind       parser.Kind
	rev        uint32
	base       spanStyle
	ok         bool
	p          para
}

type elemInfo struct {
	id    uint64
	block int
	line  int
}

func newState() *viewState { return &viewState{scrollTo: -1} }

func build(c *ui.Context, src string, o *Options, streaming bool) ui.Element {
	if o == nil {
		o = &defaultOptions
	}
	root := ui.Column(c)
	st := *ui.Local(root, stateKey{}, newState)
	st.o = o
	ext := o.Extensions
	if ext == 0 {
		ext = parser.Default
	}
	if st.doc == nil || st.ext != ext {
		st.doc, st.ext = parser.New(ext), ext
		st.paras = st.paras[:0]
	}
	st.theme(c)
	th := &st.th

	shown, revealing := st.reveal.step(c, src, streaming, o.Reveal)
	st.doc.SetStreaming(streaming || revealing)
	if !strings.HasPrefix(src[:shown], st.doc.Source()) {
		// Another text, not more of it: blocks where blocks were may
		// show other text.
		st.forget()
	}
	st.doc.Set(src[:shown])
	st.fade = st.reveal.fader(c, o.Reveal)

	root.FontSize(th.FontSize).LineHeight(th.LineHeight).TextColor(th.Text)
	if th.Font != "" {
		root.Font(th.Font)
	}
	if !o.ShareSelection {
		root.Selectable()
		root.CopyData(st.copyData)
		root.ContextMenu(func(m *ui.Menu) {
			m.EditItems()
			if m.Item(l10n.In(c.Locale()).CopyAsMarkdown()).Chosen() {
				c.Services().WriteClipboardData(markdownData(st.copyMarkdown(root.SelectedTexts())))
			}
		})
		if root.FocusWithin() && root.Shortcut(ui.Cmd|ui.Shift, ui.KeyC) {
			c.Services().WriteClipboardData(markdownData(st.copyMarkdown(root.SelectedTexts())))
		}
	}
	st.elems = st.elems[:0]
	st.numberNotes()
	root.Children(func() {
		st.blocks(c, 0, false, spanStyle{})
		st.footnotes(c)
	})
	if revealing || st.reveal.fading() {
		c.AnimationFrame()
	}
	return root
}

var defaultOptions Options

// theme resolves the theme of the frame.
func (st *viewState) theme(c *ui.Context) {
	t := c.Theme()
	if st.def == nil || st.uiTh != *t {
		st.uiTh = *t
		st.def = DefaultTheme(t)
	}
	th := resolve(st.o.Theme, st.def)
	if th != st.th {
		st.th = th
		clear(st.paras)
		for _, cs := range st.codes {
			cs.chunks = cs.chunks[:0]
		}
	}
	st.sb.d, st.sb.th = st.doc, &st.th
}

// forget drops what the view made of its blocks.
func (st *viewState) forget() {
	clear(st.paras)
	clear(st.codes)
	clear(st.tables)
}

func (st *viewState) note(id uint64, block, line int) {
	st.elems = append(st.elems, elemInfo{id, block, line})
}

// blocks builds the children of block parent, spaced; tight tells the
// items of a tight list.
func (st *viewState) blocks(c *ui.Context, parent int, tight bool, base spanStyle) {
	d := st.doc
	th := &st.th
	first := true
	prevHeading := 0
	for j := parent + 1; j < d.Block(parent).Next; j = d.Block(j).Next {
		b := d.Block(j)
		if b.Kind == parser.FootnoteDefinition {
			continue
		}
		top := float32(0)
		if !first {
			top = th.em(th.BlockSpacing)
			if tight {
				top = th.em(th.ItemSpacing)
			}
			if prevHeading > 0 {
				top = th.em(th.Headings[prevHeading-1].Bottom)
			}
			if b.Kind == parser.Heading {
				top = max(top, th.em(th.Headings[b.Level-1].Top))
			}
		}
		el := st.block(c, j, tight, base)
		st.fadeIn(el, b)
		if top > 0 {
			el.Margin(top, 0, 0, 0)
		}
		if st.scrollTo == j {
			el.ScrollIntoView()
			st.scrollTo = -1
		}
		prevHeading = 0
		if b.Kind == parser.Heading {
			prevHeading = b.Level
		}
		first = false
	}
}

// block builds block i.
func (st *viewState) block(c *ui.Context, i int, tight bool, base spanStyle) ui.Element {
	d := st.doc
	o := st.o
	th := &st.th
	b := d.Block(i)
	if o.Block != nil {
		box := ui.Column(c)
		done := false
		box.Children(func() { done = o.Block(c, d, i) })
		if done {
			return box
		}
		return box.Children(func() { st.defaultBlock(c, i, tight, base) })
	}
	_ = b
	_ = th
	return st.defaultBlock(c, i, tight, base)
}

func (st *viewState) defaultBlock(c *ui.Context, i int, tight bool, base spanStyle) ui.Element {
	d := st.doc
	th := &st.th
	b := d.Block(i)
	switch b.Kind {
	case parser.Paragraph:
		if imagesOnly(d.Inlines(i)) {
			return st.images(c, i)
		}
		return st.paragraph(c, i, base)
	case parser.Heading:
		h := th.Headings[b.Level-1]
		s := base
		s.weight, s.size, s.color = h.Weight, th.em(h.Size), h.Color
		el := st.paragraph(c, i, s).Role(ui.RoleHeading).Level(b.Level)
		if h.Rule {
			el.BorderWidth(0, 0, 1, 0).BorderColor(th.Rule).Padding(0, 0, th.em(0.3), 0)
		}
		return el
	case parser.ThematicBreak:
		return ui.Box(c).Height(1).Background(th.Rule).AlignSelf(ui.Stretch).Margin(th.em(0.25), 0)
	case parser.CodeBlock:
		return st.codeBlock(c, i)
	case parser.HTMLBlock:
		html := strings.TrimRight(d.Lines(i), "\n")
		if st.o.HTML != nil {
			box := ui.Column(c)
			return box.Children(func() { st.o.HTML(c, html) })
		}
		el := ui.Text(c, html).Font(th.CodeFont).FontSize(th.em(th.CodeSize)).TextColor(th.Muted)
		st.note(el.ID(), i, 0)
		return el
	case parser.MathBlock:
		tex := strings.TrimRight(d.Lines(i), "\n")
		if st.o.Math != nil {
			box := ui.Column(c)
			return box.Children(func() { st.o.Math(c, tex) })
		}
		box := ui.ScrollHorizontal(c).AlignSelf(ui.Stretch)
		return box.Children(func() {
			el := ui.Text(c, tex).Font(th.CodeFont).FontSize(th.em(th.CodeSize)).TextColor(th.Muted).NoWrap().Margin(0, ui.Auto)
			st.note(el.ID(), i, 0)
		})
	case parser.Quote:
		s := base
		s.color = th.QuoteText
		// The bar on the side lines start: the right right to left.
		q := ui.Column(c).BorderStartWidth(th.QuoteBarWidth).BorderColor(th.QuoteBar).PaddingStart(th.em(0.85)).TextColor(th.QuoteText)
		return q.Children(func() { st.blocks(c, i, false, s) })
	case parser.List:
		return st.list(c, i, base)
	case parser.Table:
		return st.table(c, i, base)
	}
	return ui.Box(c)
}

// fadeIn fades in a block revealed a moment ago: text fades by its
// spans, and the rest of blocks, as a code block's box or a table's
// lines, by their opacity.
func (st *viewState) fadeIn(el ui.Element, b *parser.Block) {
	if st.fade == nil {
		return
	}
	switch b.Kind {
	case parser.Paragraph, parser.Heading, parser.List:
		return
	}
	if a := st.fade(b.Start); a < 1 {
		el.Opacity(a)
	}
}

// para returns the paragraph of block i in style base, built anew only
// when it changed.
func (st *viewState) para(i int, base spanStyle) *para {
	for len(st.paras) < st.doc.Len() {
		st.paras = append(st.paras, paraCache{})
	}
	b := st.doc.Block(i)
	st.doc.Inlines(i) // their revision
	pc := &st.paras[i]
	fading := st.fade != nil && st.fade(max(b.End-1, b.Start)) < 1
	if pc.ok && pc.start == b.Start && pc.end == b.End && pc.kind == b.Kind && pc.rev == b.Revision() && pc.base == base && !pc.p.fading && !fading && !b.Healed() {
		return &pc.p
	}
	pc.start, pc.end, pc.kind, pc.rev, pc.base, pc.ok = b.Start, b.End, b.Kind, b.Revision(), base, true
	sb := &st.sb
	sb.block = i
	sb.fade = st.fade
	sb.notes = st.noteNumber
	sb.build(&pc.p, base)
	return &pc.p
}

// paragraph builds the text of block i: a paragraph, a heading or a cell.
func (st *viewState) paragraph(c *ui.Context, i int, base spanStyle) ui.Element {
	p := st.para(i, base)
	ps := p.pieces
	var el ui.Element
	switch {
	case len(ps) == 0:
		el = ui.Text(c, "")
	case ps[0].link == -1:
		el = ui.RichText(c, ps[0].spans...)
		ps = ps[1:]
	default:
		el = ui.RichText(c)
	}
	if len(ps) > 0 {
		el.Children(func() {
			for k := range ps {
				st.piece(c, i, &ps[k])
			}
		})
	}
	st.note(el.ID(), i, -1)
	return el
}

// piece builds a link, or text after one, within a paragraph.
func (st *viewState) piece(c *ui.Context, block int, pc *piece) {
	switch {
	case pc.link == -1:
		ui.RichText(c, pc.spans...)
	case pc.link < -1:
		// A footnote reference: it scrolls to the footnote.
		in := st.doc.Inlines(block)[-2-pc.link]
		l := ui.Link(c, "", "")
		l.Children(func() { ui.RichText(c, pc.spans...) })
		if l.Clicked() {
			if i, ok := st.doc.Footnote(in.Text); ok {
				st.scrollTo = i
			}
		}
	default:
		in := st.doc.Inlines(block)[pc.link]
		st.link(c, &in, func() { ui.RichText(c, pc.spans...) })
	}
}

// link builds a link to in.Dest whose text children builds.
func (st *viewState) link(c *ui.Context, in *parser.Inline, children func()) {
	dest := in.Dest
	var l ui.Element
	switch {
	case in.Pending:
		l = ui.Link(c, "", "")
	case st.o.OnLink != nil:
		l = ui.Link(c, "", "")
		if l.Clicked() {
			st.o.OnLink(dest)
		}
	case strings.HasPrefix(dest, "#"):
		l = ui.Link(c, "", "")
		if l.Clicked() {
			st.scrollTo = st.headingNamed(dest[1:])
		}
	case linkAllowed(dest, c.Router() != nil):
		l = ui.Link(c, "", dest)
	default:
		l = ui.Link(c, "", "")
	}
	l.TextColor(st.th.Link)
	l.Children(children)
	switch {
	case in.Title != "":
		l.Tooltip(in.Title)
	case !in.Auto && !in.Pending && dest != "" && !strings.HasPrefix(dest, "#"):
		l.Tooltip(dest)
	}
}

// linkAllowed reports whether a link's destination opens without
// OnLink: one ui.SafeToOpen allows, or, in a router, a path without a
// scheme, which goes there. Outside a router, a path would open as a file
// or a program, so it does nothing.
func linkAllowed(dest string, router bool) bool {
	if strings.ContainsAny(dest, "\\\x00") || strings.HasPrefix(dest, "//") {
		return false
	}
	if i := strings.IndexAny(dest, ":/?#"); i < 0 || dest[i] != ':' {
		return router
	}
	return ui.SafeToOpen(dest)
}

// headingNamed returns the heading whose slug, as GitHub makes it, is
// name, or -1.
func (st *viewState) headingNamed(name string) int {
	d := st.doc
	for i := 1; i < d.Len(); i++ {
		if d.Block(i).Kind == parser.Heading && Slug(parser.PlainText(d.Inlines(i))) == strings.ToLower(name) {
			return i
		}
	}
	return -1
}

// Slug returns the anchor of a heading of text, as GitHub makes it: in
// lower case, spaces as hyphens, without punctuation.
func Slug(text string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(strings.TrimSpace(text)) {
		switch {
		case r == ' ':
			b.WriteByte('-')
		case r == '-' || r == '_' || isAlnum(r):
			b.WriteRune(r)
		}
	}
	return b.String()
}

// list builds a list.
func (st *viewState) list(c *ui.Context, i int, base spanStyle) ui.Element {
	d := st.doc
	th := &st.th
	b := d.Block(i)
	depth := 0
	for p := b.Parent; p > 0; p = d.Block(p).Parent {
		if d.Block(p).Kind == parser.List {
			depth++
		}
	}
	width := th.em(th.ListIndent)
	if b.Ordered {
		last := b.Number + max(0, st.countChildren(i)-1)
		w, _ := c.MeasureText(0, ui.Span{Text: itoa(last) + string(b.Marker), Font: th.Font, Size: th.FontSize})
		width = max(width, w+th.em(0.5))
	}
	col := ui.Column(c).Role(ui.RoleList)
	return col.Children(func() {
		n := b.Number
		first := true
		for j := i + 1; j < d.Block(i).Next; j = d.Block(j).Next {
			item := d.Block(j)
			row := ui.Row(c).AlignItems(ui.Start).Role(ui.RoleListItem)
			if !first {
				gap := th.BlockSpacing
				if b.Tight {
					gap = th.ItemSpacing
				}
				row.Margin(th.em(gap), 0, 0, 0)
			}
			num := n
			row.Children(func() {
				st.marker(c, b, item, num, depth, width)
				ui.Column(c).Grow(1).Shrink(1).MinWidth(0).Children(func() { st.blocks(c, j, b.Tight, base) })
			})
			n++
			first = false
		}
	})
}

func (st *viewState) countChildren(i int) int {
	n := 0
	for j := i + 1; j < st.doc.Block(i).Next; j = st.doc.Block(j).Next {
		n++
	}
	return n
}

// marker builds an item's marker: a bullet, its number, or its check box.
func (st *viewState) marker(c *ui.Context, list, item *parser.Block, n, depth int, width float32) {
	th := &st.th
	box := ui.Row(c).Width(width).Shrink(0).Unselectable()
	if st.fade != nil {
		if a := st.fade(item.Start); a < 1 {
			box.Opacity(a)
		}
	}
	box.Children(func() {
		switch {
		case item.Task != parser.NoTask:
			size := th.em(0.95)
			lineH := th.FontSize * th.LineHeight * 1.2
			checked := item.Task == parser.Checked
			cb := ui.Box(c).Size(size, size).Radius(max(3, size/5)).Margin(max(0, (lineH-size)/2-1), 0, 0, 0).Role(ui.RoleCheckBox).Checked(checked)
			if checked {
				cb.Background(th.Checkbox).Draw(func(p *ui.Painter, r ui.Rect) {
					path := &ui.Path{}
					path.MoveTo(r.X+r.W*0.25, r.Y+r.H*0.52)
					path.LineTo(r.X+r.W*0.43, r.Y+r.H*0.7)
					path.LineTo(r.X+r.W*0.76, r.Y+r.H*0.32)
					p.StrokePath(path, max(1.5, r.W/9), ui.RGB(255, 255, 255))
				})
			} else {
				cb.Border(1.5, th.Muted)
			}
		case list.Ordered:
			ui.Text(c, itoa(n)+string(list.Marker)).TextColor(th.Muted).Width(width - th.em(0.4)).TextAlign(ui.End).FontFeatures("tnum")
		default:
			bullets := [...]string{"•", "◦", "▪"}
			ui.Text(c, bullets[min(depth, 2)]).TextColor(th.Muted).Width(width - th.em(0.5)).TextAlign(ui.Center)
		}
	})
}

// numberNotes numbers the footnotes by the order of their definitions.
func (st *viewState) numberNotes() {
	clear(st.notes)
	n := 0
	for i := 1; i < st.doc.Len(); i++ {
		if b := st.doc.Block(i); b.Kind == parser.FootnoteDefinition {
			if st.notes == nil {
				st.notes = map[string]int{}
			}
			key := parser.NormalizeLabel(b.Label())
			if _, ok := st.notes[key]; !ok {
				n++
				st.notes[key] = n
			}
		}
	}
}

func (st *viewState) noteNumber(label string) int { return st.notes[parser.NormalizeLabel(label)] }

// footnotes builds the footnotes, below a rule, in order.
func (st *viewState) footnotes(c *ui.Context) {
	if len(st.notes) == 0 {
		return
	}
	d := st.doc
	th := &st.th
	ui.Box(c).Height(1).Background(th.Rule).AlignSelf(ui.Stretch).Margin(th.em(1.2), 0, th.em(0.6), 0)
	base := spanStyle{color: th.Muted, size: th.em(0.875)}
	for i := 1; i < d.Len(); i++ {
		b := d.Block(i)
		if b.Kind != parser.FootnoteDefinition {
			continue
		}
		num := st.notes[parser.NormalizeLabel(b.Label())]
		row := ui.Row(c).AlignItems(ui.Start).Margin(th.em(0.25), 0, 0, 0)
		if st.scrollTo == i {
			row.ScrollIntoView()
			st.scrollTo = -1
		}
		row.Children(func() {
			ui.Text(c, itoa(num)+".").TextColor(th.Muted).FontSize(th.em(0.875)).Width(th.em(1.6)).Shrink(0).Unselectable()
			ui.Column(c).Grow(1).Shrink(1).MinWidth(0).TextColor(th.Muted).Children(func() { st.blocks(c, i, false, base) })
		})
	}
}
