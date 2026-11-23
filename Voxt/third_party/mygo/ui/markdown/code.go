package markdown

import (
	"time"

	"github.com/egoist/mygo/internal/l10n"
	"github.com/egoist/mygo/ui"
	"github.com/egoist/mygo/ui/editor"
)

// CodeBlockView is a code block as Options.CodeBlock and Options.Code build
// it.
type CodeBlockView struct {
	// Language is the language its fence names, as "go", and Info the
	// whole info string after the fence.
	Language, Info string
	// Open tells a block still being written, whose closing fence has not
	// come: heavy renderers, as of diagrams, wait for it to close.
	Open bool
	// Index is the block's in the Document.
	Index int

	st *viewState
}

// Code returns the block's code, its lines ending with newlines.
func (b *CodeBlockView) Code() string { return b.st.doc.Lines(b.Index) }

// codeState is what a view keeps of a code block: the tokens of its lines
// and the spans of its chunks of lines, to highlight only what changed.
type codeState struct {
	lang     string
	tok      editor.Tokenizer
	lines    []string
	ends     []uint64
	toks     [][]editor.Token
	chunks   []codeChunk
	copiedAt time.Time
	seen     bool
}

// codeChunk is chunkLines lines of a code block, laid out as one text, so
// that a block growing as it streams lays out only its last chunk anew.
type codeChunk struct {
	spans []ui.Span
	n     int // lines
	ok    bool
}

const chunkLines = 32

func (st *viewState) codeState(i int) *codeState {
	b := st.doc.Block(i)
	if st.codes == nil {
		st.codes = map[int]*codeState{}
	}
	cs := st.codes[b.Start]
	lang := b.Language()
	if cs == nil || cs.lang != lang {
		cs = &codeState{lang: lang}
		if l := editor.LanguageNamed(lang); l != nil && lang != "" {
			cs.tok = l.Tokenizer
		}
		st.codes[b.Start] = cs
	}
	cs.seen = true
	return cs
}

// codeBlock builds code block i, through the hooks of the options.
func (st *viewState) codeBlock(c *ui.Context, i int) ui.Element {
	b := st.doc.Block(i)
	cb := &CodeBlockView{Language: b.Language(), Info: b.Info(), Open: b.Open, Index: i, st: st}
	if f := st.o.Code[cb.Language]; f != nil {
		box := ui.Column(c)
		return box.Children(func() { f(c, cb) })
	}
	if st.o.CodeBlock != nil {
		box := ui.Column(c)
		return box.Children(func() { st.o.CodeBlock(c, cb) })
	}
	return DefaultCodeBlock(c, cb)
}

// DefaultCodeBlock builds a code block as views do without hooks: a box
// with the language and a copy button above the highlighted code.
func DefaultCodeBlock(c *ui.Context, b *CodeBlockView) ui.Element {
	th := &b.st.th
	box := ui.Column(c).Direction(ui.LeftToRight).Background(th.CodeBlockBackground).Border(1, th.CodeBlockBorder).Radius(th.CodeBlockRadius).AlignSelf(ui.Stretch).Clip()
	box.Role(ui.RoleGroup).Label(codeLabel(b.Language))
	return box.Children(func() {
		head := ui.Row(c).AlignItems(ui.Center).Padding(th.em(0.35), th.em(0.4), 0, th.em(th.CodeBlockPadding))
		head.Children(func() {
			ui.Text(c, b.Language).FontSize(th.em(0.8)).TextColor(th.Muted).Unselectable()
			ui.Spacer(c)
			b.CopyButton(c)
		})
		b.Body(c)
	})
}

func codeLabel(lang string) string {
	if lang == "" {
		return "Code"
	}
	return "Code, " + lang
}

var (
	copyIcon  = ui.MustParseSVG([]byte(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3"><rect x="5.5" y="5.5" width="8" height="8" rx="1.6"/><path d="M10.5 3.6V3.2A1.7 1.7 0 0 0 8.8 1.5H4.2A1.7 1.7 0 0 0 2.5 3.2v4.6a1.7 1.7 0 0 0 1.7 1.7h.4"/></svg>`))
	checkIcon = ui.MustParseSVG([]byte(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 8.5l3 3 6-7"/></svg>`))
)

// CopyButton builds a button copying the block's code, which shows that
// it did for two seconds, and waits for a block being written to close.
func (b *CodeBlockView) CopyButton(c *ui.Context) ui.Element {
	st := b.st
	th := &st.th
	cs := st.codeState(b.Index)
	copied := !cs.copiedAt.IsZero() && c.Now().Sub(cs.copiedAt) < 2*time.Second
	tx := l10n.In(c.Locale())
	btn := ui.ButtonBase(c).Padding(th.em(0.2), th.em(0.45)).Radius(5).Gap(th.em(0.3)).Label(tx.CopyCode()).Disabled(b.Open)
	if btn.Hovered() && !b.Open {
		btn.Background(th.Text.Alpha(0.08))
	}
	btn.Children(func() {
		icon, label := copyIcon, tx.Copy()
		if copied {
			icon, label = checkIcon, tx.Copied()
		}
		size := th.em(0.95)
		ui.Icon(c, icon).Size(size, size).TextColor(th.Muted)
		ui.Text(c, label).FontSize(th.em(0.8)).TextColor(th.Muted)
	})
	if btn.Clicked() {
		c.WriteClipboard(b.Code())
		cs.copiedAt = c.Now()
		c.After(2 * time.Second)
	}
	return btn
}

// Body builds the block's code, highlighted, selectable with the view's
// text, scrolling sideways unless Options.WrapCode.
func (b *CodeBlockView) Body(c *ui.Context) ui.Element {
	st := b.st
	th := &st.th
	cs := st.codeState(b.Index)
	st.highlight(cs, b.Index)
	pad := th.em(th.CodeBlockPadding)
	var outer ui.Element
	if st.o.WrapCode {
		outer = ui.Column(c)
	} else {
		outer = ui.ScrollHorizontal(c)
	}
	// Code reads left to right in a right-to-left interface too, as in
	// editors.
	outer.Direction(ui.LeftToRight).AlignSelf(ui.Stretch).Font(th.CodeFont).FontSize(th.em(th.CodeSize)).LineHeight(th.CodeLineHeight)
	return outer.Children(func() {
		col := ui.Column(c).Padding(th.em(0.25), pad, pad, pad)
		if !st.o.WrapCode {
			col.Shrink(0)
		}
		col.Children(func() {
			for k := range cs.chunks {
				ch := &cs.chunks[k]
				el := ui.RichText(c, ch.spans...)
				if !st.o.WrapCode {
					el.NoWrap()
				}
				st.note(el.ID(), b.Index, k*chunkLines)
			}
		})
	})
}

// highlight tokenizes the lines of code block i that changed, and makes
// the spans of the chunks they are in anew.
func (st *viewState) highlight(cs *codeState, i int) {
	d := st.doc
	n := d.LineCount(i)
	// The first line that differs from those tokenized.
	k0 := 0
	for k0 < n && k0 < len(cs.lines) {
		line, _ := d.Line(i, k0)
		if line != cs.lines[k0] {
			break
		}
		k0++
	}
	if k0 == n && len(cs.lines) == n && len(cs.chunks) == (n+chunkLines-1)/chunkLines {
		return
	}
	cs.lines, cs.ends, cs.toks = cs.lines[:k0], cs.ends[:k0], cs.toks[:k0]
	state := uint64(0)
	if k0 > 0 {
		state = cs.ends[k0-1]
	}
	for k := k0; k < n; k++ {
		line, _ := d.Line(i, k)
		var toks []editor.Token
		if cs.tok != nil {
			toks, state = cs.tok.Tokenize(line, state, nil)
		}
		cs.lines = append(cs.lines, line)
		cs.ends = append(cs.ends, state)
		cs.toks = append(cs.toks, toks)
	}
	chunks := (n + chunkLines - 1) / chunkLines
	if n == 0 {
		chunks = 1
	}
	for len(cs.chunks) < chunks {
		cs.chunks = append(cs.chunks, codeChunk{})
	}
	cs.chunks = cs.chunks[:chunks]
	for k := k0 / chunkLines; k < chunks; k++ {
		cs.chunks[k].ok = false
	}
	ct := st.th.CodeColors
	for k := range cs.chunks {
		ch := &cs.chunks[k]
		if ch.ok {
			continue
		}
		ch.spans = ch.spans[:0]
		first := k * chunkLines
		last := min(first+chunkLines, n)
		for l := first; l < last; l++ {
			if l > first {
				ch.spans = append(ch.spans, ui.Span{Text: "\n"})
			}
			line := cs.lines[l]
			start := 0
			for _, t := range cs.toks[l] {
				end := min(t.End, len(line))
				if end > start {
					sp := ui.Span{Text: line[start:end], Color: ct.TokenColor(t.Kind)}
					if s := ct.Tokens[t.Kind]; s.Bold {
						sp.Weight = 600
					} else if s.Italic {
						sp.Italic = true
					}
					ch.spans = append(ch.spans, sp)
				}
				start = max(start, end)
			}
			if start < len(line) {
				ch.spans = append(ch.spans, ui.Span{Text: line[start:], Color: ct.Text})
			}
		}
		if len(ch.spans) == 0 {
			ch.spans = append(ch.spans, ui.Span{Text: " "})
		}
		ch.n = last - first
		ch.ok = true
	}
}
