package native

import (
	"strconv"
	"unicode"

	"github.com/egoist/mygo/plugins/updater/internal/markdown"
	"github.com/egoist/mygo/ui"
)

// notes builds release notes, in the manner of the page of package
// updater. Their text is selectable, and links open in the browser.
func notes(c *ui.Context, t *ui.Theme, blocks []markdown.Block) {
	n := &notesStyle{t: t, rtl: rightToLeft(blocks)}
	ui.Column(c).Children(func() { n.blocks(c, blocks, false, 0) })
}

// notesStyle is how release notes look. Like the page's notes, whose dir
// is auto, they are laid out from the right when the language they are
// written in is written from right to left, whatever the window's.
type notesStyle struct {
	t   *ui.Theme
	rtl bool
}

// rightToLeft reports whether the first strongly directional letter of
// blocks is of a script written from right to left, as text does.
func rightToLeft(blocks []markdown.Block) bool {
	var rtl, found bool
	var inlines func([]markdown.Inline)
	text := func(s string) {
		for _, r := range s {
			if found {
				return
			}
			switch {
			case unicode.In(r, unicode.Hebrew, unicode.Arabic, unicode.Syriac, unicode.Thaana, unicode.Nko, unicode.Samaritan, unicode.Mandaic, unicode.Adlam):
				rtl, found = true, true
			case unicode.IsLetter(r):
				found = true
			}
		}
	}
	inlines = func(ins []markdown.Inline) {
		for _, in := range ins {
			text(in.Text)
			inlines(in.Children)
		}
	}
	var walk func([]markdown.Block)
	walk = func(bs []markdown.Block) {
		for _, b := range bs {
			inlines(b.Inlines)
			text(b.Text)
			for _, item := range b.Items {
				walk(item)
			}
			walk(b.Blocks)
		}
	}
	walk(blocks)
	return rtl
}

// blocks builds blocks, apart by the room between paragraphs, or by none
// in the items of a list (tight). depth counts the lists around them.
func (n *notesStyle) blocks(c *ui.Context, blocks []markdown.Block, tight bool, depth int) {
	for i, b := range blocks {
		var gap float32
		switch {
		case i == 0 || tight:
		case b.Kind == markdown.Heading:
			gap = 12
		case blocks[i-1].Kind == markdown.Heading:
			gap = 4
		default:
			gap = 8
		}
		n.block(c, b, depth).Margin(gap, 0, 0, 0)
	}
}

// headingSizes are the sizes of headings of levels 1, 2 and the others,
// relative to the text.
var headingSizes = [...]float32{16.0 / 13, 14.0 / 13, 1}

func (n *notesStyle) block(c *ui.Context, b markdown.Block, depth int) ui.Element {
	t := n.t
	switch b.Kind {
	case markdown.Heading:
		size := headingSizes[min(b.Level, len(headingSizes))-1]
		return paragraph(c, t, b.Inlines).FontSize(t.Rem(size)).Bold()
	case markdown.List:
		return ui.Column(c).Children(func() {
			for i, item := range b.Items {
				row := ui.Row(c).AlignItems(ui.Start)
				if n.rtl {
					row.Reverse()
				}
				row.Children(func() {
					marker := bullet(depth)
					if b.Ordered {
						marker = strconv.Itoa(i+1) + "."
					}
					// The marker ends next to the item.
					m := ui.Text(c, marker).Width(20).Shrink(0)
					if n.rtl {
						m.Padding(0, 0, 0, 6)
					} else {
						m.TextAlign(ui.End).Padding(0, 6, 0, 0)
					}
					ui.Column(c).Grow(1).Children(func() { n.blocks(c, item, true, depth+1) })
				})
			}
		})
	case markdown.Quote:
		quote := ui.Column(c).BorderColor(t.Border).TextColor(t.TextMuted)
		if n.rtl {
			quote.BorderWidth(0, 3, 0, 0).Padding(0, 10, 0, 0)
		} else {
			quote.BorderWidth(0, 0, 0, 3).Padding(0, 0, 0, 10)
		}
		return quote.Children(func() { n.blocks(c, b.Blocks, false, depth) })
	case markdown.Code:
		return ui.ScrollHorizontal(c).Background(track(t)).Radius(3).Children(func() {
			ui.Text(c, b.Text).Font("monospace").FontSize(t.Rem(12.0/13)).NoWrap().Padding(6, 8).Selectable()
		})
	case markdown.Rule:
		return ui.Divider(c)
	}
	return paragraph(c, t, b.Inlines)
}

// paragraph builds the text of a paragraph or heading, which the user may
// select.
func paragraph(c *ui.Context, t *ui.Theme, inlines []markdown.Inline) ui.Element {
	return ui.RichText(c).Children(func() { buildInlines(c, t, inlines) }).Selectable()
}

// bullet returns the marker of the items of a list in depth lists, as
// browsers draw them.
func bullet(depth int) string {
	switch depth {
	case 0:
		return "•"
	case 1:
		return "◦"
	}
	return "▪"
}

// buildInlines builds the text of a paragraph inside it: links open in the
// browser.
func buildInlines(c *ui.Context, t *ui.Theme, inlines []markdown.Inline) {
	for _, in := range inlines {
		switch in.Kind {
		case markdown.Text:
			ui.Text(c, in.Text)
		case markdown.CodeSpan:
			ui.Text(c, in.Text).Font("monospace").FontSize(t.Rem(12.0 / 13)).TextBackground(track(t))
		case markdown.Emphasis:
			ui.RichText(c).Italic().Children(func() { buildInlines(c, t, in.Children) })
		case markdown.Strong:
			ui.RichText(c).Bold().Children(func() { buildInlines(c, t, in.Children) })
		case markdown.Link:
			ui.Link(c, "", in.URL).Children(func() { buildInlines(c, t, in.Children) })
		}
	}
}
