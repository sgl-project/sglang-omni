package markdown

import (
	"github.com/egoist/mygo/ui"
	"github.com/egoist/mygo/ui/editor"
)

// Theme is how Markdown looks: fonts, colors and spacing for each kind
// of block and inline. What it leaves zero takes the value of
// DefaultTheme for the window's theme, so that a theme sets only what it
// changes:
//
//	app.md.Theme = &markdown.Theme{Link: ui.Hex("#7c3aed"), CodeFont: "JetBrains Mono"}
//
// Lengths in ems are times the FontSize.
type Theme struct {
	// FontSize is the size of body text in DIPs: the window theme's
	// FontSize when zero.
	FontSize float32
	// Font is the family of text, the window theme's when empty, and
	// CodeFont that of code, the system's monospaced font when empty.
	Font, CodeFont string
	// LineHeight is how many times its font's line height a line of text
	// is: 1.5 by default, and CodeLineHeight that of code, 1.4.
	LineHeight, CodeLineHeight float32

	// Text colors text, Muted what is secondary (list markers, quotes,
	// a code block's language, footnotes), Link links, and Strong strong
	// text, the Text color when zero by default.
	Text, Muted, Link, Strong Color
	// StrongWeight is the weight of strong text: 600.
	StrongWeight int

	// Code colors code spans, with CodeBackground behind them; CodeSize is
	// their size in ems: 0.875.
	Code, CodeBackground Color
	CodeSize             float32

	// BlockSpacing is the space between blocks, in ems: 0.75. ItemSpacing
	// is that between the items of a tight list: 0.25.
	BlockSpacing, ItemSpacing float32
	// Headings are the styles of headings of levels 1 to 6.
	Headings [6]HeadingStyle
	// ListIndent is how far list items are from their list's edge, their
	// markers in that room, in ems: 1.6.
	ListIndent float32

	// QuoteBar colors the bar along block quotes, QuoteBarWidth DIPs
	// wide (3), and QuoteText their text.
	QuoteBar, QuoteText Color
	QuoteBarWidth       float32

	// Rule colors thematic breaks and the rules of headings.
	Rule Color

	// CodeBlockBackground and CodeBlockBorder color code blocks, with
	// corners of CodeBlockRadius DIPs (the window theme's Radius plus 2)
	// and CodeBlockPadding ems of room inside (0.85). CodeColors colors
	// their code's tokens: editor.DefaultTheme's colors when nil.
	CodeBlockBackground, CodeBlockBorder Color
	CodeBlockRadius, CodeBlockPadding    float32
	CodeColors                           *editor.Theme

	// TableBorder colors a table's lines, TableHeader the background of
	// its header row, and TableStripe that of every other row (none by
	// default). TableCellPadding is the room in cells, in ems: 0.6 across
	// and 0.4 down. TableMaxColumn is how wide a column grows before its
	// cells wrap, in ems: 24.
	TableBorder, TableHeader, TableStripe Color
	TableCellPadding                      [2]float32
	TableMaxColumn                        float32

	// Mark is the background of text marked with <mark>.
	Mark Color
	// Checkbox colors the check boxes of task lists that are checked.
	Checkbox Color
}

// Color is a color of ui.
type Color = ui.Color

// HeadingStyle is how headings of one level look.
type HeadingStyle struct {
	// Size is their size in ems, and Weight their weight.
	Size   float32
	Weight int
	// Color colors them: Theme.Text when zero.
	Color Color
	// Top and Bottom are the room above and below them in ems, which
	// replaces Theme.BlockSpacing around them.
	Top, Bottom float32
	// Rule draws a line below them, as GitHub does below headings of
	// levels 1 and 2.
	Rule bool
}

// DefaultTheme returns the Markdown theme of a window's theme: its font
// and colors, a code span tinted with its text color, code colored as the
// code editor colors it, and the spacing of the chat apps.
func DefaultTheme(t *ui.Theme) *Theme {
	th := &Theme{
		FontSize:            t.FontSize,
		Font:                t.Font,
		CodeFont:            "monospace",
		LineHeight:          1.5,
		CodeLineHeight:      1.4,
		Text:                t.Text,
		Muted:               t.TextMuted,
		Link:                t.Accent,
		Strong:              t.Text,
		StrongWeight:        600,
		Code:                t.Text,
		CodeBackground:      t.Text.Alpha(0.07),
		CodeSize:            0.875,
		BlockSpacing:        0.75,
		ItemSpacing:         0.25,
		ListIndent:          1.6,
		QuoteBar:            t.Border,
		QuoteText:           t.TextMuted,
		QuoteBarWidth:       3,
		Rule:                t.Border,
		CodeBlockBackground: t.Text.Alpha(0.04),
		CodeBlockBorder:     t.Border,
		CodeBlockRadius:     t.Radius + 2,
		CodeBlockPadding:    0.85,
		CodeColors:          editor.DefaultTheme(t),
		TableBorder:         t.Border,
		TableHeader:         t.Text.Alpha(0.04),
		TableCellPadding:    [2]float32{0.6, 0.4},
		TableMaxColumn:      24,
		Mark:                ui.RGBA(250, 204, 21, 0.35),
		Checkbox:            t.Accent,
	}
	if t.FontSize == 0 {
		th.FontSize = 14
	}
	sizes := [6]float32{1.6, 1.35, 1.15, 1, 0.95, 0.9}
	weights := [6]int{700, 650, 600, 600, 600, 600}
	for i := range th.Headings {
		th.Headings[i] = HeadingStyle{Size: sizes[i], Weight: weights[i], Color: t.Text, Top: 1.4, Bottom: 0.55}
	}
	th.Headings[5].Color = t.TextMuted
	return th
}

// resolve returns th with what it leaves zero taken from def.
func resolve(th *Theme, def *Theme) Theme {
	if th == nil {
		return *def
	}
	out := *th
	f := func(v *float32, d float32) {
		if *v == 0 {
			*v = d
		}
	}
	c := func(v *Color, d Color) {
		if *v == (Color{}) {
			*v = d
		}
	}
	s := func(v *string, d string) {
		if *v == "" {
			*v = d
		}
	}
	f(&out.FontSize, def.FontSize)
	s(&out.Font, def.Font)
	s(&out.CodeFont, def.CodeFont)
	f(&out.LineHeight, def.LineHeight)
	f(&out.CodeLineHeight, def.CodeLineHeight)
	c(&out.Text, def.Text)
	c(&out.Muted, def.Muted)
	c(&out.Link, def.Link)
	c(&out.Strong, out.Text)
	if out.StrongWeight == 0 {
		out.StrongWeight = def.StrongWeight
	}
	c(&out.Code, out.Text)
	c(&out.CodeBackground, def.CodeBackground)
	f(&out.CodeSize, def.CodeSize)
	f(&out.BlockSpacing, def.BlockSpacing)
	f(&out.ItemSpacing, def.ItemSpacing)
	for i := range out.Headings {
		h, d := &out.Headings[i], def.Headings[i]
		f(&h.Size, d.Size)
		if h.Weight == 0 {
			h.Weight = d.Weight
		}
		if i == 5 {
			c(&h.Color, out.Muted)
		} else {
			c(&h.Color, out.Text)
		}
		f(&h.Top, d.Top)
		f(&h.Bottom, d.Bottom)
	}
	f(&out.ListIndent, def.ListIndent)
	c(&out.QuoteBar, def.QuoteBar)
	c(&out.QuoteText, out.Muted)
	f(&out.QuoteBarWidth, def.QuoteBarWidth)
	c(&out.Rule, def.Rule)
	c(&out.CodeBlockBackground, def.CodeBlockBackground)
	c(&out.CodeBlockBorder, def.CodeBlockBorder)
	f(&out.CodeBlockRadius, def.CodeBlockRadius)
	f(&out.CodeBlockPadding, def.CodeBlockPadding)
	if out.CodeColors == nil {
		out.CodeColors = def.CodeColors
	}
	c(&out.TableBorder, def.TableBorder)
	c(&out.TableHeader, def.TableHeader)
	f(&out.TableCellPadding[0], def.TableCellPadding[0])
	f(&out.TableCellPadding[1], def.TableCellPadding[1])
	f(&out.TableMaxColumn, def.TableMaxColumn)
	c(&out.Mark, def.Mark)
	c(&out.Checkbox, def.Checkbox)
	return out
}

// em returns v ems in DIPs.
func (th *Theme) em(v float32) float32 { return v * th.FontSize }
