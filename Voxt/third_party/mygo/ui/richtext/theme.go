package richtext

import "github.com/egoist/mygo/ui"

// Theme is how the editor's text looks: its colors, the font of code, the
// styles of headings, and the room of lists, quotations and blocks of
// code. What it leaves zero takes the value of DefaultTheme for the
// window's theme, so that a theme sets only what it changes:
//
//	richtext.Editor(c, doc, richtext.EditorOptions{Theme: &richtext.Theme{Link: ui.Hex("#7c3aed")}})
//
// Lengths in ems are times the editor's font size (EditorOptions.Font).
type Theme struct {
	// Text colors text, Muted the placeholder, the text of tasks done and
	// the boxes of those to do, and Link links.
	Text, Muted, Link ui.Color
	// Caret colors the caret, and Selection is behind the text selected.
	Caret, Selection ui.Color

	// CodeFont is the family of code, the system's monospaced font when
	// empty, and CodeSize its size in ems: 0.92. Code colors code, within
	// text and in blocks, the Text color by default.
	CodeFont string
	CodeSize float32
	Code     ui.Color
	// CodeBackground is behind code within text, which it hugs with
	// rounded corners.
	CodeBackground ui.Color
	// CodeBlockBackground fills blocks of code, with corners of
	// CodeBlockRadius ems (0.3) and CodeBlockPadding ems of room inside:
	// 0.7 across, and 0.5 at their top and bottom.
	CodeBlockBackground ui.Color
	CodeBlockRadius     float32
	CodeBlockPadding    [2]float32

	// Headings are the styles of headings of levels 1 to 6.
	Headings [6]HeadingStyle

	// Indent is how far a level of a list or of indentation goes, in ems,
	// a list's markers in that room: 1.7.
	Indent float32
	// Marker colors the bullets and numbers of lists, the Text color by
	// default. Checkbox fills the boxes of tasks done, and Checkmark draws
	// their checks: white.
	Marker, Checkbox, Checkmark ui.Color
	// QuoteBar colors the bar along quotations, QuoteBarWidth DIPs wide
	// (3), and QuoteText their text, the Muted color by default.
	QuoteBar, QuoteText ui.Color
	QuoteBarWidth       float32

	// Padding is the room around the text of an Editor, in DIPs across
	// and down: the window theme's Space(3) and Space(2.5). An EditorBase
	// has none, its room the app's.
	Padding [2]float32
}

// HeadingStyle is how headings of one level look.
type HeadingStyle struct {
	// Size is their size in ems, and Weight their weight: 700.
	Size   float32
	Weight int
	// Color colors them, the Text color by default.
	Color ui.Color
	// Top is the room above them in ems of their size, unless they start
	// the text: 0.6. Bottom is that below them in ems of the text: 0.25.
	Top, Bottom float32
}

// headingSizes are the default sizes of headings, in ems.
var headingSizes = [6]float32{1.8, 1.45, 1.2, 1.1, 1, 0.9}

// DefaultTheme returns the editor's theme in a window's theme: its text,
// muted, accent, selection and border colors, code tinted with its text
// color, and headings from 1.8 times the text down to 0.9.
func DefaultTheme(t *ui.Theme) *Theme {
	th := defaultTheme(t)
	return &th
}

func defaultTheme(t *ui.Theme) Theme {
	th := Theme{
		Text:                t.Text,
		Muted:               t.TextMuted,
		Link:                t.Accent,
		Caret:               t.Accent,
		Selection:           t.Selection,
		CodeFont:            "monospace",
		CodeSize:            0.92,
		Code:                t.Text,
		CodeBackground:      t.Text.Alpha(0.07),
		CodeBlockBackground: t.Text.Alpha(0.07),
		CodeBlockRadius:     0.3,
		CodeBlockPadding:    [2]float32{0.7, 0.5},
		Indent:              1.7,
		Marker:              t.Text,
		Checkbox:            t.Accent,
		Checkmark:           ui.RGB(255, 255, 255),
		QuoteBar:            t.Border,
		QuoteText:           t.TextMuted,
		QuoteBarWidth:       3,
		Padding:             [2]float32{t.Space(3), t.Space(2.5)},
	}
	for i, size := range headingSizes {
		th.Headings[i] = HeadingStyle{Size: size, Weight: 700, Color: t.Text, Top: 0.6, Bottom: 0.25}
	}
	return th
}

// resolve returns th with what it leaves zero taken from def; the colors
// that default to others of th take those.
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
	c := func(v *ui.Color, d ui.Color) {
		if *v == (ui.Color{}) {
			*v = d
		}
	}
	c(&out.Text, def.Text)
	c(&out.Muted, def.Muted)
	c(&out.Link, def.Link)
	c(&out.Caret, def.Caret)
	c(&out.Selection, def.Selection)
	if out.CodeFont == "" {
		out.CodeFont = def.CodeFont
	}
	f(&out.CodeSize, def.CodeSize)
	c(&out.Code, out.Text)
	c(&out.CodeBackground, def.CodeBackground)
	c(&out.CodeBlockBackground, def.CodeBlockBackground)
	f(&out.CodeBlockRadius, def.CodeBlockRadius)
	f(&out.CodeBlockPadding[0], def.CodeBlockPadding[0])
	f(&out.CodeBlockPadding[1], def.CodeBlockPadding[1])
	for i := range out.Headings {
		h, d := &out.Headings[i], def.Headings[i]
		f(&h.Size, d.Size)
		if h.Weight == 0 {
			h.Weight = d.Weight
		}
		c(&h.Color, out.Text)
		f(&h.Top, d.Top)
		f(&h.Bottom, d.Bottom)
	}
	f(&out.Indent, def.Indent)
	c(&out.Marker, out.Text)
	c(&out.Checkbox, def.Checkbox)
	c(&out.Checkmark, def.Checkmark)
	c(&out.QuoteBar, def.QuoteBar)
	c(&out.QuoteText, out.Muted)
	f(&out.QuoteBarWidth, def.QuoteBarWidth)
	f(&out.Padding[0], def.Padding[0])
	f(&out.Padding[1], def.Padding[1])
	return out
}
