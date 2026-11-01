package editor

import (
	"github.com/egoist/mygo/internal/gamut"
	"github.com/egoist/mygo/ui"
)

// Theme colors editors and diffs. Colors left zero take
// those of DefaultTheme for the window's appearance, which follow
// its Theme.
type Theme struct {
	Background ui.Color
	Text       ui.Color
	// LineNumber colors the numbers of the gutter, ActiveLineNumber those
	// of the lines with a caret.
	LineNumber       ui.Color
	ActiveLineNumber ui.Color
	// ActiveLine highlights the lines with a caret.
	ActiveLine ui.Color
	// Selection highlights selected text while the editor has the focus,
	// InactiveSelection while it does not.
	Selection         ui.Color
	InactiveSelection ui.Color
	Cursor            ui.Color
	// MatchingBracket outlines the brackets around the caret.
	MatchingBracket ui.Color
	// Match highlights the matches of the find bar, CurrentMatch the one
	// chosen; Occurrence other occurrences of the selected text.
	Match        ui.Color
	CurrentMatch ui.Color
	Occurrence   ui.Color
	// IndentGuide draws the indent guides, ActiveIndentGuide the one
	// around the caret; WrapGuide the guides at the columns the editor
	// marks; Whitespace the dots and arrows of spaces and tabs shown.
	IndentGuide       ui.Color
	ActiveIndentGuide ui.Color
	WrapGuide         ui.Color
	Whitespace        ui.Color
	// Error, Warning, Info and Hint color diagnostics of each severity.
	Error, Warning, Info, Hint ui.Color
	// Added, Deleted and Modified are the colors of changes, which tint
	// the lines of diffs, their gutters and the words changed in them.
	Added, Deleted, Modified ui.Color
	// Tokens styles the kinds of tokens; a zero color is Text.
	Tokens [tokenKinds]TokenStyle
}

// TokenStyle is how a kind of token shows.
type TokenStyle struct {
	Color  ui.Color
	Italic bool
	Bold   bool
}

// DefaultTheme returns the code theme for a window's theme: its
// background and text, with GitHub's colors for code.
func DefaultTheme(t *ui.Theme) *Theme {
	ct := &Theme{
		Background:  t.Background,
		Text:        t.Text,
		LineNumber:  t.TextMuted.Alpha(0.75),
		Selection:   t.Selection,
		Cursor:      t.Accent,
		WrapGuide:   t.Border.Alpha(0.6),
		IndentGuide: t.Border.Alpha(0.55),
		Whitespace:  t.TextMuted.Alpha(0.45),
	}
	ct.ActiveLineNumber = t.Text
	ct.InactiveSelection = t.Selection.Alpha(0.55)
	ct.ActiveIndentGuide = t.TextMuted.Alpha(0.6)
	tok := func(k TokenKind, hex string) { ct.Tokens[k].Color = ui.Hex(hex) }
	if t.Dark {
		ct.ActiveLine = ui.RGBA(255, 255, 255, 0.045)
		ct.MatchingBracket = ui.RGBA(255, 255, 255, 0.35)
		ct.Match = ui.RGBA(255, 210, 60, 0.22)
		ct.CurrentMatch = ui.RGBA(255, 160, 40, 0.5)
		ct.Occurrence = ui.RGBA(255, 255, 255, 0.1)
		ct.Error, ct.Warning, ct.Info, ct.Hint = ui.Hex("#f85149"), ui.Hex("#d29922"), ui.Hex("#58a6ff"), ui.Hex("#8b949e")
		ct.Added, ct.Deleted, ct.Modified = ui.Hex("#5ecc71"), ui.Hex("#ff6762"), ui.Hex("#69b1ff")
		tok(TokenKeyword, "#ff7b72")
		tok(TokenOperator, "#ff7b72")
		tok(TokenType, "#ffa657")
		tok(TokenFunction, "#d2a8ff")
		tok(TokenMeta, "#d2a8ff")
		tok(TokenProperty, "#79c0ff")
		tok(TokenConstant, "#79c0ff")
		tok(TokenNumber, "#79c0ff")
		tok(TokenAttribute, "#79c0ff")
		tok(TokenString, "#a5d6ff")
		tok(TokenEscape, "#79c0ff")
		tok(TokenRegexp, "#7ee787")
		tok(TokenTag, "#7ee787")
		tok(TokenComment, "#8b949e")
		tok(TokenHeading, "#79c0ff")
		tok(TokenLink, "#a5d6ff")
		tok(TokenCode, "#a5d6ff")
		tok(TokenInserted, "#7ee787")
		tok(TokenDeleted, "#ffa198")
		tok(TokenInvalid, "#f85149")
	} else {
		ct.ActiveLine = ui.RGBA(0, 0, 0, 0.035)
		ct.MatchingBracket = ui.RGBA(0, 0, 0, 0.3)
		ct.Match = ui.RGBA(255, 200, 0, 0.3)
		ct.CurrentMatch = ui.RGBA(255, 140, 0, 0.45)
		ct.Occurrence = ui.RGBA(0, 0, 0, 0.07)
		ct.Error, ct.Warning, ct.Info, ct.Hint = ui.Hex("#d1242f"), ui.Hex("#9a6700"), ui.Hex("#0969da"), ui.Hex("#6e7781")
		ct.Added, ct.Deleted, ct.Modified = ui.Hex("#0dbe4e"), ui.Hex("#ff2e3f"), ui.Hex("#009fff")
		tok(TokenKeyword, "#cf222e")
		tok(TokenOperator, "#cf222e")
		tok(TokenType, "#953800")
		tok(TokenFunction, "#8250df")
		tok(TokenMeta, "#8250df")
		tok(TokenProperty, "#0550ae")
		tok(TokenConstant, "#0550ae")
		tok(TokenNumber, "#0550ae")
		tok(TokenAttribute, "#0550ae")
		tok(TokenString, "#0a3069")
		tok(TokenEscape, "#0550ae")
		tok(TokenRegexp, "#116329")
		tok(TokenTag, "#116329")
		tok(TokenComment, "#6e7781")
		tok(TokenHeading, "#0550ae")
		tok(TokenLink, "#0a3069")
		tok(TokenCode, "#0a3069")
		tok(TokenInserted, "#116329")
		tok(TokenDeleted, "#82071e")
		tok(TokenInvalid, "#d1242f")
	}
	ct.Tokens[TokenHeading].Bold = true
	ct.Tokens[TokenStrong].Bold = true
	ct.Tokens[TokenEmphasis].Italic = true
	return ct
}

// resolve fills the zero colors of ct from the default theme for t.
func (ct *Theme) resolve(t *ui.Theme) *Theme {
	def := DefaultTheme(t)
	if ct == nil {
		return def
	}
	out := *ct
	fill := func(c *ui.Color, d ui.Color) {
		if *c == (ui.Color{}) {
			*c = d
		}
	}
	fill(&out.Background, def.Background)
	fill(&out.Text, def.Text)
	fill(&out.LineNumber, def.LineNumber)
	fill(&out.ActiveLineNumber, def.ActiveLineNumber)
	fill(&out.ActiveLine, def.ActiveLine)
	fill(&out.Selection, def.Selection)
	fill(&out.InactiveSelection, def.InactiveSelection)
	fill(&out.Cursor, def.Cursor)
	fill(&out.MatchingBracket, def.MatchingBracket)
	fill(&out.Match, def.Match)
	fill(&out.CurrentMatch, def.CurrentMatch)
	fill(&out.Occurrence, def.Occurrence)
	fill(&out.IndentGuide, def.IndentGuide)
	fill(&out.ActiveIndentGuide, def.ActiveIndentGuide)
	fill(&out.WrapGuide, def.WrapGuide)
	fill(&out.Whitespace, def.Whitespace)
	fill(&out.Error, def.Error)
	fill(&out.Warning, def.Warning)
	fill(&out.Info, def.Info)
	fill(&out.Hint, def.Hint)
	fill(&out.Added, def.Added)
	fill(&out.Deleted, def.Deleted)
	fill(&out.Modified, def.Modified)
	for k := range out.Tokens {
		fill(&out.Tokens[k].Color, def.Tokens[k].Color)
	}
	return &out
}

// TokenColor returns the color of a kind of token: its style's, else the
// text's.
func (ct *Theme) TokenColor(k TokenKind) ui.Color {
	if c := ct.Tokens[k].Color; c.A > 0 {
		return c
	}
	return ct.Text
}

// mixLab returns the color t of the way from a to b, mixed in Oklab, as
// CSS's color-mix(in oklab) and Pierre's diffs tint lines.
func mixLab(a, b ui.Color, t float32) ui.Color {
	lin := func(c ui.Color) (float64, float64, float64) {
		return gamut.Oklab(gamut.Decode(float64(c.R)/255), gamut.Decode(float64(c.G)/255), gamut.Decode(float64(c.B)/255))
	}
	l1, a1, b1 := lin(a)
	l2, a2, b2 := lin(b)
	f := float64(t)
	r, g, bb := gamut.LinearSRGB(l1+(l2-l1)*f, a1+(a2-a1)*f, b1+(b2-b1)*f)
	enc := func(v float64) uint8 {
		v = gamut.Encode(max(0, min(v, 1)))
		return uint8(v*255 + 0.5)
	}
	return ui.Color{R: enc(r), G: enc(g), B: enc(bb), A: uint8(float32(a.A)*(1-t) + float32(b.A)*t + 0.5)}
}

// diffColors are the tints of changes over a background.
type diffColors struct {
	addedLine, deletedLine     ui.Color // the background of changed lines
	addedGutter, deletedGutter ui.Color // and of their gutters
	addedWord, deletedWord     ui.Color // the words changed in them
	added, deleted, modified   ui.Color // bars and numbers
}

func (ct *Theme) diff(dark bool) diffColors {
	line, gutter, word := float32(0.12), float32(0.09), float32(0.15)
	if dark {
		line, gutter, word = 0.2, 0.15, 0.2
	}
	bg := ct.Background
	if bg.A == 0 {
		bg = ui.RGB(255, 255, 255)
	}
	return diffColors{
		addedLine: mixLab(bg, ct.Added, line), deletedLine: mixLab(bg, ct.Deleted, line),
		addedGutter: mixLab(bg, ct.Added, gutter), deletedGutter: mixLab(bg, ct.Deleted, gutter),
		addedWord: ct.Added.Alpha(word), deletedWord: ct.Deleted.Alpha(word),
		added: ct.Added, deleted: ct.Deleted, modified: ct.Modified,
	}
}
