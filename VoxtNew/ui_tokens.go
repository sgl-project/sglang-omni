package main

// Design tokens (AudioSmith 0.3, research/ui-design-spec.md): a neutral grey
// scale, near-black ink for the one primary action of a view, one accent
// for state (selection, progress, focus, links, what plays), and a
// red-orange for recording and destructive actions. applyTheme puts them
// into MyGo's Theme, so that MyGo's widgets follow them; views read the
// rest with tok(t).

import "github.com/egoist/mygo/ui"

// Radii, DIPs at 100 % (scale them with px).
const (
	radiusControl = 8  // buttons, inputs, list rows
	radiusCard    = 12 // cards, dialogs, popovers
)

// Type scale, DIPs at 100 % (px(t, fsBody)).
const (
	fsTitle   = 20 // page titles
	fsSection = 15 // section headings, dialog titles
	fsBody    = 13 // text
	fsCaption = 12 // secondary lines, captions
	fsMicro   = 11 // badges, keyboard hints
)

// Spacing steps, DIPs at 100 %: 4, 8, 12, 16, 20, 24, 32.
const (
	sp1 = 4
	sp2 = 8
	sp3 = 12
	sp4 = 16
	sp5 = 20
	sp6 = 24
	sp8 = 32
)

// greys are the neutral scale, from the background (0) to the text (9).
type greys [10]ui.Color

var (
	lightGreys = greys{
		ui.Hex("#ffffff"), ui.Hex("#f7f7f7"), ui.Hex("#f2f2f2"), ui.Hex("#ebebeb"), ui.Hex("#e0e0e0"),
		ui.Hex("#c7c7c7"), ui.Hex("#a3a3a3"), ui.Hex("#6e6e6e"), ui.Hex("#454545"), ui.Hex("#111111"),
	}
	darkGreys = greys{
		ui.Hex("#151515"), ui.Hex("#1c1c1c"), ui.Hex("#232323"), ui.Hex("#2b2b2b"), ui.Hex("#353535"),
		ui.Hex("#4a4a4a"), ui.Hex("#6b6b6b"), ui.Hex("#a3a3a3"), ui.Hex("#d0d0d0"), ui.Hex("#f2f2f2"),
	}
)

// accentSet is an accent in the light and the dark theme.
type accentSet struct{ light, dark string }

// accents are the brand accent candidates; AudioSmith's design review
// (2026-10-14) chose teal, and VoxtNew keeps it for now (plan D3).
// accentName is the one in use: changing it re-colours the app.
var accents = map[string]accentSet{
	"blue":   {"#2f6bf0", "#4f86f7"},
	"indigo": {"#5752e0", "#7d79f2"},
	"teal":   {"#0d9384", "#2bb3a3"},
}

// accentName is the brand accent (a key of accents).
var accentName = "teal"

// tokens are the colours of one appearance.
type tokens struct {
	G greys
	// Surfaces.
	Bg      ui.Color // pages, dialogs
	Sidebar ui.Color // the sidebar
	Subtle  ui.Color // a card's fill (cards have a fill or a hairline, not both)
	Hover   ui.Color // ghost buttons and rows under the pointer
	Pressed ui.Color // … pressed; the chosen row
	Line    ui.Color // hairlines: dividers, card borders
	// Selected: rows selected among many (the logs' lines), the accent's
	// tint, stronger than Hover; SelectedHover one under the pointer.
	Selected, SelectedHover ui.Color
	// Text.
	Text, Text2, Text3 ui.Color // primary, secondary, tertiary
	// Ink: the primary button (near-black, near-white in the dark).
	Ink, InkHover, InkPressed, OnInk ui.Color
	// Accent: state only (selection, progress, focus, links, playing);
	// AccentInk is the accent as text on the background (markup, counts),
	// darker in the light theme for contrast.
	Accent, AccentSoft, AccentInk ui.Color
	// Record: recording and destructive actions (red-orange).
	Record, RecordSoft ui.Color
	// Semantic colours, for small dots and text only.
	Success, Warning, Danger ui.Color
}

func makeTokens(dark bool) *tokens {
	g := lightGreys
	ac := accents[accentName]
	accent := ui.Hex(ac.light)
	k := &tokens{G: g}
	if dark {
		g = darkGreys
		accent = ui.Hex(ac.dark)
		k = &tokens{G: g,
			InkHover: ui.Hex("#dcdcdc"), InkPressed: ui.Hex("#c4c4c4"),
			Record: ui.Hex("#ff6b47"), Success: ui.Hex("#3ccb7f"), Warning: ui.Hex("#f2a53a"),
		}
	} else {
		k.InkHover, k.InkPressed = ui.Hex("#2e2e2e"), ui.Hex("#454545")
		k.Record, k.Success, k.Warning = ui.Hex("#eb5230"), ui.Hex("#1f9d55"), ui.Hex("#c27803")
	}
	k.Bg, k.Sidebar, k.Subtle = g[0], g[1], g[1]
	k.Hover, k.Pressed, k.Line = g[2], g[3], g[3]
	k.Text, k.Text2, k.Text3 = g[9], g[7], g[6]
	k.Ink, k.OnInk = g[9], g[0]
	k.Accent = accent
	k.AccentSoft = accent.Alpha(0.12)
	k.Selected, k.SelectedHover = accent.Alpha(0.2), accent.Alpha(0.28)
	k.AccentInk = accent
	if !dark {
		k.AccentInk = accent.Mix(ui.Hex("#000000"), 0.18)
	}
	k.RecordSoft = k.Record.Alpha(0.12)
	k.Danger = k.Record
	return k
}

// The tokens of each appearance, for accentName.
var lightTokens, darkTokens = makeTokens(false), makeTokens(true)

// tok returns the tokens of t's appearance.
func tok(t *ui.Theme) *tokens {
	if t.Dark {
		return darkTokens
	}
	return lightTokens
}

// styleTheme puts the tokens into th, MyGo's theme.
func styleTheme(th *ui.Theme) {
	k := tok(th)
	th.Background = k.Bg
	th.Surface = k.G[2]
	th.SurfaceHover = k.G[3]
	th.SurfacePressed = k.G[4]
	th.Border = k.G[3]
	if th.Dark {
		th.Border = k.G[4]
	}
	th.Text, th.TextMuted = k.Text, k.Text2
	th.Accent = k.Accent
	th.AccentHover = k.Accent.Mix(ui.Hex("#000000"), 0.12)
	th.AccentPressed = k.Accent.Mix(ui.Hex("#000000"), 0.24)
	if th.Dark {
		th.AccentHover = k.Accent.Mix(ui.Hex("#ffffff"), 0.15)
		th.AccentPressed = k.Accent.Mix(ui.Hex("#000000"), 0.15)
	}
	th.AccentText = ui.Hex("#ffffff")
	if th.Dark && accentName == "teal" {
		th.AccentText = ui.Hex("#06201c") // dark text on the light teal
	}
	th.Danger, th.Warning, th.Success = k.Danger, k.Warning, k.Success
	th.Selection = k.Accent.Alpha(0.22)
	th.Focus = k.Accent.Alpha(0.5)
	th.Inverse, th.InverseText = k.G[9], k.G[0]
	if th.Dark {
		th.Inverse, th.InverseText = k.G[3], k.G[9]
	}
	th.Scrollbar = k.Text.Alpha(0.22)
	th.Radius = radiusControl
}

// inkTheme is th with ink as its accent: MyGo's primary button in it is
// the near-black (near-white in the dark) primary action.
func inkTheme(th *ui.Theme) *ui.Theme {
	k := tok(th)
	ink := *th
	ink.Accent, ink.AccentHover, ink.AccentPressed, ink.AccentText = k.Ink, k.InkHover, k.InkPressed, k.OnInk
	ink.Focus = k.Accent.Alpha(0.5)
	return &ink
}
