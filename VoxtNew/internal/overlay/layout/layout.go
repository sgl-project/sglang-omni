// Package layout is the overlay's geometry, as pure numbers: the sizes of
// the pill, the live text, a notice and the answer card (Voxt's, in DIPs),
// the shadow's margin, the face's width for its texts, its window's frame
// and place on the display (anchored to an edge), the waveform's bars and
// the loader's dots. Package main's overlay draws with them; the ink check
// (internal/overlay/ink) measures pictures against them. No MyGo UI: only
// mygo.Rectangle, the display's geometry.
package layout

import (
	"math"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// The face's shadow: small, and whole. The window adds Pad around the
// face for it (the window has none: macOS would draw its rectangle's):
// the blur's reach plus the offset, the side it falls on.
const (
	ShadowY    = 4
	ShadowBlur = 12
	ShadowA    = 0.22
	Pad        = ShadowBlur + ShadowY + 2
)

// Sizes of the overlay's face, in DIPs: Voxt's (WaveformView.swift).
const (
	// Border is the face's hairline. MyGo's boxes are border boxes: it
	// takes this much from each side of the face's content, so the
	// paddings inside the face are Voxt's less it. (10-15: they were not,
	// and the pill's row overflowed its content box to the right and
	// down, a point each: the icon's side looked wider than the
	// waveform's.)
	Border = 1
	// The row: the feature's icon (IconSize) centred in its slot, Gap,
	// the waveform (its 16 bars, WaveW wide) or the loader centred in its
	// slot; RowH high.
	IconSlot  = 16
	IconSize  = 14
	Gap       = 10
	WaveSlotW = 94
	RowH      = 28
	RowW      = IconSlot + Gap + WaveSlotW
	// The pill, without text: the row, PillPadX and PillPadY around
	// it: 148 × 48.
	PillPadX = 14
	PillPadY = 10
	PillW    = RowW + 2*PillPadX
	PillH    = RowH + 2*PillPadY
	// With live text: the row, LiveGap, one line of the text (LiveSize,
	// medium, LineH high) as wide as it is up to LiveTextW, the two
	// centred, LivePadX and LivePadY around them: as wide as the wider
	// of the two, at most 300, and 76 high.
	LivePadX   = 20
	LivePadY   = 12
	LiveGap    = 8
	LineH      = 16
	LiveSize   = 12
	LiveWeight = 500
	LiveTextW  = 260
	LiveMaxW   = LiveTextW + 2*LivePadX
	LiveH      = RowH + LiveGap + LineH + 2*LivePadY
	FadeW      = 16 // the line's left edge fades out over this, scrolled
	// After the waveform in the row, Gap before each, when they show:
	// the time, and the matched group's name (SceneSize, at most
	// SceneMaxW wide).
	TimeW     = 30
	SceneSize = 11
	SceneMaxW = 88
	// A notice: the icon's slot, Gap, its line (NoticeSize), the
	// arrow of an action, in the pill's padding; NoticeW to NoticeW2
	// wide.
	NoticeSize = 13
	ovArrowW   = 8 + 14
	NoticeW    = 200
	NoticeW2   = 420
	// Tick paces the waveform and the loader (Voxt's 20 per second);
	// FadeStep the fade after an insert (two steps, then hidden).
	Tick     = 50 * time.Millisecond
	FadeStep = 50 * time.Millisecond
)

// Frame is the window's frame for a face of w × h in the work area
// wa: centred, its anchor edge (the bottom, or the top) Overlay.Margin
// from the work area's, the shadow's margin around the face.
func Frame(wa mygo.Rectangle, o settings.Overlay, w, h int) mygo.Rectangle {
	winW, winH := w+2*Pad, h+2*Pad
	x, y := Place(wa, o, winW, winH)
	return mygo.Rectangle{X: x, Y: y, Width: winW, Height: winH}
}

// Place is where a window of w × h goes in the work area wa:
// centred, Overlay.Margin from its bottom (or top) edge to the face.
func Place(wa mygo.Rectangle, o settings.Overlay, w, h int) (int, int) {
	margin := int(math.Round(o.Margin)) - Pad
	x := wa.X + (wa.Width-w)/2
	y := wa.Y + wa.Height - h - margin
	if o.Position == "top" {
		y = wa.Y + margin
	}
	return x, y
}

// RowWidth is the width of the pill's row: the icon and the waveform, the
// time when it shows, the group's name when one matched (scene DIPs
// wide, at most SceneMaxW; 0 for none).
func RowWidth(showTime bool, scene float32) int {
	w := RowW
	if showTime {
		w += Gap + TimeW
	}
	if scene > 0 {
		w += Gap + int(math.Ceil(float64(min(scene, SceneMaxW))))
	}
	return w
}

// LiveWidth is the face's width with live text text DIPs wide under a
// row row DIPs wide: the wider of the two, the text at most
// LiveTextW, and the padding (Voxt's).
func LiveWidth(row int, text float32) int {
	line := int(math.Ceil(float64(min(text, LiveTextW))))
	return max(row, line) + 2*LivePadX
}

// NoticeWidth fits a notice's line, text DIPs wide: the pill's padding,
// the icon, the text, the arrow of an action; at least NoticeW, at most
// NoticeW2.
func NoticeWidth(text float32, action bool) int {
	w := 2*PillPadX + IconSlot + Gap + int(math.Ceil(float64(text)))
	if action {
		w += ovArrowW
	}
	return min(max(w, NoticeW), NoticeW2)
}

// Sizes of the answer card, in DIPs: Voxt's (WaveformAnswerCard.swift,
// WaveformView.swift).
const (
	// AnswerW is the card's width, every card's: 440, the least that
	// holds the widest header (Translation, the language's name at its
	// widest, Insert, Copy, Close). Voxt's card fills its 560 window
	// (measured on the Mac, 10-15); the user found it too wide, then 480
	// too (10-15, third round).
	AnswerW  = 440
	CardPadX = 18
	CardPadY = 16 // under the text (Voxt's)
	// The header: CardPadTop over it, CardRuleTop under it to the
	// rule, the same (10-15: Voxt's 16 and 8 left twice as much room over
	// the header as under it, which the user saw as off); the rule stays
	// at Voxt's 48.
	CardPadTop  = 12
	CardHeadH   = 24
	CardHeadGap = 12 // between the header's items
	CardRuleTop = 12 // the header to the rule
	CardRuleGap = 10 // the rule to the text
	// Where the rule is.
	CardRuleY   = CardPadTop + CardHeadH + CardRuleTop
	CardBodyMax = 220
	CardInset   = 10 // the text's right inset: the scroller's room
	CardBodyW   = AnswerW - 2*CardPadX - CardInset
	// CardChrome is the card's height without its text.
	CardChrome = CardPadTop + CardHeadH + CardRuleTop + 1 + CardRuleGap + CardPadY
	CardMsgGap = 8
	CardMsgH   = 16
	// CardMaxH is the highest card: the text at its most, and a line.
	CardMaxH = CardChrome + CardBodyMax + CardMsgGap + CardMsgH

	CardIcon       = 20
	CardTitleSize  = 15
	CardTextSize   = 14
	CardTextWeight = 500
	CardBtn        = 24
	CardBtnIcon    = 15
	CardLangMaxW   = 128 // the language's name in its capsule, at most (Voxt 92: "Simplified Chinese" did not fit)
	CardWaveW      = 96  // a conversation's waveform

	// A conversation's bubbles.
	BubbleW         = 340
	BubblePadX      = 12
	BubblePadY      = 10
	BubbleGap       = 14
	BubbleTitleGap  = 7
	BubbleTextSize  = 13
	BubbleTitleSize = 11
	BubbleRadius    = 14

	ProblemIcon = 16
	ProblemGap  = 8
)

// Bars of the waveform (Voxt's): WaveW wide in all, centred in the
// waveform's slot.
const (
	WaveBarW    = 3.2
	WaveBarGap  = 2.5
	WaveBarR    = 1.5
	WaveBarMinH = 3.2
	WaveBarMaxH = 22
	WaveW       = session.Bars*WaveBarW + (session.Bars-1)*WaveBarGap // 88.7
)

// WaveBars turns the last levels (session.Level, oldest first, about 20
// a second) into the bars' heights, 0–1: each level lifted so that quiet
// speech shows and room noise does not, then followed quickly as it rises
// and slowly as it falls (Voxt's waveform: rise 0.82, fall 0.24 a frame),
// so the bars swell with a word and ebb after it. Missing levels (the
// recording's first second) are 0.
func WaveBars(levels []float64) [session.Bars]float64 {
	var out [session.Bars]float64
	pad := session.Bars - len(levels)
	if pad < 0 {
		levels, pad = levels[-pad:], 0
	}
	s := 0.0
	for i := range session.Bars {
		x := 0.0
		if i >= pad {
			x = waveLift(levels[i-pad])
		}
		if x >= s {
			s += (x - s) * 0.82
		} else {
			s += (x - s) * 0.24
		}
		out[i] = min(max(s, 0), 1)
	}
	return out
}

// waveLift maps a level (0 at −60 dBFS, 1 at 0 dBFS) to a bar: below
// −48 dBFS (a room) nothing, full at −12 dBFS, eased so that speech at
// −35 dBFS still stands up.
func waveLift(v float64) float64 {
	g := min(max((v-0.2)/0.6, 0), 1)
	return math.Pow(g, 0.8)
}

// WaveIdle is a quiet bar's breath: up to 1.6 DIPs, a slow wave along
// the bars (1.6 s), gone as the bar rises.
func WaveIdle(i int, v, phase float64) float64 {
	quiet := max(0, 1-v*8)
	return quiet * 1.6 * (0.5 + 0.5*math.Sin(2*math.Pi*phase/1.6+float64(i)*0.55))
}

// WaveGlow is a loud bar's halo, as Voxt's: none when quiet.
func WaveGlow(v float64) float64 {
	a := max(0, v-0.05)
	return min(0.24, 0.26*a)
}

// LoaderDot is a loader dot's scale and opacity at t seconds: it swells
// and brightens over the first 30 % of a 1.2 s cycle, shrinks and fades
// over the next 30 %, and rests.
func LoaderDot(t float64) (scale, alpha float64) {
	p := math.Mod(t, 1.2) / 1.2
	if p < 0 {
		p++
	}
	switch {
	case p < 0.3:
		u := p / 0.3
		return 0.6 + 0.7*u, 0.15 + 0.85*u
	case p < 0.6:
		u := (p - 0.3) / 0.3
		return 1.3 - 0.7*u, 1 - 0.85*u
	}
	return 0.6, 0.15
}

// MicRing is the starting microphone's ring at t seconds: a ping that
// grows from the icon (MicRingMin) to MicRingMax over 1.2 s, fading out;
// still (without motion), at 22 and 0.35.
func MicRing(t float64, motion bool) (d, alpha float64) {
	if !motion {
		return 22, 0.35
	}
	p := math.Mod(t, 1.2) / 1.2
	if p < 0 {
		p++
	}
	ease := 1 - (1-p)*(1-p) // ease out
	return MicRingMin + (MicRingMax-MicRingMin)*ease, 0.6 * (1 - p)
}

// MicPulse is the starting microphone icon's opacity and scale at t
// seconds (Voxt's initializing icon: 1.25 s each way, ease in and out);
// still, 0.7 and 1.
func MicPulse(t float64, motion bool) (alpha, scale float64) {
	if !motion {
		return 0.7, 1
	}
	u := 0.5 - 0.5*math.Cos(2*math.Pi*t/2.5)
	return 0.55 + 0.43*u, 0.94 + 0.1*u
}

// The ring's diameters: the icon's size to just inside the row.
const (
	MicRingMin = IconSize
	MicRingMax = RowH - 4
)

// FaceAlpha is how opaque the face's fill is (Overlay.Opacity).
func FaceAlpha(o settings.Overlay) float32 {
	return float32(min(max(o.Opacity, 0.5), 1))
}
