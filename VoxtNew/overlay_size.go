package main

// The overlay's size (overlay.go): the face's width and height in each mode,
// from its texts as the view measured them, or guessed until it did; the
// work area it shows in.

import (
	"math"
	"slices"
	"strings"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// overlaySize is the face's size for mode, and whether its width is the
// measured texts' (not guessed).
func (a *app) overlaySize(mode overlayMode) (w, h int, exact bool) {
	switch mode {
	case ovAnswer:
		h, exact := a.cardHeight() // overlay_card.go
		return layout.AnswerW, h, exact
	case ovNotice:
		if a.notice == nil {
			return layout.NoticeW, layout.PillH, true
		}
		tw, exact := a.ovTextWidth(noticeText(a.notice.text))
		return layout.NoticeWidth(tw, a.notice.action != actionNone), layout.PillH, exact
	case ovRecording, ovWorking:
		scene, exact := float32(0), true
		if name := uitext.SceneShort(a.sess); name != "" {
			scene, exact = a.ovTextWidth(sceneText(name))
		}
		row := layout.RowWidth(mode == ovRecording && a.prefs.Overlay.ShowTime, scene)
		text, _ := pillLine(a.pillSession())
		if text == "" {
			return row + 2*layout.PillPadX, layout.PillH, exact
		}
		tw, known := a.ovTextWidth(liveTextKey(text))
		w := layout.LiveWidth(row, tw)
		if a.ov.live.id == a.sess.ID {
			// Within a session the face only grows: a revised text, or the
			// model's draft after the speech, does not make it jump back.
			w = max(w, a.ov.live.w)
		}
		return w, layout.LiveH, exact && known
	}
	return layout.PillW, layout.PillH, true
}

// ovSized is a text the face's size depends on, as it draws: wrapped at
// wrap DIPs (0: one line).
type ovSized struct {
	text   string
	size   float32
	weight int
	wrap   float32
}

func (k ovSized) span() ui.Span {
	return ui.Span{Text: k.text, Size: k.size, Weight: k.weight}
}

// ovMeasured is a text's size, measured.
type ovMeasured struct{ w, h float32 }

func liveTextKey(text string) ovSized {
	return ovSized{text, layout.LiveSize, layout.LiveWeight, 0}
}

func sceneText(name string) ovSized { return ovSized{name, layout.SceneSize, 0, 0} }

func noticeText(text string) ovSized { return ovSized{text, layout.NoticeSize, 0, 0} }

// ovTextWidth is how wide k draws: as measured, or guessed (false)
// until the overlay's view measured it.
func (a *app) ovTextWidth(k ovSized) (float32, bool) {
	m, ok := a.ovText(k)
	return m.w, ok
}

// ovTextHeight is how high k draws, wrapped at k.wrap: as measured, or
// guessed (false) until the overlay's view measured it.
func (a *app) ovTextHeight(k ovSized) (float32, bool) {
	m, ok := a.ovText(k)
	return m.h, ok
}

func (a *app) ovText(k ovSized) (ovMeasured, bool) {
	if a.measureHook != nil {
		w, h := a.measureHook(k)
		return ovMeasured{w, h}, true
	}
	if m, ok := a.ov.textW[k]; ok {
		return m, true
	}
	w, h := guessText(k)
	return ovMeasured{w, h}, false
}

// guessText guesses k's size: its width (textWidth), and its lines
// (each paragraph's width over k.wrap) of 1.3 times its size.
func guessText(k ovSized) (float32, float32) {
	w := float32(textWidth(k.text, float64(k.size)))
	line := float32(math.Ceil(float64(k.size) * 1.3))
	if k.wrap <= 0 {
		return w, line
	}
	lines := 0
	for _, p := range strings.Split(k.text, "\n") {
		pw := float32(textWidth(p, float64(k.size)))
		lines += max(1, int(math.Ceil(float64(pw/k.wrap))))
	}
	return min(w, k.wrap), float32(lines) * line
}

// overlayTexts are the texts the face's size depends on in its mode.
func (a *app) overlayTexts() []ovSized {
	var out []ovSized
	switch a.ov.mode {
	case ovNotice:
		if a.notice != nil {
			out = append(out, noticeText(a.notice.text))
		}
	case ovRecording, ovWorking:
		if name := uitext.SceneShort(a.sess); name != "" {
			out = append(out, sceneText(name))
		}
		if text, _ := pillLine(a.pillSession()); text != "" {
			out = append(out, liveTextKey(text))
		}
	case ovAnswer:
		out = cardTexts(a.cardNow()) // overlay_card.go
	}
	return out
}

// overlayMeasure measures, in the overlay's build, the texts the face's
// size was guessed from; the next turn of the main loop keeps their
// widths and fits the face to them (the build pass resizes nothing).
// A window that shows again stays transparent until then: the fitting
// is posted before the reveal.
func (a *app) overlayMeasure(c *ui.Context) {
	var keys []ovSized
	var sizes []ovMeasured
	for _, k := range a.overlayTexts() {
		if _, ok := a.ov.textW[k]; ok || slices.Contains(keys, k) {
			continue
		}
		w, h := c.MeasureText(k.wrap, k.span())
		keys, sizes = append(keys, k), append(sizes, ovMeasured{w, h})
	}
	if len(keys) == 0 {
		return
	}
	a.post(func() {
		if a.ov.textW == nil || len(a.ov.textW) > 256 {
			a.ov.textW = map[ovSized]ovMeasured{}
		}
		for i, k := range keys {
			a.ov.textW[k] = sizes[i]
		}
		a.overlayRestyle()
	})
}

// textWidth guesses how wide text is at size DIPs: a wide (CJK) letter
// takes the size, any other about 0.56 of it; a little more than less.
func textWidth(text string, size float64) int {
	w := 0.0
	for _, r := range text {
		switch {
		case uitext.WideRune(r):
			w += size
		case r == ' ' || r == 'i' || r == 'l' || r == '.' || r == ',' || r == ':':
			w += 0.32 * size
		default:
			w += 0.58 * size
		}
	}
	return int(math.Ceil(w))
}

// ovTestArea is the work area the overlay takes without a display
// (tests): a 1440 × 900 one under a menu bar.
var ovTestArea = mygo.Rectangle{X: 0, Y: 25, Width: 1440, Height: 875}

// overlayArea is the work area of the display under the pointer.
func (a *app) overlayArea() mygo.Rectangle {
	if !a.overlayNative() {
		return ovTestArea
	}
	return mygo.Screen.DisplayNearestPoint(mygo.Screen.CursorScreenPoint()).WorkArea
}
