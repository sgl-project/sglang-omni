package main

// The overlay's size (overlay.go): the face's width and height in each mode,
// from its texts as the view measured them, or guessed until it did; the
// work area it shows in.

import (
	"math"
	"slices"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// overlaySize is the face's size for mode, and whether its width is the
// measured texts' (not guessed).
func (a *app) overlaySize(mode overlayMode) (w, h int, exact bool) {
	if mode != ovAnswer {
		a.refreshSessionOverlay(time.Now())
	}
	switch mode {
	case ovAnswer:
		return a.cardSize() // overlay_card.go
	case ovNotice:
		notice := a.ov.projection.view.Notice
		if notice == nil {
			return layout.NoticeW, layout.PillH, true
		}
		tw, exact := a.ovTextWidth(noticeText(notice.Text))
		return layout.NoticeWidth(tw, notice.Action != actionNone), layout.PillH, exact
	case ovRecording, ovWorking:
		pill := a.ov.projection.view.Pill
		scene, exact := float32(0), true
		if pill.SceneText != "" {
			scene, exact = a.ovTextWidth(sceneText(pill.SceneText))
		}
		row, rowExact := recordingPillRow(pill, mode == ovRecording, mode == ovRecording && a.prefs.Overlay.ShowTime, scene, a.ovTextWidth)
		exact = exact && rowExact
		if pill.CanSkip {
			w, ok := a.ovTextWidth(capsuleText(pill.SkipLabel))
			row += layout.Gap + 2*10 + int(math.Ceil(float64(w))) // cardCapsule's padding
			exact = exact && ok
		}
		if pill.CanThink {
			w, ok := a.ovTextWidth(capsuleText(pill.ThinkingLabel))
			row += layout.Gap + 2*10 + int(math.Ceil(float64(w)))
			exact = exact && ok
		}
		text, hint := pill.LiveText, pill.LiveHint
		if a.prefs.Overlay.LiveLines {
			// Wide from the start, before any text: the face does not jump
			// from the small pill at the first word (the user, 11-09).
			if text == "" || hint {
				return layout.LinesW, layout.LinesHeight(0), true
			}
			th, known := a.ovTextHeight(liveLinesKey(text))
			return layout.LinesW, layout.LinesHeight(th), known
		}
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

// liveLinesKey is the live text wrapped in lines (Overlay.LiveLines).
func liveLinesKey(text string) ovSized {
	return ovSized{text, layout.LiveSize, layout.LiveWeight, layout.LinesTextW}
}

func sceneText(name string) ovSized { return ovSized{name, layout.SceneSize, 0, 0} }

func kindTitleText(title string) ovSized { return ovSized{title, 12, 500, 0} }

func capsuleText(text string) ovSized { return ovSized{text, 11, 600, 0} }

func recordingPillRow(pill sessionPillViewState, label, showTime bool, scene float32, measure func(ovSized) (float32, bool)) (int, bool) {
	modeLabel := float32(0)
	exact := true
	if label && !pill.Starting {
		w, ok := measure(kindTitleText(pill.Title))
		modeLabel = w + float32(layout.Gap)
		exact = ok
	}
	return layout.RowWidth(showTime, scene) + int(math.Ceil(float64(modeLabel))), exact
}

// recordingPillFaceW is the face width of a recording pill without live text.
func recordingPillFaceW(s session.Snapshot, showTime bool) int {
	pill := sessionPillViewState{Title: uitext.KindTitle(s.Kind), Starting: s.Waiting != session.WaitNone}
	scene := float32(0)
	if name := uitext.SceneShort(s); name != "" {
		scene = float32(textWidth(name, float64(layout.SceneSize)))
	}
	row, _ := recordingPillRow(pill, true, showTime, scene, func(k ovSized) (float32, bool) {
		return float32(textWidth(k.text, float64(k.size))), true
	})
	return row + 2*layout.PillPadX
}

func noticeText(text string) ovSized { return ovSized{text, layout.NoticeSize, 0, 0} }

// faceRadius is the face's corner radius in mode, h DIPs high: the
// setting's, at most a capsule for the pill.
func faceRadius(o settings.Overlay, mode overlayMode, h int) float32 {
	radius := float32(o.Radius)
	if mode != ovAnswer {
		radius = min(radius, float32(h)/2)
	}
	return radius
}

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

// Text measurement lives in ui_text_measure.go.
// overlayTexts are the texts the face's size depends on in its mode.
func (a *app) overlayTexts() []ovSized {
	var out []ovSized
	switch a.ov.mode {
	case ovNotice:
		if notice := a.ov.projection.view.Notice; notice != nil {
			out = append(out, noticeText(notice.Text))
		}
	case ovRecording, ovWorking:
		pill := a.ov.projection.view.Pill
		if pill.SceneText != "" {
			out = append(out, sceneText(pill.SceneText))
		}
		if a.ov.mode == ovRecording && !pill.Starting {
			out = append(out, kindTitleText(pill.Title))
		}
		if pill.CanSkip {
			out = append(out, capsuleText(pill.SkipLabel))
		}
		if pill.CanThink {
			out = append(out, capsuleText(pill.ThinkingLabel))
		}
		if pill.LiveText != "" && a.prefs.Overlay.LiveLines && !pill.LiveHint {
			out = append(out, liveLinesKey(pill.LiveText))
		} else if pill.LiveText != "" {
			out = append(out, liveTextKey(pill.LiveText))
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

// Text measurement lives in ui_text_measure.go.
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
