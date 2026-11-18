package main

import (
	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func (a *app) overlaySize(mode overlayMode) (width, height int, exact bool) {
	if mode == ovAnswer {
		return a.cardSize()
	}
	return a.sessionOverlaySize(&a.ov.projection.view)
}

func (a *app) sessionOverlaySize(state *sessionOverlayViewState) (width, height int, exact bool) {
	previewCount := 0
	if state.Card != nil && state.Card.Conversation {
		previewCount = sessionCardBlockCount(state.Card)
	}
	a.trimBubblePreviews(previewCount)
	overlay := a.prefs.Overlay
	if a.measure == nil {
		a.measure = a.ovText
	}
	model := sessionOverlayLayoutFor(state, sessionOverlayLayoutInput{
		Measure:  a.measure,
		WorkArea: a.ov.area,
		Overlay: sessionOverlayLayoutStyle{
			Position: overlay.Position, Opacity: overlay.Opacity, Radius: overlay.Radius, Margin: overlay.Margin,
			ShowTime: overlay.ShowTime, LiveLines: overlay.LiveLines,
		},
		Memory: sessionOverlayLayoutMemory{
			LiveID: a.ov.live.id, LiveWidth: a.ov.live.w, StreamHeight: a.ov.stream.h,
		},
		Measured:       a.ov.textW,
		PreviewHeights: a.sessionOverlayPreviewHeights(state),
		Reveal:         a.ov.reveal,
	})
	a.ov.live.id, a.ov.live.w = model.LiveID, model.LiveWidth
	a.ov.stream.h = model.StreamHeight
	a.ov.layout = model
	return model.Width, model.Height, model.Exact
}

func faceRadius(overlay settings.Overlay, mode overlayMode, height int) float32 {
	radius := float32(overlay.Radius)
	if mode != ovAnswer {
		radius = min(radius, float32(height)/2)
	}
	return radius
}

func (a *app) ovTextWidth(key ovSized) (float32, bool) {
	measured, ok := a.ovText(key)
	return measured.w, ok
}

func (a *app) ovText(key ovSized) (ovMeasured, bool) {
	if a.measureHook != nil {
		width, height := a.measureHook(key)
		return ovMeasured{width, height}, true
	}
	if measured, ok := a.ov.textW[key]; ok {
		return measured, true
	}
	width, height := guessText(key)
	return ovMeasured{width, height}, false
}

func (a *app) postSessionOverlayMeasurements(measurements []sessionOverlayMeasurement) {
	if len(measurements) == 0 {
		return
	}
	a.post(func() {
		if a.ov.textW == nil || len(a.ov.textW) > 256 {
			a.ov.textW = map[ovSized]ovMeasured{}
		}
		changed := false
		for _, measurement := range measurements {
			measured := ovMeasured{measurement.Width, measurement.Height}
			if previous, ok := a.ov.textW[measurement.Key]; !ok || previous != measured {
				a.ov.textW[measurement.Key], changed = measured, true
			}
		}
		// A Markdown text reports its height on every paint
		// (overlay_card_markdown.go): only a new one resizes the face.
		if changed {
			a.overlayRestyle()
		}
	})
}

var ovTestArea = mygo.Rectangle{X: 0, Y: 25, Width: 1440, Height: 875}

func (a *app) overlayArea() mygo.Rectangle {
	if !a.overlayNative() {
		return ovTestArea
	}
	return mygo.Screen.DisplayNearestPoint(mygo.Screen.CursorScreenPoint()).WorkArea
}

func (a *app) cardSize() (width, height int, exact bool) {
	return a.sessionOverlaySize(&a.ov.projection.view)
}
