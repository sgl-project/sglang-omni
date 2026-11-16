package main

// Overlay text measurement keys and deferred view measurement. The session
// owner stores posted measurements and computes the native face size.

import (
	"math"
	"slices"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

type ovSized struct {
	text   string
	size   float32
	weight int
	wrap   float32
}

func (key ovSized) span() ui.Span {
	return ui.Span{Text: key.text, Size: key.size, Weight: key.weight}
}

type ovMeasured struct{ w, h float32 }

func liveTextKey(text string) ovSized {
	return ovSized{text, layout.LiveSize, layout.LiveWeight, 0}
}

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
		width, ok := measure(kindTitleText(pill.Title))
		modeLabel = width + float32(layout.Gap)
		exact = ok
	}
	return layout.RowWidth(showTime, scene) + int(math.Ceil(float64(modeLabel))), exact
}

func noticeText(text string) ovSized { return ovSized{text, layout.NoticeSize, 0, 0} }

func sessionOverlayTexts(state sessionOverlayViewState, layoutState sessionOverlayLayout) []ovSized {
	var texts []ovSized
	switch state.Mode {
	case ovNotice:
		if state.Notice != nil {
			texts = append(texts, noticeText(state.Notice.Text))
		}
	case ovRecording, ovWorking:
		pill := state.Pill
		if pill.SceneText != "" {
			texts = append(texts, sceneText(pill.SceneText))
		}
		if state.Mode == ovRecording && !pill.Starting {
			texts = append(texts, kindTitleText(pill.Title))
		}
		if pill.CanSkip {
			texts = append(texts, capsuleText(pill.SkipLabel))
		}
		if pill.CanThink {
			texts = append(texts, capsuleText(pill.ThinkingLabel))
		}
		if pill.LiveText != "" && layoutState.LiveLines && !pill.LiveHint {
			texts = append(texts, liveLinesKey(pill.LiveText))
		} else if pill.LiveText != "" {
			texts = append(texts, liveTextKey(pill.LiveText))
		}
	case ovAnswer:
		for index := 0; index < sessionCardBlockCount(state.Card); index++ {
			text, title, quote := sessionCardBlockText(state.Card, sessionCardBlockAt(state.Card, index))
			texts = append(texts, text)
			if title.text != "" {
				texts = append(texts, title)
			}
			if quote.text != "" {
				texts = append(texts, quote)
			}
		}
	}
	return texts
}

func sessionOverlayMeasure(context *ui.Context, state sessionOverlayViewState, layoutState sessionOverlayLayout, onMeasured func([]sessionOverlayMeasurement)) {
	var measurements []sessionOverlayMeasurement
	var keys []ovSized
	for _, key := range sessionOverlayTexts(state, layoutState) {
		if _, ok := layoutState.Measured[key]; ok || slices.Contains(keys, key) {
			continue
		}
		width, height := context.MeasureText(key.wrap, key.span())
		keys = append(keys, key)
		measurements = append(measurements, sessionOverlayMeasurement{Key: key, Width: width, Height: height})
	}
	if len(measurements) > 0 && onMeasured != nil {
		onMeasured(measurements)
	}
}
