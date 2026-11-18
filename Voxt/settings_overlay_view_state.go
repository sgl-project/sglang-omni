package main

import (
	"math"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// overlayPositions are the values of Overlay.Position, by segment.
var overlayPositions = []string{"bottom", "top"}

type settingsOverlayPreviewState struct {
	Position string  `json:"position"`
	Opacity  float64 `json:"opacity"`
	Radius   float64 `json:"radius"`
	Margin   float64 `json:"margin"`
	ShowTime bool    `json:"show_time"`

	pill       sessionOverlayViewState
	faceRadius float32
	faceAlpha  float32
}

type settingsOverlayViewState struct {
	Position          int      `json:"position"`
	PositionOptions   []string `json:"position_options"`
	Opacity           float64  `json:"opacity"`
	Radius            float64  `json:"radius"`
	Margin            float64  `json:"margin"`
	ShowTime          bool     `json:"show_time"`
	LiveText          bool     `json:"live_text"`
	LiveLayout        int      `json:"live_layout"`
	LiveLayoutOptions []string `json:"live_layout_options"`
}

func buildSettingsOverlayViewState(translator *i18n.Translator, overlay settings.Overlay) settingsOverlayViewState {
	position := 0
	if overlay.Position == "top" {
		position = 1
	}
	liveLayout := 0
	if overlay.LiveLines {
		liveLayout = 1
	}
	return settingsOverlayViewState{
		Position: position, PositionOptions: []string{translator.T("Bottom centre"), translator.T("Top centre")},
		Opacity: overlay.Opacity, Radius: overlay.Radius, Margin: overlay.Margin, ShowTime: overlay.ShowTime,
		LiveText: !overlay.HideLiveText, LiveLayout: liveLayout,
		LiveLayoutOptions: []string{translator.T("One line"), translator.T("Lines")},
	}
}

func (f *overlayForm) overlay() settings.Overlay {
	return settings.Overlay{
		Position: overlayPositions[min(max(f.position, 0), 1)],
		Opacity:  math.Round(f.opacity*100) / 100,
		Radius:   math.Round(f.radius),
		Margin:   math.Round(f.margin),
		ShowTime: f.showTime,

		HideLiveText: !f.liveText,
		LiveLines:    f.liveLayout == 1,
	}
}

// previewState is the preview's pill and face, projected through the
// session overlay's production builders so the preview draws what the
// overlay draws; the renderer stays a view and reads only this state.
func (f *overlayForm) previewState() settingsOverlayPreviewState {
	overlay := f.overlay()
	return settingsOverlayPreviewState{
		Position: overlay.Position, Opacity: overlay.Opacity, Radius: overlay.Radius,
		Margin: overlay.Margin, ShowTime: overlay.ShowTime,
		pill: buildSessionOverlayViewState(i18n.CurrentTranslator(), sessionOverlayProjectionInput{
			Snapshot: previewSession(), Mode: ovRecording, ShowTime: overlay.ShowTime,
		}, time.Now()),
		faceRadius: faceRadius(overlay, ovRecording, layout.PillH),
		faceAlpha:  layout.FaceAlpha(overlay),
	}
}
