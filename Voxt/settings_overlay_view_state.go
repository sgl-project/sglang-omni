package main

import (
	"bytes"
	"encoding/json"
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
	Position          int                          `json:"position"`
	PositionOptions   []string                     `json:"position_options"`
	Opacity           float64                      `json:"opacity"`
	Radius            float64                      `json:"radius"`
	Margin            float64                      `json:"margin"`
	ShowTime          bool                         `json:"show_time"`
	LiveText          bool                         `json:"live_text"`
	LiveLayout        int                          `json:"live_layout"`
	LiveLayoutOptions []string                     `json:"live_layout_options"`
	Preview           *settingsOverlayPreviewState `json:"preview"`
}

type settingsOverlayPreviewJSON struct {
	Position   string                  `json:"position"`
	Opacity    float64                 `json:"opacity"`
	Radius     float64                 `json:"radius"`
	Margin     float64                 `json:"margin"`
	ShowTime   bool                    `json:"show_time"`
	Pill       sessionOverlayViewState `json:"pill"`
	FaceRadius float32                 `json:"face_radius"`
	FaceAlpha  float32                 `json:"face_alpha"`
}

func (state settingsOverlayPreviewState) MarshalJSON() ([]byte, error) {
	return json.Marshal(settingsOverlayPreviewJSON{
		Position: state.Position, Opacity: state.Opacity, Radius: state.Radius,
		Margin: state.Margin, ShowTime: state.ShowTime,
		Pill: state.pill, FaceRadius: state.faceRadius, FaceAlpha: state.faceAlpha,
	})
}

func (state *settingsOverlayPreviewState) UnmarshalJSON(data []byte) error {
	var decoded settingsOverlayPreviewJSON
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&decoded); err != nil {
		return err
	}
	*state = settingsOverlayPreviewState{
		Position: decoded.Position, Opacity: decoded.Opacity, Radius: decoded.Radius,
		Margin: decoded.Margin, ShowTime: decoded.ShowTime,
		pill: decoded.Pill, faceRadius: decoded.FaceRadius, faceAlpha: decoded.FaceAlpha,
	}
	return nil
}

func buildSettingsOverlayViewState(translator *i18n.Translator, overlay settings.Overlay, now time.Time) settingsOverlayViewState {
	position := 0
	if overlay.Position == "top" {
		position = 1
	}
	liveLayout := 0
	if overlay.LiveLines {
		liveLayout = 1
	}
	state := settingsOverlayViewState{
		Position: position, PositionOptions: []string{translator.T("Bottom centre"), translator.T("Top centre")},
		Opacity: overlay.Opacity, Radius: overlay.Radius, Margin: overlay.Margin, ShowTime: overlay.ShowTime,
		LiveText: !overlay.HideLiveText, LiveLayout: liveLayout,
		LiveLayoutOptions: []string{translator.T("One line"), translator.T("Lines")},
	}
	previewForm := overlayForm{
		position: position, opacity: overlay.Opacity, radius: overlay.Radius,
		margin: overlay.Margin, showTime: overlay.ShowTime,
	}
	previewOverlay := previewForm.overlay()
	state.Preview = &settingsOverlayPreviewState{
		Position: previewOverlay.Position, Opacity: previewOverlay.Opacity, Radius: previewOverlay.Radius,
		Margin: previewOverlay.Margin, ShowTime: previewOverlay.ShowTime,
		pill: buildSessionOverlayViewState(translator, sessionOverlayProjectionInput{
			Snapshot: previewSession(), Mode: ovRecording, ShowTime: previewOverlay.ShowTime,
		}, now),
		faceRadius: faceRadius(previewOverlay, ovRecording, layout.PillH),
		faceAlpha:  layout.FaceAlpha(previewOverlay),
	}
	return state
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
