package main

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

const ovSaveDelay = 400 * time.Millisecond

func (a *app) overlaySettingsModel() settingsOverlayViewState {
	return buildSettingsOverlayViewState(i18n.CurrentTranslator(), a.prefs.Overlay)
}

// loadOverlayForm sets the controls from the settings, unless a change
// waits to be saved (the sliders move).
func (a *app) loadOverlayForm() {
	f := &a.settingsForm.ovStyle
	if !f.saveAt.IsZero() {
		return
	}
	o := a.prefs.Overlay
	f.position = 0
	if o.Position == "top" {
		f.position = 1
	}
	f.opacity, f.radius, f.margin, f.showTime = o.Opacity, o.Radius, o.Margin, o.ShowTime
	f.liveText = !o.HideLiveText
	f.liveLayout = 0
	if o.LiveLines {
		f.liveLayout = 1
	}
}

// overlayEdited applies the controls: at once to the overlay and the
// preview, to the settings after ovSaveDelay without another change.
func (a *app) overlayEdited() {
	f := &a.settingsForm.ovStyle
	o := f.overlay()
	a.prefs.Overlay = o
	a.overlayRestyle() // overlay.go
	if a.win == nil {
		a.updateSettings(func(s *settings.Settings) { s.Overlay = o })
		return
	}
	at := time.Now()
	f.saveAt = at
	time.AfterFunc(ovSaveDelay, func() {
		a.post(func() {
			if f.saveAt != at {
				return // a later change saves
			}
			f.saveAt = time.Time{}
			o := a.prefs.Overlay
			a.updateSettings(func(s *settings.Settings) { s.Overlay = o })
		})
	})
}

// previewSession is the recording the preview shows: Transcribe, seven
// seconds in, a few words on the waveform (bars 4 to 20 high, as speech
// moves them).
func previewSession() session.Snapshot {
	t0 := time.Date(2026, 1, 1, 9, 0, 0, 0, time.UTC)
	levels := []float64{0.22, 0.38, 0.52, 0.66, 0.3, 0.2, 0.36, 0.6, 0.76, 0.42, 0.22, 0.4, 0.58, 0.32, 0.24, 0.48}
	return session.Snapshot{Kind: store.KindTranscribe, State: session.Recording, StartedAt: t0, HeardAt: t0, StoppedAt: t0.Add(7 * time.Second), Levels: levels}
}
