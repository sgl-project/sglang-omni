package main

import (
	"bytes"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// TestWaveformGlides: while it moves, the recording pill's waveform draws
// the bars it is given at the time of the frame it paints (the levels
// played out, overlayWave) rather than the snapshot's own, and asks for a
// frame at every refresh; a recording's levels reach the player with their
// count.
func TestWaveformGlides(t *testing.T) {
	a, _, _ := newTestApp(t)
	now := time.Now()
	a.prefs.Overlay.LiveLines = false
	recording := session.Snapshot{State: session.Recording, StartedAt: now, HeardAt: now}
	snap(a, recording)
	var animation sessionOverlayAnimation
	tt := hudTester(func(context *ui.Context) {
		a.refreshSessionOverlay(now)
		sessionOverlayView(context, &a.ov.projection.view, a.currentSessionOverlayLayout(), a,
			sessionOverlayLocal{Overlay: &a.ov.ui, Card: &a.answer.ui}, animation,
			a.postSessionOverlayMeasurements, func() { a.ov.reveal = false })
	}, a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
	draw := func(levels []float64, gliding func(time.Time) layout.Wave) []byte {
		recording.Levels, recording.LevelCount = levels, len(levels)
		snap(a, recording)
		animation = sessionOverlayAnimation{Motion: true, Phase: 1, Wave: gliding}
		tt.Frame()
		return bytes.Clone(tt.Image().Pix)
	}
	flat := draw(nil, nil)
	if tt.FramePending() {
		t.Error("the stepping waveform asked for a frame at every refresh")
	}
	own := draw(wave(), nil)
	bars := layout.WaveBars(wave())
	var asked time.Time
	glide := draw(nil, func(at time.Time) layout.Wave { asked = at; return bars })
	if !tt.FramePending() {
		t.Error("the gliding waveform did not ask for the next refresh's frame")
	}
	if bytes.Equal(glide, flat) || !bytes.Equal(glide, own) {
		t.Error("the pill did not draw the bars it was given")
	}
	if !asked.Equal(tt.Now()) {
		t.Errorf("the bars were asked for at %v, not at the frame's time %v", asked, tt.Now())
	}

	// The levels reach the player as they come: played out, it shows them.
	recording.Levels, recording.LevelCount = wave(), 400
	snap(a, recording)
	if got := a.wave.Bars(time.Now().Add(time.Second)); got != bars {
		t.Errorf("the player's bars %v, want the levels' %v", got, bars)
	}
	// Nothing moves here (no native overlay): the bars are the levels' own.
	if a.overlayWave() != nil {
		t.Error("gliding bars without motion")
	}
	// A snapshot set as it is, not through sessionChanged (the overlay's
	// check), shows its levels.
	a.ov.native = true
	a.sess = session.Snapshot{ID: "set", State: session.Recording, Levels: wave()}
	glides := a.overlayWave()
	if glides == nil {
		t.Fatal("a snapshot set as it is: no gliding bars")
	}
	if got := glides(time.Now()); got != bars {
		t.Errorf("a snapshot set as it is: %v, want the levels' %v", got, bars)
	}
}
