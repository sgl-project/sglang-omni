package main

import (
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestPillRowKeepsItsWidthWhenWorking: when the recording turns to work,
// the pill's row stays as wide as it was and centred, so its icon keeps
// its place on the screen (the window is centred there) and the skip
// capsule ends where the waveform's slot did: with the live text in lines
// or on one line, with the time, and without live text.
func TestPillRowKeepsItsWidthWhenWorking(t *testing.T) {
	for _, tc := range []struct {
		name        string
		lines, time bool
		text        string
	}{
		{"lines", true, false, "Meet at three tomorrow, no, on Thursday"},
		{"one line", false, false, "Meet at three tomorrow"},
		{"one line with the time", false, true, "Meet at three tomorrow"},
		{"no live text", false, false, ""},
		{"no live text with the time", false, true, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, _, _ := newTestApp(t)
			a.prefs.Overlay.LiveLines, a.prefs.Overlay.ShowTime = tc.lines, tc.time
			now := time.Now()
			rec := session.Snapshot{ID: "s1", Kind: store.KindTranscribe, State: session.Recording, StartedAt: now, HeardAt: now,
				Levels: wave(), Partial: tc.text}
			recognising := rec
			recognising.State, recognising.StoppedAt = session.Recognizing, now.Add(5*time.Second)
			composing := recognising
			composing.State, composing.CanSkip, composing.Spoken = session.Composing, true, tc.text+"."
			// x of label's left and right edges from the window's middle.
			find := func(s session.Snapshot, labels ...string) [][2]float32 {
				t.Helper()
				snap(a, s)
				a.refreshSessionOverlay(now)
				w, h := a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad
				tt := hudTester(func(c *ui.Context) {
					sessionOverlayView(c, &a.ov.projection.view, a.currentSessionOverlayLayout(), a,
						sessionOverlayLocal{Overlay: &a.ov.ui, Card: &a.answer.ui}, sessionOverlayAnimation{},
						a.postSessionOverlayMeasurements, func() { a.ov.reveal = false })
				}, w, h)
				tt.Frame()
				tt.Frame()
				var out [][2]float32
				for _, label := range labels {
					r, ok := tt.Find(label)
					if !ok {
						t.Fatalf("no %q in the %s pill", label, s.State)
					}
					mid := float32(w) / 2
					out = append(out, [2]float32{r.X - mid, r.X + r.W - mid})
				}
				return out
			}
			near := func(a, b float32) bool { return a-b < 0.75 && b-a < 0.75 }
			title := i18n.T("Transcribe")
			skip := i18n.T("Use transcript")
			r := find(rec, title, i18n.T("Input level"))
			icon, slotEnd := r[0][0], r[1][1]+(layout.WaveSlotW-layout.WaveW)/2
			if got := find(recognising, title)[0][0]; !near(got, icon) {
				t.Errorf("recognising: the icon at %.1f from the middle, %.1f while recording", got, icon)
			}
			c := find(composing, title, skip)
			if !near(c[0][0], icon) {
				t.Errorf("with the skip capsule: the icon at %.1f from the middle, %.1f while recording", c[0][0], icon)
			}
			if !tc.time && !near(c[1][1], slotEnd) {
				t.Errorf("the skip capsule ends at %.1f from the middle, the waveform's slot at %.1f", c[1][1], slotEnd)
			}
		})
	}
}
