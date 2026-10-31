package session

import (
	"slices"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// TestWaiting: a recording says what it waits for once it waited longer
// than WaitNote: the microphone's first audio, then the speech model
// still starting (10-20: the first session after launch, the models
// loading, 5.9 s without audio). The recording itself goes on meanwhile
// (session_test.go's "slow services": all of it reaches the model once it
// is ready). A quick start says nothing.
func TestWaiting(t *testing.T) {
	for _, tc := range []struct {
		name      string
		micWait   time.Duration
		modelWait time.Duration
		want      []Wait // the distinct values Waiting took, in order
	}{
		{name: "quick", want: []Wait{WaitNone}},
		{name: "the microphone", micWait: 150 * time.Millisecond, want: []Wait{WaitNone, WaitMic, WaitNone}},
		{name: "the speech model", modelWait: 250 * time.Millisecond, want: []Wait{WaitNone, WaitModel, WaitNone}},
		{name: "both", micWait: 150 * time.Millisecond, modelWait: 300 * time.Millisecond, want: []Wait{WaitNone, WaitMic, WaitModel, WaitNone}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			st := &fakeStreamer{final: "hello", openWait: tc.modelWait}
			h := newHarness(t, st, nil, func(c *Config, _ *settings.Settings) { c.WaitNote = 60 * time.Millisecond })
			h.audio.startWait = tc.micWait
			id := h.start()
			h.rec.waitFor(t, "the stream", func(s Snapshot) bool { return !s.HeardAt.IsZero() && s.Waiting == WaitNone })
			h.speak(id, 5)
			end := h.rec.wait(t)
			if end.State != Done || end.Result != "hello" {
				t.Fatalf("end %+v", end)
			}
			var seen []Wait
			for _, s := range h.rec.all() {
				if len(seen) == 0 || seen[len(seen)-1] != s.Waiting {
					seen = append(seen, s.Waiting)
				}
				if s.Waiting == WaitMic && !s.HeardAt.IsZero() || s.Waiting != WaitNone && s.State != Recording && s.State != Recognizing {
					t.Errorf("waiting %q in %+v", s.Waiting, s)
				}
			}
			if !slices.Equal(seen, tc.want) {
				t.Errorf("waiting %q, want %q", seen, tc.want)
			}
		})
	}
}

// TestWaitingAfterStop: stopped before the speech model is ready, the
// session says it waits for it while it recognises.
func TestWaitingAfterStop(t *testing.T) {
	st := &fakeStreamer{final: "late", openWait: 400 * time.Millisecond}
	h := newHarness(t, st, nil, func(c *Config, _ *settings.Settings) { c.WaitNote = 30 * time.Millisecond })
	id := h.start()
	h.rec.waitFor(t, "waiting for the model", func(s Snapshot) bool { return s.Waiting == WaitModel })
	h.m.Stop(id)
	h.rec.waitFor(t, "recognising, still waiting", func(s Snapshot) bool { return s.State == Recognizing && s.Waiting == WaitModel })
	if end := h.rec.wait(t); end.State != Done || end.Result != "late" {
		t.Errorf("end %+v", end)
	}
}
