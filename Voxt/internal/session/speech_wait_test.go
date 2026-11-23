package session

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestSpeechWait: a session told to wait for the speech model does not
// record; translating a selection needs no speech model and goes on.
func TestSpeechWait(t *testing.T) {
	const wait = "still downloading (42%)"
	t.Run("speech", func(t *testing.T) {
		h := newHarness(t, &fakeFile{text: "x"}, &fakeLLM{text: "y"}, func(c *Config, _ *settings.Settings) { c.Select = &fakeSelector{} })
		if _, err := h.m.Start(StartRequest{Kind: store.KindTranslate, Target: slack, SpeechWait: wait}); err != nil {
			t.Fatal(err)
		}
		end := h.rec.wait(t)
		if end.State != Failed || end.Problem == nil || end.Problem.Code != ProblemSpeechWait || end.Problem.Err.Error() != wait {
			t.Fatalf("end %+v", end)
		}
		h.audio.mu.Lock()
		defer h.audio.mu.Unlock()
		if len(h.audio.stops) != 0 {
			t.Errorf("recorded: %v", h.audio.stops)
		}
	})
	t.Run("selection", func(t *testing.T) {
		h := newHarness(t, &fakeFile{text: "x"}, &fakeLLM{text: "早上好"}, func(c *Config, _ *settings.Settings) { c.Select = &fakeSelector{text: "Good morning"} })
		if _, err := h.m.Start(StartRequest{Kind: store.KindTranslate, Target: slack, SpeechWait: wait}); err != nil {
			t.Fatal(err)
		}
		if end := h.rec.wait(t); end.State != Done || end.Result != "早上好" {
			t.Fatalf("end %+v", end)
		}
	})
}
