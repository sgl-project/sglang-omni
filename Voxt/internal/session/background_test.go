package session

import (
	"errors"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// say records n frames of the session that started after base frames, then
// stops it (harness.speak counts every session's frames).
func (h *harness) say(id string, base, n int) {
	h.t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for h.audio.recorded() < base+n {
		if time.Now().After(deadline) {
			h.t.Fatalf("only %d frames recorded", h.audio.recorded()-base)
		}
		time.Sleep(time.Millisecond)
	}
	h.m.Stop(id)
}

// TestBackgroundComposesBesideANewSession (wall 116): a Rewrite answer
// still streaming backgrounds so the transcribe shortcut can start: the
// answer finishes into the card and the history while the new session
// records, and StopLLM still reaches the backgrounded one.
func TestBackgroundComposesBesideANewSession(t *testing.T) {
	full := "好的，这是 the shorter version：发布推迟到 Thursday。"
	h := newHarness(t, &fakeFile{text: "dictated"}, &fakeLLM{text: full, chunk: 2, pause: 15 * time.Millisecond}, func(c *Config, s *settings.Settings) {
		c.Select = &fakeSelector{text: "Dear Ann"}
		s.Features.Transcribe.Enhance = false // the dictation stays as recognised
	})
	id, _ := h.m.Start(StartRequest{Kind: store.KindRewrite, Target: slack})
	h.speak(id, 3)
	drafting := h.rec.waitFor(t, "the answer streaming", func(s Snapshot) bool { return s.ID == id && s.CanStop() })

	if !h.m.Background(id) {
		t.Fatal("a composing session would not background")
	}
	if h.m.Background(id) || h.m.Background("nobody") {
		t.Fatal("backgrounded twice or a stranger")
	}
	for _, kind := range []store.Kind{store.KindRewrite, store.KindTranslate} {
		if extra, err := h.m.Start(StartRequest{Kind: kind, Target: slack}); !errors.Is(err, ErrBusy) {
			h.m.Cancel(extra)
			t.Errorf("%s started beside the backgrounded answer: %v", kind, err)
		}
	}
	other, err := h.m.Start(StartRequest{Kind: store.KindTranscribe, Target: slack})
	if err != nil || other == id {
		t.Fatalf("the second session: %v", err)
	}
	h.rec.waitFor(t, "the new session recording", func(s Snapshot) bool { return s.ID == other && s.State == Recording })
	h.rec.waitFor(t, "the answer longer in the background", func(s Snapshot) bool {
		return s.ID == id && s.State == Composing && len(s.Draft) > len(drafting.Draft)
	})
	h.m.StopLLM(id) // the card's Stop, while the new session records
	end := h.rec.waitFor(t, "the backgrounded answer's end", func(s Snapshot) bool { return s.ID == id && s.State.Ended() })
	if end.State != Done || end.Result == "" || end.Result == full || !end.Card {
		t.Errorf("end %s %q card %v", end.State, end.Result, end.Card)
	}
	if s := h.m.Current(); s.ID != other || s.State != Recording {
		t.Errorf("the new session is the current one: %+v", s)
	}
	h.say(other, 3, 3)
	if end := h.rec.waitFor(t, "the second session's end", func(s Snapshot) bool { return s.ID == other && s.State.Ended() }); end.State != Done || end.Result != "dictated" {
		t.Errorf("the second session ended %+v", end)
	}
	if es := h.hist.all(); len(es) != 2 || es[0].ID != id || es[0].Status != store.StatusDone || es[1].ID != other {
		t.Errorf("entries %+v", es)
	}
	if got := h.m.Last(); got == "" {
		t.Error("nothing to paste as the last result")
	}
}

// TestBackgroundCancel: cancelling the backgrounded answer ends it where
// it is and keeps no result, while the new session runs on.
func TestBackgroundCancel(t *testing.T) {
	h := newHarness(t, &fakeFile{text: "dictated"}, &fakeLLM{hang: true}, func(c *Config, s *settings.Settings) {
		c.Select = &fakeSelector{text: "Dear Ann"}
		s.Features.Transcribe.Enhance = false
	})
	id, _ := h.m.Start(StartRequest{Kind: store.KindRewrite, Target: slack})
	h.speak(id, 3)
	h.rec.waitFor(t, "the answer streaming", func(s Snapshot) bool { return s.ID == id && s.CanStop() })
	if !h.m.Background(id) {
		t.Fatal("a composing session would not background")
	}
	other, err := h.m.Start(StartRequest{Kind: store.KindTranscribe, Target: slack})
	if err != nil {
		t.Fatal(err)
	}
	h.m.Cancel(id)
	if end := h.rec.waitFor(t, "the cancelled answer", func(s Snapshot) bool { return s.ID == id && s.State.Ended() }); end.State != Cancelled {
		t.Errorf("end %s", end.State)
	}
	if s := h.m.Current(); s.ID != other || !s.State.Live() {
		t.Errorf("the new session runs on: %+v", s)
	}
	h.say(other, 3, 3)
	h.rec.waitFor(t, "the second session's end", func(s Snapshot) bool { return s.ID == other && s.State.Ended() })
	// A cancelled session keeps its recording for Retry; no result.
	es := h.hist.all()
	if len(es) != 2 || es[0].ID != id || es[0].Status != store.StatusCancelled || es[0].Result != "" || es[1].ID != other || es[1].Result != "dictated" {
		t.Errorf("entries %+v", es)
	}
}

// TestBackgroundOnlyComposing: a session that is not composing cannot
// background: the earlier states need the session's place.
func TestBackgroundOnlyComposing(t *testing.T) {
	h := newHarness(t, &fakeFile{text: "dictated"}, &fakeLLM{hang: true}, nil)
	id := h.start()
	if h.m.Background(id) {
		t.Error("a recording session backgrounded")
	}
	h.speak(id, 3)
	h.rec.wait(t)
	if h.m.Background(id) {
		t.Error("an ended session backgrounded")
	}
}
