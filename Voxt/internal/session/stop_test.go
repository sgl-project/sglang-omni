package session

import (
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestStopLLM (#3886): while an answer streams into the answer card,
// StopLLM ends the model's request and the session ends done with the
// answer so far in the card: nothing pasted, the history keeps it. With
// the card off the answer goes in whole or not at all, so StopLLM does
// nothing there.
func TestStopLLM(t *testing.T) {
	for _, kind := range []store.Kind{store.KindRewrite, store.KindTranslate} {
		t.Run(string(kind), func(t *testing.T) {
			h := newHarness(t, &fakeFile{text: "shorter"}, &fakeLLM{hang: true}, func(c *Config, s *settings.Settings) {
				c.Select = &fakeSelector{text: "Dear Ann, about the release"}
				s.Features.Translate.ShowCard = true
			})
			id, _ := h.m.Start(StartRequest{Kind: kind, Target: slack})
			if kind == store.KindRewrite {
				h.speak(id, 3)
			}
			h.rec.waitFor(t, "streaming into the card", func(s Snapshot) bool { return s.CanStop() })
			h.m.StopLLM(id)
			end := h.rec.wait(t)
			if end.State != Done || end.Problem != nil || end.Result != "Half an ans" || !end.Card || end.Inserted {
				t.Errorf("end %s %+v %q card %v inserted %v", end.State, end.Problem, end.Result, end.Card, end.Inserted)
			}
			if p := h.text.pastes(); len(p) != 0 {
				t.Errorf("pasted %q", p)
			}
			if es := h.hist.all(); len(es) != 1 || es[0].Status != store.StatusDone || es[0].Result != "Half an ans" {
				t.Errorf("entries %+v", es)
			}
		})
	}
	// Mixed Chinese and English, streamed a few runes at a time: the card
	// keeps every word that came, also those since the last throttled draft.
	t.Run("mid stream", func(t *testing.T) {
		full := "好的，这是 the shorter version：发布推迟到 Thursday，原因是 QA 还没跑完。"
		h := newHarness(t, &fakeFile{text: "shorter"}, &fakeLLM{text: full, chunk: 3, pause: 20 * time.Millisecond}, func(c *Config, _ *settings.Settings) {
			c.Select = &fakeSelector{text: "Dear Ann"}
		})
		id, _ := h.m.Start(StartRequest{Kind: store.KindRewrite, Target: slack})
		h.speak(id, 3)
		h.rec.waitFor(t, "a few words", func(s Snapshot) bool { return s.CanStop() && len([]rune(s.Draft)) >= 6 })
		h.m.StopLLM(id)
		end := h.rec.wait(t)
		if end.State != Done || end.Result == "" || end.Result == full || !strings.HasPrefix(full, end.Result) || !end.Card {
			t.Errorf("end %s %q card %v", end.State, end.Result, end.Card)
		}
	})
	t.Run("card off", func(t *testing.T) {
		h := newHarness(t, &fakeFile{text: "shorter"}, &fakeLLM{hang: true}, func(c *Config, s *settings.Settings) {
			c.Select = &fakeSelector{}
			rewriteNoCard(s)
		})
		id, _ := h.m.Start(StartRequest{Kind: store.KindRewrite, Target: slack})
		h.speak(id, 3)
		s := h.rec.waitFor(t, "a draft", func(s Snapshot) bool { return s.State == Composing && s.Draft != "" })
		if s.CanStop() {
			t.Error("an answer that goes in at once can be stopped")
		}
		h.m.StopLLM(id)
		time.Sleep(50 * time.Millisecond)
		if st := h.m.Current().State; st != Composing {
			t.Errorf("StopLLM moved the session to %s", st)
		}
		h.m.Cancel(id)
		if end := h.rec.wait(t); end.State != Cancelled || len(h.text.pastes()) != 0 {
			t.Errorf("end %+v, pasted %q", end, h.text.pastes())
		}
	})
}
