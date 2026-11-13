package session

import (
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestLLMWaitSetting (#3821): the language model wait the user chose
// (settings.Models.LLMWait) is the least deadline of Transcribe's,
// Translate's and Rewrite's requests; Automatic keeps the session's own.
func TestLLMWaitSetting(t *testing.T) {
	for _, kind := range []store.Kind{store.KindTranscribe, store.KindTranslate, store.KindRewrite} {
		for wait, want := range map[string]time.Duration{settings.WaitAuto: 0, settings.Wait5Min: 5 * time.Minute, "20m": 20 * time.Minute} { // 20m: a custom wait (#3941)
			t.Run(string(kind)+"/"+wait, func(t *testing.T) {
				model := &fakeLLM{text: "Hello."}
				h := newHarness(t, &fakeFile{text: "hello"}, model, func(c *Config, s *settings.Settings) {
					c.Select = &fakeSelector{text: map[store.Kind]string{store.KindRewrite: "Dear Ann"}[kind]}
					s.Models.LLMWait = wait
				})
				id, err := h.m.Start(StartRequest{Kind: kind, Target: slack})
				if err != nil {
					t.Fatal(err)
				}
				h.speak(id, 3)
				if end := h.rec.wait(t); end.State != Done {
					t.Fatalf("end %+v", end)
				}
				reqs := model.requests()
				if len(reqs) == 0 {
					t.Fatal("no request")
				}
				got := reqs[0].Timeout
				if want > 0 && got != want || want == 0 && got >= 5*time.Minute {
					t.Errorf("deadline %v with %q", got, wait)
				}
			})
		}
	}
}

// TestSkipLLM (#3821): while the language model works on Transcribe's or
// Translate's speech, SkipLLM ends its step and the recognised text is
// inserted, with no notice; Rewrite's speech is an instruction, so its
// step cannot be skipped.
func TestSkipLLM(t *testing.T) {
	for _, kind := range []store.Kind{store.KindTranscribe, store.KindTranslate} {
		t.Run(string(kind), func(t *testing.T) {
			h := newHarness(t, &fakeFile{text: "hello aura link"}, &fakeLLM{hang: true}, func(c *Config, _ *settings.Settings) {
				c.Select = &fakeSelector{}
			})
			id, err := h.m.Start(StartRequest{Kind: kind, Target: slack})
			if err != nil {
				t.Fatal(err)
			}
			h.speak(id, 3)
			h.rec.waitFor(t, "composing", func(s Snapshot) bool { return s.State == Composing && s.CanSkip })
			h.m.SkipLLM(id)
			end := h.rec.wait(t)
			if end.State != Done || end.Problem != nil || end.Result != "hello VoxtNew" {
				t.Errorf("end %s %+v %q", end.State, end.Problem, end.Result)
			}
			if p := h.text.pastes(); len(p) != 1 || p[0] != "hello VoxtNew" {
				t.Errorf("pasted %q", p)
			}
			if es := h.hist.all(); len(es) != 1 || es[0].Status != store.StatusDone {
				t.Errorf("entries %+v", es)
			}
		})
	}
	t.Run("rewrite", func(t *testing.T) {
		h := newHarness(t, &fakeFile{text: "shorter"}, &fakeLLM{hang: true}, func(c *Config, _ *settings.Settings) {
			c.Select = &fakeSelector{text: "Dear Ann"}
		})
		id, _ := h.m.Start(StartRequest{Kind: store.KindRewrite, Target: slack})
		h.speak(id, 3)
		s := h.rec.waitFor(t, "composing", func(s Snapshot) bool { return s.State == Composing })
		if s.CanSkip {
			t.Error("Rewrite's step can be skipped")
		}
		h.m.SkipLLM(id)
		time.Sleep(50 * time.Millisecond)
		if st := h.m.Current().State; st != Composing {
			t.Errorf("SkipLLM moved Rewrite to %s", st)
		}
		h.m.Cancel(id)
		if end := h.rec.wait(t); end.State != Cancelled || len(h.text.pastes()) != 0 {
			t.Errorf("end %+v, pasted %q", end, h.text.pastes())
		}
	})
}

// TestTranslateTimeoutKeepsSpeech (#3821): a translation of speech whose
// model does not answer in time inserts the recognised speech, with the
// warning, instead of ending failed.
func TestTranslateTimeoutKeepsSpeech(t *testing.T) {
	h := newHarness(t, &fakeFile{text: "hello"}, &fakeLLM{hang: true}, func(c *Config, _ *settings.Settings) {
		c.Select = &fakeSelector{}
		c.LLMBase, c.WriteTimeout = 200*time.Millisecond, 300*time.Millisecond
	})
	id, _ := h.m.Start(StartRequest{Kind: store.KindTranslate, Target: slack})
	h.speak(id, 3)
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "hello" || end.Problem == nil || end.Problem.Code != ProblemLLM || !end.Problem.Degraded || !TimedOut(end.Problem.Err) {
		t.Errorf("end %s %+v %q", end.State, end.Problem, end.Result)
	}
	if p := h.text.pastes(); len(p) != 1 || p[0] != "hello" {
		t.Errorf("pasted %q", p)
	}
}
