package session

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// TestASRLimit: recognition after Stop gets a floor plus a share of the
// recording, so a long recording recognised slowly is not cut (the
// user's 4m51s that failed at 30 s).
func TestASRLimit(t *testing.T) {
	cfg := New(Config{}).cfg
	for _, c := range []struct{ audio, want time.Duration }{
		{5 * time.Second, 65 * time.Second},
		{291 * time.Second, 351 * time.Second},
		{10 * time.Minute, 660 * time.Second},
	} {
		if got := cfg.asrLimit(c.audio); got != c.want {
			t.Errorf("limit for %v: %v, want %v", c.audio, got, c.want)
		}
	}
}

// TestASRFallback: a Final that fails or takes too long while the live
// session had text: that text goes on (clean-up, insert), the notice says
// so, and the entry keeps the recording for a retry.
func TestASRFallback(t *testing.T) {
	for _, tc := range []struct {
		name  string
		prov  *fakeStreamer
		limit bool // the session's limit ends it
		raw   string
		kind  asr.Kind
	}{
		{"the Final fails", &fakeStreamer{finalErr: &asr.Error{Kind: asr.KindServer, Detail: "the local speech model stopped answering"}}, false, "partial", asr.KindServer},
		{"the Final takes too long", &fakeStreamer{hang: true}, true, "partial", asr.KindTimeout},
		{"the stream salvages its pieces", &fakeStreamer{salvage: "pieces so far", finalErr: &asr.Error{Kind: asr.KindServer}}, false, "pieces so far", asr.KindServer},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &fakeLLM{text: "Cleaned."}
			h := newHarness(t, tc.prov, model, nil)
			if tc.limit {
				h.m.cfg.ASRTimeout, h.m.cfg.ASRPerSecond = 100*time.Millisecond, time.Nanosecond
			}
			id := h.start()
			h.speak(id, 5)
			end := h.rec.wait(t)
			if end.State != Done || end.Raw != tc.raw || end.Result != "Cleaned." || !end.Inserted || !end.Recording {
				t.Fatalf("end %+v", end)
			}
			if p := end.Problem; p == nil || p.Code != ProblemASR || !p.Degraded || asr.KindOf(p.Err) != tc.kind {
				t.Fatalf("problem %+v", end.Problem)
			}
			if p := h.text.pastes(); !slices.Equal(p, []string{"Cleaned."}) {
				t.Errorf("pasted %q", p)
			}
			if reqs := model.requests(); len(reqs) != 1 {
				t.Errorf("%d clean-ups", len(reqs))
			}
			es := h.hist.all()
			if len(es) != 1 || es[0].Status != store.StatusDone || es[0].Error == "" || es[0].RawText != tc.raw || es[0].Audio != id+".wav" {
				t.Fatalf("entries %+v", es)
			}
			if until := time.Until(es[0].AudioUntil); until < 6*24*time.Hour || until > 7*24*time.Hour {
				t.Errorf("kept until %v", es[0].AudioUntil)
			}
			if !writeFileForTest(t, h.hist.dir, id+".wav") {
				t.Error("no recording on disk")
			}
		})
	}
}

// TestRecordingRetention: with the settings keeping no recordings, an
// entry keeps its recording for a while only when recognition failed or
// found nothing; with them keeping recordings, for as long as the entry.
func TestRecordingRetention(t *testing.T) {
	for _, tc := range []struct {
		name      string
		prov      asr.Provider
		keepAudio bool
		kept      bool
		until     bool
	}{
		{"done", &fakeFile{text: "x"}, false, false, false},
		{"done, keeping recordings", &fakeFile{text: "x"}, true, true, false},
		{"failed", &fakeFile{err: &asr.Error{Kind: asr.KindServer}}, false, true, true},
		{"failed, keeping recordings", &fakeFile{err: &asr.Error{Kind: asr.KindServer}}, true, true, false},
		{"nothing recognised", &fakeFile{text: " "}, false, true, true},
		{"the stream fails with no text", &fakeStreamer{silent: true, finalErr: &asr.Error{Kind: asr.KindServer}}, false, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, tc.prov, nil, func(_ *Config, s *settings.Settings) { s.Privacy.KeepAudio = tc.keepAudio })
			id := h.start()
			h.speak(id, 5)
			end := h.rec.wait(t)
			es := h.hist.all()
			if len(es) != 1 || (es[0].Audio != "") != tc.kept || end.Recording != tc.kept || (!es[0].AudioUntil.IsZero()) != tc.until {
				t.Fatalf("end recording %v, entries %+v", end.Recording, es)
			}
			if writeFileForTest(t, h.hist.dir, id+".wav") != tc.kept {
				t.Error("the file does not match the entry")
			}
		})
	}
}

// retryHarness is a harness with a failed entry whose recording was kept.
func retryHarness(t *testing.T, prov asr.Provider, model LLM, keepAudio bool) (*harness, store.Entry) {
	h := newHarness(t, prov, model, func(_ *Config, s *settings.Settings) {
		s.Privacy.KeepAudio = keepAudio
		s.Features.Transcribe.Enhance = model != nil
	})
	pcm := make([]byte, asr.SampleRate*2) // a second: the fake models do not listen
	for i := 0; i < len(pcm); i += 2 {
		pcm[i+1] = byte(int8(40 * ((i/64)%2*2 - 1)))
	}
	if err := os.WriteFile(filepath.Join(h.hist.dir, "e1.wav"), asr.WAV(pcm, asr.SampleRate), 0o644); err != nil {
		t.Fatal(err)
	}
	e := store.Entry{ID: "e1", Kind: store.KindTranscribe, Status: store.StatusFailed, Error: "asr : timeout: no result within 30s",
		AppBundle: slack.BundleID, AppName: "Slack", Audio: "e1.wav", AudioUntil: time.Now().Add(time.Hour), AudioMS: 1000, CreatedAt: time.Now().Add(-time.Hour)}
	return h, e
}

// TestRetry: Retry makes a failed entry again from its recording, in
// place, without inserting anything; a success drops a recording kept
// only for it when the settings keep none.
func TestRetry(t *testing.T) {
	t.Run("done", func(t *testing.T) {
		model := &fakeLLM{text: "Hello, VoxtNew."}
		h, e := retryHarness(t, &fakeFile{text: "hello aura link"}, model, false)
		var mu sync.Mutex
		var stages []State
		out, err := h.m.Retry(context.Background(), RetryRequest{Entry: e, OnStage: func(s State) { mu.Lock(); stages = append(stages, s); mu.Unlock() }})
		if err != nil {
			t.Fatal(err)
		}
		if out.ID != "e1" || out.Status != store.StatusDone || out.Error != "" || out.RawText != "hello aura link" || out.Result != "Hello, VoxtNew." ||
			out.ASRProvider != "doubao" || out.LLMProvider != "deepseek" || out.Audio != "" || !out.AudioUntil.IsZero() || out.CreatedAt != e.CreatedAt {
			t.Fatalf("entry %+v", out)
		}
		if !slices.Equal(stages, []State{Recognizing, Composing}) {
			t.Errorf("stages %v", stages)
		}
		if writeFileForTest(t, h.hist.dir, "e1.wav") {
			t.Error("the recording kept for the retry is still there")
		}
		if es := h.hist.all(); len(es) != 1 || es[0].Result != out.Result {
			t.Errorf("written %+v", es)
		}
		if len(h.text.pastes()) != 0 || h.m.Current().State != Idle {
			t.Errorf("pasted %q, session %s", h.text.pastes(), h.m.Current().State)
		}
		// The clean-up got the text with the dictionary's alias replaced.
		if reqs := model.requests(); len(reqs) != 1 || !containsText(reqs[0], "hello VoxtNew") {
			t.Errorf("requests %+v", reqs)
		}
	})
	t.Run("keeping recordings", func(t *testing.T) {
		h, e := retryHarness(t, &fakeFile{text: "x"}, nil, true)
		out, err := h.m.Retry(context.Background(), RetryRequest{Entry: e})
		if err != nil || out.Audio != "e1.wav" || !out.AudioUntil.IsZero() || out.Result != "x" || !writeFileForTest(t, h.hist.dir, "e1.wav") {
			t.Fatalf("entry %+v, %v", out, err)
		}
	})
	t.Run("fails again", func(t *testing.T) {
		h, e := retryHarness(t, &fakeFile{err: &asr.Error{Kind: asr.KindServer, Detail: "the local speech model stopped answering"}}, nil, false)
		out, err := h.m.Retry(context.Background(), RetryRequest{Entry: e})
		if err == nil || out.Status != store.StatusFailed || out.Error == e.Error || out.Audio != "e1.wav" || !out.AudioUntil.Equal(e.AudioUntil) {
			t.Fatalf("entry %+v, %v", out, err)
		}
		if !writeFileForTest(t, h.hist.dir, "e1.wav") {
			t.Error("the recording went")
		}
	})
	t.Run("the clean-up fails", func(t *testing.T) {
		h, e := retryHarness(t, &fakeFile{text: "x"}, &fakeLLM{err: errors.New("boom")}, false)
		out, err := h.m.Retry(context.Background(), RetryRequest{Entry: e})
		if err != nil || out.Status != store.StatusDone || out.Result != "x" || out.Error == "" || out.Audio != "" {
			t.Fatalf("entry %+v, %v", out, err)
		}
	})
	t.Run("no recording", func(t *testing.T) {
		h, e := retryHarness(t, &fakeFile{text: "x"}, nil, false)
		e.Audio = ""
		if _, err := h.m.Retry(context.Background(), RetryRequest{Entry: e}); !errors.Is(err, ErrNoRecording) {
			t.Fatalf("err %v", err)
		}
	})
	t.Run("beside a session", func(t *testing.T) {
		h, e := retryHarness(t, &fakeFile{text: "again"}, nil, false)
		id := h.start()
		out, err := h.m.Retry(context.Background(), RetryRequest{Entry: e})
		if err != nil || out.Result != "again" {
			t.Fatalf("entry %+v, %v", out, err)
		}
		if cur := h.m.Current(); cur.ID != id || cur.State != Recording {
			t.Fatalf("the session changed: %+v", cur)
		}
		h.speak(id, 3)
		if end := h.rec.wait(t); end.State != Done || end.Result != "again" {
			t.Errorf("session %+v", end)
		}
	})
}

// containsText reports whether a request's last message holds text.
func containsText(req llm.Request, text string) bool {
	return len(req.Messages) > 0 && strings.Contains(req.Messages[len(req.Messages)-1].Content, text)
}
