package session

import (
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestCleanupCancellationLoggingOnlyDuringCleanup(t *testing.T) {
	t.Run("recording", func(t *testing.T) {
		h := newHarness(t, &fakeFile{text: "spoken"}, nil, nil)
		id := h.start()
		h.rec.waitFor(t, "recording", func(s Snapshot) bool { return s.ID == id && s.State == Recording })
		h.m.Cancel(id)
		h.rec.waitFor(t, "cancelled recording", func(s Snapshot) bool { return s.ID == id && s.State == Cancelled })
		if h.logged("cleanup outcome=cancelled") {
			t.Fatal("recording cancellation was logged as a Cleanup outcome")
		}
	})

	t.Run("recognizing", func(t *testing.T) {
		h := newHarness(t, &fakeFile{hang: true}, nil, nil)
		id := h.start()
		h.speak(id, 3)
		h.rec.waitFor(t, "recognizing", func(s Snapshot) bool { return s.ID == id && s.State == Recognizing })
		h.m.Cancel(id)
		h.rec.waitFor(t, "cancelled recognition", func(s Snapshot) bool { return s.ID == id && s.State == Cancelled })
		if h.logged("cleanup outcome=cancelled") {
			t.Fatal("recognition cancellation was logged as a Cleanup outcome")
		}
	})

	t.Run("active Cleanup and repeated cancellation", func(t *testing.T) {
		logged := make(chan struct{})
		release := make(chan struct{})
		var count atomic.Int32
		h := newHarness(t, &fakeFile{text: "spoken"}, &fakeLLM{hang: true}, func(c *Config, s *settings.Settings) {
			s.Features.Transcribe.Enhance = true
			c.Log = func(_, message string) {
				if strings.Contains(message, "cleanup outcome=cancelled") && count.Add(1) == 1 {
					close(logged)
					<-release
				}
			}
		})
		id := h.start()
		h.speak(id, 3)
		h.rec.waitFor(t, "active Cleanup", func(s Snapshot) bool { return s.ID == id && s.State == Composing && s.CanSkip })
		cancelDone := make(chan struct{})
		go func() {
			h.m.Cancel(id)
			close(cancelDone)
		}()
		select {
		case <-logged:
		case <-time.After(5 * time.Second):
			t.Fatal("active Cleanup did not log its cancellation")
		}
		h.m.Cancel(id)
		close(release)
		<-cancelDone
		h.rec.waitFor(t, "cancelled Cleanup", func(s Snapshot) bool { return s.ID == id && s.State == Cancelled })
		if count.Load() != 1 {
			t.Fatalf("cancelled outcome logs = %d, want one", count.Load())
		}
	})
}

func TestCleanupRequestLoggingFollowsExperimentLevel(t *testing.T) {
	for _, tc := range []struct {
		name       string
		experiment bool
		requested  bool
		wantLevel  string
	}{
		{name: "disabled", requested: true, wantLevel: "debug"},
		{name: "enabled", experiment: true, requested: true, wantLevel: "info"},
		{name: "not requested", wantLevel: "debug"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &fakeLLM{text: "cleaned"}
			h := newHarness(t, &fakeFile{text: "spoken"}, model, func(c *Config, s *settings.Settings) {
				s.Features.Transcribe.Enhance = tc.requested
				if tc.experiment {
					c.CleanupLatency = CleanupLatencyPolicy{AutoFallbackAfter: cleanupSoftDeadlineTestDuration}
				}
			})
			id := h.start()
			h.speak(id, 3)
			h.rec.waitFor(t, "finished dictation", func(s Snapshot) bool { return s.ID == id && s.State.Ended() })
			h.mu.Lock()
			logs := append([]string(nil), h.logs...)
			h.mu.Unlock()
			wantRequest := tc.wantLevel + " "
			foundRequest := false
			for _, entry := range logs {
				if strings.Contains(entry, "cleanup requested=") {
					if !strings.HasPrefix(entry, wantRequest) {
						t.Fatalf("request log level = %q, want %s", entry, tc.wantLevel)
					}
					foundRequest = true
					break
				}
			}
			if !foundRequest {
				t.Fatal("missing cleanup requested diagnostic")
			}
			if !tc.requested {
				for _, entry := range logs {
					if strings.Contains(entry, "cleanup outcome=not_requested") && !strings.HasPrefix(entry, "debug ") {
						t.Fatalf("not_requested level = %q, want debug", entry)
					}
				}
				return
			}
			for _, entry := range logs {
				if strings.Contains(entry, "cleanup outcome=cleaned") && !strings.HasPrefix(entry, "info ") {
					t.Fatalf("cleanup outcome level = %q, want info", entry)
				}
			}
			if tc.experiment && (!h.logged("stop_to_choice=") || !h.logged("stop_to_delivery=")) {
				t.Fatal("enabled experiment omitted Stop-to-choice or Stop-to-delivery")
			}
		})
	}
}

func TestManagerBackgroundRejectsTranscribe(t *testing.T) {
	h := newHarness(t, &fakeFile{text: "spoken"}, &fakeLLM{hang: true}, func(_ *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
	})
	id := h.start()
	h.speak(id, 3)
	h.rec.waitFor(t, "active Cleanup", func(s Snapshot) bool { return s.ID == id && s.State == Composing })
	if h.m.Background(id) {
		t.Fatal("Transcribe entered background state")
	}
	h.m.Cancel(id)
	h.rec.waitFor(t, "cancelled Cleanup", func(s Snapshot) bool { return s.ID == id && s.State == Cancelled })
}

func TestManagerUpdateIfRejectsCompletedSession(t *testing.T) {
	h := newHarness(t, &fakeFile{text: "spoken"}, nil, nil)
	id := h.start()
	h.m.mu.Lock()
	r := h.m.cur
	h.m.mu.Unlock()
	h.m.Cancel(id)
	end := h.rec.waitFor(t, "cancelled session", func(s Snapshot) bool { return s.ID == id && s.State == Cancelled })
	before := h.m.Current()
	if h.m.updateIf(r, func(s *Snapshot) bool { s.Result = "late"; return true }) {
		t.Fatal("completed run accepted a late update")
	}
	if after := h.m.Current(); after.Rev != before.Rev || after.Result != end.Result {
		t.Fatalf("late update changed the final snapshot: before %+v, after %+v", before, after)
	}
}

func TestManagerUpdateIfRoutesBackgroundAnswers(t *testing.T) {
	for _, kind := range []store.Kind{store.KindRewrite, store.KindTranslate} {
		t.Run(string(kind), func(t *testing.T) {
			h := newHarness(t, &fakeFile{text: "spoken"}, &fakeLLM{hang: true}, func(c *Config, s *settings.Settings) {
				c.Select = &fakeSelector{text: "selected text"}
				if kind == store.KindTranslate {
					s.Features.Translate.ShowCard = true
				}
			})
			id, err := h.m.Start(StartRequest{Kind: kind, Target: slack})
			if err != nil {
				t.Fatal(err)
			}
			if kind == store.KindRewrite {
				h.speak(id, 3)
			}
			h.rec.waitFor(t, "composing answer", func(s Snapshot) bool { return s.ID == id && s.State == Composing && s.CanStop() })
			if !h.m.Background(id) {
				t.Fatal("Rewrite/Translate answer did not background")
			}
			h.m.mu.Lock()
			r := h.m.bg
			h.m.mu.Unlock()
			if !h.m.updateIf(r, func(s *Snapshot) bool { s.Draft = "background update"; return true }) {
				t.Fatal("background answer update was rejected")
			}
			if current := h.m.Current(); current.Draft == "background update" {
				t.Fatalf("background update changed the current snapshot: %+v", current)
			}
			if background := h.m.snapshot(r); background.Draft != "background update" {
				t.Fatalf("background snapshot = %+v", background)
			}
			h.m.Cancel(id)
			h.rec.waitFor(t, "cancelled background answer", func(s Snapshot) bool { return s.ID == id && s.State == Cancelled })
		})
	}
}

func TestManagerUpdateIfConditionalRevision(t *testing.T) {
	h := newHarness(t, &fakeFile{text: "spoken"}, nil, nil)
	h.audio.tick = time.Hour
	id := h.start()
	h.rec.waitFor(t, "recording", func(s Snapshot) bool { return s.ID == id && s.State == Recording && s.Device != "" })
	h.m.mu.Lock()
	r := h.m.cur
	h.m.mu.Unlock()
	before := h.m.Current()
	if !h.m.updateIf(r, func(s *Snapshot) bool { s.Profile = "accepted"; return true }) {
		t.Fatal("current session update was rejected")
	}
	accepted := h.m.Current()
	if accepted.Rev != before.Rev+1 || accepted.Profile != "accepted" {
		t.Fatalf("accepted update = %+v, before %+v", accepted, before)
	}
	if h.m.updateIf(r, func(*Snapshot) bool { return false }) {
		t.Fatal("conditional update returned true")
	}
	unchanged := h.m.Current()
	if unchanged.Rev != accepted.Rev || unchanged.Profile != accepted.Profile {
		t.Fatalf("rejected update published: accepted %+v, after %+v", accepted, unchanged)
	}
	if !h.m.update(r, func(s *Snapshot) { s.Raw = "updated through wrapper" }) {
		t.Fatal("unconditional wrapper rejected current session")
	}
	if updated := h.m.Current(); updated.Rev != unchanged.Rev+1 || updated.Raw != "updated through wrapper" {
		t.Fatalf("unconditional update = %+v", updated)
	}
	h.m.Cancel(id)
	h.rec.waitFor(t, "cancelled session", func(s Snapshot) bool { return s.ID == id && s.State == Cancelled })
}
