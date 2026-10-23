package session

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/deliver"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/vad"
)

var slack = proto.App{PID: 812, BundleID: "com.tinyspeck.slackmacgap", Name: "Slack"}

type harness struct {
	t     *testing.T
	m     *Manager
	audio *fakeAudio
	text  *fakeText
	hist  *memHistory
	rec   *recorder

	mu         sync.Mutex
	asr        asr.Provider
	llm        LLM
	resolveErr error
	logs       []string
	// vadModel is the voice activity model the services have (nil: none).
	vadModel vad.Model
}

func newHarness(t *testing.T, prov asr.Provider, model LLM, cfgFn func(*Config, *settings.Settings)) *harness {
	h := &harness{t: t, audio: newFakeAudio(), text: &fakeText{focus: proto.FocusResult{PID: 812, Role: "AXTextArea", Editable: true}},
		hist: &memHistory{dir: t.TempDir()}, rec: newRecorder(), asr: prov, llm: model}
	s := settings.Defaults()
	s.Dictionary.Enabled = true
	cfg := Config{
		Audio: h.audio, Text: h.text, History: h.hist,
		Resolve: func(ctx context.Context, kind Kind, _ settings.Settings) (Services, error) {
			h.mu.Lock()
			defer h.mu.Unlock()
			if h.resolveErr != nil {
				return Services{}, h.resolveErr
			}
			svc := Services{ASR: h.asr, ASRInfo: Model{ID: "doubao", Name: "Doubao", Model: "seed"}, VAD: h.vadModel}
			if h.llm != nil {
				svc.LLM, svc.LLMInfo = h.llm, Model{ID: "deepseek", Name: "DeepSeek", Model: "deepseek-chat"}
			}
			return svc, nil
		},
		Glossary: func(context.Context) ([]store.Term, error) {
			return []store.Term{{Term: "VoxtNew", Aliases: []string{"aura link"}, Enabled: true}, {Term: "Off", Enabled: false}}, nil
		},
		OnChange: h.rec.add,
		Log: func(level, msg string) {
			h.mu.Lock()
			h.logs = append(h.logs, level+" "+msg)
			h.mu.Unlock()
		},
		LLMFirstToken: time.Second, LLMStall: 500 * time.Millisecond, LLMPerChar: time.Millisecond, LLMTimeout: 2 * time.Second,
		ASRTimeout: 2 * time.Second, StopGrace: 200 * time.Millisecond,
	}
	if cfgFn != nil {
		cfgFn(&cfg, &s)
	}
	cfg.Settings = func() settings.Settings { return s }
	h.m = New(cfg)
	return h
}

func (h *harness) start() string {
	h.t.Helper()
	id, err := h.m.Start(StartRequest{Kind: store.KindTranscribe, Target: slack})
	if err != nil {
		h.t.Fatal(err)
	}
	return id
}

// speak waits for n frames of audio, then stops.
func (h *harness) speak(id string, n int) {
	h.t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for h.audio.recorded() < n {
		if time.Now().After(deadline) {
			h.t.Fatalf("only %d frames recorded", h.audio.recorded())
		}
		time.Sleep(time.Millisecond)
	}
	h.m.Stop(id)
}

func (h *harness) states() []State {
	var out []State
	for _, s := range h.rec.all() {
		if len(out) == 0 || out[len(out)-1] != s.State {
			out = append(out, s.State)
		}
	}
	return out
}

func statesString(ss []State) string {
	var b []string
	for _, s := range ss {
		b = append(b, string(s))
	}
	return strings.Join(b, " ")
}

// TestPipeline runs whole sessions: both kinds of speech model, the
// language model on, off, failing and timing out, and a target that takes
// no text.
func TestPipeline(t *testing.T) {
	for _, tc := range []struct {
		name     string
		stream   bool
		llm      *fakeLLM
		enhance  bool
		writable bool
		result   string
		states   string
		problem  string
		card     bool
	}{
		{"stream, llm", true, &fakeLLM{text: "Hello, VoxtNew."}, true, true, "Hello, VoxtNew.", "recording recognizing composing delivering done", "", false},
		{"stream, llm off", true, &fakeLLM{text: "unused"}, false, true, "hello VoxtNew", "recording recognizing delivering done", "", false},
		{"stream, no llm", true, nil, true, true, "hello VoxtNew", "recording recognizing delivering done", "", false},
		{"file, llm", false, &fakeLLM{text: "“Hello, VoxtNew.”"}, true, true, "Hello, VoxtNew.", "recording recognizing composing delivering done", "", false},
		{"file, llm off", false, &fakeLLM{}, false, true, "hello VoxtNew", "recording recognizing delivering done", "", false},
		{"stream, llm fails", true, &fakeLLM{err: errors.New("503")}, true, true, "hello VoxtNew", "recording recognizing composing delivering done", ProblemLLM, false},
		{"file, llm times out", false, &fakeLLM{hang: true}, true, true, "hello VoxtNew", "recording recognizing composing delivering done", ProblemLLM, false},
		{"stream, not writable", true, &fakeLLM{text: "Hello."}, true, false, "Hello.", "recording recognizing composing delivering done", "", true},
		{"file, llm says nothing", false, &fakeLLM{text: "  "}, true, true, "", "recording recognizing composing done", ProblemNoSpeech, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var prov asr.Provider = &fakeFile{text: " hello aura link "}
			if tc.stream {
				prov = &fakeStreamer{final: "hello aura link"}
			}
			var model LLM
			if tc.llm != nil {
				model = tc.llm
			}
			h := newHarness(t, prov, model, func(c *Config, s *settings.Settings) {
				s.Features.Transcribe.Enhance = tc.enhance
				s.Features.Transcribe.Prompt = "Be brief."
				c.LLMFirstToken, c.LLMTimeout = 50*time.Millisecond, 100*time.Millisecond
			})
			if !tc.writable {
				h.text.focus = proto.FocusResult{PID: 812, Role: "AXButton"}
			}
			id := h.start()
			h.speak(id, 5)
			end := h.rec.wait(t)
			if got := statesString(h.states()); got != tc.states {
				t.Errorf("states %q, want %q", got, tc.states)
			}
			if end.State != Done || end.Result != tc.result || end.Raw != "hello aura link" {
				t.Errorf("end %+v", end)
			}
			if (end.Problem == nil) != (tc.problem == "") || end.Problem != nil && end.Problem.Code != tc.problem {
				t.Errorf("problem %+v, want %q", end.Problem, tc.problem)
			}
			if tc.problem == ProblemLLM && !end.Problem.Degraded {
				t.Errorf("not degraded: %+v", end.Problem)
			}
			pastes := h.text.pastes()
			switch {
			case tc.result == "":
				if len(pastes) != 0 || end.Inserted || end.Card {
					t.Errorf("pasted %q, %+v", pastes, end)
				}
			case tc.card:
				if len(pastes) != 0 || !end.Card || end.Inserted || end.Delivery != deliver.ReasonNotEditable {
					t.Errorf("card: pasted %q, %+v", pastes, end)
				}
			default:
				if len(pastes) != 1 || pastes[0] != tc.result || !end.Inserted || end.Card {
					t.Errorf("pasted %q, %+v", pastes, end)
				}
			}
			if tc.stream {
				h.rec.waitFor(t, "with a partial", func(s Snapshot) bool { return s.Partial == "partial" && s.State == Recording })
				st := prov.(*fakeStreamer)
				if len(st.streams) != 1 || st.streams[0].bytes != h.audio.recorded()*len(frame) {
					t.Errorf("streamed %d bytes of %d frames", st.streams[0].bytes, h.audio.recorded())
				}
				if strings.Join(st.hints, ",") != "VoxtNew" {
					t.Errorf("hints %q", st.hints)
				}
			} else if f := prov.(*fakeFile); f.calls != 1 || f.bytes != h.audio.recorded()*len(frame) {
				t.Errorf("uploaded %d calls, %d bytes", f.calls, f.bytes)
			}
			if tc.llm != nil && tc.enhance {
				reqs := tc.llm.requests()
				if len(reqs) != 1 {
					t.Fatalf("%d llm requests", len(reqs))
				}
				// The system prompt, the clean-up's example turn, the input.
				msgs := reqs[0].Messages
				sys, user := msgs[0].Content, msgs[len(msgs)-1].Content
				if len(msgs) != 4 || msgs[1].Role != llm.User || msgs[2].Role != llm.Assistant || !strings.Contains(sys, "Be brief.") || !strings.Contains(sys, "- VoxtNew") || strings.Contains(sys, "Off") ||
					strings.Contains(sys, "hello") || user != "<input>\nhello VoxtNew\n</input>" || reqs[0].Skill != llm.SkillTranscribeEnhance {
					t.Errorf("request %+v", reqs[0])
				}
				if end.Profile != "transcribe:cleanup+custom+glossary" {
					t.Errorf("profile %q", end.Profile)
				}
			}
			entries := h.hist.all()
			if tc.result == "" {
				if len(entries) != 1 || entries[0].Status != store.StatusDone || entries[0].Result != "" {
					t.Errorf("entries %+v", entries)
				}
				return
			}
			if len(entries) != 1 {
				t.Fatalf("entries %+v", entries)
			}
			e := entries[0]
			wantDelivery := "insert"
			if tc.card {
				wantDelivery = "card"
			}
			if e.ID != id || e.Kind != store.KindTranscribe || e.Status != store.StatusDone || e.RawText != "hello aura link" || e.Result != tc.result ||
				e.AppBundle != slack.BundleID || e.AppName != "Slack" || e.ASRProvider != "doubao" || e.Delivery != wantDelivery || e.Audio != "" {
				t.Errorf("entry %+v", e)
			}
			if usedLLM := tc.llm != nil && tc.enhance; usedLLM != (e.LLMProvider == "deepseek") {
				t.Errorf("entry llm %q", e.LLMProvider)
			}
			if tc.problem == ProblemLLM && e.Error == "" {
				t.Error("entry without the llm's error")
			}
			if e.AudioMS < 400 || e.TotalMS <= 0 {
				t.Errorf("timings %+v", e)
			}
			if h.m.Last() != tc.result {
				t.Errorf("last %q", h.m.Last())
			}
		})
	}
}

// TestCancel cancels at each stage: nothing is pasted or kept.
func TestCancel(t *testing.T) {
	for _, tc := range []struct {
		name  string
		prov  func() asr.Provider
		model LLM
		at    State
	}{
		{"recording", func() asr.Provider { return &fakeStreamer{final: "x"} }, nil, Recording},
		{"recognizing a file", func() asr.Provider { return &fakeFile{hang: true} }, nil, Recognizing},
		{"recognizing a stream", func() asr.Provider { return &fakeStreamer{hang: true} }, nil, Recognizing},
		{"composing", func() asr.Provider { return &fakeFile{text: "x"} }, &fakeLLM{hang: true}, Composing},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, tc.prov(), tc.model, nil)
			id := h.start()
			if tc.at == Recording {
				h.rec.waitFor(t, "heard", func(s Snapshot) bool { return !s.HeardAt.IsZero() })
			} else {
				h.speak(id, 5)
			}
			h.rec.waitFor(t, "at "+string(tc.at), func(s Snapshot) bool { return s.State == tc.at })
			h.m.Cancel(id)
			end := h.rec.wait(t)
			if end.State != Cancelled || end.Problem != nil {
				t.Errorf("end %+v", end)
			}
			if p := h.text.pastes(); len(p) != 0 {
				t.Errorf("pasted %q", p)
			}
			if e := h.hist.all(); len(e) != 0 {
				t.Errorf("kept %+v", e)
			}
			if h.m.Last() != "" {
				t.Errorf("last %q", h.m.Last())
			}
			// Once ended, Stop and Cancel do nothing.
			rev := h.m.Current().Rev
			h.m.Stop(id)
			h.m.Cancel(id)
			if h.m.Current().Rev != rev {
				t.Error("an ended session changed")
			}
		})
	}
}

// TestFailures: errors with no way on end the session failed, with a
// problem the UI explains.
func TestFailures(t *testing.T) {
	type setup func(h *harness)
	for _, tc := range []struct {
		name    string
		prov    asr.Provider
		setup   setup
		speak   bool
		problem string
		kept    bool
	}{
		{"no speech model", &fakeFile{}, func(h *harness) { h.resolveErr = fmt.Errorf("transcribe: %w", ErrNoASR) }, false, ProblemNoASR, false},
		{"microphone denied", &fakeFile{}, func(h *harness) {
			h.audio.startErr = &proto.Error{Code: proto.CodePermissionDenied, Message: "denied"}
		}, false, ProblemMicrophone, false},
		{"no input", &fakeFile{}, func(h *harness) { h.audio.startErr = &proto.Error{Code: proto.CodeDeviceUnavailable} }, false, ProblemDevice, false},
		{"helper died", &fakeFile{}, func(h *harness) {
			h.audio.failAt, h.audio.failErr = 8, fmt.Errorf("the helper exited: %w", errHelperForTest)
		}, false, ProblemHelper, true},
		{"stream refused", &fakeStreamer{openErr: &asr.Error{Kind: asr.KindAuth, Detail: "401"}}, nil, false, ProblemASR, false},
		{"stream breaks while recording", &fakeStreamer{errAfter: 3}, nil, false, ProblemASR, true},
		{"stream fails at the end", &fakeStreamer{silent: true, finalErr: &asr.Error{Kind: asr.KindServer, Detail: "500"}}, nil, true, ProblemASR, true},
		{"stream times out", &fakeStreamer{silent: true, hang: true}, func(h *harness) { h.m.cfg.ASRTimeout, h.m.cfg.ASRPerSecond = 100*time.Millisecond, time.Nanosecond }, true, ProblemASR, true},
		{"upload times out", &fakeFile{hang: true}, func(h *harness) { h.m.cfg.ASRTimeout, h.m.cfg.ASRPerSecond = 100*time.Millisecond, time.Nanosecond }, true, ProblemASR, true},
		{"upload refused", &fakeFile{err: &asr.Error{Kind: asr.KindQuota, Detail: "429"}}, nil, true, ProblemASR, true},
		{"nothing said", &fakeFile{text: "  "}, nil, true, ProblemNoSpeech, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, tc.prov, &fakeLLM{text: "x"}, nil)
			if tc.setup != nil {
				tc.setup(h)
			}
			id := h.start()
			if tc.speak {
				h.speak(id, 5)
			}
			end := h.rec.wait(t)
			want := Failed
			if tc.problem == ProblemNoSpeech {
				want = Done
			}
			if end.State != want || end.Problem == nil || end.Problem.Code != tc.problem {
				t.Fatalf("end %s %+v", end.State, end.Problem)
			}
			if tc.problem == ProblemASR {
				var ae *asr.Error
				if !errors.As(end.Problem.Err, &ae) {
					t.Errorf("not an asr.Error: %v", end.Problem.Err)
				}
			}
			if p := h.text.pastes(); len(p) != 0 {
				t.Errorf("pasted %q", p)
			}
			if e := h.hist.all(); (len(e) == 1) != tc.kept {
				t.Errorf("kept %+v", e)
			} else if tc.kept && tc.problem != ProblemNoSpeech && (e[0].Status != store.StatusFailed || e[0].Error == "") {
				t.Errorf("entry %+v", e[0])
			}
			if h.audio.recorded() > 0 {
				// The recording stopped with the session: the capture
				// returns (a frame in flight when the session ended may
				// still count), then no more frames come.
				if !h.audio.waitStopped(2 * time.Second) {
					t.Fatal("still recording after the end")
				}
				n := h.audio.recorded()
				time.Sleep(20 * time.Millisecond)
				if h.audio.recorded() != n {
					t.Error("recorded after the capture returned")
				}
			}
		})
	}
}

// TestLateCallbacks: what comes for a session after it ended, or for an
// older one while a newer runs, changes nothing.
func TestLateCallbacks(t *testing.T) {
	st := &fakeStreamer{final: "one"}
	h := newHarness(t, st, nil, nil)
	first := h.start()
	h.speak(first, 3)
	end := h.rec.wait(t)
	if end.State != Done {
		t.Fatalf("first %+v", end)
	}
	rev := h.m.Current().Rev
	h.audio.lateEvents(first)
	st.streams[0].emit(asr.Event{Kind: asr.EventPartial, Text: "late"})
	if h.m.Current().Rev != rev {
		t.Fatalf("a late event changed the ended session: %+v", h.m.Current())
	}

	st.final = "two"
	second := h.start()
	h.rec.waitFor(t, "second heard", func(s Snapshot) bool { return s.ID == second && !s.HeardAt.IsZero() })
	h.audio.lateEvents(first)
	h.m.Stop(first) // an old id
	h.m.Cancel(first)
	cur := h.m.Current()
	if cur.ID != second || cur.State != Recording || cur.Device == "late" {
		t.Fatalf("the old session's events reached the new one: %+v", cur)
	}
	h.speak(second, h.audio.recorded()+3)
	end = h.rec.wait(t)
	if end.ID != second || end.Result != "two" {
		t.Errorf("second %+v", end)
	}
	if p := h.text.pastes(); len(p) != 2 || p[1] != "two" {
		t.Errorf("pasted %q", p)
	}
	// Revisions only grow.
	var last uint64
	for _, s := range h.rec.all() {
		if s.Rev <= last {
			t.Fatalf("rev %d after %d", s.Rev, last)
		}
		last = s.Rev
	}
}

func TestBusyTryShortAndMore(t *testing.T) {
	t.Run("busy", func(t *testing.T) {
		h := newHarness(t, &fakeFile{text: "x"}, nil, nil)
		id := h.start()
		if _, err := h.m.Start(StartRequest{Kind: store.KindTranscribe}); !errors.Is(err, ErrBusy) {
			t.Errorf("second start: %v", err)
		}
		h.m.Cancel(id)
		h.rec.wait(t)
	})
	t.Run("too short", func(t *testing.T) {
		f := &fakeFile{text: "x"}
		h := newHarness(t, f, nil, nil)
		h.audio.tick = 50 * time.Millisecond
		id := h.start()
		h.rec.waitFor(t, "heard", func(s Snapshot) bool { return !s.HeardAt.IsZero() })
		h.m.Stop(id)
		end := h.rec.wait(t)
		if end.State != Cancelled || end.Problem == nil || end.Problem.Code != ProblemTooShort || f.calls != 0 || len(h.hist.all()) != 0 {
			t.Errorf("end %+v, %d uploads", end, f.calls)
		}
	})
	t.Run("try", func(t *testing.T) {
		h := newHarness(t, &fakeFile{text: "tried"}, nil, nil)
		id, _ := h.m.Start(StartRequest{Kind: store.KindTranscribe, Target: slack, Try: true})
		h.speak(id, 5)
		end := h.rec.wait(t)
		if end.State != Done || end.Result != "tried" || !end.Try || end.Card || end.Inserted || len(h.text.pastes()) != 0 || len(h.hist.all()) != 0 || h.m.Last() != "" {
			t.Errorf("try %+v", end)
		}
	})
	t.Run("keeps the recording", func(t *testing.T) {
		h := newHarness(t, &fakeFile{text: "x"}, nil, func(_ *Config, s *settings.Settings) { s.Privacy.KeepAudio = true })
		id := h.start()
		h.speak(id, 5)
		h.rec.wait(t)
		e := h.hist.all()
		if len(e) != 1 || e[0].Audio != id+".wav" || !writeFileForTest(t, h.hist.dir, id+".wav") {
			t.Errorf("entry %+v", e)
		}
	})
	t.Run("no history", func(t *testing.T) {
		h := newHarness(t, &fakeFile{text: "x"}, nil, func(_ *Config, s *settings.Settings) { s.Privacy.KeepHistory = false })
		id := h.start()
		h.speak(id, 5)
		if end := h.rec.wait(t); end.State != Done || len(h.hist.all()) != 0 {
			t.Errorf("end %+v, %d entries", end, len(h.hist.all()))
		}
	})
	t.Run("the recording ends by itself", func(t *testing.T) {
		h := newHarness(t, &fakeStreamer{final: "long"}, nil, nil)
		h.audio.maxFrame = 6
		h.start()
		end := h.rec.wait(t)
		if end.State != Done || end.Result != "long" || end.StoppedAt.IsZero() {
			t.Errorf("end %+v", end)
		}
	})
	t.Run("a stop that does not stop", func(t *testing.T) {
		h := newHarness(t, &fakeFile{text: "x"}, nil, nil)
		h.audio.deaf = true
		id := h.start()
		h.speak(id, 5)
		end := h.rec.wait(t)
		if end.State != Done || end.Result != "x" {
			t.Errorf("end %+v", end)
		}
	})
	t.Run("an unknown feature", func(t *testing.T) {
		h := newHarness(t, &fakeFile{text: "x"}, nil, nil)
		h.m.Start(StartRequest{Kind: "summarize"})
		end := h.rec.wait(t)
		if end.State != Failed || end.Problem.Code != ProblemUnsupported || h.audio.recorded() != 0 {
			t.Errorf("end %+v", end)
		}
	})
	t.Run("slow services", func(t *testing.T) {
		st := &fakeStreamer{final: "late open", openWait: 100 * time.Millisecond}
		h := newHarness(t, st, nil, nil)
		id := h.start()
		h.speak(id, 3) // stops before the stream opened: the buffer catches up
		end := h.rec.wait(t)
		if end.State != Done || end.Result != "late open" || st.streams[0].bytes != h.audio.recorded()*len(frame) {
			t.Errorf("end %+v", end)
		}
	})
	t.Run("levels and the device", func(t *testing.T) {
		h := newHarness(t, &fakeFile{text: "x"}, nil, nil)
		id := h.start()
		h.speak(id, 30)
		h.rec.wait(t)
		s := h.rec.waitFor(t, "levels", func(s Snapshot) bool { return len(s.Levels) == Bars })
		if s.Levels[0] < 0.6 || s.Levels[0] > 0.7 || s.Device != "MacBook Air Microphone" {
			t.Errorf("levels %v device %q", s.Levels, s.Device)
		}
	})
}

func TestLevel(t *testing.T) {
	for db, want := range map[float64]float64{-120: 0, -60: 0, -30: 0.5, 0: 1, 6: 1} {
		if got := Level(db); got != want {
			t.Errorf("Level(%v) = %v", db, got)
		}
	}
}

func TestElapsed(t *testing.T) {
	t0 := time.Unix(100, 0)
	s := Snapshot{}
	if s.Elapsed(t0) != 0 {
		t.Error("not heard")
	}
	s.HeardAt = t0
	if s.Elapsed(t0.Add(time.Second)) != time.Second {
		t.Error("running")
	}
	s.StoppedAt = t0.Add(2 * time.Second)
	if s.Elapsed(t0.Add(time.Minute)) != 2*time.Second {
		t.Error("stopped")
	}
}
