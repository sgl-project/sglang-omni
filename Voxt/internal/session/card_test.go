package session

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestShowCard: each feature's own card switch sends its result to the
// card or inserts it (over the selection when there is one); the other
// features' switches, set the other way, change nothing. A follow-up
// stays in the card, where its conversation is.
func TestShowCard(t *testing.T) {
	for _, tc := range []struct {
		name     string
		kind     Kind
		sel      string
		answer   string
		previous []compose.Turn
		want     string
		alwaysIn bool // in the card with the switch off too
	}{
		{name: "transcribe", kind: store.KindTranscribe, answer: "Hello, VoxtNew.", want: "Hello, VoxtNew."},
		{name: "translate speech", kind: store.KindTranslate, answer: "你好", want: "你好"},
		{name: "translate the selection", kind: store.KindTranslate, sel: "Good morning", answer: "早上好", want: "早上好"},
		{name: "rewrite the selection", kind: store.KindRewrite, sel: "ship friday?", answer: "We ship on Friday.", want: "We ship on Friday."},
		{name: "write from scratch", kind: store.KindRewrite, answer: `{"title":"Friday","content":"We ship on Friday."}`, want: "We ship on Friday."},
		{name: "a follow-up", kind: store.KindRewrite, answer: "Team: Friday.", want: "Team: Friday.", alwaysIn: true,
			previous: []compose.Turn{{Instruction: "write it", Answer: compose.Answer{Content: "We ship on Friday."}}}},
	} {
		for _, on := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s, card %v", tc.name, on), func(t *testing.T) {
				h := newHarness(t, &fakeFile{text: " hello aura link "}, &fakeLLM{text: tc.answer}, func(c *Config, s *settings.Settings) {
					c.Select = &fakeSelector{text: tc.sel}
					s.Features.Transcribe.ShowCard, s.Features.Translate.ShowCard, s.Features.Rewrite.ShowCard = !on, !on, !on
					switch tc.kind {
					case store.KindTranscribe:
						s.Features.Transcribe.ShowCard = on
					case store.KindTranslate:
						s.Features.Translate.ShowCard = on
					default:
						s.Features.Rewrite.ShowCard = on
					}
					s.Features.Transcribe.Enhance = true
				})
				id, err := h.m.Start(StartRequest{Kind: tc.kind, Target: slack, Previous: tc.previous})
				if err != nil {
					t.Fatal(err)
				}
				if first := h.rec.all()[0]; first.Card != (on || tc.alwaysIn) {
					t.Errorf("first snapshot uses card %v, want %v", first.Card, on || tc.alwaysIn)
				}
				if tc.kind != store.KindTranslate || tc.sel == "" {
					h.speak(id, 5)
				}
				end := h.rec.wait(t)
				pastes := h.text.pastes()
				if end.State != Done || end.Result != tc.want {
					t.Fatalf("end %s %q (problem %+v)", end.State, end.Result, end.Problem)
				}
				if on || tc.alwaysIn {
					if !end.Card || end.Delivery != deliver.ReasonCard || len(pastes) != 0 || end.Inserted {
						t.Errorf("want the card: card %v (%s), pasted %q", end.Card, end.Delivery, pastes)
					}
					return
				}
				if end.Card || len(pastes) != 1 || pastes[0] != tc.want || !end.Inserted {
					t.Errorf("want it inserted: card %v (%s), pasted %q", end.Card, end.Delivery, pastes)
				}
			})
		}
	}
}

// TestTypedRewriteInstruction: a typed request submits without enough
// speech to transcribe, keeping the selection and the same card session.
func TestTypedRewriteInstruction(t *testing.T) {
	for name, speech := range map[string]asr.Provider{
		"file":           &fakeFile{text: "unwanted speech"},
		"opening stream": &fakeStreamer{openGate: make(chan struct{})},
	} {
		t.Run(name, func(t *testing.T) {
			model := &fakeLLM{text: "We ship Friday."}
			h := newHarness(t, speech, model, func(c *Config, _ *settings.Settings) {
				c.Select = &fakeSelector{text: "Our release is scheduled for Friday."}
				c.MinRecording = 24 * time.Hour
			})
			id, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Target: slack})
			if err != nil {
				t.Fatal(err)
			}
			h.rec.waitFor(t, "the selection", func(s Snapshot) bool { return s.Selection != "" })
			if h.m.SubmitInstruction("old-session", "ignored") || h.m.SubmitInstruction(id, "  ") {
				t.Fatal("accepted a stale or empty request")
			}
			if !h.m.SubmitInstruction(id, "  Make it shorter  ") {
				t.Fatal("the recording did not accept typed input")
			}
			end := h.rec.wait(t)
			if end.ID != id || end.State != Done || end.Raw != "Make it shorter" || end.Selection != "Our release is scheduled for Friday." || end.Result != "We ship Friday." || !end.Card || end.Inserted {
				t.Fatalf("typed rewrite: %+v", end)
			}
			if file, ok := speech.(*fakeFile); ok && file.calls != 0 {
				t.Fatalf("typed request transcribed %d times", file.calls)
			}
			if len(h.text.pastes()) != 0 {
				t.Fatal("typed request pasted a result")
			}
			request := model.reqs[0].Messages[len(model.reqs[0].Messages)-1].Content
			if !strings.Contains(request, "Make it shorter") || !strings.Contains(request, end.Selection) {
				t.Errorf("request lost the instruction or selection: %q", request)
			}
		})
	}
}
