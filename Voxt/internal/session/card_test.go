package session

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestWriteFollowUpKeepsPreviousLiteralThinkingMarker(t *testing.T) {
	const answer = "The <think> tag wraps reasoning."
	const want = "The"
	for _, tc := range []struct {
		name     string
		previous compose.Turn
	}{
		{name: "instruction", previous: compose.Turn{Instruction: "explain <think>"}},
		{name: "selection", previous: compose.Turn{Selection: "explain <think>"}},
		{name: "answer title", previous: compose.Turn{Answer: compose.Answer{Title: "explain <think>"}}},
		{name: "answer content", previous: compose.Turn{Answer: compose.Answer{Content: "explain <think>"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, &fakeFile{}, &fakeLLM{text: answer, chunk: 4}, nil)
			_, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Target: slack,
				Text: &TextInput{Raw: "shorter"}, Previous: []compose.Turn{tc.previous}})
			if err != nil {
				t.Fatal(err)
			}
			end := h.rec.wait(t)
			if end.State != Done || end.Result != want {
				t.Fatalf("end %s %q (problem %+v)", end.State, end.Result, end.Problem)
			}
			if !slices.ContainsFunc(h.rec.all(), func(s Snapshot) bool { return s.State == Composing && s.Draft == want }) {
				t.Error("the streamed draft lost the previous turn's literal thinking marker")
			}
		})
	}
}

func TestWriteFollowUpStillStripsRealControlMarkup(t *testing.T) {
	for _, tc := range []struct{ name, answer, want, forbidden string }{
		{name: "trailing token", answer: "Shorter.<|im_end|>", want: "Shorter.", forbidden: "<|im_end|>"},
		{name: "thinking block", answer: "Shorter.<think>private</think>", want: "Shorter.", forbidden: "private"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := newHarness(t, &fakeFile{}, &fakeLLM{text: tc.answer, chunk: 4}, nil)
			_, err := h.m.Start(StartRequest{Kind: store.KindRewrite, Target: slack,
				Text: &TextInput{Raw: "shorter"}, Previous: []compose.Turn{{Instruction: "explain <think> and <|im_end|>"}}})
			if err != nil {
				t.Fatal(err)
			}
			end := h.rec.wait(t)
			if end.State != Done || end.Result != tc.want {
				t.Fatalf("end %s %q (problem %+v)", end.State, end.Result, end.Problem)
			}
			for _, snapshot := range h.rec.all() {
				if snapshot.State == Composing && strings.Contains(snapshot.Draft, tc.forbidden) {
					t.Errorf("the streamed draft leaked real control markup %q", snapshot.Draft)
				}
			}
		})
	}
}

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
