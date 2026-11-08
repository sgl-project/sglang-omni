package session

import (
	"fmt"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestShowCard: the card switch sends transcribe and translate to the
// card or inserts; Rewrite always waits in the card. A follow-up stays
// in the card, where its conversation is.
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
		{name: "rewrite the selection", kind: store.KindRewrite, sel: "ship friday?", answer: "We ship on Friday.", want: "We ship on Friday.", alwaysIn: true},
		{name: "write from scratch", kind: store.KindRewrite, answer: `{"title":"Friday","content":"We ship on Friday."}`, want: "We ship on Friday.", alwaysIn: true},
		{name: "a follow-up", kind: store.KindRewrite, answer: "Team: Friday.", want: "Team: Friday.", alwaysIn: true,
			previous: []compose.Turn{{Instruction: "write it", Answer: compose.Answer{Content: "We ship on Friday."}}}},
	} {
		for _, on := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s, card %v", tc.name, on), func(t *testing.T) {
				h := newHarness(t, &fakeFile{text: " hello aura link "}, &fakeLLM{text: tc.answer}, func(c *Config, s *settings.Settings) {
					c.Select = &fakeSelector{text: tc.sel}
					s.Features.Transcribe.ShowCard = on
					s.Features.Translate.ShowCard = on
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
