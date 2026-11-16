package main

import (
	"bytes"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func testDictionaryHighlights(text string) []dict.Span {
	return dict.HighlightSpans(text, []store.Term{
		{Term: "Voxt", Enabled: true},
		{Term: "SGLang", Aliases: []string{"S G Long"}, Enabled: true},
	})
}

// TestTranscribeDictionaryHighlights: matched terms keep the whole result
// readable and selectable, with a visible tint in the Try preview and the
// answer card.
func TestTranscribeDictionaryHighlights(t *testing.T) {
	const result = "Voxt uses SGLang."
	t.Run("Try preview", func(t *testing.T) {
		a, tt := trTester(t)
		s := trySnapshot(a, session.Done)
		s.Raw, s.Result = "voxt uses s g long", result
		a.sessionChanged(s)
		frames(tt)
		before := bytes.Clone(tt.Image().Pix)
		s.Rev++
		s.Highlights = testDictionaryHighlights(result)
		a.sessionChanged(s)
		frames(tt)
		if bytes.Equal(before, tt.Image().Pix) {
			t.Error("the matched terms have no visible highlight")
		}
		if !hasExact(tt, result) {
			t.Errorf("the whole result is not readable: %q", tt.Texts())
		}
	})
	t.Run("answer card", func(t *testing.T) {
		a, _, _ := newTestApp(t)
		shortScreen(t, a)
		s := session.Snapshot{ID: "dictionary-card", Kind: store.KindTranscribe, State: session.Done, Card: true,
			Delivery: deliver.ReasonCard, Result: result}
		snap(a, s)
		before := renderOverlay(t, a)
		s.Highlights = testDictionaryHighlights(result)
		a.openAnswer(s)
		a.overlaySync()
		after := renderOverlay(t, a)
		if bytes.Equal(before.Pix, after.Pix) {
			t.Error("the matched terms have no visible highlight")
		}
	})
}
