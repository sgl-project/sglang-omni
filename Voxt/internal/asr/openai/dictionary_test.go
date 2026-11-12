package openai

import (
	"context"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/asrtest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// dictionaryTerms is a dictionary as the store returns it: A to Z, one
// off, one spelled twice.
func dictionaryTerms() []store.Term {
	t0 := time.Date(2026, 10, 15, 9, 0, 0, 0, time.UTC)
	return []store.Term{
		{Term: "voxt", Enabled: true, UpdatedAt: t0},
		{Term: "Voxt", Aliases: []string{"aura link"}, Enabled: true, UpdatedAt: t0.Add(3 * time.Hour)},
		{Term: "Disabled", Enabled: false, UpdatedAt: t0.Add(4 * time.Hour)},
		{Term: "Kubernetes", Aliases: []string{"cooper netties"}, Enabled: true, UpdatedAt: t0.Add(time.Hour)},
		{Term: "豆包", Enabled: true, UpdatedAt: t0.Add(2 * time.Hour)},
	}
}

// TestGoldenDictionary: the dictionary's hints (dict.Hints, as a session
// sends them) after the request's prompt, in the "prompt" field.
func TestGoldenDictionary(t *testing.T) {
	r := &recorder{status: []int{200}, answers: []string{"response-json.json"}}
	c := newClient(t, serve(t, r), asr.Config{})
	if _, err := c.Transcribe(context.Background(), asr.FileRequest{
		Audio: asr.Silence(time.Second), Format: "pcm", Hints: dict.Hints(dictionaryTerms()), Prompt: "Voxt, a voice input app.",
	}); err != nil {
		t.Fatal(err)
	}
	asrtest.Golden(t, "request-dictionary", r.log.String())
}
