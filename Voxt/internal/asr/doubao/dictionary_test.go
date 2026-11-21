package doubao

import (
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

// TestStreamDictionary: the dictionary's hints (dict.Hints, as a session
// sends them) become corpus.context's hot words, with the request's prompt
// as the dialog context.
func TestStreamDictionary(t *testing.T) {
	srv := newServer(t, asyncPath, loadScript(t, "short"))
	c := testClient(t, srv.URL)
	events, evs := drive(t, c, asr.StreamRequest{Hints: dict.Hints(dictionaryTerms()), Prompt: "Dictation in Chinese and English."}, pcm(400*time.Millisecond))
	got := srv.wait(t)
	checkEvents(t, evs)
	asrtest.Golden(t, "dictionary", got+"--- events\n"+events)
}
