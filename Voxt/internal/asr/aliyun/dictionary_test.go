package aliyun

import (
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
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
		{Term: "百炼", Enabled: true, UpdatedAt: t0.Add(2 * time.Hour)},
	}
}

// TestDictionaryGolden: the dictionary's hints (dict.Hints, as a session
// sends them) in run-task: input.context's text for fun-asr-realtime,
// parameters.vocabulary for qwen-audio streaming (the prompt as context).
func TestDictionaryGolden(t *testing.T) {
	hints := dict.Hints(dictionaryTerms())
	for _, sc := range []scenario{{
		name: "dictionary-context", model: "fun-asr-realtime", path: inferencePath,
		req:   asr.StreamRequest{Hints: hints, Prompt: "A note about Go."},
		audio: audioLen, chunk: 2000,
		final: "你好，世界。我在用 Voxt。Hello world.",
	}, {
		name: "dictionary-vocabulary", model: "qwen-audio-3.0-asr-flash-streaming", path: inferencePath,
		req:   asr.StreamRequest{Hints: hints, Prompt: "Engineering stand-up."},
		audio: 6400, chunk: 3200,
		final: "Kubernetes and MyGo.",
	}} {
		t.Run(sc.name, sc.run)
	}
}
