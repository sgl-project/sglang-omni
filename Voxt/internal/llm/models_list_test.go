package llm

import (
	"context"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"
)

// TestModelsOfAServiceWithAKey: a service with a key lists everything it
// sells, so Models keeps the models that chat, the newest first (OpenAI's
// created, Anthropic's created_at), and an entry dated in a way it does
// not know stays, after the dated ones. A server on this Mac keeps its
// list as it sent it.
func TestModelsOfAServiceWithAKey(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(`{"object":"list","data":[
			{"id":"fixture-video-1","created":1700001000},
			{"id":"text-embedding-3-small","created":1700000300},
			{"id":"gpt-4.1-mini","created":1700000100},
			{"id":"whisper-1","created":1700000400},
			{"id":"gpt-image-1","created":1700000500},
			{"id":"omni-moderation-latest","created":1700000600},
			{"id":"gpt-4o-realtime-preview","created":1700000700},
			{"id":"tts-1-hd","created":1700000800},
			{"id":"dall-e-3","created":1700000900},
			{"id":"gpt-6.1-sol","created":1700000200},
			{"id":"claude-fable-5-2","created_at":"2026-05-01T00:00:00Z"},
			{"id":"odd-dates","created":"yesterday"},
			{"id":"no-date"}
		]}`))
	}))
	t.Cleanup(srv.Close)

	got, err := NewOpenAI(presetAt(t, "openai", srv.URL), "sk-test", srv.Client()).Models(context.Background())
	if want := []string{"claude-fable-5-2", "gpt-6.1-sol", "gpt-4.1-mini", "odd-dates", "no-date"}; err != nil || !slices.Equal(got, want) {
		t.Errorf("with a key: %q, %v; want %q", got, err, want)
	}

	got, err = NewOpenAI(presetAt(t, "ollama", srv.URL), "", srv.Client()).Models(context.Background())
	if err != nil || len(got) != 13 || got[0] != "fixture-video-1" || got[1] != "text-embedding-3-small" || got[12] != "no-date" {
		t.Errorf("on this Mac: %q, %v; want all 13 in the server's order", got, err)
	}
}
