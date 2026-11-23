package llm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// ollamaLike serves chat completions under /v1 only, answers an unknown
// model with Ollama's 404 body, and any other path with Go's plain 404,
// as Ollama does.
func ollamaLike(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		var body struct{ Model string }
		_ = json.NewDecoder(r.Body).Decode(&body)
		if body.Model != "qwen3" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(404)
			_, _ = w.Write([]byte(`{"error":{"message":"model \"` + body.Model + `\" not found, try pulling it first","type":"api_error","param":null,"code":null}}`))
			return
		}
		jsonReply(200, map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": "ok"}, "finish_reason": "stop"}}})(w, r)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestOllamaBareRoot (#3950): the root Ollama prints, typed with no /v1,
// passes Test connection; a missing model still says the model is
// missing, and an address with nothing behind it says the address.
func TestOllamaBareRoot(t *testing.T) {
	srv := ollamaLike(t)
	for _, tc := range []struct{ base, model, code string }{
		{srv.URL, "qwen3", ""},
		{srv.URL + "/", "qwen3", ""},
		{srv.URL + "/v1", "qwen3", ""},
		{srv.URL + "/v1/chat/completions", "qwen3", ""},
		{srv.URL, "qwen3:8b", CodeModelNotFound},
		{srv.URL + "/api", "qwen3", CodeAddress},
		{srv.URL + "/api/chat", "qwen3", CodeAddress},
	} {
		p, err := ProviderConfig{ID: "ollama", BaseURL: tc.base, Model: tc.model}.Resolve()
		if err != nil {
			t.Fatalf("%s: %v", tc.base, err)
		}
		_, err = Probe(context.Background(), NewOpenAI(p, "", srv.Client()))
		if Code(err) != tc.code {
			t.Errorf("%s model %s: got %v, want %q", tc.base, tc.model, err, tc.code)
		}
	}
}

// TestResolveBareRootTakesPresetPath: an address on this Mac with no path
// gets the preset's API path; a remote one, one with a path, and the
// custom preset's are kept.
func TestResolveBareRootTakesPresetPath(t *testing.T) {
	for _, tc := range []struct{ id, base, want string }{
		{"ollama", "http://localhost:11434", "http://localhost:11434/v1"},
		{"ollama", "http://127.0.0.1:11434/", "http://127.0.0.1:11434/v1"},
		{"lmstudio", "http://localhost:1234", "http://localhost:1234/v1"},
		{"zhipu", "https://open.bigmodel.cn", "https://open.bigmodel.cn"}, // a remote root may be a proxy's
		{"ollama", "http://localhost:11434/api", "http://localhost:11434/api"},
		{Custom, "http://localhost:8080", "http://localhost:8080"},
	} {
		p, err := ProviderConfig{ID: tc.id, BaseURL: tc.base}.Resolve()
		if err != nil || p.BaseURL != tc.want {
			t.Errorf("%s %s: %q %v, want %q", tc.id, tc.base, p.BaseURL, err, tc.want)
		}
	}
}
