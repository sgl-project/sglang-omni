package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
)

// ollamaServer is Ollama with qwen3:8b and llama3.2:3b pulled: its
// OpenAI-compatible API under /v1 only, a model it lacks answered with
// its 404, any other path with Go's plain 404.
func ollamaServer(t *testing.T) *httptest.Server {
	t.Helper()
	have := []string{"qwen3:8b", "llama3.2:3b"}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			var list struct {
				Data []map[string]string `json:"data"`
			}
			for _, m := range have {
				list.Data = append(list.Data, map[string]string{"id": m, "object": "model"})
			}
			_ = json.NewEncoder(w).Encode(list)
		case "/v1/chat/completions":
			var body struct{ Model string }
			_ = json.NewDecoder(r.Body).Decode(&body)
			if !slices.Contains(have, body.Model) {
				w.WriteHeader(http.StatusNotFound)
				_, _ = w.Write([]byte(`{"error":{"message":"model \"` + body.Model + `\" not found, try pulling it first","type":"api_error"}}`))
				return
			}
			_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(srv.Close)
	return srv
}

// TestOllamaModelMenuListsServer (#3950): with Ollama's root as the
// address and the preset's qwen3 (not pulled), Test connection says the
// model is missing and names the pulled ones, and the Model menu offers
// them; with one of them chosen the test passes.
func TestOllamaModelMenuListsServer(t *testing.T) {
	useKeys(t, &memKeys{m: map[string]string{}})
	srv := ollamaServer(t)
	a, _, tt := newTestApp(t)
	a.navigate(remotePath)
	frames(tt)
	a.openProviderDialog(local.LLM, "ollama")
	frames(tt)
	d := &a.provDlg
	if d.model != "qwen3" || d.served != nil {
		t.Fatalf("opened with model %q, served %v", d.model, d.served)
	}
	runListModels = providers.Models
	t.Cleanup(func() { runListModels = failListModels })
	d.endpoint = srv.URL // the root Ollama prints, no /v1
	click(t, tt, "Test connection")
	if want := "The server does not have this model. It has: qwen3:8b, llama3.2:3b"; !has(tt, want) {
		t.Errorf("no %q: %q", want, firstTexts(tt))
	}
	if got := d.modelOptions(lookupService(local.LLM, "ollama").models); !slices.Equal(got, []string{"qwen3:8b", "llama3.2:3b", "qwen3", otherModel()}) {
		t.Errorf("menu %q", got)
	}
	d.model = "qwen3:8b"
	click(t, tt, "Test connection")
	if d.test.res.Err != nil || !hasPrefix(tt, "Connection OK") {
		t.Errorf("with qwen3:8b: %v %q", d.test.res.Err, firstTexts(tt))
	}

	// A wrong path says the address, not the model; the menu goes back to
	// the preset's list rather than keep the other address's.
	d.endpoint = srv.URL + "/api"
	click(t, tt, "Test connection")
	if llm.Code(d.test.res.Err) != llm.CodeAddress || d.served != nil {
		t.Errorf("at /api: %v, served %v", d.test.res.Err, d.served)
	}
}

// TestProviderDialogOffer: a new, untouched setup takes the server's
// first model when it lacks the default; a typed model ID the server has
// becomes its menu item; a setup the user changed keeps its model.
func TestProviderDialogOffer(t *testing.T) {
	useKeys(t, &memKeys{m: map[string]string{}})
	a, _, _ := newTestApp(t)
	served := []string{"qwen3:8b", "llama3.2:3b"}

	a.openProviderDialog(local.LLM, "ollama")
	a.provDlg.offer(served)
	if d := a.provDlg; d.model != "qwen3:8b" || d.unsaved() {
		t.Errorf("new setup: model %q, unsaved %v", d.model, d.unsaved())
	}

	a.openProviderDialog(local.LLM, "ollama")
	a.provDlg.model, a.provDlg.customModel = otherModel(), " llama3.2:3b "
	a.provDlg.offer(served)
	if d := a.provDlg; d.model != "llama3.2:3b" || d.customModel != "" {
		t.Errorf("typed id: model %q, custom %q", d.model, d.customModel)
	}

	a.openProviderDialog(local.LLM, "ollama")
	a.provDlg.endpoint = "http://127.0.0.1:11434"
	a.provDlg.offer(served)
	if d := a.provDlg; d.model != "qwen3" {
		t.Errorf("changed setup: model %q", d.model)
	}
	if got := a.provDlg.modelOptions(nil); !slices.Equal(got, []string{"qwen3:8b", "llama3.2:3b", "qwen3", otherModel()}) {
		t.Errorf("menu %q", got)
	}
}
