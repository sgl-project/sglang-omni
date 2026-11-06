package providers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// TestReady: a setup is offered once it has what it needs, in the
// presets' order: a key (unless its service needs none), an address
// (custom) and a model.
func TestReady(t *testing.T) {
	var s settings.Settings
	Put(&s, settings.Provider{ID: llm.Custom, Model: "qwen3"})                                    // no address
	Put(&s, settings.Provider{ID: "openai"})                                                      // no key
	Put(&s, settings.Provider{ID: "deepseek", Model: "deepseek-v4-pro"})                          // ready
	Put(&s, settings.Provider{ID: "ollama"})                                                      // keyless: ready
	Put(&s, settings.Provider{ID: "lmstudio"})                                                    // no model
	Put(&s, settings.Provider{ID: llm.Custom, Endpoint: "http://127.0.0.1:9/v1", Model: "qwen3"}) // replaces the first
	hasKey := func(id string) bool { return id == "deepseek" }
	var got []string
	for _, p := range Ready(s, hasKey) {
		got = append(got, p.ID+"="+Model(p))
	}
	if strings.Join(got, " ") != "ollama=qwen3 deepseek=deepseek-v4-pro custom=qwen3" || len(s.Providers.LLM) != 5 {
		t.Fatalf("ready %q, setups %+v", got, s.Providers.LLM)
	}
	for id, want := range map[string]string{"openai": MissingKey, "lmstudio": MissingModel} {
		p, _ := Setup(s, id)
		if m := Missing(p, false); m != want {
			t.Errorf("%s: missing %q, want %q", id, m, want)
		}
	}
	if m := Missing(settings.Provider{ID: llm.Custom, Model: "x"}, true); m != MissingEndpoint {
		t.Errorf("custom without an address: %q", m)
	}
	// Removing a setup takes it off the features that used it.
	s.Features.Translate.LLM, s.Features.Rewrite.LLM = "deepseek", "llm.fast"
	Remove(&s, "deepseek")
	if _, ok := Setup(s, "deepseek"); ok || s.Features.Translate.LLM != "" || s.Features.Rewrite.LLM != "llm.fast" {
		t.Fatalf("after Remove: %+v %+v", s.Providers.LLM, s.Features)
	}
}

// localModels is a local.Services with one language model.
type localModels struct{ local.NotInstalled }

var localLLM = local.Model{Kind: local.LLM, ID: "llm.balanced", Name: "Qwen3.5 4B OptiQ (4bit)"}

func (localModels) Installed(kind local.Kind) []local.Model {
	if kind == local.LLM {
		return []local.Model{localLLM}
	}
	return nil
}

func (localModels) LLM(_ context.Context, id string) (local.Chat, local.Model, error) {
	return nil, localLLM, nil
}

// TestServices: the ready remote models come after the local ones; a
// feature's choice opens it (with its key, over the client), automatic
// keeps the local one, and a remote model that is no longer ready falls
// back to the local one.
func TestServices(t *testing.T) {
	var got []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		got = append(got, fmt.Sprint(r.URL.Path, " ", r.Header.Get("Authorization"), " ", body["model"]))
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, `data: {"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}`+"\n\ndata: [DONE]\n\n")
	}))
	defer srv.Close()
	p := settings.Provider{ID: llm.Custom, Endpoint: srv.URL + "/v1", Model: "qwen3"}
	keys := map[string]string{llm.Custom: "sk-local"}
	svc := NewServices(localModels{}, func(p settings.Provider) string { return "Custom · " + Model(p) },
		func(_ context.Context, p settings.Provider) (local.Chat, error) {
			return Open(p, keys[p.ID], srv.Client())
		})
	svc.SetReady([]settings.Provider{p})
	list := svc.Installed(local.LLM)
	if len(list) != 2 || list[0].ID != localLLM.ID || list[1].ID != llm.Custom || list[1].Name != "Custom · qwen3" {
		t.Fatalf("installed %+v", list)
	}
	if _, m, err := svc.LLM(context.Background(), ""); err != nil || m.ID != localLLM.ID {
		t.Fatalf("automatic: %+v %v", m, err)
	}
	chat, m, err := svc.LLM(context.Background(), llm.Custom)
	if err != nil || m.ID != llm.Custom {
		t.Fatalf("remote: %+v %v", m, err)
	}
	var deltas []string
	resp, err := chat.Chat(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "x"}}, OnDelta: func(s string) { deltas = append(deltas, s) }})
	if err != nil || resp.Text != "hi" || strings.Join(deltas, "") != "hi" || strings.Join(got, ";") != "/v1/chat/completions Bearer sk-local qwen3" {
		t.Fatalf("chat %+v %v %q %q", resp, err, deltas, got)
	}
	svc.SetReady(nil)
	if _, m, err := svc.LLM(context.Background(), llm.Custom); err != nil || m.ID != localLLM.ID || len(svc.Installed(local.LLM)) != 1 {
		t.Fatalf("not ready any more: %+v %v", m, err)
	}
}

// TestTestLog: a connection test's log line names the service and model,
// never the key, even when the server's error quotes it.
func TestTestLog(t *testing.T) {
	p := settings.Provider{ID: "openai", Model: "gpt-x"}
	_, msg := TestLog(p, TestResult{Err: errors.New(`llm.bad_request detail="bad key sk-secret-123"`)}, "sk-secret-123")
	if strings.Contains(msg, "sk-secret-123") || !strings.Contains(msg, "openai (model gpt-x)") || !strings.Contains(msg, "[key]") {
		t.Fatalf("log %q", msg)
	}
}
