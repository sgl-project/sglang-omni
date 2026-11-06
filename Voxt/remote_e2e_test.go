package main

// The remote language models in the app: a session through a remote
// model over a local mock of an OpenAI-compatible server, and the Remote
// models page's dialog saving a setup and its key.

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// memKeys is a keyring in memory (the Keychain's stand-in).
type memKeys struct {
	mu sync.Mutex
	m  map[string]string
}

func (k *memKeys) Get(id string) (string, error) {
	k.mu.Lock()
	defer k.mu.Unlock()
	return k.m[id], nil
}

func (k *memKeys) Set(id, _, value string) error {
	k.mu.Lock()
	defer k.mu.Unlock()
	if value == "" {
		delete(k.m, id)
	} else {
		k.m[id] = value
	}
	return nil
}

// useKeys makes new apps keep their keys in k.
func useKeys(t *testing.T, k *memKeys) {
	old := newKeyring
	newKeyring = func() keyring { return k }
	t.Cleanup(func() { newKeyring = old })
}

// mockOpenAI is an OpenAI-compatible server that streams answer, and the
// requests it got (path, Authorization, body).
func mockOpenAI(t *testing.T, answer string) (*httptest.Server, func() []string) {
	var mu sync.Mutex
	var got []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		raw, _ := json.Marshal(body)
		mu.Lock()
		got = append(got, r.URL.Path+" "+r.Header.Get("Authorization")+" "+string(raw))
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		for _, part := range strings.SplitAfter(answer, "，") {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", part)
			w.(http.Flusher).Flush()
		}
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)
	return srv, func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), got...)
	}
}

// TestRemoteModelEndToEnd: with no local language model, Translate's
// remote model (its key in the keyring) translates the selection: one
// streamed request to /v1/chat/completions with the key and the model,
// its answer pasted and kept in the history.
func TestRemoteModelEndToEnd(t *testing.T) {
	srv, requests := mockOpenAI(t, "早上好，团队。")
	useKeys(t, &memKeys{m: map[string]string{"llm." + llm.Custom: "sk-mock-123"}})
	ev := script(t, hotkeyLine(0, "translate/0", proto.PhaseTrigger))
	a := newSessionApp(t, helpertest.Opt("selection", "Good morning, team.")+","+ev, func(s *settings.Settings) {
		s.Features.Translate.Target = "zh-Hans"
		s.Features.Translate.LLM = llm.Custom
		s.Providers.LLM = []settings.Provider{{ID: llm.Custom, Endpoint: srv.URL + "/v1", Model: "mock-model"}}
	}, local.NotInstalled{})
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if end := a.sess; end.State != session.Done || end.Result != "早上好，团队。" || !end.Inserted {
		t.Fatalf("end %+v", end)
	}
	if p := a.pasted(); len(p) != 1 || p[0].Text != "早上好，团队。" {
		t.Errorf("pasted %+v", p)
	}
	got := requests()
	if len(got) != 1 || !strings.HasPrefix(got[0], "/v1/chat/completions Bearer sk-mock-123 ") ||
		!strings.Contains(got[0], `"model":"mock-model"`) || !strings.Contains(got[0], `"stream":true`) || !strings.Contains(got[0], "Good morning, team.") {
		t.Fatalf("requests %q", got)
	}
	if es := a.history(); len(es) != 1 || es[0].LLMProvider != llm.Custom {
		t.Errorf("history %+v", es)
	}
}

// TestRemoteErrorKeepsNoKey: a remote model that quotes the key in its
// error fails the session, and the error kept in the history has the key
// blanked out.
func TestRemoteErrorKeepsNoKey(t *testing.T) {
	const key = "sk-mock-secret-77"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintf(w, `{"error":{"message":"invalid header %s"}}`, r.Header.Get("Authorization"))
	}))
	t.Cleanup(srv.Close)
	useKeys(t, &memKeys{m: map[string]string{llm.Custom: key}})
	ev := script(t, hotkeyLine(0, "translate/0", proto.PhaseTrigger))
	a := newSessionApp(t, helpertest.Opt("selection", "Good morning, team.")+","+ev, func(s *settings.Settings) {
		s.Features.Translate.Target = "zh-Hans"
		s.Features.Translate.LLM = llm.Custom
		s.Providers.LLM = []settings.Provider{{ID: llm.Custom, Endpoint: srv.URL + "/v1", Model: "mock-model"}}
	}, local.NotInstalled{})
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if a.sess.State != session.Failed {
		t.Fatalf("end %+v", a.sess)
	}
	es := a.history()
	if len(es) != 1 || !strings.Contains(es[0].Error, "invalid header Bearer [key]") || strings.Contains(es[0].Error, key) {
		t.Fatalf("history %+v", es)
	}
}

// TestRemotePage: setting a service up on the Language models page saves
// its key in the keyring (never the settings) and offers its model to the
// features; Remove takes both away again.
func TestRemotePage(t *testing.T) {
	keys := &memKeys{m: map[string]string{}}
	useKeys(t, keys)
	a, _, tt := newTestApp(t)
	a.navigate(remotePath)
	frames(tt)
	if !has(tt, "On this Mac") || !has(tt, "Remote") || !has(tt, "Online services") { // Ollama's card and group, DeepSeek's
		t.Errorf("no where labels: %q", tt.Texts())
	}
	click(t, tt, "DeepSeek")
	frames(tt)
	a.provDlg.key = "sk-deep-0123456789"
	click(t, tt, "Save")
	frames(tt)
	if keys.m["llm.deepseek"] != "sk-deep-0123456789" {
		t.Fatalf("keys %v", keys.m)
	}
	if p, ok := a.lib.settings.Get().Providers.LLM, true; !ok || len(p) != 1 || p[0].ID != "deepseek" || p[0].Model != "" {
		t.Fatalf("setups %+v", p)
	}
	raw, _ := json.Marshal(a.lib.settings.Get())
	if strings.Contains(string(raw), "sk-deep") {
		t.Fatal("the key is in the settings")
	}
	if !has(tt, "sk-…6789") || !has(tt, "Used by Transcribe · Translate · Rewrite") {
		t.Errorf("no mask of the saved key, or where it is used: %q", tt.Texts())
	}
	m, ok := a.featureModel(local.LLM, hotkey.Rewrite)
	if !ok || m.ID != "deepseek" || m.Name != "DeepSeek · deepseek-flash" {
		t.Fatalf("Rewrite's model %+v %v", m, ok)
	}
	click(t, tt, "DeepSeek")
	frames(tt)
	click(t, tt, "Remove DeepSeek")
	frames(tt)
	if len(keys.m) != 0 || len(a.lib.settings.Get().Providers.LLM) != 0 || len(a.installed(local.LLM)) != 0 {
		t.Fatalf("after Remove: keys %v, setups %+v", keys.m, a.lib.settings.Get().Providers.LLM)
	}
}
