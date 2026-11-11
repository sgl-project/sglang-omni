package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// fakeLegacyPlutil replaces the preferences' reader by one that answers
// values by key, and fails for a key it does not have. On Linux, where
// there is no plutil, the import can then be tested end to end.
func fakeLegacyPlutil(t *testing.T, values map[string]string) {
	t.Helper()
	dir := t.TempDir()
	for k, v := range values {
		if err := os.WriteFile(filepath.Join(dir, k), []byte(v), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	script := filepath.Join(dir, "plutil")
	body := "#!/bin/sh\n[ \"$1\" = -lint ] && exit 0\nexec cat \"" + dir + "/$2\" 2>/dev/null\n"
	if err := os.WriteFile(script, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	old := legacyPlutil
	legacyPlutil = script
	t.Cleanup(func() { legacyPlutil = old })
}

// importSetups runs the import over llmSetups (the preferences' JSON array),
// with the Transcribe clean-up and the rest of the features left unset.
func importSetups(t *testing.T, llmSetups string) legacyReport {
	t.Helper()
	fakeLegacyPlutil(t, map[string]string{"remoteLLMProviderConfigurations": llmSetups})
	a, _, _ := newTestApp(t)
	a.importLegacyFrom(copyFixture(t, "real"))
	return a.ob.legacy
}

func oneLLM(t *testing.T, setup string) (settings.Provider, legacySettings) {
	t.Helper()
	l := parseLegacySettings("["+setup+"]", "", "", "")
	if len(l.llm) != 1 {
		t.Fatalf("setups %+v from %s", l.llm, setup)
	}
	return l.llm[0], l
}

// pathServer answers every request and records the last one's path.
type pathServer struct {
	*httptest.Server
	mu   sync.Mutex
	last string
}

func newPathServer(t *testing.T) *pathServer {
	t.Helper()
	s := &pathServer{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		s.last = r.URL.RequestURI()
		s.mu.Unlock()
		io.WriteString(w, `{"model":"m","choices":[{"message":{"content":"ok"},"finish_reason":"stop"}]}`)
	}))
	t.Cleanup(s.Close)
	return s
}

// requestPath is where a request with setup p goes.
func (s *pathServer) requestPath(t *testing.T, p settings.Provider) string {
	t.Helper()
	b, err := providers.NewLLM(p, "k", s.Client())
	if err != nil {
		t.Fatalf("%s at %q: %v", p.ID, p.Endpoint, err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if _, err := b.Chat(ctx, llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "hi"}}, Timeout: 5 * time.Second}); err != nil {
		t.Fatalf("%s: %v", p.ID, err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.last
}

func TestLegacyBareAddressGetsTheServiceRoot(t *testing.T) {
	srv := newPathServer(t)
	for _, c := range []struct{ service, want string }{
		{"openAI", "/v1/chat/completions"}, {"deepseek", "/v1/chat/completions"}, {"openrouter", "/v1/chat/completions"},
		{"grok", "/v1/chat/completions"}, {"zai", "/v1/chat/completions"}, {"kimi", "/v1/chat/completions"},
		{"stepFun", "/v1/chat/completions"}, {"xiaomiMiMo", "/v1/chat/completions"},
		{"anthropic", "/v1/chat/completions"}, {"minimax", "/v1/chat/completions"}, {"omlx", "/v1/chat/completions"}, {"lmStudio", "/v1/chat/completions"},
		{"volcengine", "/api/v3/chat/completions"}, {"aliyunBailian", "/compatible-mode/v1/chat/completions"},
		{"google", "/v1beta/openai/chat/completions"},
	} {
		p, l := oneLLM(t, fmt.Sprintf(`{"providerID":%q,"model":"m","endpoint":%q}`, c.service, srv.URL))
		if got := srv.requestPath(t, p); got != c.want {
			t.Errorf("%s: requests go to %s, want %s", c.service, got, c.want)
		}
		if !slices.Contains(l.checkLLM, p.ID) {
			t.Errorf("%s: a bare address is not to check", c.service)
		}
	}
	for _, c := range []struct {
		service, endpoint, want string
		check                   bool
	}{
		{"stepFun", "https://api.stepfun.com", "", false},                                              // the preset's root once the path is added
		{"ollama", "http://127.0.0.1:11434", "http://127.0.0.1:11434/v1", false},                       // its own rule
		{"deepseek", "https://relay.example.com/chat/completions", "https://relay.example.com", false}, // had a path: not bare
		{"openAI", "https://relay.example.com/", "https://relay.example.com/v1", true},
	} {
		if got, check := legacyEndpoint(c.service, c.endpoint); got != c.want || check != c.check {
			t.Errorf("%s at %s: %q, check %v; want %q, %v", c.service, c.endpoint, got, check, c.want, c.check)
		}
	}
}

// A query string is not written into, is not kept, and the user is told.
func TestLegacyAddressWithQuery(t *testing.T) {
	srv := newPathServer(t)
	p, l := oneLLM(t, fmt.Sprintf(`{"providerID":"deepseek","model":"m","endpoint":%q}`, srv.URL+"/openai/deployments/d/chat/completions?api-version=2024-10-21"))
	if strings.Contains(p.Endpoint, "?") || !slices.Contains(l.checkLLM, "deepseek") {
		t.Errorf("endpoint %q, to check %v", p.Endpoint, l.checkLLM)
	}
	if got := srv.requestPath(t, p); got != "/openai/deployments/d/chat/completions" {
		t.Errorf("requests go to %s", got)
	}
	if got, check := legacyEndpoint("openAI", "https://api.openai.com/v1/responses?x=1"); got != "" || !check {
		t.Errorf("the service's default with a query: %q, check %v", got, check)
	}
}

func TestLegacyOpenAIGeneration(t *testing.T) {
	all := `"maxOutputTokens":300,"temperature":0.2,"topP":0.8,"topK":4,"minP":0.1,"seed":7,"stop":["x"],"presencePenalty":0.1,"frequencyPenalty":0.2,"repetitionPenalty":1.1,"logprobs":true,"topLogprobs":3,"responseFormat":"json"`
	p, l := oneLLM(t, fmt.Sprintf(`{"providerID":"openAI","model":"gpt-5","openAITextVerbosity":"low","ollamaKeepAlive":"5m","generationSettings":{%s,"thinking":{"mode":"effort","effort":"low"},"extraOptionsJSON":"{\"a\":1}"}}`, all))
	g := p.Generation
	if g.Temperature != nil || g.TopP != nil || g.TopK != nil || g.MinP != nil || g.Seed != nil || len(g.Stop) != 0 || g.PresencePenalty != nil ||
		g.FrequencyPenalty != nil || g.RepetitionPenalty != nil || g.Logprobs || g.TopLogprobs != nil || g.KeepAlive != "" || strings.Contains(g.ExtraBodyJSON, `"options"`) {
		t.Errorf("OpenAI kept what the previous Voxt never sent: %+v", g)
	}
	if g.MaxOutputTokens == nil || *g.MaxOutputTokens != 300 || g.ResponseFormat != "json" || g.Thinking != llm.ThinkEffort || g.ThinkingEffort != "low" || g.Verbosity != "low" {
		t.Errorf("OpenAI lost what the previous Voxt sent: %+v", g)
	}
	if len(l.checkLLM) != 0 {
		t.Errorf("to check %v", l.checkLLM)
	}

	// Not only OpenAI's: another service keeps them.
	d, _ := oneLLM(t, fmt.Sprintf(`{"providerID":"deepseek","model":"deepseek-chat","generationSettings":{%s}}`, all))
	if g := d.Generation; g.Temperature == nil || g.TopP == nil || g.Seed == nil || len(g.Stop) != 1 || !g.Logprobs {
		t.Errorf("deepseek lost generation settings: %+v", g)
	}
	// The effort and verbosity filters, by model.
	for _, c := range []struct {
		model, effort string
		kept          bool
		verbosity     string
	}{
		{"gpt-4.1", "high", false, ""},
		{"gpt-5.2", "none", true, "low"},
		{"gpt-5", "none", false, "low"},
		{"gpt-5.2-pro", "low", false, "low"},
		{"gpt-5.2-pro-2025-12-11", "medium", true, "low"},
		{"gpt-5-pro", "medium", false, "low"},
		{"gpt-5.1", "xhigh", false, "low"},
		{"gpt-5.1", "none", true, "low"},
		{"o3-mini", "low", true, ""},
		{"GPT-5.2", "high", true, "low"},
		{"gpt-5.3-codex-spark", "none", true, "low"},
		{"gpt-5.2-codex", "none", false, "low"},
		{"gpt-5.1-codex-max", "low", false, "low"},
	} {
		p, _ := oneLLM(t, fmt.Sprintf(`{"providerID":"openAI","model":%q,"openAITextVerbosity":"low","generationSettings":{"thinking":{"mode":"effort","effort":%q}}}`, c.model, c.effort))
		g := p.Generation
		if g.Verbosity != c.verbosity {
			t.Errorf("%s: verbosity %q, want %q", c.model, g.Verbosity, c.verbosity)
		}
		if kept := g.Thinking == llm.ThinkEffort && g.ThinkingEffort == c.effort; kept != c.kept {
			t.Errorf("%s effort %s: thinking %q %q, want kept=%v", c.model, c.effort, g.Thinking, g.ThinkingEffort, c.kept)
		}
	}
}

func TestLegacyOpenAIToCheck(t *testing.T) {
	for _, c := range []struct {
		setup string
		check bool
	}{
		{`{"providerID":"openAI","model":"gpt-5","generationSettings":{"extraBodyJSON":"{\"text\":{}}"}}`, true},
		{`{"providerID":"openAI","model":"gpt-5","generationSettings":{"extraBodyJSON":""}}`, false},
		{`{"providerID":"openAI","model":"gpt-5.2-pro"}`, true},
		{`{"providerID":"openAI","model":"gpt-5-pro-2025-10-06"}`, true},
		{`{"providerID":"openAI","model":"GPT-5.2-Pro"}`, true},
		{`{"providerID":"openAI","model":"gpt-5.2"}`, false},
		{`{"providerID":"openrouter","model":"gpt-5.2-pro","generationSettings":{"extraBodyJSON":"{\"a\":1}"}}`, false},
	} {
		_, l := oneLLM(t, c.setup)
		if got := len(l.checkLLM) == 1; got != c.check {
			t.Errorf("%s: to check %v, want %v", c.setup, l.checkLLM, c.check)
		}
	}
}

func hasTodo(r legacyReport, id, want string) bool {
	return slices.ContainsFunc(r.todo, func(d legacyTodo) bool { return d.id == id && strings.Contains(d.text, want) })
}

// A key that existed there is asked for again, where the setup would
// otherwise read as ready.
func TestLegacyImportAsksForOptionalKey(t *testing.T) {
	setups := `[
	 {"providerID":"ollama","model":"qwen3","endpoint":"http://localhost:11434/api/chat","storedCredentialPresence":{"apiKey":true}},
	 {"providerID":"omlx","model":"m","storedCredentialPresence":{"apiKey":true}},
	 {"providerID":"lmStudio","model":"m","storedCredentialPresence":{"apiKey":true}}
	]`
	r := importSetups(t, setups)
	for _, id := range []string{"ollama", "omlx", "lmstudio"} {
		if !hasTodo(r, id, "needs its API key") {
			t.Errorf("%s: no to-do for its key: %q", id, r.text())
		}
	}
	r = importSetups(t, `[{"providerID":"ollama","model":"qwen3","endpoint":"http://localhost:11434/api/chat","storedCredentialPresence":{"apiKey":false}},
	 {"providerID":"deepseek","model":"deepseek-chat","storedCredentialPresence":{"apiKey":true}}]`)
	if hasTodo(r, "ollama", "API key") || len(r.todo) != 1 || !hasTodo(r, "deepseek", "needs its API key") {
		t.Errorf("without a saved key it is left alone, and a service that needs one asks once: %q", r.text())
	}
}

// An address this app would refuse says so, as a to-do; Resolve keeps
// refusing it.
func TestLegacyImportPlainHTTPElsewhere(t *testing.T) {
	r := importSetups(t, `[{"providerID":"ollama","model":"qwen3","endpoint":"http://192.168.1.20:11434/api/chat"},
	 {"providerID":"lmStudio","model":"m","endpoint":"http://127.0.0.1:1234/v1"}]`)
	if !hasTodo(r, "ollama", "http to this Mac") || hasTodo(r, "ollama", "may not work") {
		t.Errorf("no to-do naming the address rule: %q", r.text())
	}
	if hasTodo(r, "lmstudio", "http") {
		t.Errorf("a server on this Mac was flagged: %q", r.text())
	}
	r = importSetups(t, `[{"providerID":"ollama","model":"qwen3","endpoint":"http://192.168.1.20:11434/api/chat","storedCredentialPresence":{"apiKey":true}}]`)
	if len(r.todo) != 2 || !hasTodo(r, "ollama", "needs its API key") || !hasTodo(r, "ollama", "http to this Mac") {
		t.Errorf("a key that existed there and a refused address are two to-dos: %q", r.text())
	}
	p, _ := oneLLM(t, `{"providerID":"ollama","model":"qwen3","endpoint":"http://192.168.1.20:11434/api/chat"}`)
	if _, err := providers.Config(p).Resolve(); err == nil {
		t.Errorf("%q resolved", p.Endpoint)
	}
}

// A setup at a service's default address, with its settings untouched,
// comes out as before: asked only for its key.
func TestLegacyDefaultAddressesUnchanged(t *testing.T) {
	r := importSetups(t, `[{"providerID":"openAI","model":"gpt-5.2","endpoint":"https://api.openai.com/v1/responses","storedCredentialPresence":{"apiKey":true},"openAITextVerbosity":"automatic"},
	 {"providerID":"aliyunBailian","model":"qwen3","endpoint":"https://dashscope.aliyuncs.com/compatible-mode/v1/responses","storedCredentialPresence":{"apiKey":true}}]`)
	if len(r.todo) != 2 || !hasTodo(r, "openai", "needs its API key") || !hasTodo(r, "dashscope", "needs its API key") {
		t.Errorf("default-address setups: %q", r.text())
	}
}

// A key already saved here is not asked for again; one that is not, is.
func TestLegacyImportKeySavedHere(t *testing.T) {
	const setups = `[{"providerID":"openrouter","model":"m","storedCredentialPresence":{"apiKey":true}},
	 {"providerID":"deepseek","model":"deepseek-chat","storedCredentialPresence":{"apiKey":true}}]`
	const speech = `[{"providerID":"glmASR","model":"glm-asr-2512","storedCredentialPresence":{"apiKey":true}},
	 {"providerID":"stepFunASR","model":"step-asr","storedCredentialPresence":{"apiKey":true}}]`
	useKeys(t, &memKeys{m: map[string]string{providers.Account(local.LLM, "openrouter"): "sk-saved", providers.Account(local.ASR, providers.ZhipuASR): "sk-saved"}})
	fakeLegacyPlutil(t, map[string]string{"remoteLLMProviderConfigurations": setups, "remoteASRProviderConfigurations": speech})
	a, _, _ := newTestApp(t)
	a.importLegacyFrom(copyFixture(t, "real"))
	r := a.ob.legacy
	if hasTodo(r, "openrouter", "needs its API key") || hasTodo(r, providers.ZhipuASR, "needs its API key") {
		t.Errorf("asked again for a key saved here: %q", r.text())
	}
	if !hasTodo(r, "deepseek", "needs its API key") || !hasTodo(r, providers.StepFunASR, "needs its API key") {
		t.Errorf("no to-do for a key not saved here: %q", r.text())
	}
}

// A setup at an address Resolve refuses is not ready, is not what Automatic
// picks, and its card says so with Complete setup.
func TestLegacyRefusedAddressNotReady(t *testing.T) {
	useKeys(t, &memKeys{m: map[string]string{}})
	a, _, tt := newTestApp(t)
	lan := settings.Provider{ID: "ollama", Endpoint: "http://192.168.1.20:11434/v1", Model: "qwen3"}
	a.updateSettings(func(s *settings.Settings) { s.Providers.LLM = []settings.Provider{lan} })
	a.syncRemote()
	if got := providers.Missing(lan, false); got != providers.MissingAddress {
		t.Errorf("Missing %q, want %q", got, providers.MissingAddress)
	}
	if ready := providers.Ready(a.lib.settings.Get(), func(string) bool { return true }); len(ready) != 0 {
		t.Errorf("ready: %+v", ready)
	}
	if list := a.installed(local.LLM); slices.ContainsFunc(list, func(m local.Model) bool { return m.ID == "ollama" }) {
		t.Errorf("Automatic can pick it: %+v", list)
	}
	a.navigate(remotePath)
	frames(tt)
	if !has(tt, "Address not allowed") || !slices.Contains(tt.AccessLabels(), "Complete setup of Ollama") {
		t.Fatalf("card: %q", tt.Texts())
	}
	if err := tt.ClickControl("Complete setup of Ollama"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if !a.provDlg.open || a.provDlg.id != "ollama" {
		t.Errorf("Complete setup: %+v", a.provDlg.open)
	}
	if got := setupNeedsText(local.LLM, lan, providers.MissingAddress); !strings.Contains(got, "http to this Mac") {
		t.Errorf("the notice says %q", got)
	}
}
