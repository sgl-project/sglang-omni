package main

// The ChatGPT (Codex) sign-in in the app: over a fake OAuth service and a
// fake Responses endpoint. No test talks to OpenAI, and every token here
// is made up.

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/codexauth"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/providers"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

const (
	testAccess  = "at-FAKE-access-token-1234"
	testRefresh = "rt-FAKE-refresh-token-5678"
	testIDToken = "eyJh.eyJemailpart.sigFAKEidtoken"
)

func testJWT(claims map[string]any) string {
	raw, _ := json.Marshal(claims)
	return "x." + base64.RawURLEncoding.EncodeToString(raw) + ".sig"
}

// fakeOAuth is the OAuth service: token answers, and the requests it got.
type fakeOAuth struct {
	*httptest.Server
	mu      sync.Mutex
	reqs    int
	fail    bool
	refresh bool // a refresh got a token
}

func useFakeOAuth(t *testing.T) *fakeOAuth {
	f := &fakeOAuth{}
	f.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		f.mu.Lock()
		defer f.mu.Unlock()
		f.reqs++
		if f.fail {
			w.WriteHeader(400)
			fmt.Fprint(w, `{"error":"invalid_grant"}`)
			return
		}
		f.refresh = true
		id := testJWT(map[string]any{"email": "me@example.com", "https://api.openai.com/auth": map[string]any{"chatgpt_account_id": "acct-9"}})
		json.NewEncoder(w).Encode(map[string]string{"id_token": id, "access_token": testAccess, "refresh_token": testRefresh})
	}))
	t.Cleanup(f.Close)
	old := chatgptEndpoints
	chatgptEndpoints = codexauth.Endpoints{Authorize: f.URL + "/authorize", Token: f.URL + "/token"}
	t.Cleanup(func() { chatgptEndpoints = old })
	return f
}

// browserAnswers makes openBrowser come back to Voxt with the code.
func browserAnswers(t *testing.T) {
	old := openBrowser
	openBrowser = func(u string) error {
		q, err := url.Parse(u)
		if err != nil {
			return err
		}
		go func() {
			resp, err := http.Get(q.Query().Get("redirect_uri") + "?code=c1&state=" + q.Query().Get("state"))
			if err == nil {
				resp.Body.Close()
			}
		}()
		return nil
	}
	t.Cleanup(func() { openBrowser = old })
}

func waitFrames(t *testing.T, tt *ui.Tester, cond func() bool) {
	t.Helper()
	for deadline := time.Now().Add(5 * time.Second); !cond(); {
		if time.Now().After(deadline) {
			t.Fatalf("timed out: %q", tt.Texts())
		}
		frames(tt)
		time.Sleep(10 * time.Millisecond)
	}
}

// TestChatGPTSignIn: the card's dialog has no key field; signing in saves
// only the Keychain item and makes the setup ready; signing out deletes
// the item and keeps the setup.
func TestChatGPTSignIn(t *testing.T) {
	useFakeOAuth(t)
	browserAnswers(t)
	keys := &memKeys{m: map[string]string{}}
	useKeys(t, keys)
	a, _, tt := newTestApp(t)
	a.navigate(remotePath)
	frames(tt)
	a.openProviderDialog(local.LLM, providers.ChatGPT) // its card is further down the page
	frames(tt)
	if hasExact(tt, "API key") || hasExact(tt, "Address") || has(tt, "Get a key") || has(tt, "Generation settings") || !has(tt, "Sign in with ChatGPT") {
		t.Fatalf("dialog: %q", tt.Texts())
	}
	click(t, tt, "Sign in with ChatGPT")
	waitFrames(t, tt, func() bool { return has(tt, "Signed in as me@example.com") })
	if !has(tt, "Manage usage") || !has(tt, "Sign out") || has(tt, "Sign in with ChatGPT") {
		t.Errorf("signed in: %q", tt.Texts())
	}
	item := keys.m["llm.codex"]
	if codexauth.Email(item) != "me@example.com" || strings.Contains(item, testAccess) {
		t.Errorf("item %q", item)
	}
	raw, _ := json.Marshal(a.lib.settings.Get())
	for _, s := range []string{testAccess, testRefresh, testIDToken} {
		if strings.Contains(string(raw), s) {
			t.Errorf("settings hold %q", s)
		}
	}
	if m, ok := a.featureModel(local.LLM, hotkey.Rewrite); !ok || m.ID != "codex" || m.Name != "ChatGPT (Codex) · gpt-6.1-sol" {
		t.Fatalf("Rewrite's model %+v %v", m, ok)
	}
	click(t, tt, "Sign out")
	waitFrames(t, tt, func() bool { return has(tt, "Sign in with ChatGPT") })
	if len(keys.m) != 0 {
		t.Errorf("keys %v", keys.m)
	}
	if ps := a.lib.settings.Get().Providers.LLM; len(ps) != 1 || ps[0].ID != "codex" {
		t.Errorf("setups %+v", ps)
	}
	if len(a.installed(local.LLM)) != 0 {
		t.Errorf("still offered: %+v", a.installed(local.LLM))
	}
}

// TestChatGPTRemove: removing the setup also forgets the sign-in in
// memory, not just the Keychain item.
func TestChatGPTRemove(t *testing.T) {
	useFakeOAuth(t)
	browserAnswers(t)
	keys := &memKeys{m: map[string]string{}}
	useKeys(t, keys)
	a, _, tt := newTestApp(t)
	a.navigate(remotePath)
	frames(tt)
	a.openProviderDialog(local.LLM, providers.ChatGPT)
	frames(tt)
	click(t, tt, "Sign in with ChatGPT")
	waitFrames(t, tt, func() bool { return has(tt, "Signed in as me@example.com") })
	a.removeProvider()
	waitFrames(t, tt, func() bool { return len(a.lib.settings.Get().Providers.LLM) == 0 })
	if len(keys.m) != 0 {
		t.Errorf("keys %v", keys.m)
	}
	if _, err := providers.ChatGPTTokens(a.chatgpt)(context.Background(), ""); llm.Code(err) != llm.CodeSignIn {
		t.Errorf("still signed in: %v", err)
	}
}

// TestChatGPTMask: the app keeps the account's email, never any of the
// item, which holds a token.
func TestChatGPTMask(t *testing.T) {
	if got := maskFor(chatgptAccount(), signedInItem(t)); got != "me@example.com" {
		t.Errorf("%q", got)
	}
	if got := maskFor(chatgptAccount(), "garbage"); got != "•••" {
		t.Errorf("no email: %q", got)
	}
	if got := maskFor("llm.openai", "sk-abcdefghijkl"); got != maskKey("sk-abcdefghijkl") {
		t.Errorf("other key: %q", got)
	}
	if got := maskFor(chatgptAccount(), ""); got != "" {
		t.Errorf("none: %q", got)
	}
}

// TestChatGPTPortInUse: the dialog says what to do when the callback's
// port is taken.
func TestChatGPTPortInUseText(t *testing.T) {
	msg, _ := chatgptLoginText(&codexauth.Error{Kind: codexauth.KindPortInUse, Port: 1455})
	if !strings.Contains(msg, "1455") || !strings.Contains(msg, "codex login") {
		t.Errorf("%q", msg)
	}
	for _, k := range []codexauth.Kind{codexauth.KindDenied, codexauth.KindTimeout, codexauth.KindExchange, codexauth.KindNetwork, codexauth.KindStore} {
		if msg, _ := chatgptLoginText(&codexauth.Error{Kind: k}); msg == "" {
			t.Errorf("%s: no text", k)
		}
	}
}

// mockCodex is the fake Responses endpoint: it answers 1.x's event shapes.
func mockCodex(t *testing.T, answer string, status int, body string) (*httptest.Server, func() []string) {
	var mu sync.Mutex
	var got []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		_ = json.NewDecoder(r.Body).Decode(&req)
		raw, _ := json.Marshal(req)
		mu.Lock()
		got = append(got, r.URL.Path+" "+r.Header.Get("Authorization")+" "+r.Header.Get("ChatGPT-Account-ID")+" "+r.Header.Get("originator")+" "+string(raw))
		mu.Unlock()
		if status != 0 {
			w.WriteHeader(status)
			fmt.Fprint(w, body)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":%q}\n\n", answer)
		fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":3,\"output_tokens\":2,\"total_tokens\":5}}}\n\n")
	}))
	t.Cleanup(srv.Close)
	return srv, func() []string {
		mu.Lock()
		defer mu.Unlock()
		return slices.Clone(got)
	}
}

func signedInItem(t *testing.T) string {
	t.Helper()
	raw, _ := json.Marshal(map[string]string{"r": testRefresh, "a": "acct-9", "e": "me@example.com"})
	return base64.RawURLEncoding.EncodeToString(raw)
}

func chatgptRun(t *testing.T, keys *memKeys, srv *httptest.Server) *sessionApp {
	useKeys(t, keys)
	ev := script(t, hotkeyLine(0, "translate/0", proto.PhaseTrigger))
	return newSessionApp(t, helpertest.Opt("selection", "Good morning, team.")+","+ev, func(s *settings.Settings) {
		s.Features.Translate.Target = "zh-Hans"
		s.Features.Translate.LLM = providers.ChatGPT
		s.Providers.LLM = []settings.Provider{{ID: providers.ChatGPT, Endpoint: srv.URL + "/backend-api/codex"}}
	}, local.NotInstalled{})
}

// TestChatGPTSessionEndToEnd: a session refreshes the token, streams from
// the Responses endpoint with it, and no token lands in the history or the
// settings.
func TestChatGPTSessionEndToEnd(t *testing.T) {
	useFakeOAuth(t)
	srv, requests := mockCodex(t, "早上好，团队。", 0, "")
	keys := &memKeys{m: map[string]string{"llm.codex": signedInItem(t)}}
	a := chatgptRun(t, keys, srv)
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if end := a.sess; end.State != session.Done || end.Result != "早上好，团队。" {
		t.Fatalf("end %+v", end)
	}
	got := requests()
	if len(got) != 1 || !strings.HasPrefix(got[0], "/backend-api/codex/responses Bearer "+testAccess+" acct-9 codex_cli_rs ") ||
		!strings.Contains(got[0], `"store":false`) || !strings.Contains(got[0], `"stream":true`) || strings.Contains(got[0], "max_output_tokens") {
		t.Fatalf("requests %q", got)
	}
	raw, _ := json.Marshal(a.history())
	cfg, _ := json.Marshal(a.lib.settings.Get())
	for _, s := range []string{testAccess, testRefresh} {
		if strings.Contains(string(raw), s) || strings.Contains(string(cfg), s) {
			t.Errorf("%q kept", s)
		}
	}
	if codexauth.Email(keys.m["llm.codex"]) != "me@example.com" {
		t.Errorf("item %q", keys.m["llm.codex"])
	}
}

// TestChatGPTRevokedSignIn: a sign-in the service refuses fails the
// session asking to sign in again, removes it, and does not try again.
func TestChatGPTRevokedSignIn(t *testing.T) {
	o := useFakeOAuth(t)
	o.fail = true
	srv, requests := mockCodex(t, "x", 0, "")
	keys := &memKeys{m: map[string]string{"llm.codex": signedInItem(t)}}
	a := chatgptRun(t, keys, srv)
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if a.sess.State != session.Failed || llm.Code(a.sess.Problem.Err) != llm.CodeSignIn {
		t.Fatalf("end %+v", a.sess)
	}
	msg, _, action := sessionProblemText(a.sess.Problem)
	if action != actionSignIn || !strings.Contains(msg, "sign in again") || noticeActionLabel(action) != "Sign in" {
		t.Errorf("%q %q", msg, action)
	}
	if len(requests()) != 0 {
		t.Errorf("sent %q", requests())
	}
	if _, set := keys.m["llm.codex"]; set {
		t.Error("the sign-in is still saved")
	}
	a.pump("the card", func() bool {
		return a.setupMissing(local.LLM, settings.Provider{ID: providers.ChatGPT}) == providers.MissingKey
	})
	if txt := setupNeedsText(local.LLM, settings.Provider{ID: providers.ChatGPT}, providers.MissingKey); !strings.Contains(txt, "sign in again") {
		t.Errorf("%q", txt)
	}
}

// TestChatGPTUsageNotice: a used-up plan's notice: as recognised text, a
// way to the usage page, not a fallback model.
func TestChatGPTUsageNotice(t *testing.T) {
	spent := &llm.Error{Code: llm.CodeUsageLimit}
	msg, _, action := sessionProblemText(&session.Problem{Code: session.ProblemLLM, Err: spent, Degraded: true})
	if !strings.Contains(msg, "Inserted as recognised") || !strings.Contains(msg, "used up") || action != actionUsage || noticeActionLabel(action) != "Manage usage" {
		t.Errorf("degraded %q %q", msg, action)
	}
	if txt := degradedLLMText(&session.Problem{Err: spent}, true); !strings.Contains(txt, "Notes") {
		t.Errorf("note %q", txt)
	}
	if _, _, a := sessionProblemText(&session.Problem{Code: session.ProblemLLM, Err: spent}); a != actionUsage {
		t.Errorf("failed: %q", a)
	}
}

func TestChatGPTPrivacyLine(t *testing.T) {
	useKeys(t, &memKeys{m: map[string]string{}})
	a, _, tt := newTestApp(t)
	a.navigate("/settings/privacy")
	frames(tt)
	if !hasPrefix(tt, "• With ChatGPT (Codex), that text goes to OpenAI") {
		t.Errorf("no line: %q", tt.Texts())
	}
}

// TestChatGPTRetiredModel: a saved model that left the built-in list stays
// in the dialog as a typed-in model, unchanged.
func TestChatGPTRetiredModel(t *testing.T) {
	useKeys(t, &memKeys{m: map[string]string{}})
	a, _, tt := newTestApp(t)
	a.updateSettings(func(s *settings.Settings) {
		s.Providers.LLM = []settings.Provider{{ID: providers.ChatGPT, Model: "gpt-5.5"}}
	})
	a.navigate(remotePath)
	frames(tt)
	a.openProviderDialog(local.LLM, providers.ChatGPT)
	frames(tt)
	d := &a.provDlg
	if d.model != otherModel() || d.customModel != "gpt-5.5" || !has(tt, "Model ID") {
		t.Fatalf("model %q custom %q: %q", d.model, d.customModel, tt.Texts())
	}
	if got := d.setup().Model; got != "gpt-5.5" || d.unsaved() {
		t.Errorf("setup model %q, unsaved %v", got, d.unsaved())
	}
	if msg, _ := d.check(d.setup()); msg != "" {
		t.Errorf("check %q", msg)
	}
}
