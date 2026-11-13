package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

const fakeToken = "at-FAKE-access-token-0001"

// signedInBackend is a preset like the ChatGPT one, at srv.
func signedInBackend(t *testing.T, srv *httptest.Server, src TokenSource) *OpenAI {
	t.Helper()
	p, err := ProviderConfig{ID: "codex", BaseURL: srv.URL + "/backend-api/codex"}.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	o := NewOpenAI(p, "", srv.Client())
	if src != nil {
		o.WithTokens(src)
	}
	return o
}

func fixedToken(token string) TokenSource {
	return func(context.Context, string) (Credential, error) {
		return Credential{Token: token, AccountID: "acct-1"}, nil
	}
}

func completedStream(w http.ResponseWriter, deltas ...string) {
	w.Header().Set("Content-Type", "text/event-stream")
	fmt.Fprint(w, "event: response.created\ndata: {\"type\":\"response.created\",\"response\":{\"status\":\"in_progress\"}}\n\n")
	for _, d := range deltas {
		fmt.Fprintf(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":%q}\n\n", d)
	}
	fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":3,\"output_tokens\":2,\"total_tokens\":5}}}\n\n")
}

// A signed-in request is the bare, streamed shape Voxt 1.x sent, however
// much Voxt asked for, with the session's headers.
func TestSignedInRequest(t *testing.T) {
	var got map[string]any
	var hdr http.Header
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path, hdr = r.URL.Path, r.Header.Clone()
		_ = json.NewDecoder(r.Body).Decode(&got)
		completedStream(w, "hel", "lo")
	}))
	defer srv.Close()
	o := signedInBackend(t, srv, fixedToken(fakeToken))
	temp, top := 0.3, 0.8
	maxTok := 4096
	o.preset.Generation = Generation{MaxOutputTokens: &maxTok, Temperature: &temp, TopP: &top, Verbosity: "low", Thinking: Thinking{Mode: ThinkEffort, Effort: "high"}, ExtraBody: map[string]any{"service_tier": "priority", "temperature": 0.1}}
	res, err := o.Chat(context.Background(), Request{
		Messages:    []Message{{Role: System, Content: "be brief"}, {Role: User, Content: "ping"}},
		MaxTokens:   500,
		Temperature: &temp, TopP: &top, Stop: []string{"x"},
		Format: FormatJSON,
	})
	if err != nil || res.Text != "hello" {
		t.Fatalf("%q %v", res.Text, err)
	}
	if path != "/backend-api/codex/responses" {
		t.Errorf("path %s", path)
	}
	want := map[string]bool{"model": true, "input": true, "instructions": true, "store": true, "stream": true}
	for k := range got {
		if !want[k] {
			t.Errorf("sent %s = %v", k, got[k])
		}
	}
	if got["store"] != false || got["stream"] != true || got["model"] != "gpt-6.1-sol" {
		t.Errorf("body %v", got)
	}
	if ins, _ := got["instructions"].(string); !strings.Contains(ins, "be brief") || !strings.Contains(ins, "single JSON value") {
		t.Errorf("instructions %q", got["instructions"])
	}
	input, _ := got["input"].([]any)
	if len(input) != 1 || input[0].(map[string]any)["content"] != "ping" {
		t.Errorf("input %v", got["input"])
	}
	for k, v := range map[string]string{"Authorization": "Bearer " + fakeToken, "ChatGPT-Account-ID": "acct-1", "originator": "codex_cli_rs", "User-Agent": "codex_cli_rs/0.0.0 (Voxt)", "Accept": "text/event-stream, application/json"} {
		if hdr.Get(k) != v {
			t.Errorf("header %s = %q, want %q", k, hdr.Get(k), v)
		}
	}
}

func TestSignedInDefaultInstructions(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		completedStream(w, "ok")
	}))
	defer srv.Close()
	o := signedInBackend(t, srv, fixedToken(fakeToken))
	if _, err := o.Chat(context.Background(), Request{Messages: []Message{{Role: User, Content: "ping"}}}); err != nil {
		t.Fatal(err)
	}
	if got["instructions"] != bareInstructions {
		t.Errorf("instructions %v", got["instructions"])
	}
}

// Probe does not stream for other services; it must for this one, with
// no limit on the output.
func TestSignedInProbeStreams(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		completedStream(w, "ok")
	}))
	defer srv.Close()
	if _, err := Probe(context.Background(), signedInBackend(t, srv, fixedToken(fakeToken))); err != nil {
		t.Fatal(err)
	}
	if got["stream"] != true {
		t.Errorf("stream %v", got["stream"])
	}
	if _, set := got["max_output_tokens"]; set {
		t.Errorf("max_output_tokens %v", got["max_output_tokens"])
	}
}

// The Codex backend may send the whole text in response.output_text.done,
// and an empty output in response.completed.
func TestSignedInDoneEventCarriesText(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"response.output_text.done\",\"text\":\"whole\"}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":{\"output\":[]}}\n\n")
	}))
	defer srv.Close()
	var deltas []string
	res, err := signedInBackend(t, srv, fixedToken(fakeToken)).Chat(context.Background(), Request{
		Messages: []Message{{Role: User, Content: "hi"}}, OnDelta: func(d string) { deltas = append(deltas, d) },
	})
	if err != nil || res.Text != "whole" || len(deltas) != 1 || deltas[0] != "whole" {
		t.Fatalf("%q %v deltas %v", res.Text, err, deltas)
	}
}

func TestDoneEventAfterDeltasIsNotRepeated(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ab\"}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"response.output_text.done\",\"text\":\"ab\"}\n\n")
		fmt.Fprint(w, "data: {\"type\":\"response.completed\",\"response\":{\"output\":[]}}\n\n")
	}))
	defer srv.Close()
	res, err := signedInBackend(t, srv, fixedToken(fakeToken)).Chat(context.Background(), Request{Messages: []Message{{Role: User, Content: "hi"}}})
	if err != nil || res.Text != "ab" {
		t.Fatalf("%q %v", res.Text, err)
	}
}

// A 401 asks for a fresh token once, naming the refused one, and tries
// again; a second 401 is the end of the sign-in, not a loop.
func TestSignedIn401Refreshes(t *testing.T) {
	var sends atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if sends.Add(1) == 1 || r.Header.Get("Authorization") != "Bearer fresh-token-2" {
			w.WriteHeader(401)
			fmt.Fprint(w, `{"error":{"message":"expired"}}`)
			return
		}
		completedStream(w, "ok")
	}))
	defer srv.Close()
	var stales []string
	src := func(_ context.Context, stale string) (Credential, error) {
		stales = append(stales, stale)
		if stale != "" {
			return Credential{Token: "fresh-token-2"}, nil
		}
		return Credential{Token: fakeToken}, nil
	}
	res, err := signedInBackend(t, srv, src).Chat(context.Background(), Request{Messages: []Message{{Role: User, Content: "hi"}}})
	if err != nil || res.Text != "ok" {
		t.Fatalf("%q %v", res.Text, err)
	}
	if len(stales) != 2 || stales[0] != "" || stales[1] != fakeToken {
		t.Errorf("stale tokens %q", stales)
	}

	sends.Store(0)
	srv2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sends.Add(1)
		w.WriteHeader(401)
	}))
	defer srv2.Close()
	_, err = signedInBackend(t, srv2, fixedToken(fakeToken)).Chat(context.Background(), Request{Messages: []Message{{Role: User, Content: "hi"}}})
	if Code(err) != CodeSignIn {
		t.Fatalf("err %v", err)
	}
	if n := sends.Load(); n != 2 {
		t.Errorf("%d sends", n)
	}
}

func TestSignedInPlanErrors(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		body   string
		code   string
		retry  bool
	}{
		{"usage used up", 429, `{"error":{"type":"usage_limit_reached","message":"The usage limit has been reached","plan_type":"plus"}}`, CodeUsageLimit, false},
		{"not in plan", 403, `{"error":{"type":"usage_not_included","message":"Upgrade"}}`, CodeNotEligible, false},
		{"region", 403, `{"error":{"code":"unsupported_country_region_territory"}}`, CodeNotEligible, false},
		{"a plain 429", 429, `{"error":{"type":"rate_limit_exceeded","message":"slow down"}}`, CodeRateLimited, true},
		{"server trouble", 503, `{"error":{"message":"busy"}}`, CodeServer, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
				fmt.Fprint(w, tc.body)
			}))
			defer srv.Close()
			_, err := signedInBackend(t, srv, fixedToken(fakeToken)).Chat(context.Background(), Request{Messages: []Message{{Role: User, Content: "hi"}}})
			var e *Error
			if !errors.As(err, &e) || e.Code != tc.code || e.Retryable != tc.retry {
				t.Fatalf("err %v (retryable %v)", err, e.Retryable)
			}
		})
	}
}

// A failure reported inside the stream is the same failure, with the
// access token blanked out.
func TestSignedInStreamFailures(t *testing.T) {
	for _, tc := range []struct {
		name, event, code string
	}{
		{"usage in response.failed", `{"type":"response.failed","response":{"error":{"code":"usage_limit_reached","message":"limit for ` + fakeToken + `"}}}`, CodeUsageLimit},
		{"not eligible in error event", `{"type":"error","code":"usage_not_included","message":"upgrade"}`, CodeNotEligible},
		{"server error", `{"type":"response.failed","response":{"error":{"code":"server_error","message":"x"}}}`, CodeServer},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprintf(w, "data: %s\n\n", tc.event)
			}))
			defer srv.Close()
			_, err := signedInBackend(t, srv, fixedToken(fakeToken)).Chat(context.Background(), Request{Messages: []Message{{Role: User, Content: "hi"}}})
			if Code(err) != tc.code {
				t.Fatalf("err %v", err)
			}
			if strings.Contains(err.Error(), fakeToken) {
				t.Errorf("the token is in the error: %v", err)
			}
		})
	}
}

func TestSignedInTokenNeverInErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(400)
		fmt.Fprintf(w, `{"error":{"message":"bad header %s"}}`, r.Header.Get("Authorization"))
	}))
	defer srv.Close()
	_, err := signedInBackend(t, srv, fixedToken(fakeToken)).Chat(context.Background(), Request{Messages: []Message{{Role: User, Content: "hi"}}})
	if err == nil || strings.Contains(err.Error(), fakeToken) || !strings.Contains(err.Error(), "[key]") {
		t.Fatalf("err %v", err)
	}
}

func TestSignedInSourceErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { t.Error("sent") }))
	defer srv.Close()
	o := signedInBackend(t, srv, func(context.Context, string) (Credential, error) {
		return Credential{}, SignInError("codex")
	})
	if _, err := o.Chat(context.Background(), Request{Messages: []Message{{Role: User, Content: "hi"}}}); Code(err) != CodeSignIn {
		t.Fatalf("sign-in: %v", err)
	}
	o = signedInBackend(t, srv, func(context.Context, string) (Credential, error) {
		return Credential{}, errors.New("no network")
	})
	if _, err := o.Chat(context.Background(), Request{Messages: []Message{{Role: User, Content: "hi"}}}); Code(err) != CodeNetwork {
		t.Fatalf("network: %v", err)
	}
	if _, err := signedInBackend(t, srv, nil).Chat(context.Background(), Request{Messages: []Message{{Role: User, Content: "hi"}}}); Code(err) != CodeSignIn {
		t.Fatalf("no source: %v", err)
	}
}

// Concurrent requests share the backend's list of tokens safely.
func TestSignedInConcurrent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { completedStream(w, "ok") }))
	defer srv.Close()
	var n atomic.Int32
	o := signedInBackend(t, srv, func(context.Context, string) (Credential, error) {
		return Credential{Token: fmt.Sprintf("token-%d", n.Add(1))}, nil
	})
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := o.Chat(context.Background(), Request{Messages: []Message{{Role: User, Content: "hi"}}}); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
}

// A model the server refuses keeps the server's words, which say why, with
// the token blanked out before they are clipped.
func TestSignedInModelNotFoundDetail(t *testing.T) {
	long := strings.Repeat("x", 290) + " " + fakeToken + strings.Repeat("y", 400)
	for _, tc := range []struct {
		name, body, want string
		status           int
	}{
		{"words", `{"error":{"message":"The model is not supported when using Codex with a ChatGPT account."}}`, "not supported when using Codex", 404},
		{"token in the words", `{"error":{"message":"no route for ` + fakeToken + `"}}`, "no route for [key]", 404},
		{"token at the clip", `{"error":{"message":"` + long + `"}}`, "[key]", 404},
		{"400 that names the model", `{"error":{"message":"model not found for ` + fakeToken + `"}}`, "model not found for [key]", 400},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
				fmt.Fprint(w, tc.body)
			}))
			defer srv.Close()
			_, err := signedInBackend(t, srv, fixedToken(fakeToken)).Chat(context.Background(), Request{Messages: []Message{{Role: User, Content: "hi"}}})
			var e *Error
			if !errors.As(err, &e) || e.Code != CodeModelNotFound || e.Status != tc.status || e.Params["status"] != fmt.Sprint(tc.status) {
				t.Fatalf("err %#v", err)
			}
			if d := e.Params["detail"]; !strings.Contains(d, tc.want) || len(d) > 300+len("…") {
				t.Errorf("detail %q", d)
			}
			if !strings.Contains(err.Error(), tc.want) || strings.Contains(err.Error(), fakeToken) {
				t.Errorf("err %v", err)
			}
		})
	}
}

func TestHTTPErrorModelNotFoundClipsDetail(t *testing.T) {
	e := HTTPError("p", "m", 404, strings.Repeat("x", 900), 0)
	if e.Code != CodeModelNotFound || len(e.Params["detail"]) != 300+len("…") {
		t.Errorf("%s detail of %d bytes", e.Code, len(e.Params["detail"]))
	}
}
