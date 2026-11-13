package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// fakeServer is a scripted OpenAI-compatible server: each request gets
// the next handler (the last one repeats). It records the requests.
type fakeServer struct {
	srv   *httptest.Server
	mu    sync.Mutex
	steps []http.HandlerFunc
	reqs  []recorded
}

type recorded struct {
	Path   string
	Header http.Header
	Body   map[string]any
}

func newFakeServer(t *testing.T, steps ...http.HandlerFunc) *fakeServer {
	f := &fakeServer{steps: steps}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		f.mu.Lock()
		h := f.steps[min(len(f.reqs), len(f.steps)-1)]
		f.reqs = append(f.reqs, recorded{r.URL.Path, r.Header.Clone(), body})
		f.mu.Unlock()
		h(w, r)
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeServer) requests() []recorded {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]recorded(nil), f.reqs...)
}

func jsonReply(status int, v any) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(v)
	}
}

func oaText(text string) map[string]any {
	return map[string]any{
		"model":   "m-reported",
		"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": text}, "finish_reason": "stop"}},
		"usage":   map[string]any{"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
	}
}

func sse(events ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range events {
			fmt.Fprint(w, e)
			w.(http.Flusher).Flush()
		}
	}
}

// presetAt is preset id served by base (a loopback test server).
func presetAt(t *testing.T, id, base string) Preset {
	t.Helper()
	p, err := ProviderConfig{ID: id, BaseURL: base + "/v1"}.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func TestOpenAIChat(t *testing.T) {
	f := newFakeServer(t, jsonReply(200, oaText("hola")))
	c := &Client{Backend: NewOpenAI(presetAt(t, "openai", f.srv.URL), "sk-test", f.srv.Client())}
	resp, err := c.Chat(context.Background(), Request{
		Model:       "gpt-x",
		Messages:    []Message{{Role: System, Content: "sys"}, {Role: User, Content: "hello"}},
		Temperature: ptr(0.2), MaxTokens: 50, Stop: []string{"\n\n"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "hola" || resp.Model != "m-reported" || resp.Provider != "openai" || resp.FinishReason != FinishStop || resp.Usage.TotalTokens != 18 {
		t.Fatalf("resp %+v", resp)
	}
	r := f.requests()[0]
	if r.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer sk-test" || r.Header.Get("User-Agent") != UserAgent {
		t.Fatalf("req %+v", r)
	}
	if r.Body["model"] != "gpt-x" || r.Body["temperature"] != 0.2 || r.Body["max_completion_tokens"] != 50.0 || r.Body["stream"] != nil {
		t.Fatalf("body %v", r.Body)
	}
	if msgs := r.Body["messages"].([]any); len(msgs) != 2 || msgs[0].(map[string]any)["role"] != "system" {
		t.Fatalf("messages %v", msgs)
	}
}

// TestOpenAIRelaxesGeneration: a rejected temperature or top_p that came
// from the setup's settings (or its Extra Body) is dropped too; a rejection
// naming something the body does not carry is not retried.
func TestOpenAIRelaxesGeneration(t *testing.T) {
	reject := jsonReply(400, map[string]any{"error": map[string]any{"message": "Unsupported value: 'temperature' does not support 0.2"}})
	for name, g := range map[string]Generation{
		"temperature": {Temperature: ptr(0.2)},
		"top_p":       {TopP: ptr(0.9)},
		"extra body":  {ExtraBody: map[string]any{"temperature": 0.2}},
	} {
		f := newFakeServer(t, reject, jsonReply(200, oaText("ok")))
		p := presetAt(t, "openai", f.srv.URL)
		p.Generation = g
		c := NewOpenAI(p, "k", f.srv.Client())
		if _, err := c.Chat(context.Background(), Request{Messages: msgs("hi")}); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		rs := f.requests()
		if len(rs) != 2 || rs[1].Body["temperature"] != nil || rs[1].Body["top_p"] != nil {
			t.Fatalf("%s: requests %+v", name, rs)
		}
	}
	// Nothing to drop: not resent.
	f := newFakeServer(t, reject)
	c := NewOpenAI(presetAt(t, "openai", f.srv.URL), "k", f.srv.Client())
	if _, err := c.Chat(context.Background(), Request{Messages: msgs("hi")}); err == nil || len(f.requests()) != 1 {
		t.Fatalf("err %v, %d requests", err, len(f.requests()))
	}
}

// TestOpenAIStreaming: deltas in order on the caller's goroutine, keep-
// alive comments skipped, the usage of the last chunk; the progress
// deadlines stream too.
func TestOpenAIStreaming(t *testing.T) {
	f := newFakeServer(t, sse(
		": keep-alive\n\n",
		`data: {"model":"m1","choices":[{"delta":{"role":"assistant"}}]}`+"\n\n",
		`data: {"choices":[{"delta":{"content":"Hel"}}]}`+"\n\n",
		`data: {"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}`+"\n\n",
		`data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`+"\n\n",
		"data: [DONE]\n\n",
	))
	var deltas []string
	c := &Client{Backend: NewOpenAI(presetAt(t, "openai", f.srv.URL), "sk-test", f.srv.Client())}
	resp, err := c.Chat(context.Background(), Request{Messages: []Message{{Role: User, Content: "hi"}}, OnDelta: func(s string) { deltas = append(deltas, s) }})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "Hello" || strings.Join(deltas, "|") != "Hel|lo" || resp.Usage.TotalTokens != 5 || resp.Model != "m1" {
		t.Fatalf("resp %+v deltas %q", resp, deltas)
	}
	if b := f.requests()[0].Body; b["stream"] != true || b["stream_options"] == nil || b["model"] != "gpt-4.1-mini" {
		t.Fatalf("body %v", b)
	}
	if _, err := c.Chat(context.Background(), Request{Messages: []Message{{Role: User, Content: "hi"}}, FirstTokenTimeout: 1 << 30}); err != nil {
		t.Fatal(err)
	}
	if b := f.requests()[1].Body; b["stream"] != true {
		t.Fatalf("a request with a first-token deadline did not stream: %v", b)
	}
}

// TestOpenAIRelaxed: parameters a server rejects are dropped, one at a
// time: stream_options, temperature, then the preset's thinking switch.
func TestOpenAIRelaxed(t *testing.T) {
	f := newFakeServer(t,
		jsonReply(400, map[string]any{"error": map[string]any{"message": "Unrecognized request argument: stream_options"}}),
		jsonReply(400, map[string]any{"error": map[string]any{"message": "Unsupported value: 'temperature' does not support 0.3"}}),
		jsonReply(400, map[string]any{"error": map[string]any{"message": "unknown parameter: thinking"}}),
		sse(`data: {"choices":[{"delta":{"content":"好的"},"finish_reason":"stop"}]}`+"\n\n", "data: [DONE]\n\n"),
	)
	c := &Client{Backend: NewOpenAI(presetAt(t, "deepseek", f.srv.URL), "sk-ds", f.srv.Client())}
	resp, err := c.Chat(context.Background(), Request{Messages: []Message{{Role: User, Content: "hi"}}, Temperature: ptr(0.3), OnDelta: func(string) {}})
	if err != nil || resp.Text != "好的" || !resp.Usage.Estimated {
		t.Fatalf("%v %+v", err, resp)
	}
	rs := f.requests()
	th, _ := rs[0].Body["thinking"].(map[string]any)
	if len(rs) != 4 || th["type"] != "disabled" || rs[0].Body["model"] != "deepseek-flash" {
		t.Fatalf("first request %+v", rs[0].Body)
	}
	if rs[1].Body["stream_options"] != nil || rs[2].Body["temperature"] != nil || rs[3].Body["thinking"] != nil {
		t.Fatalf("requests %+v", rs)
	}
	// Request fields win over ExtraBody's.
	p := presetAt(t, "deepseek", f.srv.URL)
	p.ExtraBody = map[string]any{"model": "evil"}
	if b := NewOpenAI(p, "k", nil).body(Request{}, "deepseek-flash", false, false, true); b["model"] != "deepseek-flash" {
		t.Errorf("ExtraBody overrode the model: %v", b["model"])
	}
}

// TestOpenAIStreamTruncated: a stream cut before [DONE] or a finish
// reason fails; text was streamed, so it is not retried.
func TestOpenAIStreamTruncated(t *testing.T) {
	f := newFakeServer(t, sse(`data: {"choices":[{"delta":{"content":"par"}}]}`+"\n\n"))
	c := &Client{Backend: NewOpenAI(presetAt(t, "openai", f.srv.URL), "sk", f.srv.Client()), Sleep: noSleep, Retry: RetryPolicy{MaxAttempts: 2}}
	_, err := c.Chat(context.Background(), Request{Messages: msgs("hi"), OnDelta: func(string) {}})
	if Code(err) != CodeNetwork || len(f.requests()) != 1 {
		t.Fatalf("err %v, %d requests", err, len(f.requests()))
	}
}

// TestOpenAIKeys: a service that needs a key sends nothing without one;
// a keyless server gets no Authorization header, and a key when one is
// saved.
func TestOpenAIKeys(t *testing.T) {
	f := newFakeServer(t, jsonReply(200, oaText("x")))
	if _, err := NewOpenAI(presetAt(t, "openai", f.srv.URL), " ", f.srv.Client()).Chat(context.Background(), Request{Messages: msgs("x")}); Code(err) != CodeNoKey {
		t.Fatalf("no key: %v", err)
	}
	p, err := ProviderConfig{ID: Custom, BaseURL: f.srv.URL + "/v1/chat/completions/", Model: "qwen"}.Resolve()
	if err != nil || p.BaseURL != f.srv.URL+"/v1" {
		t.Fatalf("custom: %v %q", err, p.BaseURL)
	}
	for _, key := range []string{"", "lm-key"} {
		if _, err := NewOpenAI(p, key, f.srv.Client()).Chat(context.Background(), Request{Messages: msgs("x")}); err != nil {
			t.Fatal(err)
		}
	}
	rs := f.requests()
	if len(rs) != 2 || rs[0].Header.Get("Authorization") != "" || rs[1].Header.Get("Authorization") != "Bearer lm-key" || rs[0].Body["model"] != "qwen" {
		t.Fatalf("requests %+v", rs)
	}
}

// TestOpenAIErrorsHideKey: a server that quotes the key in an error (a
// rejection, a body that is not JSON, a stream event that is not JSON) gets
// it blanked out of the error, also when the quote is long enough for the
// detail's clip (300 or 200 bytes) to cut the key in two.
func TestOpenAIErrorsHideKey(t *testing.T) {
	const key = "sk-secret-4242-0123456789abcdefghijklmnopqrstuvwxyz"
	for _, long := range []bool{false, true} {
		pad := func(n int) string { // puts the key 10 bytes before the clip
			if !long {
				return ""
			}
			return strings.Repeat("x", n)
		}
		for name, h := range map[string]http.HandlerFunc{
			"400":    jsonReply(400, map[string]any{"error": map[string]any{"message": pad(256) + "bad header: Authorization: Bearer " + key}}),
			"body":   func(w http.ResponseWriter, r *http.Request) { fmt.Fprint(w, "<html>"+pad(179)+"echo "+key+"</html>") },
			"stream": sse("data: " + pad(181) + "rejected " + key + "\n\n"),
		} {
			f := newFakeServer(t, h)
			req := Request{Messages: msgs("x")}
			if name == "stream" {
				req.OnDelta = func(string) {}
			}
			_, err := NewOpenAI(presetAt(t, "openai", f.srv.URL), key, f.srv.Client()).Chat(context.Background(), req)
			if err == nil || strings.Contains(err.Error(), key[:12]) || !strings.Contains(err.Error(), "[key]") {
				t.Errorf("%s (long %v): %v", name, long, err)
			}
		}
	}
}

// TestResolveAddress: keys go over https, plain http only to this Mac,
// never with credentials in the URL; a custom server needs an address.
func TestResolveAddress(t *testing.T) {
	for base, ok := range map[string]bool{
		"":                                           true, // the preset's
		"https://example.com/v1":                     true,
		"http://localhost:11434/v1":                  true,
		"http://127.0.0.1:8080/v1":                   true,
		"http://[::1]:8080/v1":                       true,
		"http://example.com/v1":                      false,
		"http://192.168.1.5:11434/v1":                false,
		"https://user:pass@example.com/v1":           false,
		"ftp://example.com/v1":                       false,
		"example.com/v1":                             false,
		"https://api.openai.com/v1/":                 true,
		"https://api.openai.com/v1/chat/completions": true,
	} {
		_, err := ProviderConfig{ID: "openai", BaseURL: base}.Resolve()
		if (err == nil) != ok {
			t.Errorf("%q: %v", base, err)
		}
	}
	if _, err := (ProviderConfig{ID: Custom}).Resolve(); Code(err) != CodeConfig {
		t.Errorf("custom without an address: %v", err)
	}
	if _, err := (ProviderConfig{ID: "nope"}).Resolve(); Code(err) != CodeUnknown {
		t.Errorf("unknown preset: %v", err)
	}
	for _, p := range Presets() {
		if p.ID != Custom && p.BaseURL == "" || p.Name == "" {
			t.Errorf("preset %+v", p)
		}
		if _, err := (ProviderConfig{ID: p.ID, BaseURL: "https://example.com/v1"}).Resolve(); err != nil {
			t.Errorf("%s: %v", p.ID, err)
		}
	}
}

func TestOpenAIJSONModes(t *testing.T) {
	schema := Object(map[string]*Schema{"a": String()}, "a")
	schema.Name = "thing"
	for _, mode := range []JSONMode{JSONSchemaMode, JSONObjectMode, JSONPromptMode} {
		f := newFakeServer(t, jsonReply(200, oaText(`{"a":"x"}`)))
		p := presetAt(t, "openai", f.srv.URL)
		p.JSONMode = mode
		c := &Client{Backend: NewOpenAI(p, "sk", f.srv.Client())}
		var out struct{ A string }
		if _, err := c.JSON(context.Background(), Request{Messages: msgs("x"), Schema: schema}, &out, 0); err != nil || out.A != "x" {
			t.Fatalf("%s: %v %+v", mode, err, out)
		}
		b := f.requests()[0].Body
		rf, _ := b["response_format"].(map[string]any)
		sys := b["messages"].([]any)[0].(map[string]any)
		switch mode {
		case JSONSchemaMode:
			if rf["type"] != "json_schema" || rf["json_schema"].(map[string]any)["name"] != "thing" || strings.Contains(sys["content"].(string), "JSON Schema") {
				t.Fatalf("schema mode body %v", b)
			}
		case JSONObjectMode:
			if rf["type"] != "json_object" || !strings.Contains(sys["content"].(string), `"required":["a"]`) {
				t.Fatalf("object mode body %v", b)
			}
		case JSONPromptMode:
			if rf != nil || sys["role"] != "system" || !strings.Contains(sys["content"].(string), "JSON Schema") {
				t.Fatalf("prompt mode body %v", b)
			}
		}
	}
}

// TestProbe: a reply cut short (even empty) proves the key and the
// model; a refusal carries its code and the server's detail.
func TestProbe(t *testing.T) {
	f := newFakeServer(t, jsonReply(200, map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": ""}, "finish_reason": "length"}}}))
	if d, err := Probe(context.Background(), NewOpenAI(presetAt(t, "deepseek", f.srv.URL), "sk", f.srv.Client())); err != nil || d < 0 {
		t.Fatalf("length: %v %v", d, err)
	}
	if r := f.requests()[0]; r.Body["max_tokens"] != float64(8) || r.Body["stream"] != nil {
		t.Errorf("probe body %+v", r.Body)
	}
	for _, tc := range []struct {
		status int
		msg    string
		code   string
	}{
		{401, "Authentication Fails, Your api key is invalid", CodeAuth},
		{400, "Model Not Exist", CodeModelNotFound},
		{422, "Invalid max_tokens value", CodeBadRequest},
	} {
		f := newFakeServer(t, jsonReply(tc.status, map[string]any{"error": map[string]any{"message": tc.msg}}))
		_, err := Probe(context.Background(), NewOpenAI(presetAt(t, "deepseek", f.srv.URL), "sk", f.srv.Client()))
		if Code(err) != tc.code {
			t.Errorf("%d %q: %v", tc.status, tc.msg, err)
		}
		if e, _ := err.(*Error); tc.code == CodeBadRequest && (e == nil || e.Params["detail"] == "") {
			t.Errorf("no detail: %v", err)
		}
	}
}
