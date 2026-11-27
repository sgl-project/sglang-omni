package voxtcore

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// fakeChat is an OpenAI-compatible chat completions server on loopback:
// it answers the requests in order with answers (the last one repeats),
// and keeps what each request sent.
type fakeChat struct {
	t       *testing.T
	answers []fakeAnswer
	// block, when set, holds every request until the request's context
	// ends; arrived gets a value when one comes in.
	block   bool
	arrived chan struct{}

	mu       sync.Mutex
	requests []chatBody
}

type fakeAnswer struct {
	status int
	text   string
}

type chatBody struct {
	Model    string `json:"model"`
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	ResponseFormat map[string]any `json:"response_format"`
}

func (b chatBody) system() string { return b.Messages[0].Content }
func (b chatBody) user() string   { return b.Messages[len(b.Messages)-1].Content }

func (f *fakeChat) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
		http.NotFound(w, r)
		return
	}
	data, _ := io.ReadAll(r.Body) // a short body of a test request
	var body chatBody
	if err := json.Unmarshal(data, &body); err != nil {
		f.t.Errorf("request body: %v", err)
	}
	f.mu.Lock()
	n := len(f.requests)
	f.requests = append(f.requests, body)
	f.mu.Unlock()
	if f.block {
		f.arrived <- struct{}{}
		<-r.Context().Done()
		return
	}
	a := f.answers[min(n, len(f.answers)-1)]
	w.Header().Set("Content-Type", "application/json")
	if a.status != 0 && a.status != http.StatusOK {
		w.WriteHeader(a.status)
		_, _ = io.WriteString(w, `{"error":{"message":"refused"}}`) // the client reads the status
		return
	}
	_ = json.NewEncoder(w).Encode(map[string]any{ // a response writer in a test
		"model":   body.Model,
		"choices": []any{map[string]any{"index": 0, "message": map[string]any{"role": "assistant", "content": a.text}, "finish_reason": "stop"}},
		"usage":   map[string]any{"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
	})
}

func (f *fakeChat) calls() []chatBody {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]chatBody(nil), f.requests...)
}

// serveChat starts f and returns a provider that reaches it.
func serveChat(t *testing.T, answers ...string) (*fakeChat, Provider) {
	t.Helper()
	f := &fakeChat{t: t, arrived: make(chan struct{}, 1)}
	for _, a := range answers {
		f.answers = append(f.answers, fakeAnswer{text: a})
	}
	return f, startChat(t, f)
}

func startChat(t *testing.T, f *fakeChat) Provider {
	t.Helper()
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	return Provider{Preset: "custom", BaseURL: srv.URL + "/v1", Model: "test-model"}
}
