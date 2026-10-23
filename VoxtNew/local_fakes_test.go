package main

// In-process local models for the app's tests (local.Services): a speech
// model that transcribes recordings (fakeTranscriptions) or streams
// (fakeStreamer), and a language model (fakeChat) behind llm.Client, as
// internal/llmrt's backend is. They stand where the runtimes' processes
// run in the app, and keep what the tests read: the chat requests in the
// shape of an OpenAI request ({"messages": […], "stream": …}).

import (
	"context"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
)

// The models the fakes say they are.
var (
	fakeASRModel = local.Model{Kind: local.ASR, ID: models.ASRAccurate, Name: "Accurate speech recognition"}
	fakeLLMModel = local.Model{Kind: local.LLM, ID: models.LLMBalanced, Name: "Balanced language model"}
)

// fakeLocal is local.Services over fakes: a nil ASR or Chat is a model
// that is not installed.
type fakeLocal struct {
	ASRP asr.Provider
	Chat local.Chat
	// MoreLLM are more language models installed (their opens get Chat).
	MoreLLM []local.Model
}

// readyLocal is a speech model and a language model, both installed,
// which the tests that only read the pages never call.
func readyLocal() *fakeLocal {
	return &fakeLocal{ASRP: &fakeTranscriptions{}, Chat: &llm.Client{Backend: &fakeChat{}}}
}

func (f *fakeLocal) Installed(kind local.Kind) []local.Model {
	switch {
	case kind == local.ASR && f.ASRP != nil:
		return []local.Model{fakeASRModel}
	case kind == local.LLM && f.Chat != nil:
		return append([]local.Model{fakeLLMModel}, f.MoreLLM...)
	}
	return nil
}

func (f *fakeLocal) ASR(_ context.Context, id string) (asr.Provider, local.Model, error) {
	if f.ASRP == nil {
		return nil, local.Model{}, local.NotInstalledError(local.ASR, id)
	}
	return f.ASRP, fakeASRModel, nil
}

func (f *fakeLocal) LLM(_ context.Context, id string) (local.Chat, local.Model, error) {
	if f.Chat == nil {
		return nil, local.Model{}, local.NotInstalledError(local.LLM, id)
	}
	return f.Chat, fakeLLMModel, nil
}

// withChat is s with chat as its language model (a fakeChat behind
// llm.Client, two attempts as sessions had).
func withChat(s *fakeLocal, chat *fakeChat) *fakeLocal {
	s.Chat = &llm.Client{Backend: chat, Retry: llm.RetryPolicy{MaxAttempts: 2}}
	return s
}

// fakeChat is a language model (llm.Backend): it answers each request
// with text (or texts in turn, the last repeating; or fails with an HTTP
// status), streamed in pieces when the request streams, and keeps the
// requests.
type fakeChat struct {
	mu     sync.Mutex
	text   string
	texts  []string
	status int
	delay  time.Duration
	bodies []map[string]any
	// onRequest runs as each request comes (mute_e2e_test.go: is the
	// output back by then?).
	onRequest func()
}

func (f *fakeChat) set(text string, status int) {
	f.mu.Lock()
	f.text, f.texts, f.status = text, nil, status
	f.mu.Unlock()
}

func (f *fakeChat) Info() llm.BackendInfo { return llm.BackendInfo{ID: fakeLLMModel.ID} }

func (f *fakeChat) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	if f.onRequest != nil {
		f.onRequest()
	}
	f.mu.Lock()
	f.bodies = append(f.bodies, requestBody(req))
	text, status, delay := f.text, f.status, f.delay
	if len(f.texts) > 0 {
		text = f.texts[min(len(f.bodies), len(f.texts))-1]
	}
	f.mu.Unlock()
	if delay > 0 {
		select {
		case <-time.After(delay):
		case <-ctx.Done():
			return llm.Response{}, &llm.Error{Code: llm.CodeCancelled, Err: ctx.Err()}
		}
	}
	if status != 0 {
		return llm.Response{}, llm.HTTPError(fakeLLMModel.ID, req.Model, status, "overloaded", 0)
	}
	if req.OnDelta != nil {
		rs := []rune(text)
		for i := 0; i < len(rs); i += 6 {
			req.OnDelta(string(rs[i:min(i+6, len(rs))]))
			time.Sleep(5 * time.Millisecond)
		}
	}
	return llm.Response{Text: text, FinishReason: llm.FinishStop, Model: fakeLLMModel.ID,
		Usage: llm.Usage{PromptTokens: 120, CompletionTokens: 6, TotalTokens: 126}}, nil
}

// requestBody is req as the tests read it: an OpenAI chat request.
func requestBody(req llm.Request) map[string]any {
	msgs := make([]any, 0, len(req.Messages))
	for _, m := range req.Messages {
		msgs = append(msgs, map[string]any{"role": string(m.Role), "content": m.Content})
	}
	body := map[string]any{"messages": msgs, "stream": req.OnDelta != nil}
	if req.Model != "" {
		body["model"] = req.Model
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.Format == llm.FormatJSON {
		body["response_format"] = map[string]any{"type": "json_object"}
	}
	return body
}

func (f *fakeChat) requests() []map[string]any {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]map[string]any(nil), f.bodies...)
}

// fakeTranscriptions is a speech model that transcribes a recording.
type fakeTranscriptions struct {
	mu    sync.Mutex
	text  string
	calls int
	size  int
}

func (f *fakeTranscriptions) Caps() asr.Caps { return asr.Caps{File: true} }

func (f *fakeTranscriptions) Transcribe(_ context.Context, req asr.FileRequest) (asr.Result, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls++
	f.size = len(req.Audio)
	return asr.Result{Text: f.text, Model: fakeASRModel.ID}, nil
}

func (f *fakeTranscriptions) setText(text string) {
	f.mu.Lock()
	f.text = text
	f.mu.Unlock()
}

func (f *fakeTranscriptions) count() int {
	n, _ := f.uploads()
	return n
}

func (f *fakeTranscriptions) uploads() (calls, size int) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls, f.size
}

// fakeStreamer is a streaming speech model: a partial "hello" after
// three pieces of audio, the final text on Finish.
type fakeStreamer struct{ final string }

func (fakeStreamer) Caps() asr.Caps { return asr.Caps{Stream: true} }

func (f fakeStreamer) Open(context.Context, asr.StreamRequest) (asr.Stream, error) {
	return &fakeStream{final: f.final, ev: make(chan asr.Event, 16)}, nil
}

type fakeStream struct {
	mu     sync.Mutex
	final  string
	sends  int
	ev     chan asr.Event
	closed bool
}

func (s *fakeStream) Send([]byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.sends++
	if s.sends == 3 && !s.closed {
		s.ev <- asr.Event{Kind: asr.EventPartial, Text: "hello"}
	}
	return nil
}

func (s *fakeStream) Finish() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.closed {
		s.ev <- asr.Event{Kind: asr.EventFinal, Text: s.final, Result: asr.Result{Text: s.final, Model: fakeASRModel.ID}}
		s.closed = true
		close(s.ev)
	}
	return nil
}

func (s *fakeStream) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.closed {
		s.closed = true
		close(s.ev)
	}
}

func (s *fakeStream) Events() <-chan asr.Event { return s.ev }
