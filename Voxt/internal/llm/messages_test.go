package llm

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
)

// The fake below follows Anthropic's Messages API reference, "Streaming
// messages": named events, message_start with the usage so far, content
// blocks (a thinking block before the text), ping, message_delta with the
// stop reason and the output tokens, then message_stop.

func msEventText(typ string, fields map[string]any) string {
	ev := map[string]any{"type": typ}
	for k, v := range fields {
		ev[k] = v
	}
	raw, _ := json.Marshal(ev)
	return "event: " + typ + "\ndata: " + string(raw) + "\n\n"
}

var msStartUsage = map[string]any{"input_tokens": 12, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1100, "output_tokens": 1}

// msEvents streams a thinking block and then parts of text; end ends the
// stream (nil: the connection is cut after the text).
func msEvents(end []string, parts ...string) []string {
	out := []string{
		msEventText("message_start", map[string]any{"message": map[string]any{
			"id": "msg_1", "type": "message", "role": "assistant", "content": []any{}, "model": "claude-reported",
			"stop_reason": nil, "stop_sequence": nil, "usage": msStartUsage,
		}}),
		msEventText("content_block_start", map[string]any{"index": 0, "content_block": map[string]any{"type": "thinking", "thinking": ""}}),
		msEventText("content_block_delta", map[string]any{"index": 0, "delta": map[string]any{"type": "thinking_delta", "thinking": "not the answer"}}),
		msEventText("content_block_delta", map[string]any{"index": 0, "delta": map[string]any{"type": "signature_delta", "signature": "sig"}}),
		msEventText("content_block_stop", map[string]any{"index": 0}),
		msEventText("ping", nil),
		msEventText("content_block_start", map[string]any{"index": 1, "content_block": map[string]any{"type": "text", "text": ""}}),
	}
	for _, p := range parts {
		out = append(out, msEventText("content_block_delta", map[string]any{"index": 1, "delta": map[string]any{"type": "text_delta", "text": p}}))
	}
	return append(out, end...)
}

func msFinished(stopReason string) []string {
	return []string{
		msEventText("content_block_stop", map[string]any{"index": 1}),
		msEventText("message_delta", map[string]any{"delta": map[string]any{"stop_reason": stopReason, "stop_sequence": nil}, "usage": map[string]any{"output_tokens": 7}}),
		msEventText("message_stop", nil),
	}
}

func msBody(stopReason string, blocks ...map[string]any) map[string]any {
	return map[string]any{
		"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-reported", "content": blocks,
		"stop_reason": stopReason, "stop_sequence": nil, "usage": map[string]any{"input_tokens": 12, "cache_creation_input_tokens": 1100, "cache_read_input_tokens": 0, "output_tokens": 7},
	}
}

func msPreset(t *testing.T, base string) Preset {
	t.Helper()
	p, err := ProviderConfig{ID: "anthropic", BaseURL: base + "/v1/messages", API: APIMessages}.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	return p
}

// TestMessagesRequest: the path, the key in x-api-key with the version,
// the system messages joined into system, max_tokens always there, stop as
// stop_sequences, the preset's thinking off, and no chat completions field
// the API does not take; the cache only when asked for.
func TestMessagesRequest(t *testing.T) {
	for _, promptCache := range []bool{false, true} {
		f := newFakeServer(t, sse(msEvents(msFinished("end_turn"), "o", "k")...))
		p := msPreset(t, f.srv.URL)
		p.Generation = Generation{PromptCache: promptCache, Seed: ptr(int64(7)), PresencePenalty: ptr(0.5), TopK: ptr(4)}
		_, err := NewOpenAI(p, "sk-ant", f.srv.Client()).Chat(context.Background(), Request{
			Messages: []Message{{Role: System, Content: "one"}, {Role: User, Content: "hi"}, {Role: Assistant, Content: "yes"}, {Role: System, Content: "two"}, {Role: User, Content: "go"}},
			Stop:     []string{"\n\n"}, OnDelta: rsNoop,
		})
		if err != nil {
			t.Fatalf("cache %v: %v", promptCache, err)
		}
		r := f.requests()[0]
		if r.Path != "/v1/messages" || r.Header.Get("x-api-key") != "sk-ant" || r.Header.Get("anthropic-version") != anthropicVersion || r.Header.Get("Authorization") != "" {
			t.Fatalf("cache %v: request %s %v", promptCache, r.Path, r.Header)
		}
		b := r.Body
		if b["model"] != "claude-haiku-5-5" || b["max_tokens"] != float64(messagesMaxTokens) || b["stream"] != true || b["top_k"] != 4.0 {
			t.Errorf("cache %v: body %v", promptCache, b)
		}
		if thinking, _ := b["thinking"].(map[string]any); thinking["type"] != "disabled" {
			t.Errorf("cache %v: thinking %v", promptCache, b["thinking"])
		}
		system, _ := json.Marshal(b["system"])
		want := `"one\n\ntwo"`
		if promptCache {
			want = `[{"cache_control":{"type":"ephemeral"},"text":"one\n\ntwo","type":"text"}]`
		}
		if string(system) != want {
			t.Errorf("cache %v: system %s", promptCache, system)
		}
		messages, _ := json.Marshal(b["messages"])
		if string(messages) != `[{"role":"user","content":"hi"},{"role":"assistant","content":"yes"},{"role":"user","content":"go"}]` {
			t.Errorf("cache %v: messages %s", promptCache, messages)
		}
		if stop, _ := json.Marshal(b["stop_sequences"]); string(stop) != `["\n\n"]` {
			t.Errorf("cache %v: stop_sequences %s", promptCache, stop)
		}
		for _, k := range []string{"stop", "stream_options", "seed", "presence_penalty", "response_format"} {
			if v, set := b[k]; set {
				t.Errorf("cache %v: %s sent: %v", promptCache, k, v)
			}
		}
	}
}

// TestMessagesMaxTokens: the request's limit is kept, and the default
// leaves a thinking budget its room.
func TestMessagesMaxTokens(t *testing.T) {
	p := msPreset(t, "http://127.0.0.1:1")
	o := NewOpenAI(p, "k", nil)
	if b := o.body(Request{Messages: msgs("x"), MaxTokens: 50}, "m", false, false, true); b["max_tokens"] != 50 {
		t.Errorf("max_tokens %v", b["max_tokens"])
	}
	p.Generation.Thinking = Thinking{Mode: ThinkBudget, Budget: 2048}
	if b := NewOpenAI(p, "k", nil).body(Request{Messages: msgs("x")}, "claude-haiku-4-5", false, false, true); b["max_tokens"] != messagesMaxTokens+2048 {
		t.Errorf("max_tokens with a budget %v", b["max_tokens"])
	}
}

// TestMessagesStream: only text deltas are the answer's, the prompt
// tokens count the cached ones, and the stop reason is normalised.
func TestMessagesStream(t *testing.T) {
	f := newFakeServer(t, sse(msEvents(msFinished("max_tokens"), "Hel", "lo")...))
	var deltas []string
	resp, err := (&Client{Backend: NewOpenAI(msPreset(t, f.srv.URL), "k", f.srv.Client())}).Chat(context.Background(), Request{Messages: msgs("x"), OnDelta: func(s string) { deltas = append(deltas, s) }})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "Hello" || len(deltas) != 2 || resp.Model != "claude-reported" || resp.FinishReason != FinishLength {
		t.Errorf("response %+v, deltas %q", resp, deltas)
	}
	if u := resp.Usage; u.PromptTokens != 1112 || u.CachedTokens != 1100 || u.CompletionTokens != 7 || u.TotalTokens != 1119 || u.Estimated {
		t.Errorf("usage %+v", u)
	}
}

// TestMessagesFailures: an error event takes its type's status (an
// overloaded service is retried), and a stream without message_stop was
// cut.
func TestMessagesFailures(t *testing.T) {
	overloaded := msEventText("error", map[string]any{"error": map[string]any{"type": "overloaded_error", "message": "Overloaded"}})
	f := newFakeServer(t, sse(msEvents([]string{overloaded}, "par")...), sse(msEvents(nil, "par")...))
	o := NewOpenAI(msPreset(t, f.srv.URL), "k", f.srv.Client())
	var e *Error
	if _, err := o.Chat(context.Background(), Request{Messages: msgs("x"), OnDelta: rsNoop}); !errors.As(err, &e) || e.Code != CodeOverloaded || !e.Retryable {
		t.Errorf("error event: %v", err)
	}
	if _, err := o.Chat(context.Background(), Request{Messages: msgs("x"), OnDelta: rsNoop}); !errors.As(err, &e) || e.Code != CodeNetwork || !e.Retryable {
		t.Errorf("cut stream: %v", err)
	}
}

// TestMessagesReply: a non-streamed answer is its text blocks, with the
// usage; a refusal of a field is relaxed as on the other APIs.
func TestMessagesReply(t *testing.T) {
	refusal := jsonReply(400, map[string]any{"type": "error", "error": map[string]any{"type": "invalid_request_error", "message": "temperature: not supported on this model"}})
	reply := jsonReply(200, msBody("end_turn", map[string]any{"type": "thinking", "thinking": "not the answer", "signature": "s"}, map[string]any{"type": "text", "text": "ok"}))
	f := newFakeServer(t, refusal, reply)
	resp, err := NewOpenAI(msPreset(t, f.srv.URL), "k", f.srv.Client()).Chat(context.Background(), Request{Messages: msgs("x"), Temperature: ptr(0.2)})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "ok" || resp.FinishReason != FinishStop || resp.Usage.PromptTokens != 1112 || resp.Usage.CachedTokens != 0 {
		t.Errorf("response %+v", resp)
	}
	rs := f.requests()
	if len(rs) != 2 || rs[0].Body["temperature"] != 0.2 || rs[1].Body["temperature"] != nil {
		t.Errorf("not relaxed: %d requests", len(rs))
	}
}

// TestMessagesImages: pictures go as base64 image blocks beside the text.
func TestMessagesImages(t *testing.T) {
	b := NewOpenAI(msPreset(t, "http://127.0.0.1:1"), "k", nil).body(Request{Messages: msgs("what is this"), Images: [][]byte{{0xff, 0xd8}}}, "m", false, false, true)
	messages, _ := json.Marshal(b["messages"])
	if string(messages) != `[{"role":"user","content":[{"text":"what is this","type":"text"},{"source":{"data":"/9g=","media_type":"image/jpeg","type":"base64"},"type":"image"}]}]` {
		t.Errorf("messages %s", messages)
	}
}
