package llm

import (
	"context"
	"encoding/json"
	"maps"
	"net/http"
	"strings"
	"testing"
)

// The fake below follows OpenAI's API reference, "Streaming events":
// named events, a sequence_number on each, the lifecycle events around the
// text deltas, one event that ends the response and no [DONE].

type rsStream struct {
	seq int
	out []string
}

func (s *rsStream) emit(typ string, fields map[string]any) {
	ev := map[string]any{"type": typ, "sequence_number": s.seq}
	maps.Copy(ev, fields)
	s.seq++
	raw, _ := json.Marshal(ev)
	s.out = append(s.out, "event: "+typ+"\ndata: "+string(raw)+"\n\n")
}

// rsObject is a response object; the fields a client should not need are
// there as the reference's examples have them.
func rsObject(status string, output []any, usage, details, failure any) map[string]any {
	if output == nil {
		output = []any{}
	}
	return map[string]any{
		"id": "resp_123", "object": "response", "created_at": 1740855869, "status": status,
		"error": failure, "incomplete_details": details, "instructions": nil, "max_output_tokens": nil,
		"model": "gpt-reported", "output": output, "previous_response_id": nil, "store": false, "temperature": 1,
		"text": map[string]any{"format": map[string]any{"type": "text"}}, "tool_choice": "auto", "tools": []any{},
		"top_p": 1, "truncation": "disabled", "usage": usage, "user": nil, "metadata": map[string]any{}, "parallel_tool_calls": true,
	}
}

var rsUsed = map[string]any{
	"input_tokens": 11, "input_tokens_details": map[string]any{"cached_tokens": 0},
	"output_tokens": 7, "output_tokens_details": map[string]any{"reasoning_tokens": 2}, "total_tokens": 18,
}

var rsReasoning = map[string]any{"id": "rs_1", "type": "reasoning", "summary": []any{}, "content": []any{map[string]any{"type": "reasoning_text", "text": "not the answer"}}}

func rsMessage(part map[string]any) map[string]any {
	return map[string]any{"id": "msg_1", "type": "message", "role": "assistant", "status": "completed", "content": []any{part}}
}

func rsTextPart(text string) map[string]any {
	return map[string]any{"type": "output_text", "text": text, "annotations": []any{}, "logprobs": []any{}}
}

// rsEvents streams an answer in parts (none: only a reasoning item), then
// whatever end writes; an end that writes nothing is a cut connection.
func rsEvents(end func(s *rsStream, output []any), parts ...string) http.HandlerFunc {
	s := &rsStream{}
	s.emit("response.created", map[string]any{"response": rsObject("in_progress", nil, nil, nil, nil)})
	s.emit("response.in_progress", map[string]any{"response": rsObject("in_progress", nil, nil, nil, nil)})
	s.emit("response.output_item.added", map[string]any{"output_index": 0, "item": rsReasoning})
	s.emit("response.output_item.done", map[string]any{"output_index": 0, "item": rsReasoning})
	output := []any{rsReasoning}
	if len(parts) > 0 {
		text := strings.Join(parts, "")
		loc := map[string]any{"item_id": "msg_1", "output_index": 1, "content_index": 0}
		with := func(k string, v any) map[string]any {
			m := maps.Clone(loc)
			m[k] = v
			return m
		}
		item := rsMessage(rsTextPart(text))
		s.emit("response.output_item.added", map[string]any{"output_index": 1, "item": rsMessage(rsTextPart(""))})
		s.emit("response.content_part.added", with("part", rsTextPart("")))
		for _, p := range parts {
			d := with("delta", p)
			d["logprobs"] = []any{}
			s.emit("response.output_text.delta", d)
		}
		s.emit("response.output_text.done", with("text", text))
		s.emit("response.content_part.done", with("part", rsTextPart(text)))
		s.emit("response.output_item.done", map[string]any{"output_index": 1, "item": item})
		output = append(output, item)
	}
	end(s, output)
	return sse(s.out...)
}

func rsCompleted(s *rsStream, output []any) {
	s.emit("response.completed", map[string]any{"response": rsObject("completed", output, rsUsed, nil, nil)})
}

func rsIncomplete(reason string) func(*rsStream, []any) {
	return func(s *rsStream, output []any) {
		s.emit("response.incomplete", map[string]any{"response": rsObject("incomplete", output, nil, map[string]any{"reason": reason}, nil)})
	}
}

func rsFailed(code, message string) func(*rsStream, []any) {
	return func(s *rsStream, _ []any) {
		s.emit("response.failed", map[string]any{"response": rsObject("failed", nil, nil, nil, map[string]any{"code": code, "message": message})})
	}
}

func rsErrorEvent(code any, message string) func(*rsStream, []any) {
	return func(s *rsStream, _ []any) {
		s.emit("error", map[string]any{"code": code, "message": message, "param": nil})
	}
}

func rsCut(*rsStream, []any) {}

func rsPreset(t *testing.T, id, base string) Preset {
	t.Helper()
	cfg := ProviderConfig{ID: id, BaseURL: base + "/v1", Model: "m", API: APIResponses}
	p, err := cfg.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func rsChat(t *testing.T, f *fakeServer, req Request) (Response, error) {
	t.Helper()
	return NewOpenAI(rsPreset(t, "openai", f.srv.URL), "sk-test", f.srv.Client()).Chat(context.Background(), req)
}

var rsNoop = func(string) {}

// TestResponsesRequest: every row of the mapping from chat completions, as
// sent, for both services that offer the format.
func TestResponsesRequest(t *testing.T) {
	for _, id := range []string{"openai", Custom} {
		f := newFakeServer(t, jsonReply(200, rsObject("completed", []any{rsMessage(rsTextPart("ok"))}, rsUsed, nil, nil)))
		p := rsPreset(t, id, f.srv.URL)
		p.Generation = Generation{Verbosity: "low", Thinking: Thinking{Mode: ThinkEffort, Effort: "high"}, TopK: ptr(4), Seed: ptr(int64(7)), Logprobs: true, TopLogprobs: ptr(2)}
		_, err := NewOpenAI(p, "sk-test", f.srv.Client()).Chat(context.Background(), Request{
			Messages:    []Message{{Role: System, Content: "one"}, {Role: User, Content: "hi"}, {Role: Assistant, Content: "yes"}, {Role: System, Content: "two"}, {Role: User, Content: "go"}},
			Temperature: ptr(0.2), TopP: ptr(0.9), MaxTokens: 50, Stop: []string{"\n\n"},
		})
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		r := f.requests()[0]
		if r.Path != "/v1/responses" || r.Header.Get("Authorization") != "Bearer sk-test" {
			t.Fatalf("%s: request %+v", id, r)
		}
		b := r.Body
		if b["instructions"] != "one\n\ntwo" || b["model"] != "m" || b["temperature"] != 0.2 || b["top_p"] != 0.9 || b["max_output_tokens"] != 50.0 || b["store"] != false {
			t.Errorf("%s: body %v", id, b)
		}
		input, _ := json.Marshal(b["input"])
		if string(input) != `[{"content":"hi","role":"user"},{"content":"yes","role":"assistant"},{"content":"go","role":"user"}]` {
			t.Errorf("%s: input %s", id, input)
		}
		if text := b["text"].(map[string]any); text["verbosity"] != "low" {
			t.Errorf("%s: text %v", id, text)
		}
		if reasoning := b["reasoning"].(map[string]any); reasoning["effort"] != "high" {
			t.Errorf("%s: reasoning %v", id, reasoning)
		}
		for _, k := range []string{"messages", "max_completion_tokens", "max_tokens", "verbosity", "reasoning_effort", "response_format", "stream", "stream_options", "stop", "seed", "top_k", "logprobs", "top_logprobs", "previous_response_id"} {
			if v, set := b[k]; set {
				t.Errorf("%s: %s sent: %v", id, k, v)
			}
		}
	}
}

// TestResponsesStoreOff: every request, streamed or not, the first or one
// sent again after a refusal, and whatever the Extra Body says.
func TestResponsesStoreOff(t *testing.T) {
	f := newFakeServer(t,
		jsonReply(400, map[string]any{"error": map[string]any{"message": "Unsupported parameter: 'temperature'"}}),
		rsEvents(rsCompleted, "ok"))
	p := rsPreset(t, "openai", f.srv.URL)
	p.Generation.ExtraBody = map[string]any{"store": true}
	if _, err := NewOpenAI(p, "k", f.srv.Client()).Chat(context.Background(), Request{Messages: msgs("x"), Temperature: ptr(0.3), OnDelta: rsNoop}); err != nil {
		t.Fatal(err)
	}
	_, _ = NewOpenAI(p, "k", f.srv.Client()).Chat(context.Background(), Request{Messages: msgs("x")}) // not streamed; only the request matters
	rs := f.requests()
	if len(rs) != 3 {
		t.Fatalf("%d requests", len(rs))
	}
	for i, r := range rs {
		if r.Body["store"] != false {
			t.Errorf("request %d: store %v", i, r.Body["store"])
		}
	}
	if rs[1].Body["temperature"] != nil || rs[1].Path != "/v1/responses" {
		t.Errorf("not relaxed: %+v", rs[1])
	}
}

// TestResponsesExtraBody: the user's keys win over what the rewrite writes
// and survive its list of fields to drop, and their map is left as it was.
func TestResponsesExtraBody(t *testing.T) {
	extra := map[string]any{
		"reasoning": map[string]any{"summary": "auto"}, "text": map[string]any{"verbosity": "high"},
		"seed": 7.0, "max_output_tokens": 4096.0, "instructions": "mine", "custom": "x",
	}
	p := rsPreset(t, "openai", "http://127.0.0.1:1")
	p.Generation = Generation{Verbosity: "low", Thinking: Thinking{Mode: ThinkEffort, Effort: "high"}, ExtraBody: extra}
	b := NewOpenAI(p, "k", nil).body(Request{Messages: []Message{{Role: System, Content: "sys"}, {Role: User, Content: "x"}}, MaxTokens: 50}, "m", false, false, false)
	raw, _ := json.Marshal(b)
	var got map[string]any
	_ = json.Unmarshal(raw, &got)
	if got["seed"] != 7.0 || got["max_output_tokens"] != 4096.0 || got["instructions"] != "mine" || got["custom"] != "x" {
		t.Errorf("body %v", got)
	}
	if r := got["reasoning"].(map[string]any); r["summary"] != "auto" || r["effort"] != "high" {
		t.Errorf("reasoning %v", r)
	}
	if text := got["text"].(map[string]any); text["verbosity"] != "high" {
		t.Errorf("text %v", text)
	}
	if len(extra["reasoning"].(map[string]any)) != 1 {
		t.Errorf("the rewrite changed the user's map: %v", extra)
	}
}

// TestResponsesJSONFormats: text.format has the schema's fields flat, and
// the prompt carries the schema only when the API does not.
func TestResponsesJSONFormats(t *testing.T) {
	schema := Object(map[string]*Schema{"a": String()}, "a")
	schema.Name = "thing"
	for _, mode := range []JSONMode{JSONSchemaMode, JSONObjectMode, JSONPromptMode} {
		f := newFakeServer(t, jsonReply(200, rsObject("completed", []any{rsMessage(rsTextPart(`{"a":"x"}`))}, rsUsed, nil, nil)))
		p := rsPreset(t, "openai", f.srv.URL)
		p.JSONMode = mode
		var out struct{ A string }
		c := &Client{Backend: NewOpenAI(p, "sk", f.srv.Client())}
		if _, err := c.JSON(context.Background(), Request{Messages: msgs("x"), Schema: schema}, &out, 0); err != nil || out.A != "x" {
			t.Fatalf("%s: %v %+v", mode, err, out)
		}
		b := f.requests()[0].Body
		text, _ := b["text"].(map[string]any)
		format, _ := text["format"].(map[string]any)
		instr, _ := b["instructions"].(string)
		switch mode {
		case JSONSchemaMode:
			if format["type"] != "json_schema" || format["name"] != "thing" || format["strict"] != false || format["schema"] == nil || format["json_schema"] != nil || strings.Contains(instr, "JSON Schema") {
				t.Errorf("schema mode %v", b)
			}
		case JSONObjectMode:
			if format["type"] != "json_object" || len(format) != 1 || !strings.Contains(instr, "JSON Schema") {
				t.Errorf("object mode %v", b)
			}
		case JSONPromptMode:
			if format != nil || !strings.Contains(instr, "JSON Schema") {
				t.Errorf("prompt mode %v", b)
			}
		}
	}
}

// TestResponsesProbe: the connection test's 8 tokens go out as the API's
// minimum, and an answer cut short by them, even empty, proves the setup.
func TestResponsesProbe(t *testing.T) {
	f := newFakeServer(t, jsonReply(200, rsObject("incomplete", []any{rsReasoning}, nil, map[string]any{"reason": "max_output_tokens"}, nil)))
	if d, err := Probe(context.Background(), NewOpenAI(rsPreset(t, "openai", f.srv.URL), "sk", f.srv.Client())); err != nil || d < 0 {
		t.Fatalf("%v %v", d, err)
	}
	if b := f.requests()[0].Body; b["max_output_tokens"] != 16.0 || b["stream"] != nil {
		t.Errorf("probe body %v", b)
	}
	p := rsPreset(t, "openai", f.srv.URL)
	p.Generation.MaxOutputTokens = ptr(3)
	if b := NewOpenAI(p, "k", nil).body(Request{Messages: msgs("x")}, "m", false, false, false); b["max_output_tokens"] != 16 {
		t.Errorf("a user's 3 tokens: %v", b["max_output_tokens"])
	}
}

// TestResponsesStream: only the text deltas reach the caller, in order,
// through the documented sequence with its reasoning item; the event that
// ends the response gives the usage and the model.
func TestResponsesStream(t *testing.T) {
	f := newFakeServer(t, rsEvents(rsCompleted, "Hel", "lo"))
	var deltas []string
	c := &Client{Backend: NewOpenAI(rsPreset(t, "openai", f.srv.URL), "sk", f.srv.Client())}
	resp, err := c.Chat(context.Background(), Request{Messages: msgs("hi"), OnDelta: func(s string) { deltas = append(deltas, s) }})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "Hello" || strings.Join(deltas, "|") != "Hel|lo" || resp.FinishReason != FinishStop || resp.Model != "gpt-reported" || resp.Provider != "openai" {
		t.Fatalf("resp %+v deltas %q", resp, deltas)
	}
	if u := resp.Usage; u.PromptTokens != 11 || u.CompletionTokens != 7 || u.TotalTokens != 18 || u.Estimated {
		t.Errorf("usage %+v", u)
	}
	if b := f.requests()[0].Body; b["stream"] != true || b["stream_options"] != nil {
		t.Errorf("body %v", b)
	}
}

// TestResponsesStreamNoDeltas: a server that sends only the final event
// still gives its text.
func TestResponsesStreamNoDeltas(t *testing.T) {
	f := newFakeServer(t, func(w http.ResponseWriter, r *http.Request) {
		s := &rsStream{}
		rsCompleted(s, []any{rsMessage(rsTextPart("whole"))})
		sse(s.out...)(w, r)
	})
	if resp, err := rsChat(t, f, Request{Messages: msgs("hi"), OnDelta: rsNoop}); err != nil || resp.Text != "whole" {
		t.Fatalf("%v %+v", err, resp)
	}
}

// TestResponsesIncomplete: response.incomplete is the end of an answer,
// not a failure, with its reason as the finish reason. Without usage (the
// reference's example has none) the Client estimates it.
func TestResponsesIncomplete(t *testing.T) {
	for _, tc := range []struct {
		reason, finish string
	}{
		{"max_output_tokens", FinishLength},
		{"content_filter", FinishFilter},
		{"max_messages", FinishOther},
	} {
		f := newFakeServer(t, rsEvents(rsIncomplete(tc.reason), "par", "tial"))
		c := &Client{Backend: NewOpenAI(rsPreset(t, "openai", f.srv.URL), "sk", f.srv.Client())}
		resp, err := c.Chat(context.Background(), Request{Messages: msgs("hi"), OnDelta: rsNoop})
		if err != nil || resp.Text != "partial" || resp.FinishReason != tc.finish || !resp.Usage.Estimated {
			t.Errorf("%s: %v %+v", tc.reason, err, resp)
		}
	}
	// No text at all: the token limit is still an answer; the filter is not.
	f := newFakeServer(t, rsEvents(rsIncomplete("max_output_tokens")))
	if resp, err := rsChat(t, f, Request{Messages: msgs("hi"), OnDelta: rsNoop}); err != nil || resp.FinishReason != FinishLength {
		t.Errorf("empty, cut short: %v %+v", err, resp)
	}
	f = newFakeServer(t, rsEvents(rsIncomplete("content_filter")))
	if _, err := rsChat(t, f, Request{Messages: msgs("hi"), OnDelta: rsNoop}); Code(err) != CodeFiltered {
		t.Errorf("empty, filtered: %v", err)
	}
}

// TestResponsesStreamRefusal: the documented refusal events end in a
// response.completed whose message holds a refusal part.
func TestResponsesStreamRefusal(t *testing.T) {
	part := map[string]any{"type": "refusal", "refusal": "I can't help with that."}
	s := &rsStream{}
	s.emit("response.created", map[string]any{"response": rsObject("in_progress", nil, nil, nil, nil)})
	s.emit("response.output_item.added", map[string]any{"output_index": 0, "item": rsMessage(map[string]any{"type": "refusal", "refusal": ""})})
	s.emit("response.refusal.delta", map[string]any{"item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": "I can't"})
	s.emit("response.refusal.done", map[string]any{"item_id": "msg_1", "output_index": 0, "content_index": 0, "refusal": "I can't help with that."})
	s.emit("response.content_part.done", map[string]any{"item_id": "msg_1", "output_index": 0, "content_index": 0, "part": part})
	s.emit("response.output_item.done", map[string]any{"output_index": 0, "item": rsMessage(part)})
	rsCompleted(s, []any{rsMessage(part)})
	f := newFakeServer(t, sse(s.out...))
	deltas := 0
	if _, err := rsChat(t, f, Request{Messages: msgs("hi"), OnDelta: func(string) { deltas++ }}); Code(err) != CodeFiltered || deltas != 0 {
		t.Fatalf("%v, %d deltas", err, deltas)
	}
}

// TestResponsesStreamFailures: response.failed (error inside the response)
// and the error event (code and message at the top level) become the same
// errors an HTTP status would.
func TestResponsesStreamFailures(t *testing.T) {
	for name, tc := range map[string]struct {
		end       func(*rsStream, []any)
		code      string
		retryable bool
	}{
		"failed server":     {rsFailed("server_error", "The model failed to generate a response."), CodeServer, true},
		"failed rate":       {rsFailed("rate_limit_exceeded", "slow down"), CodeRateLimited, true},
		"failed prompt":     {rsFailed("invalid_prompt", "not allowed"), CodeBadRequest, false},
		"failed quota":      {rsFailed("insufficient_quota", "out of credit"), CodeRateLimited, false},
		"error event":       {rsErrorEvent("ERR_SOMETHING", "Something went wrong"), CodeBadRequest, false},
		"error null code":   {rsErrorEvent(nil, "Something went wrong"), CodeServer, true},
		"error too long":    {rsErrorEvent("context_length_exceeded", "maximum context length"), CodeContextLength, false},
		"error overloaded":  {rsErrorEvent("server_is_overloaded", "overloaded"), CodeOverloaded, true},
		"error server code": {rsErrorEvent("server_error", "oops"), CodeServer, true},
	} {
		f := newFakeServer(t, rsEvents(tc.end, "par"))
		_, err := rsChat(t, f, Request{Messages: msgs("hi"), OnDelta: rsNoop})
		e, _ := err.(*Error)
		if e == nil || e.Code != tc.code || e.Retryable != tc.retryable {
			t.Errorf("%s: %v (retryable %v)", name, err, e != nil && e.Retryable)
		}
	}
	f := newFakeServer(t, rsEvents(rsErrorEvent("ERR_SOMETHING", "Something went wrong")))
	_, err := rsChat(t, f, Request{Messages: msgs("hi"), OnDelta: rsNoop})
	if e, _ := err.(*Error); e == nil || e.Params["detail"] != "ERR_SOMETHING: Something went wrong" {
		t.Errorf("detail lost: %v", err)
	}
}

// TestResponsesStreamCut: a stream without an event that ends the response
// is a retryable failure; with text already shown it is not retried.
func TestResponsesStreamCut(t *testing.T) {
	f := newFakeServer(t, rsEvents(rsCut), rsEvents(rsCompleted, "ok"))
	c := &Client{Backend: NewOpenAI(rsPreset(t, "openai", f.srv.URL), "sk", f.srv.Client()), Sleep: noSleep, Retry: RetryPolicy{MaxAttempts: 2}}
	resp, err := c.Chat(context.Background(), Request{Messages: msgs("hi"), OnDelta: rsNoop})
	if err != nil || resp.Text != "ok" || resp.Attempts != 2 {
		t.Fatalf("%v %+v", err, resp)
	}
	f = newFakeServer(t, rsEvents(rsCut, "par"))
	c = &Client{Backend: NewOpenAI(rsPreset(t, "openai", f.srv.URL), "sk", f.srv.Client()), Sleep: noSleep, Retry: RetryPolicy{MaxAttempts: 2}}
	if _, err := c.Chat(context.Background(), Request{Messages: msgs("hi"), OnDelta: rsNoop}); Code(err) != CodeNetwork || len(f.requests()) != 1 {
		t.Fatalf("%v, %d requests", err, len(f.requests()))
	}
}

// TestResponsesNotStreamed: the text of the output_text parts only, the
// usage, and the failures a 200 body can carry.
func TestResponsesNotStreamed(t *testing.T) {
	f := newFakeServer(t, jsonReply(200, rsObject("completed", []any{rsReasoning, rsMessage(rsTextPart("hola")), rsMessage(rsTextPart(" mundo"))}, rsUsed, nil, nil)))
	resp, err := rsChat(t, f, Request{Messages: msgs("hi")})
	if err != nil || resp.Text != "hola mundo" || resp.FinishReason != FinishStop || resp.Usage.TotalTokens != 18 || resp.Model != "gpt-reported" {
		t.Fatalf("%v %+v", err, resp)
	}
	for name, tc := range map[string]struct {
		reply any
		code  string
	}{
		"refusal": {rsObject("completed", []any{rsMessage(map[string]any{"type": "refusal", "refusal": "no"})}, rsUsed, nil, nil), CodeFiltered},
		"failed":  {rsObject("failed", nil, nil, nil, map[string]any{"code": "server_error", "message": "x"}), CodeServer},
		"empty":   {map[string]any{"id": "resp_1"}, CodeBadResponse},
	} {
		f := newFakeServer(t, jsonReply(200, tc.reply))
		if _, err := rsChat(t, f, Request{Messages: msgs("hi")}); Code(err) != tc.code {
			t.Errorf("%s: %v", name, err)
		}
	}
	both := map[string]any{"id": "msg_1", "type": "message", "role": "assistant", "content": []any{rsTextPart("some"), map[string]any{"type": "refusal", "refusal": "no more"}}}
	f = newFakeServer(t, jsonReply(200, rsObject("completed", []any{both}, rsUsed, nil, nil)))
	if resp, err := rsChat(t, f, Request{Messages: msgs("hi")}); err != nil || resp.Text != "some" || resp.FinishReason != FinishStop {
		t.Errorf("text and a refusal: %v %+v", err, resp)
	}
	f = newFakeServer(t, jsonReply(200, rsObject("incomplete", []any{rsMessage(rsTextPart("par"))}, nil, map[string]any{"reason": "max_output_tokens"}, nil)))
	if resp, err := rsChat(t, f, Request{Messages: msgs("hi")}); err != nil || resp.FinishReason != FinishLength || resp.Text != "par" {
		t.Errorf("incomplete: %v %+v", err, resp)
	}
}

// TestResponsesErrorsHideKey: a key quoted in a stream's error, or in a
// stream event that is not JSON, is blanked out before the detail is cut.
func TestResponsesErrorsHideKey(t *testing.T) {
	const key = "sk-secret-4242-0123456789abcdefghijklmnopqrstuvwxyz"
	for name, h := range map[string]http.HandlerFunc{
		"error event": rsEvents(rsErrorEvent("ERR_X", strings.Repeat("x", 270)+" "+key)),
		"failed":      rsEvents(rsFailed("invalid_prompt", strings.Repeat("x", 270)+" "+key)),
		"not json":    sse("data: " + strings.Repeat("x", 181) + "rejected " + key + "\n\n"),
	} {
		f := newFakeServer(t, h)
		_, err := NewOpenAI(rsPreset(t, "openai", f.srv.URL), key, f.srv.Client()).Chat(context.Background(), Request{Messages: msgs("x"), OnDelta: rsNoop})
		if err == nil || strings.Contains(err.Error(), key[:12]) || !strings.Contains(err.Error(), "[key]") {
			t.Errorf("%s: %v", name, err)
		}
	}
}

// TestResponsesOnlyModel: on chat completions, a model that says it needs
// the Responses API gets its own error whichever status says so. The same
// words over the Responses API are an ordinary refusal.
func TestResponsesOnlyModel(t *testing.T) {
	const words = "This model is only supported in v1/responses and not in v1/chat/completions."
	for _, status := range []int{400, 404} {
		reply := jsonReply(status, map[string]any{"error": map[string]any{"message": words}})
		f := newFakeServer(t, reply)
		_, err := NewOpenAI(presetAt(t, "openai", f.srv.URL), "sk", f.srv.Client()).Chat(context.Background(), Request{Messages: msgs("x")})
		if e, _ := err.(*Error); e == nil || e.Code != CodeResponsesOnly || e.Status != status {
			t.Errorf("%d on chat completions: %v", status, err)
		}
		if len(f.requests()) != 1 {
			t.Errorf("%d: %d requests", status, len(f.requests()))
		}
		if _, err := rsChat(t, f, Request{Messages: msgs("x")}); Code(err) == CodeResponsesOnly {
			t.Errorf("%d on Responses: %v", status, err)
		}
	}
	f := newFakeServer(t, jsonReply(500, map[string]any{"error": map[string]any{"message": "v1/responses is down"}}))
	if _, err := NewOpenAI(presetAt(t, "openai", f.srv.URL), "sk", f.srv.Client()).Chat(context.Background(), Request{Messages: msgs("x")}); Code(err) != CodeServer {
		t.Errorf("500: %v", err)
	}
}

// TestResolveResponsesURL: a full Responses URL is taken for its API root.
func TestResolveResponsesURL(t *testing.T) {
	for _, base := range []string{"https://example.com/v1/responses", "https://example.com/v1/responses/", "https://example.com/v1"} {
		p, err := ProviderConfig{ID: Custom, BaseURL: base, Model: "m", API: APIResponses}.Resolve()
		if err != nil || p.BaseURL != "https://example.com/v1" || p.API != APIResponses {
			t.Errorf("%s: %v %+v", base, err, p)
		}
	}
}

// TestChatBodyUnchanged: with no API format a request is, to the byte, the
// chat completions body.
func TestChatBodyUnchanged(t *testing.T) {
	p, err := ProviderConfig{ID: "openai", BaseURL: "https://example.com/v1"}.Resolve()
	if err != nil {
		t.Fatal(err)
	}
	p.Generation = Generation{Verbosity: "low", Thinking: Thinking{Mode: ThinkEffort, Effort: "high"}}
	req := Request{Messages: []Message{{Role: System, Content: "sys"}, {Role: User, Content: "hi"}}, Temperature: ptr(0.2), MaxTokens: 50, Stop: []string{"x"}}
	raw, _ := json.Marshal(NewOpenAI(p, "k", nil).body(req, "m", true, true, false))
	const want = `{"max_completion_tokens":50,"messages":[{"role":"system","content":"sys"},{"role":"user","content":"hi"}],"model":"m","reasoning_effort":"high","stop":["x"],"stream":true,"stream_options":{"include_usage":true},"temperature":0.2,"verbosity":"low"}`
	if string(raw) != want {
		t.Errorf("body\n%s\nwant\n%s", raw, want)
	}
	if got := NewOpenAI(p, "k", nil).path(); got != "/chat/completions" {
		t.Errorf("path %s", got)
	}
}
