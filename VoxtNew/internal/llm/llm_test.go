package llm

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
)

// step is one scripted answer of fakeBackend: deltas streamed first (when
// the request streams), then resp or err; block waits for the attempt's
// context instead.
type step struct {
	deltas []string
	resp   Response
	err    error
	block  bool
}

// fakeBackend answers each call with the next step (the last repeats)
// and keeps the requests.
type fakeBackend struct {
	mu    sync.Mutex
	steps []step
	reqs  []Request
}

func (f *fakeBackend) Info() BackendInfo { return BackendInfo{ID: "fake"} }

func (f *fakeBackend) Chat(ctx context.Context, req Request) (Response, error) {
	f.mu.Lock()
	st := f.steps[min(len(f.reqs), len(f.steps)-1)]
	f.reqs = append(f.reqs, req)
	f.mu.Unlock()
	if req.OnDelta != nil {
		for _, d := range st.deltas {
			req.OnDelta(d)
			if ctx.Err() != nil {
				return Response{}, ctxErr(ctx, "fake", ctx.Err(), req.Timeout)
			}
		}
	}
	if st.block {
		<-ctx.Done()
		return Response{}, ctxErr(ctx, "fake", ctx.Err(), req.Timeout)
	}
	return st.resp, st.err
}

func (f *fakeBackend) requests() []Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]Request(nil), f.reqs...)
}

func text(s string) step {
	return step{resp: Response{Text: s, FinishReason: FinishStop, Usage: Usage{PromptTokens: 11, CompletionTokens: 7, TotalTokens: 18}}}
}

func noSleep(context.Context, time.Duration) error { return nil }

func msgs(s string) []Message { return []Message{{Role: User, Content: s}} }

func TestRetries(t *testing.T) {
	var slept []time.Duration
	sleep := func(_ context.Context, d time.Duration) error { slept = append(slept, d); return nil }
	f := &fakeBackend{steps: []step{
		{err: HTTPError("fake", "m", 429, "rate limit per minute", 2*time.Second)},
		{err: HTTPError("fake", "m", 503, "busy", 0)},
		text("done"),
	}}
	m := &Meter{}
	c := &Client{Backend: f, Sleep: sleep, Meter: m}
	resp, err := c.Chat(context.Background(), Request{Messages: msgs("x"), Skill: SkillTranslateText})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Attempts != 3 || len(slept) != 2 || slept[0] != 2*time.Second {
		t.Fatalf("attempts %d slept %v", resp.Attempts, slept)
	}
	if slept[1] < 500*time.Millisecond || slept[1] > time.Second {
		t.Fatalf("backoff %v", slept[1])
	}
	snap := m.Snapshot()
	if len(snap) != 1 || snap[0].Calls != 1 || snap[0].Attempts != 3 || snap[0].Skill != SkillTranslateText || snap[0].Usage.TotalTokens != 18 {
		t.Fatalf("meter %+v", snap)
	}
}

func TestRetryGivesUp(t *testing.T) {
	f := &fakeBackend{steps: []step{{err: HTTPError("fake", "m", 500, "boom", 0)}}}
	c := &Client{Backend: f, Sleep: noSleep, Retry: RetryPolicy{MaxAttempts: 4}}
	_, err := c.Chat(context.Background(), Request{Messages: msgs("x")})
	var e *Error
	if !errors.As(err, &e) || e.Code != CodeServer || e.Status != 500 || len(f.requests()) != 4 {
		t.Fatalf("err %v reqs %d", err, len(f.requests()))
	}
	// A long Retry-After fails at once.
	f2 := &fakeBackend{steps: []step{{err: HTTPError("fake", "m", 429, "", time.Hour)}}}
	c.Backend = f2
	_, err = c.Chat(context.Background(), Request{Messages: msgs("x")})
	if Code(err) != CodeRateLimited || len(f2.requests()) != 1 || !errors.Is(err, core.ErrBusy) {
		t.Fatalf("err %v", err)
	}
	// A request the model refuses is not retried.
	f3 := &fakeBackend{steps: []step{{err: HTTPError("fake", "m", 400, "maximum context length exceeded", 0)}}}
	c.Backend = f3
	if _, err = c.Chat(context.Background(), Request{Messages: msgs("x")}); Code(err) != CodeContextLength || len(f3.requests()) != 1 {
		t.Fatalf("err %v reqs %d", err, len(f3.requests()))
	}
}

func TestTimeoutAndCancel(t *testing.T) {
	f := &fakeBackend{steps: []step{{block: true}}}
	c := &Client{Backend: f, Timeout: 50 * time.Millisecond, Sleep: noSleep, Retry: RetryPolicy{MaxAttempts: 2}}
	_, err := c.Chat(context.Background(), Request{Messages: msgs("x")})
	if Code(err) != CodeTimeout || len(f.requests()) != 2 {
		t.Fatalf("err %v calls %d", err, len(f.requests()))
	}

	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(30 * time.Millisecond); cancel() }()
	c.Timeout = 10 * time.Second
	_, err = c.Chat(ctx, Request{Messages: msgs("x")})
	if Code(err) != CodeCancelled || !errors.Is(err, core.ErrCancelled) || !errors.Is(err, context.Canceled) {
		t.Fatalf("err %v", err)
	}
}

func TestStreamingNotRetriedOnceStarted(t *testing.T) {
	f := &fakeBackend{steps: []step{{deltas: []string{"<think>x</think>", "Hel", "lo"}, err: HTTPError("fake", "m", 503, "reset", 0)}, text("unused")}}
	c := &Client{Backend: f, Sleep: noSleep}
	var got strings.Builder
	_, err := c.Chat(context.Background(), Request{Messages: msgs("x"), OnDelta: func(d string) { got.WriteString(d) }})
	if Code(err) != CodeServer || len(f.requests()) != 1 || got.String() != "Hello" {
		t.Fatalf("err %v calls %d streamed %q", err, len(f.requests()), got.String())
	}
	// Cancelling while it streams.
	ctx, cancel := context.WithCancel(context.Background())
	f2 := &fakeBackend{steps: []step{{deltas: []string{"a", "b"}, block: true}}}
	c.Backend = f2
	_, err = c.Chat(ctx, Request{Messages: msgs("x"), OnDelta: func(string) { cancel() }})
	if Code(err) != CodeCancelled {
		t.Fatalf("err %v", err)
	}
}

func TestEmptyAnswer(t *testing.T) {
	c := &Client{Backend: &fakeBackend{steps: []step{text("<think>only this</think>")}}}
	if _, err := c.Chat(context.Background(), Request{Messages: msgs("x")}); Code(err) != CodeEmpty {
		t.Fatalf("err %v", err)
	}
	if _, err := (&Client{}).Chat(context.Background(), Request{Messages: msgs("x"), Skill: SkillRewrite}); Code(err) != CodeNoRoute {
		t.Fatalf("no backend: %v", err)
	}
}

func TestJSONRepair(t *testing.T) {
	schema := Object(map[string]*Schema{
		"lines": Array(Object(map[string]*Schema{"id": String(), "text": String()}, "id", "text")).Count(2, 2),
	}, "lines")
	f := &fakeBackend{steps: []step{
		text("Sure! Here you go: no json"),
		text("```json\n{\"lines\":[{\"id\":\"1\",\"text\":\"a\"}]}\n```"),
		text(`{"lines":[{"id":"1","text":"a"},{"id":"2","text":"b"}]}`),
	}}
	c := &Client{Backend: f}
	var out struct {
		Lines []struct{ ID, Text string }
	}
	resp, err := c.JSON(context.Background(), Request{Messages: []Message{{Role: System, Content: "s"}, {Role: User, Content: "u"}}, Schema: schema}, &out, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Lines) != 2 || resp.Repairs != 2 || resp.Usage.TotalTokens != 54 || resp.Attempts != 3 {
		t.Fatalf("out %+v resp %+v", out, resp)
	}
	last := f.requests()[2].Messages
	if len(last) != 4 {
		t.Fatalf("repair turn has %d messages", len(last))
	}
	if fix := last[3].Content; !strings.Contains(fix, "/lines: must have at least 2 items") {
		t.Fatalf("repair prompt %q", fix)
	}
	// Out of repairs → schema error with the path.
	c.Backend = &fakeBackend{steps: []step{text(`{"lines":[]}`)}}
	_, err = c.JSON(context.Background(), Request{Messages: msgs("u"), Schema: schema}, nil, -1)
	var e *Error
	if !errors.As(err, &e) || e.Code != CodeSchema || e.Params["path"] != "/lines" || !errors.Is(err, core.ErrInvalid) {
		t.Fatalf("err %v", err)
	}
}

func TestThinkingStripped(t *testing.T) {
	if got := StripThinking("<think>\nhmm\n</think>\n\nAnswer"); got != "Answer" {
		t.Fatal(got)
	}
	if got := StripThinking("Answer <think>"); got != "Answer <think>" {
		t.Fatal(got)
	}
	var out strings.Builder
	f := &thinkFilter{emit: func(s string) { out.WriteString(s) }}
	for _, d := range []string{"<th", "ink>reason", "ing</thi", "nk>", "\n  ", "Hi", " you"} {
		f.write(d)
	}
	f.flush()
	if out.String() != "Hi you" {
		t.Fatalf("%q", out.String())
	}
	out.Reset()
	f.reset()
	for _, d := range []string{"<", "b>bold"} {
		f.write(d)
	}
	f.flush()
	if out.String() != "<b>bold" {
		t.Fatalf("%q", out.String())
	}
	// Gemma 4's thought channel, empty or not, split anywhere.
	if got := StripThinking("<|channel>thought\n<channel|>Answer"); got != "Answer" {
		t.Fatal(got)
	}
	out.Reset()
	f.reset()
	for _, d := range []string{"<|", "channel>thought\nhm", "m<chan", "nel|>", "Hi", " <think>"} {
		f.write(d)
	}
	f.flush()
	if out.String() != "Hi <think>" {
		t.Fatalf("%q", out.String())
	}
	out.Reset()
	f.reset()
	for _, d := range []string{"<", "|x"} {
		f.write(d)
	}
	f.flush()
	if out.String() != "<|x" {
		t.Fatalf("%q", out.String())
	}
}

func TestExtractJSONAndSchema(t *testing.T) {
	cases := map[string]string{
		`{"a":1}`:                          `{"a":1}`,
		"```json\n[1,2]\n```":              `[1,2]`,
		`Here: {"a":"}"} trailing`:         `{"a":"}"}`,
		"<think>{x}</think>\n{\"b\":true}": `{"b":true}`,
		`text [not json] {"ok":1}`:         `{"ok":1}`,
		"nothing":                          ``,
	}
	for in, want := range cases {
		if got := string(ExtractJSON(in)); got != want {
			t.Errorf("ExtractJSON(%q) = %q, want %q", in, got, want)
		}
	}
	s := Object(map[string]*Schema{
		"e":    Enum("a", "b"),
		"n":    Number().Range(0, 1),
		"i":    Integer(),
		"opt":  {Type: "string", Nullable: true},
		"list": Array(Boolean()).Count(1, -1),
	}, "e", "n", "list")
	errs := s.Validate(json.RawMessage(`{"e":"c","n":2,"i":1.5,"opt":null,"list":[],"x":1}`))
	var paths []string
	for _, e := range errs {
		paths = append(paths, e.Path)
	}
	if strings.Join(paths, ",") != "/e,/i,/list,/n,/" {
		t.Fatalf("%v", errs)
	}
	if errs := s.Validate(json.RawMessage(`{"e":"a","n":0.5,"list":[true],"opt":"x"}`)); len(errs) != 0 {
		t.Fatal(errs)
	}
	b, _ := json.Marshal(s)
	if !strings.Contains(string(b), `"additionalProperties":false`) || !strings.Contains(string(b), `"type":["string","null"]`) {
		t.Fatal(string(b))
	}
}

func TestEstimateTokens(t *testing.T) {
	if n := EstimateTokens("你好世界"); n != 4 {
		t.Fatal(n)
	}
	if n := EstimateTokens("abcdefgh"); n != 2 {
		t.Fatal(n)
	}
}
