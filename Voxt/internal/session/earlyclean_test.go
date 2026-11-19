package session

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// earlySeg is a stretch of dictation of about 290 characters, distinct
// per week: four of them are over the clean-up's piece limit together.
func earlySeg(week, quarter string) string {
	return strings.Repeat(fmt.Sprintf("嗯，那个，我们周%s下午三点开个会，讨论一下第%s季度的预算。", week, quarter), 10)
}

// earlyFixture is a dictation whose speech model finalised three
// stretches while recording went on (asr.EventPiece) and answers with
// the fourth at Stop.
type earlyFixture struct {
	segs   []string
	stable []string // the EventPiece texts, cumulative
	full   string
}

func newEarlyFixture(t *testing.T) earlyFixture {
	t.Helper()
	segs := []string{earlySeg("一", "一"), earlySeg("二", "二"), earlySeg("三", "三"), earlySeg("四", "四")}
	f := earlyFixture{segs: segs, full: strings.Join(segs, "")}
	for i, s := range segs {
		if n := utf8.RuneCountInString(s); n < 200 || n > 300 {
			t.Fatalf("segment %d has %d characters", i+1, n)
		}
		f.stable = append(f.stable, strings.Join(segs[:i+1], ""))
	}
	f.stable = f.stable[:3]
	if n := len(compose.Pieces(f.full)); n != 4 {
		t.Fatalf("the whole text is %d clean-up pieces, want 4 (the revert makes 4 requests after Stop)", n)
	}
	return f
}

// waitEarly waits until the language model answered n requests while
// the recording still goes on; it fails when they do not come (with the
// early clean-up reverted, none do before Stop).
func waitEarly(t *testing.T, model *fakeLLM, n int) {
	t.Helper()
	deadline := time.Now().Add(30 * time.Second)
	for model.answeredCount() < n {
		if time.Now().After(deadline) {
			t.Fatalf("only %d requests answered while recording, want %d", model.answeredCount(), n)
		}
		time.Sleep(20 * time.Millisecond)
	}
}

// TestEarlyCleanWhileRecording: on a stream that finalises stretches
// while recording continues, each finished stretch is cleaned by the
// language model right away: at Stop only the tail is left, the wait
// after Stop is about one request, not one per stretch, and the text
// is the stretches' answers in order.
func TestEarlyCleanWhileRecording(t *testing.T) {
	f := newEarlyFixture(t)
	const delay = 3 * time.Second
	model := &fakeLLM{texts: []string{"整理一。", "整理二。", "整理三。", "整理四。"}, pause: delay}
	h := newHarness(t, &fakeStreamer{final: f.full, pieces: f.stable}, model, func(c *Config, s *settings.Settings) {
		c.LLMTimeout = 30 * time.Second
	})
	id := h.start()
	waitEarly(t, model, 3)
	if n := len(model.requests()); n != 3 {
		t.Fatalf("%d requests before Stop, want the 3 finished stretches'", n)
	}
	stop := time.Now()
	h.m.Stop(id)
	end := h.rec.wait(t)
	waited := time.Since(stop)
	if waited >= 2*delay {
		t.Errorf("waited %v after Stop, want about one request (%v), not one per stretch", waited.Round(time.Millisecond), delay)
	}
	if waited < delay/2 {
		t.Errorf("waited only %v after Stop: the last stretch was not cleaned", waited.Round(time.Millisecond))
	}
	if want := "整理一。整理二。整理三。整理四。"; end.Result != want || end.Raw != f.full {
		t.Errorf("result %q, raw %q, want %q", end.Result, end.Raw, want)
	}
	if end.Problem != nil {
		t.Errorf("problem %+v", end.Problem)
	}
	reqs := model.requests()
	if len(reqs) != 4 {
		t.Fatalf("%d requests", len(reqs))
	}
	for i, req := range reqs {
		m := req.Messages
		if len(m) != 4 || m[3].Content != compose.UserMessage(f.segs[i]) {
			t.Errorf("request %d cleaned %q, want segment %d", i+1, m[len(m)-1].Content, i+1)
		}
	}
}

// TestEarlyCleanFailure: a stretch whose clean-up fails keeps its text
// as recognised, the other stretches stay cleaned, with a warning.
func TestEarlyCleanFailure(t *testing.T) {
	f := newEarlyFixture(t)
	model := &fakeLLM{texts: []string{"整理一。", "不用。", "整理三。", "整理四。"}, err: errors.New("503"), failOn: 2, pause: time.Millisecond}
	h := newHarness(t, &fakeStreamer{final: f.full, pieces: f.stable}, model, nil)
	id := h.start()
	waitEarly(t, model, 2) // the second stretch's request failed
	h.speak(id, 5)
	end := h.rec.wait(t)
	if want := "整理一。" + f.segs[1] + "整理三。整理四。"; end.Result != want {
		t.Errorf("result %q, want %q", end.Result, want)
	}
	if end.Problem == nil || end.Problem.Code != ProblemLLM || !end.Problem.Degraded {
		t.Errorf("problem %+v, want a degraded language-model one", end.Problem)
	}
	if n := len(model.requests()); n != 4 {
		t.Errorf("%d requests, want 4", n)
	}
}

// TestEarlyCleanFallback: when the final text does not start with the
// stretches reported while recording (their join broke), the whole
// text is cleaned at Stop, as without the events.
func TestEarlyCleanFallback(t *testing.T) {
	f := newEarlyFixture(t)
	// The final text differs from what the stretches promised (a live
	// result of its own, say): none of the early answers applies.
	full := f.segs[0] + "改过的第二段。" + f.segs[2] + f.segs[3]
	model := &fakeLLM{text: "清理。"}
	h := newHarness(t, &fakeStreamer{final: full, pieces: f.stable}, model, nil)
	id := h.start()
	waitEarly(t, model, 3)
	h.speak(id, 5)
	end := h.rec.wait(t)
	parts := compose.Pieces(full)
	var want string
	for range parts {
		want += "清理。"
	}
	// JoinParts keeps the pieces' separators; the fixture's text has none.
	if end.Result != want {
		t.Errorf("result %q, want %q", end.Result, want)
	}
	reqs := model.requests()
	if len(reqs) != 3+len(parts) {
		t.Fatalf("%d requests, want the 3 early ones and %d at Stop", len(reqs), len(parts))
	}
	last := reqs[len(reqs)-1].Messages
	if last[len(last)-1].Content != compose.UserMessage(parts[len(parts)-1].Text) {
		t.Errorf("the last request did not clean the whole text's last piece")
	}
}

func newEarlyUnitRun(t *testing.T, model LLM, terms []store.Term) *run {
	t.Helper()
	h := newHarness(t, nil, model, nil)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	s := settings.Defaults()
	s.Features.Transcribe.Enhance = true
	s.Dictionary.Enabled = true
	s.Dictionary.Glossary = true
	r := &run{
		m: h.m, id: "early-test", req: StartRequest{Kind: store.KindTranscribe}, settings: s,
		ctx: ctx, cancel: cancel, svcResolved: make(chan struct{}),
		svc: Services{LLM: model, LLMInfo: Model{ID: "deepseek"}}, terms: terms,
	}
	r.llmCtx, r.llmStop = context.WithCancel(ctx)
	r.startScene()
	close(r.svcResolved)
	return r
}

func TestEarlyCleanShortInputThinking(t *testing.T) {
	full := strings.Repeat("a", 1000)
	model := &fakeLLM{text: "clean"}
	r := newEarlyUnitRun(t, model, nil)
	r.settings.Providers.LLM = []settings.Provider{{
		ID: "deepseek", Generation: settings.Generation{ThinkingOffUnder: 200},
	}}
	r.hearPiece(full[:100])
	_, _, err := r.cleanUpFirst(full)
	if err != nil {
		t.Fatal(err)
	}
	reqs := model.requests()
	if len(reqs) != 4 {
		t.Fatalf("%d requests, want the 4 whole-text requests", len(reqs))
	}
	for i, req := range reqs {
		if req.InputChars != len([]rune(full)) || req.OnDelta == nil {
			t.Errorf("request %d has InputChars %d and streaming %t, want 1000 and whole-text streaming", i+1, req.InputChars, req.OnDelta != nil)
		}
	}
}

func TestEarlyCleanThinkingCountsReplacedText(t *testing.T) {
	model := &fakeLLM{text: "clean"}
	r := newEarlyUnitRun(t, model, []store.Term{{Term: "X", Aliases: []string{"aaaaaaaaaaaaaaaaaaaa"}, Enabled: true}})
	r.settings.Providers.LLM = []settings.Provider{{
		ID: "deepseek", Generation: settings.Generation{ThinkingOffUnder: 20},
	}}
	r.hearPiece("aaaaaaaaaaaaaaaaaaaa")
	_, _, err := r.cleanUpFirst("X")
	if err != nil {
		t.Fatal(err)
	}
	reqs := model.requests()
	if len(reqs) != 1 || reqs[0].InputChars != 1 || reqs[0].OnDelta == nil {
		t.Fatalf("requests %+v, want one whole-text request with the replaced length 1", reqs)
	}
}

func TestEarlyCleanLargeGlossary(t *testing.T) {
	terms := make([]store.Term, 31)
	for i := range terms {
		terms[i] = store.Term{Term: fmt.Sprintf("Term%02d", i), Enabled: true}
	}
	first := strings.Repeat("a", 295) + "Term00"
	full := first + strings.Repeat("b", 293) + "Term30"
	model := &fakeLLM{text: "clean"}
	r := newEarlyUnitRun(t, model, terms)
	r.hearPiece(first)
	_, _, err := r.cleanUpFirst(full)
	if err != nil {
		t.Fatal(err)
	}
	reqs := model.requests()
	if len(reqs) != 2 {
		t.Fatalf("%d requests, want the 2 whole-text requests", len(reqs))
	}
	for i, req := range reqs {
		message := req.Messages[len(req.Messages)-1].Content
		if req.InputChars != 600 || !strings.Contains(message, "- Term00") || !strings.Contains(message, "- Term30") {
			t.Errorf("request %d lacks the whole-text length or glossary: InputChars %d, message %q", i+1, req.InputChars, message)
		}
	}
}

func TestEarlyCleanRequestCountFollowsSpeechStretches(t *testing.T) {
	first := strings.Repeat("a", 301)
	full := first + strings.Repeat("b", 299)
	model := &fakeLLM{text: "clean"}
	r := newEarlyUnitRun(t, model, nil)
	r.hearPiece(first)
	_, _, err := r.cleanUpFirst(full)
	if err != nil {
		t.Fatal(err)
	}
	reqs := model.requests()
	if len(reqs) != 3 {
		t.Fatalf("%d requests, want 2 for the first stretch and 1 for the tail", len(reqs))
	}
	if reqs[0].InputChars != 301 || reqs[1].InputChars != 301 || reqs[2].InputChars != 600 {
		t.Errorf("request lengths are %d, %d, %d, want 301, 301, 600", reqs[0].InputChars, reqs[1].InputChars, reqs[2].InputChars)
	}
}

type earlyFallbackLLM struct {
	mu            sync.Mutex
	calls         int
	answered      chan struct{}
	secondStarted chan struct{}
	secondErr     error
	thirdStarted  bool
}

func (f *earlyFallbackLLM) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	if req.OnDelta != nil {
		return llm.Response{Text: "fallback"}, nil
	}
	f.mu.Lock()
	f.calls++
	call := f.calls
	f.mu.Unlock()
	switch call {
	case 1:
		if f.answered != nil {
			close(f.answered)
		}
		return llm.Response{Text: "early one"}, nil
	case 2:
		close(f.secondStarted)
		<-ctx.Done()
		f.mu.Lock()
		f.secondErr = ctx.Err()
		f.mu.Unlock()
		return llm.Response{}, ctx.Err()
	default:
		f.mu.Lock()
		f.thirdStarted = true
		f.mu.Unlock()
		return llm.Response{Text: "early three"}, nil
	}
}

func TestEarlyCleanFallbackCancelsBacklog(t *testing.T) {
	model := &earlyFallbackLLM{secondStarted: make(chan struct{})}
	r := newEarlyUnitRun(t, model, nil)
	r.hearPiece("one")
	r.hearPiece("one two")
	r.hearPiece("one two three")
	<-model.secondStarted
	result, _, err := r.cleanUpFirst("one changed three")
	if err != nil {
		t.Fatal(err)
	}
	if result != "fallback" {
		t.Errorf("result %q, want the whole-text fallback", result)
	}
	model.mu.Lock()
	defer model.mu.Unlock()
	if !errors.Is(model.secondErr, context.Canceled) {
		t.Errorf("the discarded request ended with %v, want cancellation", model.secondErr)
	}
	if model.thirdStarted {
		t.Error("the queued third request started after the final-text mismatch")
	}
}

func waitEarlyWorker(t *testing.T, r *run) {
	t.Helper()
	select {
	case <-r.early.workerDone:
	case <-time.After(3 * time.Second):
		t.Fatal("the early clean-up worker did not stop")
	}
}

func TestEarlyCleanCancelStopsWorker(t *testing.T) {
	model := &earlyFallbackLLM{answered: make(chan struct{})}
	r := newEarlyUnitRun(t, model, nil)
	r.hearPiece("one")
	<-model.answered
	r.cancel()
	waitEarlyWorker(t, r)
}

func TestEarlyCleanBrokenPieceCancelsBacklog(t *testing.T) {
	model := &earlyFallbackLLM{secondStarted: make(chan struct{})}
	r := newEarlyUnitRun(t, model, nil)
	r.hearPiece("one")
	r.hearPiece("one two")
	r.hearPiece("one two three")
	<-model.secondStarted
	r.hearPiece("changed")
	waitEarlyWorker(t, r)
	model.mu.Lock()
	defer model.mu.Unlock()
	if !errors.Is(model.secondErr, context.Canceled) {
		t.Errorf("the discarded request ended with %v, want cancellation", model.secondErr)
	}
	if model.thirdStarted {
		t.Error("the queued third request started after the broken event")
	}
}
