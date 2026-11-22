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
	timeout := time.After(30 * time.Second)
	for {
		answered := model.answerSignal() // taken before the count, so no answer slips between them
		if model.answeredCount() >= n {
			return
		}
		select {
		case <-answered:
		case <-timeout:
			t.Fatalf("only %d requests answered while recording, want %d", model.answeredCount(), n)
		}
	}
}

type earlyOrderedLLM struct {
	mu       sync.Mutex
	calls    int
	started  [4]chan struct{}
	finished [4]chan struct{}
	release  [4]chan struct{}
	reqs     []llm.Request
}

func newEarlyOrderedLLM() *earlyOrderedLLM {
	model := &earlyOrderedLLM{}
	for i := range model.started {
		model.started[i] = make(chan struct{})
		model.finished[i] = make(chan struct{})
		model.release[i] = make(chan struct{})
	}
	return model
}

func (m *earlyOrderedLLM) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	m.mu.Lock()
	m.calls++
	call := m.calls
	m.reqs = append(m.reqs, req)
	m.mu.Unlock()
	if call > len(m.started) {
		return llm.Response{Text: "extra"}, nil
	}
	close(m.started[call-1])
	defer close(m.finished[call-1])
	select {
	case <-m.release[call-1]:
		return llm.Response{Text: []string{"整理一。", "整理二。", "整理三。", "整理四。"}[call-1]}, nil
	case <-ctx.Done():
		return llm.Response{}, ctx.Err()
	}
}

func (m *earlyOrderedLLM) finish(t *testing.T, call int) {
	t.Helper()
	close(m.release[call])
}

func (m *earlyOrderedLLM) waitStarted(t *testing.T, call int) {
	t.Helper()
	select {
	case <-m.started[call]:
	case <-time.After(3 * time.Second):
		t.Fatalf("request %d did not start", call+1)
	}
}

func (m *earlyOrderedLLM) waitFinished(t *testing.T, call int) {
	t.Helper()
	select {
	case <-m.finished[call]:
	case <-time.After(3 * time.Second):
		t.Fatalf("request %d did not finish", call+1)
	}
}

func (m *earlyOrderedLLM) requests() []llm.Request {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]llm.Request(nil), m.reqs...)
}

// TestEarlyCleanWhileRecording: on a stream that finalises stretches
// while recording continues, each finished stretch is cleaned by the
// language model right away: at Stop only the tail is left, the wait
// after Stop is about one request, not one per stretch, and the text
// is the stretches' answers in order.
func TestEarlyCleanWhileRecording(t *testing.T) {
	f := newEarlyFixture(t)
	model := newEarlyOrderedLLM()
	h := newHarness(t, &fakeStreamer{final: f.full, pieces: f.stable}, model, nil)
	id := h.start()
	model.finish(t, 0)
	model.waitStarted(t, 1)
	model.finish(t, 1)
	model.waitStarted(t, 2)
	model.finish(t, 2)
	h.m.Stop(id)
	model.waitStarted(t, 3)
	model.finish(t, 3)
	end := h.rec.wait(t)
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
		if len(m) != 2 || m[1].Content != compose.UserMessage(f.segs[i]) { // system and the segment: Transcribe sends no example (#4244)
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

func TestEarlyCleanLeavesUnfinishedPartsForStop(t *testing.T) {
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
	if reqs[0].InputChars != 301 || reqs[1].InputChars != 600 || reqs[2].InputChars != 600 {
		t.Errorf("request lengths are %d, %d, %d, want one early part and two whole-text tail requests", reqs[0].InputChars, reqs[1].InputChars, reqs[2].InputChars)
	}
}

type earlyFallbackLLM struct {
	mu            sync.Mutex
	calls         int
	answered      chan struct{}
	secondStarted chan struct{}
	secondRelease chan struct{}
	secondErr     error
	thirdStarted  bool
	tailStarted   chan struct{}
	tailRelease   chan struct{}
	tailOnce      sync.Once
}

func (f *earlyFallbackLLM) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	if req.OnDelta != nil {
		if f.tailStarted != nil {
			f.tailOnce.Do(func() { close(f.tailStarted) })
			<-f.tailRelease
		}
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
		if f.secondRelease != nil {
			select {
			case <-f.secondRelease:
				return llm.Response{Text: "early two"}, nil
			case <-ctx.Done():
			}
		} else {
			<-ctx.Done()
		}
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

func TestEarlyCleanMatchingFinalCancelsBacklog(t *testing.T) {
	model := &earlyFallbackLLM{secondStarted: make(chan struct{})}
	r := newEarlyUnitRun(t, model, nil)
	r.m.cfg.LLMBase, r.m.cfg.LLMStall, r.m.cfg.LLMTimeout = 50*time.Millisecond, 50*time.Millisecond, 50*time.Millisecond
	r.hearPiece("one")
	r.hearPiece("one two")
	r.hearPiece("one two three")
	<-model.secondStarted
	result, _, err := r.cleanUpFirst("one two three tail")
	if err != nil {
		t.Fatal(err)
	}
	if result != "early one fallback" {
		t.Errorf("result %q, want the completed early answer and the fallback tail", result)
	}
	model.mu.Lock()
	calls := model.calls
	model.mu.Unlock()
	if calls != 2 {
		t.Errorf("%d requests, want the first and the cancelled second; the queued early request must not run", calls)
	}
}

func TestEarlyCleanMatchingFinalSkipsRemainingParts(t *testing.T) {
	model := &earlyFallbackLLM{secondStarted: make(chan struct{}), secondRelease: make(chan struct{})}
	r := newEarlyUnitRun(t, model, nil)
	lastJob := strings.Repeat("a", 301)
	full := "one " + lastJob + " tail"
	r.hearPiece("one")
	r.hearPiece("one " + lastJob)
	<-model.secondStarted
	r.early.mu.Lock()
	r.early.closed = true
	r.early.mu.Unlock()
	close(model.secondRelease)
	waitEarlyWorker(t, r)
	parts, answers, tail, _, ok := r.early.drain(full)
	if !ok || len(parts) != 1 || len(answers) != 1 || answers[0] != "early one" {
		t.Fatalf("early parts = %d, answers = %q, ok = %v; want only the completed first job", len(parts), answers, ok)
	}
	if want := " " + lastJob + " tail"; tail != want {
		t.Errorf("tail = %q, want the unfinished job and tail %q", tail, want)
	}
	model.mu.Lock()
	calls := model.calls
	model.mu.Unlock()
	if calls != 2 {
		t.Errorf("%d requests, want only the completed job and the active part", calls)
	}
}

func TestEarlyCleanMatchingFinalSkipAndRethink(t *testing.T) {
	for _, tc := range []struct {
		name string
		act  func(*Manager, string)
	}{
		{name: "skip", act: func(m *Manager, id string) { m.SkipLLM(id) }},
		{name: "rethink", act: func(m *Manager, id string) { m.Rethink(id, &llm.Thinking{Mode: llm.ThinkOff}) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &earlyFallbackLLM{secondStarted: make(chan struct{}), tailStarted: make(chan struct{}), tailRelease: make(chan struct{})}
			h := newHarness(t, &fakeStreamer{final: "one two three tail", pieces: []string{"one", "one two", "one two three"}}, model, nil)
			h.m.cfg.LLMBase, h.m.cfg.LLMStall, h.m.cfg.LLMTimeout = 50*time.Millisecond, 50*time.Millisecond, 50*time.Millisecond
			id := h.start()
			<-model.secondStarted
			h.speak(id, 3)
			h.m.Stop(id)
			<-model.tailStarted
			tc.act(h.m, id)
			close(model.tailRelease)
			end := h.rec.wait(t)
			if tc.name == "skip" && end.Result != "one two three tail" {
				t.Errorf("skip end %+v, want the raw transcript", end)
			}
			if tc.name == "rethink" && (end.Result != "fallback" || end.Thinking == nil || end.Thinking.Mode != llm.ThinkOff) {
				t.Errorf("rethink end %+v", end)
			}
		})
	}
}

func TestEarlyCleanDrainSkipAndRethink(t *testing.T) {
	for _, tc := range []struct {
		name string
		act  func(*Manager, string)
	}{
		{name: "skip", act: func(m *Manager, id string) { m.SkipLLM(id) }},
		{name: "rethink", act: func(m *Manager, id string) { m.Rethink(id, &llm.Thinking{Mode: llm.ThinkOff}) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := newEarlyOrderedLLM()
			h := newHarness(t, &fakeStreamer{final: "one tail", pieces: []string{"one"}}, model, nil)
			h.m.cfg.LLMTimeout = 10 * time.Second
			id := h.start()
			model.waitStarted(t, 0)
			h.speak(id, 3)
			h.rec.waitFor(t, "early cleanup draining", func(s Snapshot) bool { return s.State == Composing && s.CanSkip })
			tc.act(h.m, id)
			model.waitFinished(t, 0)
			if tc.name == "rethink" {
				model.waitStarted(t, 2)
				model.finish(t, 2)
			}
			end := h.rec.wait(t)
			if tc.name == "skip" && end.Result != "one tail" {
				t.Errorf("skip end %+v, want the raw transcript", end)
			}
			if tc.name == "rethink" && (end.Result != "整理三。" || end.Thinking == nil || end.Thinking.Mode != llm.ThinkOff) {
				t.Errorf("rethink end %+v", end)
			}
		})
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
