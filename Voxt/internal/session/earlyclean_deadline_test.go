package session

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func waitEarlyDrainStarted(t *testing.T, early *earlyClean) time.Time {
	t.Helper()
	started := make(chan time.Time, 1)
	go func() {
		early.mu.Lock()
		for !early.closed {
			early.cond.Wait()
		}
		early.mu.Unlock()
		started <- time.Now()
	}()
	select {
	case at := <-started:
		return at
	case <-time.After(3 * time.Second):
		t.Fatal("early clean-up drain did not start")
		return time.Time{}
	}
}

func TestEarlyCleanWaitDefaultsToDisabled(t *testing.T) {
	h := newHarness(t, nil, nil, nil)
	if got := h.m.cfg.EarlyCleanWait; got != 0 {
		t.Fatalf("EarlyCleanWait default = %v, want disabled", got)
	}
	if ExperimentalEarlyCleanWait != 250*time.Millisecond {
		t.Fatalf("experimental wait = %v, want the former 250 ms value", ExperimentalEarlyCleanWait)
	}
}

func TestEarlyCleanTranscriptCoveredByCompletedSegment(t *testing.T) {
	model := newEarlyOrderedLLM()
	r := newEarlyUnitRun(t, model, nil)
	r.hearPiece("whole transcript")
	model.waitStarted(t, 0)

	r.early.mu.Lock()
	r.early.closed = true
	r.early.mu.Unlock()
	r.early.cond.Broadcast()
	model.finish(t, 0)
	waitEarlyWorker(t, r)

	got, _, err := r.cleanUpFirst("whole transcript")
	if err != nil || got != "整理一。" {
		t.Fatalf("cleanUpFirst = %q, %v; want the completed answer without degradation", got, err)
	}
}

func TestEarlyCleanDrainReusesOneInFlightSegment(t *testing.T) {
	for _, tc := range []struct {
		name        string
		wait        time.Duration
		releaseWait time.Duration
	}{
		{name: "default-disabled"},
		{name: "negative-disabled", wait: -time.Millisecond},
		{name: "bounded-enabled", wait: time.Second, releaseWait: 750 * time.Millisecond},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := newEarlyOrderedLLM()
			r := newEarlyUnitRun(t, model, nil)
			r.m.cfg.EarlyCleanWait = tc.wait
			if got := r.m.cfg.EarlyCleanWait; got != tc.wait {
				t.Fatalf("EarlyCleanWait = %v, want %v", got, tc.wait)
			}
			r.hearPiece("one")
			model.waitStarted(t, 0)
			result := make(chan struct {
				parts    []compose.Part
				answers  []string
				tail     string
				deadline bool
				ok       bool
			}, 1)
			go func() {
				parts, answers, tail, _, ok, deadline := r.early.drain("one")
				result <- struct {
					parts    []compose.Part
					answers  []string
					tail     string
					deadline bool
					ok       bool
				}{parts, answers, tail, deadline, ok}
			}()
			drainStart := waitEarlyDrainStarted(t, r.early)
			if tc.releaseWait > 0 {
				timer := time.NewTimer(tc.releaseWait)
				<-timer.C
				timer.Stop()
			}
			model.finish(t, 0)
			drained := <-result
			drainEnd := time.Now()
			if !drained.ok || drained.deadline || len(drained.parts) != 1 || len(drained.answers) != 1 || drained.answers[0] != "整理一。" || drained.tail != "" {
				t.Fatalf("drain %+v, want the in-flight segment reused without degradation", drained)
			}
			t.Logf("drain-wait=%s reused=1 discarded=0 exact-output=%q degraded=false", drainEnd.Sub(drainStart), compose.JoinParts(drained.parts, drained.answers)+drained.tail)
		})
	}
}

func TestEarlyCleanCancellationRacesWithWorkerCompletion(t *testing.T) {
	model := &earlyCleanCompletionRaceLLM{
		secondStarted: make(chan struct{}),
		releaseSecond: make(chan struct{}),
		firstFinished: make(chan struct{}),
	}
	r := newEarlyUnitRun(t, model, nil)
	r.m.cfg.EarlyCleanWait = time.Second
	r.hearPiece("one")
	r.hearPiece("one two")
	r.hearPiece("one two three")
	<-model.firstFinished
	<-model.secondStarted

	result := make(chan struct {
		parts    []compose.Part
		answers  []string
		tail     string
		ok       bool
		deadline bool
	}, 1)
	go func() {
		parts, answers, tail, _, ok, deadline := r.early.drain("one two three tail")
		result <- struct {
			parts    []compose.Part
			answers  []string
			tail     string
			ok       bool
			deadline bool
		}{parts, answers, tail, ok, deadline}
	}()
	_ = waitEarlyDrainStarted(t, r.early)
	// Releasing as drain closes worker intake races a successful response
	// against cancellation; either ordering must leave the same final prefix.
	close(model.releaseSecond)
	drained := <-result
	waitEarlyWorker(t, r)
	if !drained.ok || drained.deadline || len(drained.parts) != 1 || len(drained.answers) != 1 || drained.answers[0] != "整理一。" || drained.tail != " two three tail" {
		t.Fatalf("drain %+v, want only the first contiguous completed segment and raw suffix", drained)
	}
	if got := compose.JoinParts(drained.parts, drained.answers) + drained.tail; got != "整理一。 two three tail" {
		t.Errorf("raced output %q lost, duplicated, or reordered transcript text", got)
	}
	r.early.mu.Lock()
	defer r.early.mu.Unlock()
	if len(r.early.segs) != 1 {
		t.Errorf("worker published %d segments after finalization, want exactly the first", len(r.early.segs))
	}
}

func TestEarlyCleanDeadlineKeepsCompletedContiguousPrefix(t *testing.T) {
	model := newEarlyOrderedLLM()
	r := newEarlyUnitRun(t, model, nil)
	r.m.cfg.EarlyCleanWait = 25 * time.Millisecond
	r.hearPiece("one")
	r.hearPiece("one two")
	r.hearPiece("one two three")
	model.waitStarted(t, 0)
	model.finish(t, 0)
	model.waitStarted(t, 1)
	model.finish(t, 1)
	model.waitStarted(t, 2)
	stopAt := time.Now()
	asrFinalAt := time.Now()
	drainStart := time.Now()
	parts, answers, tail, failed, ok, deadline := r.early.drain("one two three tail")
	drainEnd := time.Now()
	if !ok || !deadline || failed != nil || len(parts) != 2 || len(answers) != 2 || answers[0] != "整理一。" || answers[1] != "整理二。" || tail != " three tail" {
		t.Fatalf("drain parts=%d answers=%q tail=%q failed=%v ok=%t deadline=%t", len(parts), answers, tail, failed, ok, deadline)
	}
	if got := compose.JoinParts(parts, answers) + tail; got != "整理一。整理二。 three tail" {
		t.Errorf("partial output %q lost, duplicated, or reordered transcript text", got)
	}
	waitEarlyWorker(t, r)
	if got := len(r.early.segs); got != 2 {
		t.Errorf("the unfinished segment published after finalization: got %d segments, want the two completed ones", got)
	}
	t.Logf("Stop=%s ASR-Final=%s drain=%s..%s reused=2 cancelled/discarded=1 post-Stop-LLM-requests=0", stopAt.Format(time.RFC3339Nano), asrFinalAt.Format(time.RFC3339Nano), drainStart.Format(time.RFC3339Nano), drainEnd.Format(time.RFC3339Nano))
}

type deadlineAssemblyLLM struct {
	mu             sync.Mutex
	calls          int
	firstAnswer    string
	secondStarted  chan struct{}
	secondCanceled chan struct{}
	releaseSecond  chan struct{}
}

func (model *deadlineAssemblyLLM) Chat(ctx context.Context, req llm.Request) (llm.Response, error) {
	model.mu.Lock()
	model.calls++
	call := model.calls
	model.mu.Unlock()
	if req.OnDelta != nil {
		return llm.Response{Text: "unexpected tail cleanup"}, nil
	}
	switch call {
	case 1:
		return llm.Response{Text: model.firstAnswer}, nil
	case 2:
		close(model.secondStarted)
		<-ctx.Done()
		close(model.secondCanceled)
		<-model.releaseSecond
		return llm.Response{Text: "stale second answer"}, nil
	default:
		return llm.Response{Text: "unexpected early cleanup"}, nil
	}
}

func TestEarlyCleanDeadlineAssemblesExactRecognizedSuffix(t *testing.T) {
	chain := []store.Term{
		{Term: "A", Aliases: []string{"raw"}, Enabled: true},
		{Term: "C", Aliases: []string{"A B"}, Enabled: true},
	}
	for _, tc := range []struct {
		name, first, stable, finalRaw, firstAnswer, want string
		terms                                            []store.Term
	}{
		{name: "English spaces", first: "hello", stable: "hello world", finalRaw: "hello world today", firstAnswer: "Hello,", want: "Hello, world today"},
		{name: "Chinese without spaces", first: "你好", stable: "你好世界", finalRaw: "你好世界今天", firstAnswer: "你好，", want: "你好，世界今天"},
		{name: "mixed Chinese and English", first: "你好", stable: "你好 world", finalRaw: "你好 world today", firstAnswer: "你好", want: "你好 world today"},
		{name: "punctuation boundary", first: "hello,", stable: "hello, world", finalRaw: "hello, world!", firstAnswer: "Hello,", want: "Hello, world!"},
		{name: "dictionary replacement is not repeated", first: "prefix", stable: "prefix raw", finalRaw: "prefix raw B tail", firstAnswer: "Prefix,", want: "Prefix, A B tail", terms: chain},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &deadlineAssemblyLLM{
				firstAnswer:    tc.firstAnswer,
				secondStarted:  make(chan struct{}),
				secondCanceled: make(chan struct{}),
				releaseSecond:  make(chan struct{}),
			}
			r := newEarlyUnitRun(t, model, tc.terms)
			r.m.cfg.EarlyCleanWait = 25 * time.Millisecond
			r.hearPiece(tc.first)
			r.hearPiece(tc.stable)
			<-model.secondStarted // the first answer is committed; the second is held

			final := r.spoken(tc.finalRaw) // run.main applies dictionary aliases before transcribe
			got, _, err := r.cleanUpFirst(final)
			if !errors.Is(err, errEarlyCleanDeadline) {
				t.Fatalf("cleanUpFirst error = %v, want the bounded deadline", err)
			}
			if got != tc.want {
				t.Errorf("cleanUpFirst = %q, want exact prefix/suffix %q", got, tc.want)
			}
			<-model.secondCanceled
			close(model.releaseSecond)
			waitEarlyWorker(t, r)
			r.early.mu.Lock()
			defer r.early.mu.Unlock()
			if len(r.early.segs) != 1 {
				t.Errorf("late segment changed finalized prefix: have %d segments, want 1", len(r.early.segs))
			}
		})
	}
}

func TestEarlyCleanDeadlineMarksPartialOutputDegraded(t *testing.T) {
	model := &cancellationIgnoringEarlyLLM{
		requestStarted:  make(chan struct{}),
		requestCanceled: make(chan struct{}),
		releaseRequest:  make(chan struct{}),
	}
	finalAt := make(chan time.Time, 1)
	stream := &fakeStreamer{final: "one", pieces: []string{"one"}, finalAt: func(at time.Time) { finalAt <- at }}
	h := newHarness(t, stream, model, func(config *Config, _ *settings.Settings) {
		config.EarlyCleanWait = 25 * time.Millisecond
	})
	h.audio.notifyAt, h.audio.notified = 3, make(chan struct{})
	id := h.start()
	<-model.requestStarted
	<-h.audio.notified
	h.m.mu.Lock()
	r, _ := h.m.find(id)
	h.m.mu.Unlock()
	stopAt := time.Now()
	h.m.Stop(id)
	finalTimestamp := <-finalAt
	drainStart := waitEarlyDrainStarted(t, r.early)
	end := h.rec.wait(t)
	drainEnd := time.Now()
	if !finalTimestamp.After(stopAt) {
		t.Fatalf("ASR Final at %s did not follow Stop at %s", finalTimestamp, stopAt)
	}
	if end.State != Done || end.Result != "one" || end.Raw != "one" || end.Problem == nil || end.Problem.Code != ProblemLLM || !end.Problem.Degraded {
		t.Fatalf("end %+v, want the raw suffix with an explicit degraded-cleanup problem", end)
	}
	<-model.requestCanceled
	canceledObservedAt := time.Now()
	select {
	case <-r.early.workerDone:
		t.Fatal("the cancellation-ignoring request unexpectedly ended before release")
	default:
	}
	close(model.releaseRequest)
	waitEarlyWorker(t, r)
	workerDoneAt := time.Now()
	current := h.m.Current()
	if current.Result != "one" || current.Problem == nil || !current.Problem.Degraded {
		t.Errorf("late LLM completion changed the finalized session: %+v", current)
	}
	if len(r.early.segs) != 0 {
		t.Errorf("the cancelled request published %d stale segments", len(r.early.segs))
	}
	model.mu.Lock()
	requestCount := model.requests
	model.mu.Unlock()
	t.Logf("bounded Stop-to-ASR-Final=%s Stop-to-delivery=%s drain-wait<=%s cancellation-after-drain-start=%s worker-done-after-release=%s reused=0 cancelled/discarded=1 requests-after-Stop=%d exact-output=%q degraded=true", finalTimestamp.Sub(stopAt), drainEnd.Sub(stopAt), drainEnd.Sub(drainStart), canceledObservedAt.Sub(drainStart), workerDoneAt.Sub(canceledObservedAt), requestCount-1, end.Result)
}

func TestEarlyCleanDefaultWaitDoesNotDegradeLateCompletion(t *testing.T) {
	model := newEarlyOrderedLLM()
	finalAt := make(chan time.Time, 1)
	stream := &fakeStreamer{final: "one", pieces: []string{"one"}, finalAt: func(at time.Time) { finalAt <- at }}
	h := newHarness(t, stream, model, nil)
	h.audio.notifyAt, h.audio.notified = 3, make(chan struct{})
	id := h.start()
	model.waitStarted(t, 0)
	<-h.audio.notified
	h.m.mu.Lock()
	r, _ := h.m.find(id)
	h.m.mu.Unlock()
	stopAt := time.Now()
	h.m.Stop(id)
	asrFinalAt := <-finalAt
	drainStart := waitEarlyDrainStarted(t, r.early)
	model.finish(t, 0)
	end := h.rec.wait(t)
	deliveredAt := time.Now()
	if end.State != Done || end.Result != "整理一。" || end.Raw != "one" || end.Problem != nil {
		t.Fatalf("end %+v, want the normally cleaned answer without a default degradation", end)
	}
	t.Logf("default Stop-to-ASR-Final=%s Stop-to-delivery=%s drain-start=%s reused=1 discarded=0 exact-output=%q degraded=false", asrFinalAt.Sub(stopAt), deliveredAt.Sub(stopAt), drainStart.Format(time.RFC3339Nano), end.Result)
}

type cancellationIgnoringEarlyLLM struct {
	requestStarted  chan struct{}
	requestCanceled chan struct{}
	releaseRequest  chan struct{}
	fallbackStarted chan struct{}
	mu              sync.Mutex
	requests        int
}

type cancellationHonoringEarlyLLM struct {
	requestStarted  chan struct{}
	requestCanceled chan struct{}
	fallbackStarted chan struct{}
}

type earlyCleanCompletionRaceLLM struct {
	mu            sync.Mutex
	calls         int
	firstFinished chan struct{}
	secondStarted chan struct{}
	releaseSecond chan struct{}
}

func (model *earlyCleanCompletionRaceLLM) Chat(ctx context.Context, _ llm.Request) (llm.Response, error) {
	model.mu.Lock()
	model.calls++
	call := model.calls
	model.mu.Unlock()
	switch call {
	case 1:
		close(model.firstFinished)
		return llm.Response{Text: "整理一。"}, nil
	case 2:
		close(model.secondStarted)
		select {
		case <-model.releaseSecond:
			return llm.Response{Text: "整理二。"}, nil
		case <-ctx.Done():
			return llm.Response{}, ctx.Err()
		}
	default:
		return llm.Response{Text: "unexpected"}, nil
	}
}

func (model *cancellationIgnoringEarlyLLM) Chat(ctx context.Context, request llm.Request) (llm.Response, error) {
	model.mu.Lock()
	model.requests++
	requestNumber := model.requests
	model.mu.Unlock()
	if request.OnDelta != nil {
		if model.fallbackStarted != nil {
			close(model.fallbackStarted)
		}
		return llm.Response{Text: "fallback"}, nil
	}
	if requestNumber != 1 {
		return llm.Response{Text: "unexpected early request"}, nil
	}
	close(model.requestStarted)
	<-ctx.Done()
	close(model.requestCanceled)
	<-model.releaseRequest
	return llm.Response{Text: "stale early answer"}, nil
}

func (model *cancellationHonoringEarlyLLM) Chat(ctx context.Context, request llm.Request) (llm.Response, error) {
	if request.OnDelta != nil {
		close(model.fallbackStarted)
		return llm.Response{Text: "fallback"}, nil
	}
	close(model.requestStarted)
	<-ctx.Done()
	close(model.requestCanceled)
	return llm.Response{}, ctx.Err()
}

func TestEarlyCleanPrefixMismatchCancelsHonoringLLM(t *testing.T) {
	model := &cancellationHonoringEarlyLLM{
		requestStarted:  make(chan struct{}),
		requestCanceled: make(chan struct{}),
		fallbackStarted: make(chan struct{}),
	}
	r := newEarlyUnitRun(t, model, nil)
	r.hearPiece("stable")
	<-model.requestStarted

	type cleanupResult struct {
		text string
		err  error
	}
	result := make(chan cleanupResult, 1)
	go func() {
		text, _, err := r.cleanUpFirst("changed")
		result <- cleanupResult{text: text, err: err}
	}()
	<-model.requestCanceled
	<-model.fallbackStarted
	outcome := <-result
	if outcome.err != nil || outcome.text != "fallback" {
		t.Fatalf("whole-text cleanup = %q, %v; want fallback", outcome.text, outcome.err)
	}
	waitEarlyWorker(t, r)
}

func TestEarlyCleanPrefixMismatchDoesNotWaitForCancellationIgnoringLLM(t *testing.T) {
	model := &cancellationIgnoringEarlyLLM{
		requestStarted:  make(chan struct{}),
		requestCanceled: make(chan struct{}),
		releaseRequest:  make(chan struct{}),
		fallbackStarted: make(chan struct{}),
	}
	r := newEarlyUnitRun(t, model, nil)
	if r.m.cfg.EarlyCleanWait != 0 {
		t.Fatalf("EarlyCleanWait = %v, want disabled by default", r.m.cfg.EarlyCleanWait)
	}
	r.hearPiece("stable")
	<-model.requestStarted

	type cleanupResult struct {
		text string
		err  error
	}
	result := make(chan cleanupResult, 1)
	go func() {
		text, _, err := r.cleanUpFirst("changed")
		result <- cleanupResult{text: text, err: err}
	}()
	<-model.requestCanceled
	<-model.fallbackStarted
	select {
	case outcome := <-result:
		if outcome.err != nil {
			t.Fatal(outcome.err)
		}
		if outcome.text != "fallback" {
			t.Errorf("result %q, want whole-text fallback", outcome.text)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("whole-text cleanup did not finish while the canceled early request was held")
	}
	select {
	case <-r.early.workerDone:
		t.Fatal("the cancellation-ignoring request unexpectedly finished before release")
	default:
	}
	close(model.releaseRequest)
	waitEarlyWorker(t, r)
	r.early.mu.Lock()
	defer r.early.mu.Unlock()
	if len(r.early.segs) != 0 {
		t.Errorf("the canceled request published %d stale segments", len(r.early.segs))
	}
}

func TestEarlyCleanPrefixMismatchAfterCompletedSegmentUsesWholeText(t *testing.T) {
	model := &earlyFallbackLLM{secondStarted: make(chan struct{})}
	r := newEarlyUnitRun(t, model, nil)
	r.hearPiece("one")
	r.hearPiece("one two")
	<-model.secondStarted // the first segment has completed; the second is in flight

	result, _, err := r.cleanUpFirst("changed transcript")
	if err != nil {
		t.Fatal(err)
	}
	if result != "fallback" {
		t.Fatalf("result %q, want whole-text cleanup", result)
	}
	model.mu.Lock()
	defer model.mu.Unlock()
	if !errors.Is(model.secondErr, context.Canceled) {
		t.Errorf("second early request ended with %v, want cancellation", model.secondErr)
	}
}
