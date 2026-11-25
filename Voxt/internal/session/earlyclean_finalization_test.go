package session

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
)

func waitEarlyDrainStarted(t *testing.T, early *earlyClean) {
	t.Helper()
	started := make(chan struct{})
	go func() {
		early.mu.Lock()
		for !early.closed {
			early.cond.Wait()
		}
		early.mu.Unlock()
		close(started)
	}()
	select {
	case <-started:
	case <-time.After(3 * time.Second):
		t.Fatal("early clean-up drain did not start")
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

	output := r.cleanUpFirst("whole transcript")
	if output.err != nil || output.text != "整理一。" {
		t.Fatalf("cleanUpFirst = %q, %v; want the completed answer", output.text, output.err)
	}
}

func TestEarlyCleanMatchingPrefixWaitsForInFlightRequest(t *testing.T) {
	model := newEarlyOrderedLLM()
	h := newHarness(t, &fakeStreamer{final: "one", pieces: []string{"one"}}, model, nil)
	h.audio.notifyAt, h.audio.notified = 3, make(chan struct{})
	id := h.start()
	model.waitStarted(t, 0)
	<-h.audio.notified
	h.m.Stop(id)
	h.m.mu.Lock()
	r, _ := h.m.find(id)
	h.m.mu.Unlock()
	waitEarlyDrainStarted(t, r.early)
	model.finish(t, 0)

	end := h.rec.wait(t)
	if end.State != Done || end.Result != "整理一。" || end.Raw != "one" || end.Problem != nil {
		t.Fatalf("end %+v, want the normal cleaned result without degradation", end)
	}
}

func TestEarlyCleanMatchingPrefixCancellationRacesWithCompletion(t *testing.T) {
	model := &earlyCleanCompletionRaceLLM{
		firstFinished: make(chan struct{}),
		secondStarted: make(chan struct{}),
		releaseSecond: make(chan struct{}),
	}
	r := newEarlyUnitRun(t, model, nil)
	r.hearPiece("one")
	r.hearPiece("one two")
	r.hearPiece("one two three")
	<-model.firstFinished
	<-model.secondStarted

	result := make(chan struct {
		parts   []compose.Part
		answers []string
		tail    string
		ok      bool
	}, 1)
	go func() {
		parts, answers, tail, _, _, ok := r.early.drain("one two three tail")
		result <- struct {
			parts   []compose.Part
			answers []string
			tail    string
			ok      bool
		}{parts, answers, tail, ok}
	}()
	waitEarlyDrainStarted(t, r.early)
	close(model.releaseSecond)
	drained := <-result
	waitEarlyWorker(t, r)
	if !drained.ok || len(drained.parts) != 1 || len(drained.answers) != 1 || drained.answers[0] != "整理一。" || drained.tail != " two three tail" {
		t.Fatalf("drain %+v, want the same finalized prefix and raw suffix", drained)
	}
	if got := compose.JoinParts(drained.parts, drained.answers) + drained.tail; got != "整理一。 two three tail" {
		t.Errorf("raced output %q lost, duplicated, or reordered transcript text", got)
	}
	r.early.mu.Lock()
	defer r.early.mu.Unlock()
	if len(r.early.segs) != 1 {
		t.Errorf("worker published %d segments after finalization, want the first only", len(r.early.segs))
	}
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

type cancellationIgnoringEarlyLLM struct {
	requestStarted  chan struct{}
	requestCanceled chan struct{}
	releaseRequest  chan struct{}
	fallbackStarted chan struct{}
}

func (model *cancellationIgnoringEarlyLLM) Chat(ctx context.Context, request llm.Request) (llm.Response, error) {
	if request.OnDelta != nil {
		close(model.fallbackStarted)
		return llm.Response{Text: "fallback"}, nil
	}
	close(model.requestStarted)
	<-ctx.Done()
	close(model.requestCanceled)
	<-model.releaseRequest
	return llm.Response{Text: "stale early answer"}, errors.New("late provider failure")
}

type cancellationHonoringEarlyLLM struct {
	requestStarted  chan struct{}
	requestCanceled chan struct{}
	fallbackStarted chan struct{}
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
		output := r.cleanUpFirst("changed")
		result <- cleanupResult{text: output.text, err: output.err}
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
	r.hearPiece("stable")
	<-model.requestStarted

	type cleanupResult struct {
		text string
		err  error
	}
	result := make(chan cleanupResult, 1)
	go func() {
		output := r.cleanUpFirst("changed")
		result <- cleanupResult{text: output.text, err: output.err}
	}()
	<-model.requestCanceled
	<-model.fallbackStarted
	select {
	case outcome := <-result:
		if outcome.err != nil || outcome.text != "fallback" {
			t.Fatalf("whole-text cleanup = %q, %v; want fallback", outcome.text, outcome.err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("whole-text cleanup waited for the held early request")
	}
	select {
	case <-r.early.workerDone:
		t.Fatal("the cancellation-ignoring request ended before release")
	default:
	}
	r.early.mu.Lock()
	if !r.early.finalized || len(r.early.segs) != 0 || r.early.failed != nil {
		t.Errorf("finalized state changed before release: finalized=%t segments=%d error=%v", r.early.finalized, len(r.early.segs), r.early.failed)
	}
	r.early.mu.Unlock()

	close(model.releaseRequest)
	waitEarlyWorker(t, r)
	r.early.mu.Lock()
	defer r.early.mu.Unlock()
	if len(r.early.segs) != 0 || r.early.failed != nil {
		t.Errorf("late result changed finalized state: segments=%d error=%v", len(r.early.segs), r.early.failed)
	}
}

func TestEarlyCleanPrefixMismatchAfterCompletedSegmentUsesWholeText(t *testing.T) {
	model := &earlyFallbackLLM{secondStarted: make(chan struct{})}
	r := newEarlyUnitRun(t, model, nil)
	r.hearPiece("one")
	r.hearPiece("one two")
	<-model.secondStarted

	output := r.cleanUpFirst("changed transcript")
	if output.err != nil {
		t.Fatal(output.err)
	}
	if output.text != "fallback" {
		t.Fatalf("result %q, want whole-text cleanup", output.text)
	}
	waitEarlyWorker(t, r)
	model.mu.Lock()
	if !errors.Is(model.secondErr, context.Canceled) {
		t.Errorf("second early request ended with %v, want cancellation", model.secondErr)
	}
	model.mu.Unlock()
	r.early.mu.Lock()
	defer r.early.mu.Unlock()
	if len(r.early.segs) != 1 || !r.early.finalized {
		t.Errorf("mismatch finalization has %d segments and finalized=%t, want one completed segment fenced", len(r.early.segs), r.early.finalized)
	}
}
