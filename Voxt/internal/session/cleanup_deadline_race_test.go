package session

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

const cleanupSoftDeadlineTestDuration = 1500 * time.Millisecond

type stagedCleanupLLM struct {
	started  chan int
	requests chan llm.Request
	contexts chan context.Context
	returned chan int
	release  []chan struct{}
	calls    atomic.Int32
}

type cleanupLLMReply struct {
	text string
	err  error
}

type barrierCleanupLLM struct {
	started   chan llm.Request
	responses chan cleanupLLMReply
	calls     atomic.Int32
}

func newBarrierCleanupLLM() *barrierCleanupLLM {
	return &barrierCleanupLLM{started: make(chan llm.Request, 4), responses: make(chan cleanupLLMReply, 4)}
}

func (model *barrierCleanupLLM) Chat(ctx context.Context, request llm.Request) (llm.Response, error) {
	model.calls.Add(1)
	model.started <- request
	select {
	case reply := <-model.responses:
		if request.OnDelta != nil && reply.text != "" {
			request.OnDelta(reply.text)
		}
		return llm.Response{Text: reply.text}, reply.err
	case <-ctx.Done():
		return llm.Response{}, ctx.Err()
	}
}

type rethinkAtFallbackCheckError struct {
	calls    atomic.Int32
	checking chan struct{}
	resume   chan struct{}
}

type lateImageDropLLM struct {
	started chan struct{}
	release chan struct{}
}

func (model *lateImageDropLLM) Chat(context.Context, llm.Request) (llm.Response, error) {
	close(model.started)
	<-model.release
	return llm.Response{ImagesDropped: true}, nil
}

func newRethinkAtFallbackCheckError() *rethinkAtFallbackCheckError {
	return &rethinkAtFallbackCheckError{checking: make(chan struct{}), resume: make(chan struct{})}
}

func (err *rethinkAtFallbackCheckError) Error() string { return "primary provider failed" }

func (err *rethinkAtFallbackCheckError) As(target any) bool {
	if _, ok := target.(**llm.Error); !ok {
		return false
	}
	if err.calls.Add(1) == 2 {
		close(err.checking)
		<-err.resume
	}
	return false
}

func newStagedCleanupLLM(count int) *stagedCleanupLLM {
	model := &stagedCleanupLLM{started: make(chan int, count), requests: make(chan llm.Request, count), contexts: make(chan context.Context, count), returned: make(chan int, count), release: make([]chan struct{}, count)}
	for index := range model.release {
		model.release[index] = make(chan struct{})
	}
	return model
}

func (model *stagedCleanupLLM) Chat(ctx context.Context, request llm.Request) (llm.Response, error) {
	call := int(model.calls.Add(1) - 1)
	model.started <- call
	model.requests <- request
	model.contexts <- ctx
	<-model.release[call]
	if request.OnDelta != nil {
		request.OnDelta("late cleanup draft")
	}
	model.returned <- call
	return llm.Response{Text: "staged cleanup result"}, nil
}

func (model *stagedCleanupLLM) releaseCall(call int) { close(model.release[call]) }

func waitStagedCall(t *testing.T, model *stagedCleanupLLM, call int) {
	t.Helper()
	select {
	case got := <-model.started:
		if got != call {
			t.Fatalf("started call %d, want %d", got, call)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for cleanup call %d", call)
	}
}

func waitStagedReturn(t *testing.T, model *stagedCleanupLLM, call int) {
	t.Helper()
	select {
	case got := <-model.returned:
		if got != call {
			t.Fatalf("returned call %d, want %d", got, call)
		}
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for cleanup call %d to return", call)
	}
}

func waitStagedRequest(t *testing.T, model *stagedCleanupLLM, what string) llm.Request {
	t.Helper()
	select {
	case request := <-model.requests:
		return request
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
		return llm.Request{}
	}
}

func waitStagedContext(t *testing.T, model *stagedCleanupLLM, what string) context.Context {
	t.Helper()
	select {
	case ctx := <-model.contexts:
		return ctx
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
		return nil
	}
}

func waitCleanupCall(t *testing.T, calls <-chan llm.Request, what string) llm.Request {
	t.Helper()
	select {
	case request := <-calls:
		return request
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
		return llm.Request{}
	}
}

func TestCleanupRethinkBetweenFallbackEligibilityAndAdmission(t *testing.T) {
	for _, enabled := range []bool{false, true} {
		name := "disabled"
		if enabled {
			name = "enabled"
		}
		t.Run(name, func(t *testing.T) {
			clock := newSoftDeadlineClock()
			primary := newBarrierCleanupLLM()
			fallback := newBarrierCleanupLLM()
			barrier := newRethinkAtFallbackCheckError()
			h := newHarness(t, &fakeFile{text: "spoken words"}, primary, cleanupLatencyConfig(clock, enabled))
			h.fallback = fallback
			id := h.start()
			h.m.mu.Lock()
			r := h.m.cur
			h.m.mu.Unlock()
			h.speak(id, 3)

			waitCleanupCall(t, primary.started, "initial primary request")
			primary.responses <- cleanupLLMReply{err: barrier}
			select {
			case <-barrier.checking:
			case <-time.After(5 * time.Second):
				t.Fatal("worker did not reach the fallback eligibility check")
			}
			off := &llm.Thinking{Mode: llm.ThinkOff}
			h.m.Rethink(id, off)
			close(barrier.resume)

			var retry llm.Request
			select {
			case retry = <-primary.started:
			case <-fallback.started:
				t.Fatal("fallback was admitted after Rethink won")
			case <-time.After(5 * time.Second):
				t.Fatal("Rethink was lost before a new primary request")
			}
			if retry.Thinking == nil || *retry.Thinking != *off {
				t.Fatalf("Rethink request thinking = %+v, want %+v", retry.Thinking, off)
			}
			primary.responses <- cleanupLLMReply{text: "cleaned words"}

			end := h.rec.waitFor(t, "Rethink cleanup delivery", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
			if end.State != Done || end.Result != "cleaned words" || end.LLM.ID != "deepseek" || end.Problem != nil {
				t.Fatalf("end %+v", end)
			}
			if primary.calls.Load() != 2 || fallback.calls.Load() != 0 {
				t.Fatalf("primary calls %d, fallback calls %d", primary.calls.Load(), fallback.calls.Load())
			}
			if pasted := h.text.pastes(); len(pasted) != 1 || pasted[0] != "cleaned words" {
				t.Fatalf("pasted %q", pasted)
			}
			if entries := h.hist.all(); len(entries) != 1 || entries[0].Result != "cleaned words" {
				t.Fatalf("history entries %+v", entries)
			}
			h.m.mu.Lock()
			defer h.m.mu.Unlock()
			if r.cleanupGeneration != 2 || r.cleanupOutput.generation != 2 || r.cleanupOutcome != cleanupOutcomeCleaned {
				t.Fatalf("generation %d, output generation %d, outcome %q", r.cleanupGeneration, r.cleanupOutput.generation, r.cleanupOutcome)
			}
		})
	}
}

func TestCleanupSoftDeadlineRethinkUsesOriginalBudgetAndFencesLateResponses(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newStagedCleanupLLM(2)
	h := newHarness(t, &fakeFile{text: "spoken words"}, model, cleanupLatencyConfig(clock, true))
	id := h.start()
	h.speak(id, 3)
	waitStagedCall(t, model, 0)
	clock.Advance(cleanupSoftDeadlineTestDuration - time.Nanosecond)
	h.m.Rethink(id, &llm.Thinking{Mode: llm.ThinkOn})
	model.releaseCall(0)
	waitStagedReturn(t, model, 0)
	waitStagedCall(t, model, 1)
	if clock.timerCount() != 1 || clock.timerDelay(0) != cleanupSoftDeadlineTestDuration {
		t.Fatalf("Rethink reset the deadline: timers %d, delay %v", clock.timerCount(), clock.timerDelay(0))
	}
	clock.Advance(time.Nanosecond)
	clock.Fire(t, 0)
	end := h.rec.waitFor(t, "raw result at original deadline", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
	if end.Result != "spoken words" || len(h.text.pastes()) != 1 || h.text.pastes()[0] != "spoken words" {
		t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
	}
	h.m.Rethink(id, &llm.Thinking{Mode: llm.ThinkOff})
	model.releaseCall(1)
	waitStagedReturn(t, model, 1)
	for _, snapshot := range h.rec.all() {
		if snapshot.ID == id && snapshot.State.Ended() && snapshot.Draft != "" {
			t.Fatalf("late response changed terminal draft: %+v", snapshot)
		}
	}
	if model.calls.Load() != 2 || len(h.text.pastes()) != 1 {
		t.Fatalf("cleanup calls %d, pasted %q", model.calls.Load(), h.text.pastes())
	}
}

func TestCleanupConsecutiveRethinksKeepGenerationContextAndThinkingTogether(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newStagedCleanupLLM(4)
	h := newHarness(t, &fakeFile{text: "spoken words"}, model, cleanupLatencyConfig(clock, true))
	id := h.start()
	h.m.mu.Lock()
	r := h.m.cur
	h.m.mu.Unlock()
	h.speak(id, 3)
	waitStagedCall(t, model, 0)
	waitStagedRequest(t, model, "initial Cleanup request")
	ctx0 := waitStagedContext(t, model, "initial Cleanup context")

	levels := []*llm.Thinking{
		{Mode: llm.ThinkOff},
		{Mode: llm.ThinkOn},
		{Mode: llm.ThinkOff},
	}
	for call, level := range levels {
		h.m.Rethink(id, level)
		if ctx0.Err() == nil {
			t.Fatalf("Rethink %d did not cancel the prior generation context", call+1)
		}
		model.releaseCall(call)
		waitStagedReturn(t, model, call)
		waitStagedCall(t, model, call+1)
		request := waitStagedRequest(t, model, "Rethink generation")
		ctx0 = waitStagedContext(t, model, "Rethink generation context")
		if ctx0.Err() != nil {
			t.Fatalf("Rethink generation %d started with canceled context: %v", call+2, ctx0.Err())
		}
		if request.Thinking == nil || *request.Thinking != *level {
			t.Fatalf("generation %d thinking = %+v, want %+v", call+2, request.Thinking, level)
		}
		if clock.timerCount() != 1 || clock.timerDelay(0) != cleanupSoftDeadlineTestDuration {
			t.Fatalf("Rethink reset the deadline: timers %d, delay %v", clock.timerCount(), clock.timerDelay(0))
		}
	}
	model.releaseCall(3)
	waitStagedReturn(t, model, 3)
	end := h.rec.waitFor(t, "final Rethink delivery", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
	if end.State != Done || end.Result != "staged cleanup result" || end.Problem != nil || len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
		t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
	}
	if entries := h.hist.all(); len(entries) != 1 || entries[0].Result != end.Result {
		t.Fatalf("history entries %+v", entries)
	}
	h.m.mu.Lock()
	defer h.m.mu.Unlock()
	if r.cleanupGeneration != 4 || r.cleanupOutput.generation != 4 || r.cleanupOutcome != cleanupOutcomeCleaned || model.calls.Load() != 4 {
		t.Fatalf("generation %d, output generation %d, outcome %q, calls %d", r.cleanupGeneration, r.cleanupOutput.generation, r.cleanupOutcome, model.calls.Load())
	}
}

func TestCleanupSoftDeadlineWinsRethinkAtDeadline(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newStagedCleanupLLM(1)
	h := newHarness(t, &fakeFile{text: "spoken words"}, model, cleanupLatencyConfig(clock, true))
	id := h.start()
	h.speak(id, 3)
	waitStagedCall(t, model, 0)
	clock.Advance(cleanupSoftDeadlineTestDuration)
	h.m.Rethink(id, &llm.Thinking{Mode: llm.ThinkOn})
	clock.Fire(t, 0)
	end := h.rec.waitFor(t, "raw result at deadline", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
	h.m.Rethink(id, &llm.Thinking{Mode: llm.ThinkOff})
	model.releaseCall(0)
	waitStagedReturn(t, model, 0)
	if end.Result != "spoken words" || model.calls.Load() != 1 || len(h.text.pastes()) != 1 {
		t.Fatalf("end %+v, cleanup calls %d, pasted %q", end, model.calls.Load(), h.text.pastes())
	}
}

func TestCleanupSoftDeadlineCancellationWhileRethinkIsPending(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newStagedCleanupLLM(2)
	h := newHarness(t, &fakeFile{text: "spoken words"}, model, cleanupLatencyConfig(clock, true))
	id := h.start()
	h.speak(id, 3)
	waitStagedCall(t, model, 0)
	h.m.Rethink(id, &llm.Thinking{Mode: llm.ThinkOff})
	model.releaseCall(0)
	waitStagedReturn(t, model, 0)
	waitStagedCall(t, model, 1)

	h.m.Cancel(id)
	end := h.rec.waitFor(t, "cancelled Rethink", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
	if end.State != Cancelled || end.Result != "" || len(h.text.pastes()) != 0 {
		t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
	}
	model.releaseCall(1)
	waitStagedReturn(t, model, 1)
	if model.calls.Load() != 2 || len(h.text.pastes()) != 0 {
		t.Fatalf("cleanup calls %d, pasted %q", model.calls.Load(), h.text.pastes())
	}
	for _, snapshot := range h.rec.all() {
		if snapshot.ID == id && snapshot.State == Cancelled && (snapshot.Result != "" || snapshot.Draft != "") {
			t.Fatalf("late Rethink response changed cancelled snapshot: %+v", snapshot)
		}
	}
}

func TestCleanupRethinkAfterRawCommitCannotReopenDecision(t *testing.T) {
	manager, r, clock, deadline := cleanupArbiter(t)
	clock.Advance(cleanupSoftDeadlineTestDuration)
	manager.notifyCleanupDeadline(r)
	choice, _ := manager.chooseCleanup(r)
	if choice != cleanupRaw || r.cleanupOutcome != cleanupOutcomeSoftDeadline {
		t.Fatalf("choice %v, outcome %q", choice, r.cleanupOutcome)
	}
	generation := r.cleanupGeneration
	manager.Rethink(r.id, &llm.Thinking{Mode: llm.ThinkOn})
	if r.cleanupGeneration != generation || r.rethink.Load() || r.cleanupOutcome != cleanupOutcomeSoftDeadline {
		t.Fatalf("Rethink changed committed Raw outcome: generation %d, rethink %t, outcome %q", r.cleanupGeneration, r.rethink.Load(), r.cleanupOutcome)
	}
	manager.publishCleanupCandidate(r, cleanupOutput{text: "late cleanup", completedAt: deadline.Add(-time.Nanosecond)}, generation, deadline.Add(-time.Nanosecond))
	choice, output := manager.chooseCleanup(r)
	if choice != cleanupRaw || output.text != "" || r.cleanupOutcome != cleanupOutcomeSoftDeadline {
		t.Fatalf("late candidate changed committed Raw: choice %v, output %+v, outcome %q", choice, output, r.cleanupOutcome)
	}
}

func TestCleanupLateImageHintCannotPublishAfterRawChoice(t *testing.T) {
	manager, r, clock, _ := cleanupArbiter(t)
	provider := &lateImageDropLLM{started: make(chan struct{}), release: make(chan struct{})}
	r.shot = func() []byte { return []byte{1, 2, 3} }
	manager.mu.Lock()
	r.svc.LLM = provider
	manager.mu.Unlock()
	parent := r.currentLLMContext()
	returned := make(chan struct{})
	go func() {
		_, _ = r.llmChatContext(parent, llm.Request{Timeout: time.Second})
		close(returned)
	}()
	select {
	case <-provider.started:
	case <-time.After(5 * time.Second):
		t.Fatal("provider did not start")
	}
	before := manager.snapshot(r)
	if before.ImageHint != "window" || before.ImagePreviewKind != "window" {
		t.Fatalf("request image state %+v", before)
	}
	clock.Advance(cleanupSoftDeadlineTestDuration)
	manager.notifyCleanupDeadline(r)
	choice, _ := manager.chooseCleanup(r)
	if choice != cleanupRaw {
		t.Fatalf("choice %v, want Raw", choice)
	}
	committed := manager.snapshot(r)
	close(provider.release)
	select {
	case <-returned:
	case <-time.After(5 * time.Second):
		t.Fatal("late provider did not return")
	}
	after := manager.snapshot(r)
	if after.Rev != committed.Rev || after.ImageHint != committed.ImageHint || after.ImagePreviewKind != committed.ImagePreviewKind {
		t.Fatalf("late provider changed snapshot after Raw: before %+v, after %+v", committed, after)
	}
}

func cleanupArbiter(t *testing.T) (*Manager, *run, *softDeadlineClock, time.Time) {
	t.Helper()
	clock := newSoftDeadlineClock()
	manager := New(Config{Now: clock.Now})
	ctx, cancel := context.WithCancel(context.Background())
	llmCtx, llmStop := context.WithCancel(ctx)
	startedAt := clock.Now()
	r := &run{
		m: manager, id: "race", req: StartRequest{Kind: store.KindTranscribe}, ctx: ctx, cancel: cancel, llmCtx: llmCtx, llmStop: llmStop,
		cleanupWake: make(chan struct{}, 1), cleanupGeneration: 1,
	}
	r.recordStopAt(startedAt)
	manager.mu.Lock()
	manager.cur = r
	manager.snap = Snapshot{ID: r.id, Kind: store.KindTranscribe, State: Composing, CanSkip: true}
	manager.mu.Unlock()
	deadline := startedAt.Add(cleanupSoftDeadlineTestDuration)
	manager.setCleanupDeadline(r, deadline)
	t.Cleanup(func() {
		cancel()
		manager.Close()
	})
	return manager, r, clock, deadline
}

func TestCleanupCandidatePublicationArbitration(t *testing.T) {
	t.Run("published before deadline wins after owner delay", func(t *testing.T) {
		manager, r, clock, deadline := cleanupArbiter(t)
		manager.publishCleanupCandidate(r, cleanupOutput{text: "cleaned", completedAt: deadline.Add(-time.Nanosecond)}, r.cleanupGeneration, deadline.Add(-time.Nanosecond))
		clock.Advance(cleanupSoftDeadlineTestDuration + time.Second)
		manager.notifyCleanupDeadline(r)
		choice, output := manager.chooseCleanup(r)
		if choice != cleanupComplete || output.text != "cleaned" || r.cleanupOutcome != cleanupOutcomeCleaned {
			t.Fatalf("choice %v, output %+v, outcome %s", choice, output, r.cleanupOutcome)
		}
	})

	for _, tc := range []struct {
		name string
		at   time.Duration
	}{
		{name: "exact deadline", at: cleanupSoftDeadlineTestDuration},
		{name: "after deadline", at: cleanupSoftDeadlineTestDuration + time.Nanosecond},
	} {
		t.Run(tc.name, func(t *testing.T) {
			manager, r, clock, deadline := cleanupArbiter(t)
			at := deadline.Add(tc.at - cleanupSoftDeadlineTestDuration)
			clock.Advance(tc.at)
			manager.publishCleanupCandidate(r, cleanupOutput{text: "late", completedAt: at}, r.cleanupGeneration, at)
			manager.notifyCleanupDeadline(r)
			choice, output := manager.chooseCleanup(r)
			if choice != cleanupRaw || output.text != "" || r.cleanupOutcome != cleanupOutcomeSoftDeadline {
				t.Fatalf("choice %v, output %+v, outcome %s", choice, output, r.cleanupOutcome)
			}
		})
	}

	t.Run("raw commitment is irreversible", func(t *testing.T) {
		manager, r, clock, deadline := cleanupArbiter(t)
		clock.Advance(cleanupSoftDeadlineTestDuration)
		manager.notifyCleanupDeadline(r)
		choice, _ := manager.chooseCleanup(r)
		if choice != cleanupRaw {
			t.Fatalf("first choice %v, want Raw", choice)
		}
		manager.publishCleanupCandidate(r, cleanupOutput{text: "backdated", completedAt: deadline.Add(-time.Nanosecond)}, r.cleanupGeneration, deadline.Add(-time.Nanosecond))
		choice, output := manager.chooseCleanup(r)
		if choice != cleanupRaw || output.text != "" || r.cleanupOutcome != cleanupOutcomeSoftDeadline {
			t.Fatalf("late publication changed decision: %v, %+v, %s", choice, output, r.cleanupOutcome)
		}
	})

	t.Run("manual Skip wins before owner arbitration", func(t *testing.T) {
		manager, r, _, deadline := cleanupArbiter(t)
		manager.publishCleanupCandidate(r, cleanupOutput{text: "cleaned", completedAt: deadline.Add(-time.Nanosecond)}, r.cleanupGeneration, deadline.Add(-time.Nanosecond))
		manager.SkipLLM(r.id)
		choice, _ := manager.chooseCleanup(r)
		if choice != cleanupRaw || r.cleanupOutcome != cleanupOutcomeManualSkip {
			t.Fatalf("choice %v, outcome %s", choice, r.cleanupOutcome)
		}
	})

	t.Run("provider failure has its own outcome", func(t *testing.T) {
		manager, r, _, deadline := cleanupArbiter(t)
		manager.publishCleanupCandidate(r, cleanupOutput{text: "usable raw text", err: errors.New("provider unavailable"), completedAt: deadline.Add(-time.Nanosecond)}, r.cleanupGeneration, deadline.Add(-time.Nanosecond))
		choice, output := manager.chooseCleanup(r)
		if choice != cleanupComplete || output.err == nil || r.cleanupOutcome != cleanupOutcomeLLMError {
			t.Fatalf("choice %v, output %+v, outcome %s", choice, output, r.cleanupOutcome)
		}
		manager.publishCleanupCandidate(r, cleanupOutput{text: "late recovery", completedAt: deadline.Add(-time.Nanosecond)}, r.cleanupGeneration, deadline.Add(-time.Nanosecond))
		if choice, output := manager.chooseCleanup(r); choice != cleanupComplete || output.err == nil || r.cleanupOutcome != cleanupOutcomeLLMError {
			t.Fatalf("late result changed provider failure: %v, %+v, %s", choice, output, r.cleanupOutcome)
		}
	})

	t.Run("Cancel wins before owner arbitration", func(t *testing.T) {
		manager, r, _, deadline := cleanupArbiter(t)
		manager.publishCleanupCandidate(r, cleanupOutput{text: "cleaned", completedAt: deadline.Add(-time.Nanosecond)}, r.cleanupGeneration, deadline.Add(-time.Nanosecond))
		manager.Cancel(r.id)
		choice, _ := manager.chooseCleanup(r)
		if choice != cleanupCancelled || r.cleanupOutcome != cleanupOutcomeCancelled {
			t.Fatalf("choice %v, outcome %s", choice, r.cleanupOutcome)
		}
	})

	t.Run("Rethink invalidates the previous generation", func(t *testing.T) {
		manager, r, _, deadline := cleanupArbiter(t)
		manager.publishCleanupCandidate(r, cleanupOutput{text: "stale", completedAt: deadline.Add(-time.Nanosecond)}, r.cleanupGeneration, deadline.Add(-time.Nanosecond))
		manager.Rethink(r.id, nil)
		if choice, _ := manager.chooseCleanup(r); choice != cleanupRetry {
			t.Fatalf("Rethink choice %v, want retry", choice)
		}
		if !r.askAgain() {
			t.Fatal("Rethink did not ready a new generation")
		}
		manager.publishCleanupCandidate(r, cleanupOutput{text: "fresh", completedAt: deadline.Add(-time.Nanosecond)}, r.cleanupGeneration, deadline.Add(-time.Nanosecond))
		if choice, output := manager.chooseCleanup(r); choice != cleanupComplete || output.text != "fresh" {
			t.Fatalf("choice %v, output %+v", choice, output)
		}
	})

	t.Run("an expired deadline does not restart Rethink", func(t *testing.T) {
		manager, r, clock, _ := cleanupArbiter(t)
		manager.Rethink(r.id, nil)
		clock.Advance(cleanupSoftDeadlineTestDuration)
		manager.notifyCleanupDeadline(r)
		choice, _ := manager.chooseCleanup(r)
		if choice != cleanupRaw || r.cleanupOutcome != cleanupOutcomeSoftDeadline {
			t.Fatalf("choice %v, outcome %s", choice, r.cleanupOutcome)
		}
	})

	t.Run("delayed timer cannot admit a late result", func(t *testing.T) {
		manager, r, clock, deadline := cleanupArbiter(t)
		clock.Advance(cleanupSoftDeadlineTestDuration)
		manager.publishCleanupCandidate(r, cleanupOutput{text: "late", completedAt: deadline}, r.cleanupGeneration, deadline)
		// No timer notification yet: publication itself is still rejected
		// by the absolute deadline, and the owner observes expired time.
		choice, _ := manager.chooseCleanup(r)
		if choice != cleanupRaw || r.cleanupOutcome != cleanupOutcomeSoftDeadline {
			t.Fatalf("choice %v, outcome %s", choice, r.cleanupOutcome)
		}
		manager.notifyCleanupDeadline(r)
		choice, _ = manager.chooseCleanup(r)
		if choice != cleanupRaw {
			t.Fatalf("repeated timer changed decision to %v", choice)
		}
	})

	t.Run("repeated notifications keep one committed result", func(t *testing.T) {
		manager, r, _, deadline := cleanupArbiter(t)
		manager.publishCleanupCandidate(r, cleanupOutput{text: "cleaned", completedAt: deadline.Add(-time.Nanosecond)}, r.cleanupGeneration, deadline.Add(-time.Nanosecond))
		firstChoice, firstOutput := manager.chooseCleanup(r)
		manager.publishCleanupCandidate(r, cleanupOutput{text: "duplicate", completedAt: deadline.Add(-time.Nanosecond)}, r.cleanupGeneration, deadline.Add(-time.Nanosecond))
		manager.notifyCleanupDeadline(r)
		secondChoice, secondOutput := manager.chooseCleanup(r)
		if firstChoice != cleanupComplete || secondChoice != firstChoice || firstOutput.text != secondOutput.text || r.cleanupOutcome != cleanupOutcomeCleaned {
			t.Fatalf("decisions %v/%v, outputs %+v/%+v, outcome %s", firstChoice, secondChoice, firstOutput, secondOutput, r.cleanupOutcome)
		}
	})
}

func TestCleanupSoftDeadlineExpiredASRFinalReusesOnlyCompleteEarlyCleanup(t *testing.T) {
	t.Run("completed and compatible", func(t *testing.T) {
		clock := newSoftDeadlineClock()
		model := &fakeLLM{text: "cleaned while recording"}
		provider := &clockAdvancingStreamer{fakeStreamer: fakeStreamer{final: "spoken words", pieces: []string{"spoken words"}}, clock: clock, delay: cleanupSoftDeadlineTestDuration + time.Millisecond}
		h := newHarness(t, provider, model, cleanupLatencyConfig(clock, true))
		id := h.start()
		waitEarly(t, model, 1)
		h.speak(id, 5)
		end := h.rec.waitFor(t, "reused early cleanup", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
		if end.Result != "cleaned while recording" || len(model.requests()) != 1 || len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
			t.Fatalf("end %+v, requests %d, pasted %q", end, len(model.requests()), h.text.pastes())
		}
	})

	t.Run("unfinished early prefix", func(t *testing.T) {
		clock := newSoftDeadlineClock()
		model := newCleanupGateLLM("", "", "too late")
		provider := &clockAdvancingStreamer{fakeStreamer: fakeStreamer{final: "spoken words", pieces: []string{"spoken words"}}, clock: clock, delay: cleanupSoftDeadlineTestDuration + time.Millisecond}
		h := newHarness(t, provider, model, cleanupLatencyConfig(clock, true))
		id := h.start()
		waitCleanupRequest(t, model, "early cleanup request")
		h.speak(id, 5)
		end := h.rec.waitFor(t, "raw result without waiting for early cleanup", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
		if end.Result != "spoken words" || len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
			t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
		}
		model.Release()
		waitCleanupSignal(t, model.returned, "cancelled early cleanup")
		if len(model.started) != 0 {
			t.Fatalf("new cleanup request started after late ASR Final: %d", len(model.started))
		}
	})

	t.Run("incompatible early prefix", func(t *testing.T) {
		clock := newSoftDeadlineClock()
		model := &fakeLLM{text: "irrelevant early cleanup"}
		provider := &clockAdvancingStreamer{fakeStreamer: fakeStreamer{final: "spoken words", pieces: []string{"different words"}}, clock: clock, delay: cleanupSoftDeadlineTestDuration + time.Millisecond}
		h := newHarness(t, provider, model, cleanupLatencyConfig(clock, true))
		id := h.start()
		waitEarly(t, model, 1)
		h.speak(id, 5)
		end := h.rec.waitFor(t, "raw result for incompatible early prefix", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
		if end.Result != "spoken words" || len(model.requests()) != 1 || len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
			t.Fatalf("end %+v, requests %d, pasted %q", end, len(model.requests()), h.text.pastes())
		}
	})
}

type clockAdvancingStreamer struct {
	fakeStreamer
	clock *softDeadlineClock
	delay time.Duration
}

func (provider *clockAdvancingStreamer) Open(ctx context.Context, request asr.StreamRequest) (asr.Stream, error) {
	stream, err := provider.fakeStreamer.Open(ctx, request)
	if err != nil {
		return nil, err
	}
	return &clockAdvancingStream{Stream: stream, clock: provider.clock, delay: provider.delay}, nil
}

type clockAdvancingStream struct {
	asr.Stream
	clock *softDeadlineClock
	delay time.Duration
}

func (stream *clockAdvancingStream) Finish() error {
	stream.clock.Advance(stream.delay)
	return stream.Stream.Finish()
}
