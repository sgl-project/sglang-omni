package session

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
)

type stagedCleanupLLM struct {
	started  chan int
	returned chan int
	release  []chan struct{}
	calls    atomic.Int32
}

func newStagedCleanupLLM(count int) *stagedCleanupLLM {
	model := &stagedCleanupLLM{started: make(chan int, count), returned: make(chan int, count), release: make([]chan struct{}, count)}
	for index := range model.release {
		model.release[index] = make(chan struct{})
	}
	return model
}

func (model *stagedCleanupLLM) Chat(_ context.Context, request llm.Request) (llm.Response, error) {
	call := int(model.calls.Add(1) - 1)
	model.started <- call
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

func TestCleanupSoftDeadlineRethinkUsesOriginalBudgetAndFencesLateResponses(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newStagedCleanupLLM(2)
	h := newHarness(t, &fakeFile{text: "spoken words"}, model, cleanupLatencyConfig(clock, true))
	id := h.start()
	h.speak(id, 3)
	waitStagedCall(t, model, 0)
	clock.Advance(CandidateCleanupSoftDeadline - time.Nanosecond)
	h.m.Rethink(id, &llm.Thinking{Mode: llm.ThinkOn})
	model.releaseCall(0)
	waitStagedReturn(t, model, 0)
	waitStagedCall(t, model, 1)
	if clock.timerCount() != 1 || clock.timerDelay(0) != CandidateCleanupSoftDeadline {
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

func TestCleanupSoftDeadlineWinsRethinkAtDeadline(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newStagedCleanupLLM(1)
	h := newHarness(t, &fakeFile{text: "spoken words"}, model, cleanupLatencyConfig(clock, true))
	id := h.start()
	h.speak(id, 3)
	waitStagedCall(t, model, 0)
	clock.Advance(CandidateCleanupSoftDeadline)
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
