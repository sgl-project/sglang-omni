package session

import (
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
)

func TestLivePolishMismatchDetachesCancellationIgnoringFlight(t *testing.T) {
	model := newControlledLiveLLM(true)
	r := newEarlyUnitRun(t, model, nil)
	r.hearLive("Old transcript.")
	oldCall := waitLiveCall(t, model, "the mismatched speculative request")
	oldCall.sendDelta(t, "private old text")
	result := make(chan string, 1)
	go func() {
		got, _ := liveCleanUp(r, "Final transcript.")
		result <- got
	}()
	newCall := waitLiveCall(t, model, "the normal final cleanup")
	oldCall.sendDelta(t, " late")
	if draft := r.m.Current().Draft; draft != "" {
		t.Fatalf("mismatched flight published %q", draft)
	}
	newCall.complete(t, "Final cleaned text.", nil)
	select {
	case got := <-result:
		if got != "Final cleaned text." {
			t.Fatalf("final cleanup %q", got)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("mismatched flight blocked final cleanup")
	}
	oldCall.complete(t, "Old completed answer.", nil)
	if model.requestCount() != 2 {
		t.Fatalf("requests %d, want one speculative and one final", model.requestCount())
	}
}

func TestLivePolishRevisionCannotReAdoptCancelledMatchingSource(t *testing.T) {
	model := newControlledLiveLLM(true)
	r := newEarlyUnitRun(t, model, nil)
	r.hearLive("Old transcript.")
	oldCall := waitLiveCall(t, model, "the original speculative request")
	oldCall.sendDelta(t, "private stale text")
	r.hearLive("Revised transcript.")
	r.hearLive("Old transcript.")
	result := make(chan string, 1)
	go func() {
		got, _ := liveCleanUp(r, "Old transcript.")
		result <- got
	}()
	replacement := waitLiveCall(t, model, "the final-text replacement request")
	if got := inputOf(replacement.request); got != "Old transcript." {
		t.Fatalf("replacement input %q", got)
	}
	oldCall.sendDelta(t, " late stale text")
	if draft := r.m.Current().Draft; draft != "" {
		t.Fatalf("invalidated flight published Draft %q", draft)
	}
	replacement.complete(t, "Current answer", nil)
	select {
	case got := <-result:
		if got != "Current answer" {
			t.Fatalf("cleanup result %q", got)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("final cleanup waited for the invalidated provider")
	}
	oldCall.complete(t, "Stale completed answer", nil)
	if model.requestCount() != 2 {
		t.Fatalf("requests %d, want invalidated speculative call and one final call", model.requestCount())
	}
}

func TestLivePolishSkipStopsAdoptionWithoutWaitingForProvider(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newControlledLiveLLM(true)
	fixture := newLiveSessionFixture(t, "spoken words.", model, clock, false)
	fixture.stopAndReleaseFinal(t, 300*time.Millisecond)
	fixture.harness.rec.waitFor(t, "ASR Final", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State == Composing
	})
	waitLiveAdoption(t, fixture.run)
	fixture.call.sendDelta(t, "visible partial")
	fixture.harness.rec.waitFor(t, "the adopted partial", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.Draft == "visible partial"
	})
	fixture.harness.m.SkipLLM(fixture.id)
	end := fixture.harness.rec.waitFor(t, "raw Skip result", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State.Ended()
	})
	select {
	case <-fixture.call.returned:
		t.Fatal("Skip waited for a cancellation-ignoring provider")
	default:
	}
	if end.Result != "spoken words." || end.Draft != "" {
		t.Fatalf("Skip result %+v", end)
	}
	beforeLateDelta := len(fixture.harness.rec.all())
	fixture.call.sendDelta(t, " late")
	fixture.call.complete(t, "late full answer", nil)
	if afterLateDelta := len(fixture.harness.rec.all()); afterLateDelta != beforeLateDelta {
		t.Fatalf("late stream published %d snapshots", afterLateDelta-beforeLateDelta)
	}
	if pastes := fixture.harness.text.pastes(); len(pastes) != 1 || pastes[0] != "spoken words." {
		t.Fatalf("pasted %q", pastes)
	}
}

func TestLivePolishSoftDeadlineClearsAdoptedStream(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newControlledLiveLLM(true)
	fixture := newLiveSessionFixture(t, "spoken words.", model, clock, true)
	fixture.stopAndReleaseFinal(t, 300*time.Millisecond)
	fixture.harness.rec.waitFor(t, "ASR Final", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State == Composing
	})
	waitLiveAdoption(t, fixture.run)
	fixture.call.sendDelta(t, "validated partial")
	fixture.harness.rec.waitFor(t, "the adopted partial", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.Draft == "validated partial"
	})
	if clock.timerCount() != 1 || clock.timerDelay(0) != cleanupSoftDeadlineTestDuration-300*time.Millisecond {
		t.Fatalf("deadline timers %d, delay %v", clock.timerCount(), clock.timerDelay(0))
	}
	clock.Advance(cleanupSoftDeadlineTestDuration - 300*time.Millisecond)
	clock.Fire(t, 0)
	end := fixture.harness.rec.waitFor(t, "raw deadline result", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State.Ended()
	})
	if end.Result != "spoken words." || end.Draft != "" {
		t.Fatalf("deadline result %+v", end)
	}
	fixture.call.sendDelta(t, " late")
	fixture.call.complete(t, "late cleaned answer", nil)
	if got := fixture.harness.m.Current().Draft; got != "" {
		t.Fatalf("late adopted stream restored Draft %q", got)
	}
	if pastes := fixture.harness.text.pastes(); len(pastes) != 1 || pastes[0] != "spoken words." {
		t.Fatalf("pasted %q", pastes)
	}
	fixture.harness.m.mu.Lock()
	outcome := fixture.run.cleanupOutcome
	fixture.harness.m.mu.Unlock()
	if outcome != cleanupOutcomeSoftDeadline {
		t.Fatalf("cleanup outcome %q", outcome)
	}
}

func TestLivePolishRethinkFencesTheAdoptedGeneration(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newControlledLiveLLM(true)
	fixture := newLiveSessionFixture(t, "spoken words.", model, clock, false)
	fixture.stopAndReleaseFinal(t, 300*time.Millisecond)
	fixture.harness.rec.waitFor(t, "ASR Final", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State == Composing
	})
	waitLiveAdoption(t, fixture.run)
	fixture.call.sendDelta(t, "old generation draft")
	fixture.harness.rec.waitFor(t, "the old generation draft", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.Draft == "old generation draft"
	})
	thinking := &llm.Thinking{Mode: llm.ThinkOff}
	fixture.harness.m.Rethink(fixture.id, thinking)
	newCall := waitLiveCall(t, model, "the Rethink request")
	if got := inputOf(newCall.request); got != "spoken words." {
		t.Fatalf("Rethink reused only a tail: %q", got)
	}
	if newCall.request.Thinking == nil || *newCall.request.Thinking != *thinking {
		t.Fatalf("Rethink thinking %+v, want %+v", newCall.request.Thinking, thinking)
	}
	newCall.complete(t, "new generation answer", nil)
	end := fixture.harness.rec.waitFor(t, "the Rethink result", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State.Ended()
	})
	fixture.harness.m.mu.Lock()
	generation := fixture.run.cleanupGeneration
	fixture.harness.m.mu.Unlock()
	if end.Result != "new generation answer" || generation != 2 {
		t.Fatalf("Rethink result %+v, generation %d", end, generation)
	}
	stableDraft := fixture.harness.m.Current().Draft
	beforeLateDelta := len(fixture.harness.rec.all())
	fixture.call.sendDelta(t, " stale old text")
	fixture.call.complete(t, "old completed answer", nil)
	if got := fixture.harness.m.Current().Draft; got != stableDraft {
		t.Fatalf("old generation changed Draft from %q to %q", stableDraft, got)
	}
	if afterLateDelta := len(fixture.harness.rec.all()); afterLateDelta != beforeLateDelta {
		t.Fatalf("old generation published %d late snapshots", afterLateDelta-beforeLateDelta)
	}
	if pastes := fixture.harness.text.pastes(); len(pastes) != 1 || pastes[0] != "new generation answer" {
		t.Fatalf("pasted %q", pastes)
	}
}

func TestLivePolishCompletionRacingWithRethinkCannotPublishOldGeneration(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newControlledLiveLLM(true)
	fixture := newLiveSessionFixture(t, "spoken words.", model, clock, false)
	flight := liveFlightOf(fixture.run)
	commitEntered, releaseCommit := make(chan struct{}), make(chan struct{})
	fixture.run.live.mu.Lock()
	fixture.run.live.testHooks = &livePolishTestHooks{beforeFlightCommit: func() {
		close(commitEntered)
		<-releaseCommit
	}}
	fixture.run.live.mu.Unlock()
	fixture.call.complete(t, "Old completed answer", nil)
	select {
	case <-commitEntered:
	case <-time.After(5 * time.Second):
		t.Fatal("the completed old flight did not reach terminal publication")
	}
	fixture.stopAndReleaseFinal(t, 300*time.Millisecond)
	fixture.harness.rec.waitFor(t, "the composing session", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State == Composing
	})
	fixture.harness.m.Rethink(fixture.id, &llm.Thinking{Mode: llm.ThinkOff})
	newCall := waitLiveCall(t, model, "the new-generation cleanup request")
	newCall.complete(t, "New generation answer", nil)
	end := fixture.harness.rec.waitFor(t, "the Rethink result", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State.Ended()
	})
	if end.Result != "New generation answer" || model.requestCount() != 2 {
		t.Fatalf("Rethink result %+v, requests %d", end, model.requestCount())
	}
	snapshotsBeforeOldCommit := len(fixture.harness.rec.all())
	close(releaseCommit)
	waitFlightDone(t, flight)
	for _, snapshot := range fixture.harness.rec.all()[snapshotsBeforeOldCommit:] {
		if snapshot.ID == fixture.id && snapshot.Draft == "Old completed answer" {
			t.Fatalf("old generation published after Rethink: %+v", snapshot)
		}
	}
	assertSinglePaste(t, fixture, "New generation answer")
}

func TestLivePolishCompletionRacingWithSoftDeadlineKeepsRawResult(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newControlledLiveLLM(true)
	fixture := newLiveSessionFixture(t, "spoken words.", model, clock, true)
	flight := liveFlightOf(fixture.run)
	commitEntered, releaseCommit := make(chan struct{}), make(chan struct{})
	fixture.run.live.mu.Lock()
	fixture.run.live.testHooks = &livePolishTestHooks{beforeFlightCommit: func() {
		close(commitEntered)
		<-releaseCommit
	}}
	fixture.run.live.mu.Unlock()
	fixture.call.complete(t, "Late cleaned answer", nil)
	select {
	case <-commitEntered:
	case <-time.After(5 * time.Second):
		t.Fatal("the completed flight did not reach terminal publication")
	}
	fixture.stopAndReleaseFinal(t, 300*time.Millisecond)
	fixture.harness.rec.waitFor(t, "the composing session", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State == Composing
	})
	if clock.timerCount() != 1 || clock.timerDelay(0) != cleanupSoftDeadlineTestDuration-300*time.Millisecond {
		t.Fatalf("deadline timers %d, delay %v", clock.timerCount(), clock.timerDelay(0))
	}
	clock.Advance(cleanupSoftDeadlineTestDuration - 300*time.Millisecond)
	clock.Fire(t, 0)
	end := fixture.harness.rec.waitFor(t, "the raw deadline result", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State.Ended()
	})
	if end.Result != "spoken words." || end.Draft != "" || len(fixture.harness.text.pastes()) != 1 || fixture.harness.text.pastes()[0] != "spoken words." {
		t.Fatalf("deadline result %+v, pasted %q", end, fixture.harness.text.pastes())
	}
	snapshotsBeforeOldCommit := len(fixture.harness.rec.all())
	close(releaseCommit)
	waitFlightDone(t, flight)
	for _, snapshot := range fixture.harness.rec.all()[snapshotsBeforeOldCommit:] {
		if snapshot.ID == fixture.id && snapshot.Draft != "" {
			t.Fatalf("late completion restored Draft after deadline: %+v", snapshot)
		}
	}
	if model.requestCount() != 1 {
		t.Fatalf("requests %d, want only the speculative request", model.requestCount())
	}
}

func TestLivePolishOldSessionCannotPublishIntoReplacement(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newControlledLiveLLM(true)
	fixture := newLiveSessionFixture(t, "first session.", model, clock, false)
	fixture.stopAndReleaseFinal(t, 300*time.Millisecond)
	fixture.harness.rec.waitFor(t, "first ASR Final", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State == Composing
	})
	waitLiveAdoption(t, fixture.run)
	fixture.call.sendDelta(t, "old visible draft")
	fixture.harness.rec.waitFor(t, "the first session's draft", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.Draft == "old visible draft"
	})
	fixture.harness.m.Cancel(fixture.id)
	end := fixture.harness.rec.waitFor(t, "the cancelled first session", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State == Cancelled
	})
	if end.Result != "" || len(fixture.harness.text.pastes()) != 0 {
		t.Fatalf("cancelled session result %+v, pasted %q", end, fixture.harness.text.pastes())
	}
	if !fixture.harness.audio.waitStopped(2 * time.Second) {
		t.Fatal("the cancelled session's recording did not stop")
	}
	fixture.harness.audio.notifyAt = 3
	fixture.harness.audio.notified = make(chan struct{})
	fixture.harness.mu.Lock()
	fixture.harness.asr = &fakeStreamer{final: "second session.", partials: []string{"second session."}}
	fixture.harness.mu.Unlock()
	secondID := fixture.harness.start()
	fixture.harness.rec.waitFor(t, "the replacement recording", func(snapshot Snapshot) bool {
		return snapshot.ID == secondID && snapshot.State == Recording
	})
	select {
	case <-fixture.harness.audio.notified:
	case <-time.After(5 * time.Second):
		t.Fatal("the replacement recording did not send three frames")
	}
	newCall := waitLiveCall(t, model, "the replacement session's request")
	fixture.harness.m.Stop(secondID)
	fixture.harness.rec.waitFor(t, "the replacement ASR Final", func(snapshot Snapshot) bool {
		return snapshot.ID == secondID && snapshot.State == Composing
	})
	fixture.harness.m.mu.Lock()
	secondRun := fixture.harness.m.cur
	fixture.harness.m.mu.Unlock()
	waitLiveAdoption(t, secondRun)
	newCall.sendDelta(t, "new visible draft")
	fixture.harness.rec.waitFor(t, "the replacement draft", func(snapshot Snapshot) bool {
		return snapshot.ID == secondID && snapshot.Draft == "new visible draft"
	})
	fixture.call.sendDelta(t, " old late delta")
	if got := fixture.harness.m.Current().Draft; got != "new visible draft" {
		t.Fatalf("old session changed replacement Draft to %q", got)
	}
	newCall.complete(t, "new final answer", nil)
	end = fixture.harness.rec.waitFor(t, "the replacement result", func(snapshot Snapshot) bool {
		return snapshot.ID == secondID && snapshot.State.Ended()
	})
	fixture.call.complete(t, "old final answer", nil)
	if end.Result != "new final answer" || model.requestCount() != 2 {
		t.Fatalf("replacement result %+v, requests %d", end, model.requestCount())
	}
	if pastes := fixture.harness.text.pastes(); len(pastes) != 1 || pastes[0] != "new final answer" {
		t.Fatalf("pasted %q", pastes)
	}
}

var _ LLM = (*controlledLiveLLM)(nil)
