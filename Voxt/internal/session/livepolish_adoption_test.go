package session

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type liveDelta struct {
	text string
	ack  chan struct{}
}

type liveReply struct {
	text string
	err  error
}

type controlledLiveCall struct {
	request  llm.Request
	ctx      context.Context
	deltas   chan liveDelta
	replies  chan liveReply
	returned chan struct{}
}

func (call *controlledLiveCall) sendDelta(t *testing.T, text string) {
	t.Helper()
	ack := make(chan struct{})
	select {
	case call.deltas <- liveDelta{text: text, ack: ack}:
	case <-call.returned:
		t.Fatalf("LLM call ended before the delta was accepted (context error: %v)", call.ctx.Err())
	case <-time.After(5 * time.Second):
		t.Fatal("LLM call did not accept a delta")
	}
	select {
	case <-ack:
	case <-call.returned:
		t.Fatalf("LLM call ended before the delta callback ran (context error: %v)", call.ctx.Err())
	case <-time.After(5 * time.Second):
		t.Fatal("LLM delta callback did not run")
	}
}

func (call *controlledLiveCall) complete(t *testing.T, text string, err error) {
	t.Helper()
	select {
	case call.replies <- liveReply{text: text, err: err}:
	case <-call.returned:
		return
	case <-time.After(5 * time.Second):
		t.Fatal("LLM call did not accept its completion")
	}
	select {
	case <-call.returned:
	case <-time.After(5 * time.Second):
		t.Fatal("LLM call did not return")
	}
}

type controlledLiveLLM struct {
	calls              chan *controlledLiveCall
	closed             chan struct{}
	closeOnce          sync.Once
	mu                 sync.Mutex
	requests           []llm.Request
	ignoreCancellation bool
}

func newControlledLiveLLM(ignoreCancellation bool) *controlledLiveLLM {
	return &controlledLiveLLM{
		calls: make(chan *controlledLiveCall, 16), closed: make(chan struct{}),
		ignoreCancellation: ignoreCancellation,
	}
}

func (model *controlledLiveLLM) Chat(ctx context.Context, request llm.Request) (llm.Response, error) {
	call := &controlledLiveCall{
		request: request, ctx: ctx, deltas: make(chan liveDelta), replies: make(chan liveReply), returned: make(chan struct{}),
	}
	model.mu.Lock()
	model.requests = append(model.requests, request)
	model.mu.Unlock()
	model.calls <- call
	defer close(call.returned)
	for {
		if model.ignoreCancellation {
			select {
			case delta := <-call.deltas:
				if request.OnDelta != nil {
					request.OnDelta(delta.text)
				}
				close(delta.ack)
			case reply := <-call.replies:
				return llm.Response{Text: reply.text}, reply.err
			case <-model.closed:
				return llm.Response{}, context.Canceled
			}
		} else {
			select {
			case delta := <-call.deltas:
				if request.OnDelta != nil {
					request.OnDelta(delta.text)
				}
				close(delta.ack)
			case reply := <-call.replies:
				return llm.Response{Text: reply.text}, reply.err
			case <-ctx.Done():
				return llm.Response{}, ctx.Err()
			case <-model.closed:
				return llm.Response{}, context.Canceled
			}
		}
	}
}

func (model *controlledLiveLLM) close() {
	model.closeOnce.Do(func() { close(model.closed) })
}

func (model *controlledLiveLLM) requestCount() int {
	model.mu.Lock()
	defer model.mu.Unlock()
	return len(model.requests)
}

func waitLiveCall(t *testing.T, model *controlledLiveLLM, what string) *controlledLiveCall {
	t.Helper()
	select {
	case call := <-model.calls:
		return call
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
		return nil
	}
}

type liveSessionFixture struct {
	harness   *harness
	run       *run
	model     *controlledLiveLLM
	streamer  *fakeStreamer
	id        string
	clock     *softDeadlineClock
	finalGate chan struct{}
	finalOnce sync.Once
	call      *controlledLiveCall
}

func newLiveSessionFixture(t *testing.T, text string, model *controlledLiveLLM, clock *softDeadlineClock, softDeadline bool, thinking ...*llm.Thinking) *liveSessionFixture {
	t.Helper()
	finalGate := make(chan struct{})
	streamer := &fakeStreamer{
		final: text, partials: []string{text}, finishGate: finalGate,
		finishStarted: make(chan struct{}, 1), finishEmitted: make(chan bool, 1),
	}
	h := newHarness(t, streamer, model, func(config *Config, currentSettings *settings.Settings) {
		config.Now = clock.Now
		config.AfterFunc = clock.AfterFunc
		currentSettings.Audio.VAD.Off = true
		if softDeadline {
			config.CleanupLatency = CleanupLatencyPolicy{AutoFallbackAfter: cleanupSoftDeadlineTestDuration}
		}
		currentSettings.Features.Transcribe.Enhance = true
	})
	h.audio.notifyAt = 3 // meet DefaultMinRecording before the fake Stop
	h.audio.notified = make(chan struct{})
	request := StartRequest{Kind: store.KindTranscribe, Target: slack}
	if len(thinking) > 0 {
		request.Thinking = thinking[0]
	}
	id, err := h.m.Start(request)
	if err != nil {
		t.Fatal(err)
	}
	h.rec.waitFor(t, "recording", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State == Recording })
	select {
	case <-h.audio.notified:
	case <-time.After(5 * time.Second):
		t.Fatal("recording did not send three frames")
	}
	call := waitLiveCall(t, model, "the speculative request")
	h.m.mu.Lock()
	r := h.m.cur
	h.m.mu.Unlock()
	fixture := &liveSessionFixture{harness: h, run: r, model: model, streamer: streamer, id: id, clock: clock, finalGate: finalGate, call: call}
	t.Cleanup(func() {
		model.close()
		fixture.releaseFinal()
	})
	return fixture
}

func (fixture *liveSessionFixture) releaseFinal() {
	fixture.finalOnce.Do(func() { close(fixture.finalGate) })
}

func (fixture *liveSessionFixture) beginStop(t *testing.T) {
	t.Helper()
	fixture.harness.m.Stop(fixture.id)
	select {
	case <-fixture.streamer.finishStarted:
	case <-time.After(5 * time.Second):
		fixture.harness.mu.Lock()
		logs := append([]string(nil), fixture.harness.logs...)
		fixture.harness.mu.Unlock()
		t.Fatalf("ASR did not begin finalization: current=%+v snapshots=%+v logs=%v", fixture.harness.m.Current(), tail(fixture.harness.rec.all()), logs)
	}
}

func (fixture *liveSessionFixture) releaseFinalAndWait(t *testing.T) {
	t.Helper()
	fixture.releaseFinal()
	select {
	case emitted := <-fixture.streamer.finishEmitted:
		if !emitted {
			t.Fatal("ASR Finish did not emit its Final event")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("ASR Finish did not return after releasing its Final gate")
	}
}

func (fixture *liveSessionFixture) stopAndReleaseFinal(t *testing.T, afterStop time.Duration) {
	t.Helper()
	fixture.beginStop(t)
	fixture.clock.Advance(afterStop)
	fixture.releaseFinalAndWait(t)
}

func waitLiveAdoption(t *testing.T, r *run) {
	t.Helper()
	waitLiveFor(t, r, "ASR Final validation", func(lp *livePolish) bool { return lp.closed })
}

func TestLivePolishAdoptsBufferedStreamBeforeCompletion(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newControlledLiveLLM(false)
	fixture := newLiveSessionFixture(t, "aura link is ready.", model, clock, false)
	if got := inputOf(fixture.call.request); got != "VoxtNew is ready." {
		t.Fatalf("normalized speculative source %q", got)
	}
	fixture.call.sendDelta(t, " \n")
	fixture.call.sendDelta(t, "<out")
	if draft := fixture.harness.m.Current().Draft; draft != "" {
		t.Fatalf("speculative draft %q escaped while recording", draft)
	}
	fixture.beginStop(t)
	clock.Advance(100 * time.Millisecond)
	fixture.call.sendDelta(t, "put>VoxtNew")
	if draft := fixture.harness.m.Current().Draft; draft != "" {
		t.Fatalf("speculative draft %q escaped before ASR Final validation", draft)
	}
	clock.Advance(200 * time.Millisecond)
	fixture.releaseFinalAndWait(t)
	fixture.harness.rec.waitFor(t, "the adopted buffered draft", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State == Composing && snapshot.Draft == "VoxtNew"
	})
	select {
	case <-fixture.call.returned:
		t.Fatal("adoption waited for the LLM completion gate")
	default:
	}
	if len(fixture.harness.text.pastes()) != 0 {
		t.Fatal("the session delivered before cleanup completed")
	}
	clock.Advance(draftEvery)
	fixture.call.sendDelta(t, " is ready</output>")
	fixture.harness.rec.waitFor(t, "the continued stream", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.Draft == "VoxtNew is ready"
	})
	clock.Advance(660 * time.Millisecond)
	fixture.call.complete(t, "<output>VoxtNew is ready</output>", nil)
	end := fixture.harness.rec.wait(t)
	if end.State != Done || end.Result != "VoxtNew is ready" || fixture.model.requestCount() != 1 {
		t.Fatalf("end %+v, requests %d", end, fixture.model.requestCount())
	}
	if pastes := fixture.harness.text.pastes(); len(pastes) != 1 || pastes[0] != "VoxtNew is ready" {
		t.Fatalf("pasted %q", pastes)
	}
}

func TestLivePolishCompletedFlightBeforeFirstAdoptReadIsDeliveredOnce(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newControlledLiveLLM(false)
	fixture := newLiveSessionFixture(t, "spoken words.", model, clock, false)
	flight := liveFlightOf(fixture.run)
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	fixture.run.live.mu.Lock()
	fixture.run.live.testHooks = &livePolishTestHooks{afterFinishAdopted: func() {
		once.Do(func() { close(entered); <-release })
	}}
	fixture.run.live.mu.Unlock()
	fixture.stopAndReleaseFinal(t, 300*time.Millisecond)
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("Stop did not adopt the matching flight")
	}
	fixture.call.complete(t, "Completed answer", nil)
	waitFlightDone(t, flight)
	close(release)
	end := fixture.harness.rec.waitFor(t, "completed-flight delivery", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State.Ended()
	})
	if end.Result != "Completed answer" || end.Draft != "Completed answer" || fixture.model.requestCount() != 1 {
		t.Fatalf("result %+v, requests %d", end, fixture.model.requestCount())
	}
	assertSinglePaste(t, fixture, "Completed answer")
}

func TestLivePolishCompletionBetweenSnapshotAndWakeUsesFreshState(t *testing.T) {
	for _, test := range []struct {
		name        string
		updateReady bool
		wantPending int
	}{
		{name: "completion notification ready"},
		{name: "completion and update notifications both ready", updateReady: true, wantPending: 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			clock := newSoftDeadlineClock()
			model := newControlledLiveLLM(false)
			fixture := newLiveSessionFixture(t, "spoken words.", model, clock, false)
			flight := liveFlightOf(fixture.run)
			drainFlightUpdates(flight)
			entered, release := make(chan struct{}), make(chan struct{})
			var once sync.Once
			fixture.run.live.mu.Lock()
			fixture.run.live.testHooks = &livePolishTestHooks{afterAdoptSnapshot: func() {
				once.Do(func() { close(entered); <-release })
			}}
			fixture.run.live.mu.Unlock()
			fixture.stopAndReleaseFinal(t, 300*time.Millisecond)
			select {
			case <-entered:
			case <-time.After(5 * time.Second):
				t.Fatal("adoption did not take its active-state snapshot")
			}
			if test.updateReady {
				fixture.call.sendDelta(t, "Update after snapshot")
			}
			fixture.call.complete(t, "Completed authoritative answer", nil)
			waitFlightDone(t, flight)
			if !test.updateReady {
				drainFlightUpdates(flight)
			}
			if len(flight.updates) != test.wantPending {
				t.Fatalf("pending update notifications %d, want %d", len(flight.updates), test.wantPending)
			}
			close(release)
			var end Snapshot
			select {
			case extra := <-model.calls:
				t.Fatalf("completion snapshot caused replacement request %+v", extra.request)
			case end = <-fixture.harness.rec.ended:
			case <-time.After(5 * time.Second):
				t.Fatal("session did not finish after completion notification")
			}
			if end.Result != "Completed authoritative answer" || fixture.model.requestCount() != 1 {
				t.Fatalf("result %+v, requests %d", end, fixture.model.requestCount())
			}
			assertSinglePaste(t, fixture, "Completed authoritative answer")
		})
	}
}

func TestLivePolishCompletionDuringPartialDraftPublication(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newControlledLiveLLM(false)
	fixture := newLiveSessionFixture(t, "spoken words.", model, clock, false)
	flight := liveFlightOf(fixture.run)
	fixture.call.sendDelta(t, "Partial text")
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	fixture.run.live.mu.Lock()
	fixture.run.live.testHooks = &livePolishTestHooks{beforeAdoptDraftPublish: func() {
		once.Do(func() { close(entered); <-release })
	}}
	fixture.run.live.mu.Unlock()
	fixture.stopAndReleaseFinal(t, 300*time.Millisecond)
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("the partial Draft publication did not start")
	}
	fixture.call.complete(t, "Authoritative completed answer", nil)
	waitFlightDone(t, flight)
	close(release)
	end := fixture.harness.rec.waitFor(t, "completed response after partial publication", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State.Ended()
	})
	if end.Result != "Authoritative completed answer" || fixture.model.requestCount() != 1 {
		t.Fatalf("result %+v, requests %d", end, fixture.model.requestCount())
	}
	assertSinglePaste(t, fixture, "Authoritative completed answer")
}

func TestLivePolishNormalCompletionContextDisposalDoesNotRejectFlight(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newControlledLiveLLM(false)
	fixture := newLiveSessionFixture(t, "spoken words.", model, clock, false)
	flight := liveFlightOf(fixture.run)
	entered, release := make(chan struct{}), make(chan struct{})
	var once sync.Once
	fixture.run.live.mu.Lock()
	fixture.run.live.testHooks = &livePolishTestHooks{beforeFlightCommit: func() {
		once.Do(func() { close(entered); <-release })
	}}
	fixture.run.live.mu.Unlock()
	fixture.call.complete(t, "Normal completed answer", nil)
	select {
	case <-entered:
	case <-time.After(5 * time.Second):
		t.Fatal("worker did not reach terminal-state publication")
	}
	if err := flight.ctx.Err(); err != nil {
		t.Fatalf("normal completion disposed the context before terminal state: %v", err)
	}
	fixture.stopAndReleaseFinal(t, 300*time.Millisecond)
	fixture.harness.rec.waitFor(t, "ASR Final", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State == Composing
	})
	waitLiveAdoption(t, fixture.run)
	if err := flight.ctx.Err(); err != nil {
		t.Fatalf("finish cancelled a matching successful flight: %v", err)
	}
	close(release)
	end := fixture.harness.rec.waitFor(t, "normal completion after Stop", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State.Ended()
	})
	if end.Result != "Normal completed answer" || fixture.model.requestCount() != 1 {
		t.Fatalf("result %+v, requests %d", end, fixture.model.requestCount())
	}
	assertSinglePaste(t, fixture, "Normal completed answer")
}

func liveFlightOf(r *run) *liveFlight {
	r.live.mu.Lock()
	defer r.live.mu.Unlock()
	return r.live.flight
}

func waitFlightDone(t *testing.T, flight *liveFlight) {
	t.Helper()
	select {
	case <-flight.done:
	case <-time.After(5 * time.Second):
		t.Fatal("the live flight did not complete")
	}
}

func drainFlightUpdates(flight *liveFlight) {
	for {
		select {
		case <-flight.updates:
		default:
			return
		}
	}
}

func assertSinglePaste(t *testing.T, fixture *liveSessionFixture, want string) {
	t.Helper()
	if pastes := fixture.harness.text.pastes(); len(pastes) != 1 || pastes[0] != want {
		t.Fatalf("pastes %q, want one %q", pastes, want)
	}
}

func TestLivePolishAdoptsFirstDeltaAfterFinalValidation(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newControlledLiveLLM(false)
	fixture := newLiveSessionFixture(t, "spoken words.", model, clock, false)
	fixture.stopAndReleaseFinal(t, 300*time.Millisecond)
	fixture.harness.rec.waitFor(t, "ASR Final", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State == Composing
	})
	waitLiveAdoption(t, fixture.run)
	if draft := fixture.harness.m.Current().Draft; draft != "" {
		t.Fatalf("empty stream published %q before its first delta", draft)
	}
	fixture.call.sendDelta(t, " \t\n")
	if draft := fixture.harness.m.Current().Draft; draft != "" {
		t.Fatalf("whitespace-only delta published %q", draft)
	}
	fixture.call.sendDelta(t, "First visible answer")
	fixture.harness.rec.waitFor(t, "the first post-Final delta", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.Draft == "First visible answer"
	})
	select {
	case <-fixture.call.returned:
		t.Fatal("the first delta waited for the full answer")
	default:
	}
	clock.Advance(700 * time.Millisecond)
	fixture.call.complete(t, "First visible answer", nil)
	end := fixture.harness.rec.wait(t)
	if end.Result != "First visible answer" || fixture.model.requestCount() != 1 {
		t.Fatalf("end %+v, requests %d", end, fixture.model.requestCount())
	}
}

func TestLivePolishAdoptsMixedLanguagePartsInOrder(t *testing.T) {
	model := newControlledLiveLLM(false)
	h, r := newEarlyUnitHarnessRun(t, model, nil)
	text := "Hello there. 你好。"
	r.hearLive(text)
	first := waitLiveCall(t, model, "the English part")
	if got := inputOf(first.request); got != "Hello there." {
		t.Fatalf("first part %q", got)
	}
	first.complete(t, "Hello there.", nil)
	second := waitLiveCall(t, model, "the Chinese part")
	if got := inputOf(second.request); got != "你好。" {
		t.Fatalf("second part %q", got)
	}
	result := make(chan string, 1)
	go func() {
		got, _ := liveCleanUp(r, text)
		result <- got
	}()
	waitLiveAdoption(t, r)
	h.rec.waitFor(t, "the completed part draft", func(snapshot Snapshot) bool {
		return snapshot.ID == r.id && snapshot.Draft == "Hello there."
	})
	second.sendDelta(t, "你好")
	second.complete(t, "你好。", nil)
	select {
	case got := <-result:
		if got != "Hello there. 你好。" {
			t.Fatalf("final output %q", got)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the adopted multi-part request did not finish")
	}
	if model.requestCount() != 2 {
		t.Fatalf("requests %d, want one per language part", model.requestCount())
	}
}

func TestLivePolishAdoptedDraftExpandsMarkedSnippet(t *testing.T) {
	model := newControlledLiveLLM(false)
	h, r := newEarlyUnitHarnessRun(t, model, nil)
	r.snippets = []snippet.Snippet{{ID: "email", Trigger: "my email", Text: "hello@example.com"}}
	text := "My email please."
	r.hearLive(text)
	call := waitLiveCall(t, model, "the snippet-marked request")
	if got := inputOf(call.request); !strings.Contains(got, snippet.Open+"My email"+snippet.Close) {
		t.Fatalf("request did not preserve snippet markers: %q", got)
	}
	result := make(chan string, 1)
	go func() {
		got, _ := liveCleanUp(r, text)
		result <- got
	}()
	waitLiveAdoption(t, r)
	call.sendDelta(t, snippet.Open+"My email"+snippet.Close+" please.")
	h.rec.waitFor(t, "the expanded snippet draft", func(snapshot Snapshot) bool {
		return snapshot.ID == r.id && snapshot.Draft == "hello@example.com please."
	})
	call.complete(t, snippet.Open+"My email"+snippet.Close+" please.", nil)
	select {
	case got := <-result:
		if expanded := r.expandSnippets(text, got); expanded != "hello@example.com please." {
			t.Fatalf("expanded final output %q", expanded)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("snippet cleanup did not finish")
	}
}
