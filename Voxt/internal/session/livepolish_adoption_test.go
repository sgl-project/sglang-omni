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

func newLiveSessionFixture(t *testing.T, text string, model *controlledLiveLLM, clock *softDeadlineClock, softDeadline bool) *liveSessionFixture {
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
	id := h.start()
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

func (fixture *liveSessionFixture) beginStop(t *testing.T) time.Time {
	t.Helper()
	stopAt := fixture.clock.Now()
	fixture.harness.m.Stop(fixture.id)
	select {
	case <-fixture.streamer.finishStarted:
	case <-time.After(5 * time.Second):
		fixture.harness.mu.Lock()
		logs := append([]string(nil), fixture.harness.logs...)
		fixture.harness.mu.Unlock()
		t.Fatalf("ASR did not begin finalization: current=%+v snapshots=%+v logs=%v", fixture.harness.m.Current(), tail(fixture.harness.rec.all()), logs)
	}
	return stopAt
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

func (fixture *liveSessionFixture) stopAndReleaseFinal(t *testing.T, afterStop time.Duration) time.Time {
	t.Helper()
	stopAt := fixture.beginStop(t)
	fixture.clock.Advance(afterStop)
	fixture.releaseFinalAndWait(t)
	return stopAt
}

func waitLiveAdoption(t *testing.T, r *run) {
	t.Helper()
	waitLiveFor(t, r, "ASR Final validation", func(lp *livePolish) bool { return lp.closed })
}

func TestLivePolishAdoptsBufferedStreamBeforeCompletion(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newControlledLiveLLM(false)
	fixture := newLiveSessionFixture(t, "spoken words.", model, clock, false)
	fixture.call.sendDelta(t, " \n")
	fixture.call.sendDelta(t, "<out")
	if draft := fixture.harness.m.Current().Draft; draft != "" {
		t.Fatalf("speculative draft %q escaped while recording", draft)
	}
	stopAt := fixture.beginStop(t)
	clock.Advance(100 * time.Millisecond)
	fixture.call.sendDelta(t, "put>Cleaned")
	if draft := fixture.harness.m.Current().Draft; draft != "" {
		t.Fatalf("speculative draft %q escaped before ASR Final validation", draft)
	}
	clock.Advance(200 * time.Millisecond)
	fixture.releaseFinalAndWait(t)
	fixture.harness.rec.waitFor(t, "the adopted buffered draft", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State == Composing && snapshot.Draft == "Cleaned"
	})
	firstDraftAt := clock.Now()
	select {
	case <-fixture.call.returned:
		t.Fatal("adoption waited for the LLM completion gate")
	default:
	}
	if len(fixture.harness.text.pastes()) != 0 {
		t.Fatal("the session delivered before cleanup completed")
	}
	clock.Advance(draftEvery)
	fixture.call.sendDelta(t, " text</output>")
	fixture.harness.rec.waitFor(t, "the continued stream", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.Draft == "Cleaned text"
	})
	clock.Advance(660 * time.Millisecond)
	baselineFirstDraftAt := clock.Now() // the pre-adoption path first published only after this completion.
	fixture.call.complete(t, "<output>Cleaned text</output>", nil)
	end := fixture.harness.rec.wait(t)
	completeAt := clock.Now()
	if end.State != Done || end.Result != "Cleaned text" || fixture.model.requestCount() != 1 {
		t.Fatalf("end %+v, requests %d", end, fixture.model.requestCount())
	}
	if pastes := fixture.harness.text.pastes(); len(pastes) != 1 || pastes[0] != "Cleaned text" {
		t.Fatalf("pasted %q", pastes)
	}
	if got := firstDraftAt.Sub(stopAt); got != 300*time.Millisecond {
		t.Fatalf("stop-to-first-draft = %v, want 300 ms", got)
	}
	if got := completeAt.Sub(stopAt); got != time.Second {
		t.Fatalf("stop-to-completion = %v, want 1 s", got)
	}
	t.Logf("scenario=matching_prebuffered stop_to_first_cleaned_draft_ms=%d asr_final_to_first_cleaned_draft_ms=0 stop_to_cleanup_complete_ms=%d stop_to_paste_ms=%d speculative_flight_adopted=true buffer_nonempty_at_adoption=true completed_prefix_published=false llm_request_count=1 cleanup_outcome=cleaned baseline_wait_for_completion_stop_to_first_cleaned_draft_ms=%d", firstDraftAt.Sub(stopAt).Milliseconds(), completeAt.Sub(stopAt).Milliseconds(), completeAt.Sub(stopAt).Milliseconds(), baselineFirstDraftAt.Sub(stopAt).Milliseconds())
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
	firstDraftAt := clock.Now()
	select {
	case <-fixture.call.returned:
		t.Fatal("the first delta waited for the full answer")
	default:
	}
	clock.Advance(700 * time.Millisecond)
	baselineFirstDraftAt := clock.Now() // the pre-adoption path first published only after this completion.
	fixture.call.complete(t, "First visible answer", nil)
	end := fixture.harness.rec.wait(t)
	completeAt := clock.Now()
	if end.Result != "First visible answer" || fixture.model.requestCount() != 1 {
		t.Fatalf("end %+v, requests %d", end, fixture.model.requestCount())
	}
	if got := firstDraftAt.Sub(fixture.run.stoppedAt()); got != 300*time.Millisecond {
		t.Fatalf("stop-to-first-draft = %v, want 300 ms", got)
	}
	if got := completeAt.Sub(fixture.run.stoppedAt()); got != time.Second {
		t.Fatalf("stop-to-completion = %v, want 1 s", got)
	}
	t.Logf("scenario=matching_post_final_delta stop_to_first_cleaned_draft_ms=%d asr_final_to_first_cleaned_draft_ms=0 stop_to_cleanup_complete_ms=%d stop_to_paste_ms=%d speculative_flight_adopted=true buffer_nonempty_at_adoption=false completed_prefix_published=false llm_request_count=1 cleanup_outcome=cleaned baseline_wait_for_completion_stop_to_first_cleaned_draft_ms=%d", firstDraftAt.Sub(fixture.run.stoppedAt()).Milliseconds(), completeAt.Sub(fixture.run.stoppedAt()).Milliseconds(), completeAt.Sub(fixture.run.stoppedAt()).Milliseconds(), baselineFirstDraftAt.Sub(fixture.run.stoppedAt()).Milliseconds())
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
