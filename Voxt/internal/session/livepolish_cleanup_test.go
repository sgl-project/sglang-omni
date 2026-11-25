package session

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestLivePolishAdoptionConsumesFlightCompletedBeforeFirstRead(t *testing.T) {
	model := newControlledLiveLLM(false)
	r := newEarlyUnitRun(t, model, nil)
	text := "Hello there."
	r.hearLive(text)
	call := waitLiveCall(t, model, "the speculative request")
	r.m.mu.Lock()
	r.m.cur = r
	r.m.rev++
	r.m.snap = Snapshot{Rev: r.m.rev, ID: r.id, Kind: store.KindTranscribe, State: Composing}
	request := cleanupRequest{generation: r.cleanupGeneration, ctx: r.llmCtx, thinking: r.think}
	r.m.mu.Unlock()
	_, _, _, _, flight := r.live.finish(text)
	if flight == nil {
		t.Fatal("the matching flight was not adopted")
	}
	call.complete(t, "<output>Cleaned.</output>", nil)
	select {
	case <-flight.done:
	case <-time.After(5 * time.Second):
		t.Fatal("the speculative flight did not finish")
	}
	result, adopted := r.live.adopt(text, flight, request)
	if !adopted || len(result.answers) != 1 || result.answers[0] != "Cleaned." {
		t.Fatalf("completed flight result %+v, adopted %t", result, adopted)
	}
	if draft := r.m.Current().Draft; draft != "Cleaned." || model.requestCount() != 1 {
		t.Fatalf("draft %q, requests %d", draft, model.requestCount())
	}
}

func TestLivePolishPublishesOnlyTheRetainedCompletedPrefix(t *testing.T) {
	for _, test := range []struct {
		name      string
		reclean   int
		answer    string
		wantDraft string
		wantInput string
	}{
		{name: "none recleaned", reclean: 0, answer: "Alpha. Beta. Gamma.", wantDraft: "Alpha. Beta. Gamma.", wantInput: strings.Repeat("z", 285) + "."},
		{name: "one recleaned", reclean: 1, answer: "Alpha. Beta. Gamma.", wantDraft: "Alpha. Beta.", wantInput: "Gamma."},
		{name: "two recleaned", reclean: 2, answer: "Alpha. Beta. Gamma.", wantDraft: "Alpha.", wantInput: "Beta. Gamma."},
		{name: "no retained sentences", reclean: 2, answer: "Alpha. Beta.", wantDraft: "", wantInput: "Alpha. Beta."},
	} {
		t.Run(test.name, func(t *testing.T) {
			clock := newSoftDeadlineClock()
			model := newControlledLiveLLM(false)
			r := newEarlyUnitRun(t, model, nil)
			r.m.cfg.Now = clock.Now
			r.settings.Features.Transcribe.RecleanSentences = test.reclean
			block := "One. Two. Three."
			openTail := strings.Repeat("z", 285)
			r.hearLive(block + " " + openTail)
			first := waitLiveCall(t, model, "the completed live block")
			first.complete(t, test.answer, nil)
			waitLiveFor(t, r, "the committed block", func(lp *livePolish) bool { return len(lp.blocks) == 1 })
			finalText := block + " " + openTail + "."
			stopAt := clock.Now()
			clock.Advance(300 * time.Millisecond)
			result := make(chan string, 1)
			go func() {
				got, _ := liveCleanUp(r, finalText)
				result <- got
			}()
			tail := waitLiveCall(t, model, "the tail request")
			if got := r.m.Current().Draft; got != test.wantDraft {
				t.Fatalf("draft before tail completion %q, want retained prefix %q", got, test.wantDraft)
			}
			if got := inputOf(tail.request); !strings.HasPrefix(got, test.wantInput) {
				t.Fatalf("tail input %q, want prefix %q", got, test.wantInput)
			}
			firstDraftAt := clock.Now()
			tailInput := inputOf(tail.request)
			clock.Advance(700 * time.Millisecond)
			baselineFirstDraftAt := clock.Now() // the previous path first published only after the tail completed.
			tail.complete(t, tailInput, nil)
			wantFinal := strings.TrimSpace(test.wantDraft)
			if wantFinal != "" {
				wantFinal += " "
			}
			wantFinal += tailInput
			select {
			case got := <-result:
				if got != wantFinal {
					t.Fatalf("final output %q, want %q", got, wantFinal)
				}
				if test.name == "one recleaned" {
					t.Logf("scenario=completed_prefix_slow_tail stop_to_first_cleaned_draft_ms=%d asr_final_to_first_cleaned_draft_ms=0 stop_to_cleanup_complete_ms=%d speculative_flight_adopted=false buffer_nonempty_at_adoption=false completed_prefix_published=true llm_request_count=%d cleanup_outcome=cleaned baseline_wait_for_tail_stop_to_first_cleaned_draft_ms=%d", firstDraftAt.Sub(stopAt).Milliseconds(), clock.Now().Sub(stopAt).Milliseconds(), model.requestCount(), baselineFirstDraftAt.Sub(stopAt).Milliseconds())
				}
			case <-time.After(5 * time.Second):
				t.Fatal("tail cleanup did not finish")
			}
			if model.requestCount() != 2 {
				t.Fatalf("requests %d, want one live block and one tail", model.requestCount())
			}
		})
	}
}

type speculativeRetryBackend struct {
	calls       atomic.Int32
	emitOnError bool
}

func (backend *speculativeRetryBackend) Info() llm.BackendInfo {
	return llm.BackendInfo{ID: "retry-test"}
}

func (backend *speculativeRetryBackend) Chat(_ context.Context, request llm.Request) (llm.Response, error) {
	call := backend.calls.Add(1)
	if call == 1 {
		if backend.emitOnError && request.OnDelta != nil {
			request.OnDelta("partial attempt")
		}
		return llm.Response{}, &llm.Error{Code: llm.CodeServer, Retryable: true}
	}
	return llm.Response{Text: "complete answer"}, nil
}

func TestLivePolishStreamingCallbackRetrySemantics(t *testing.T) {
	for _, test := range []struct {
		name         string
		emitOnError  bool
		wantAttempts int32
	}{
		{name: "retry before first delta", wantAttempts: 2},
		{name: "do not combine after private delta", emitOnError: true, wantAttempts: 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			backend := &speculativeRetryBackend{emitOnError: test.emitOnError}
			client := &llm.Client{
				Backend: backend,
				Retry:   llm.RetryPolicy{MaxAttempts: 2, Base: time.Nanosecond, Max: time.Nanosecond},
				Sleep:   func(context.Context, time.Duration) error { return nil },
			}
			r := newEarlyUnitRun(t, client, nil)
			r.hearLive("Hello there.")
			if test.emitOnError {
				waitLiveFor(t, r, "the speculative request failure", func(lp *livePolish) bool { return lp.closed })
				if got := backend.calls.Load(); got != 1 {
					t.Fatalf("attempts after a private delta %d, want one", got)
				}
				if got, err := liveCleanUp(r, "Hello there."); err != nil || got != "complete answer" {
					t.Fatalf("fallback cleanup %q, %v", got, err)
				}
			} else {
				waitLive(t, r, "Hello there.")
				if got, err := liveCleanUp(r, "Hello there."); err != nil || got != "complete answer" {
					t.Fatalf("cleanup %q, %v", got, err)
				}
			}
			if got := backend.calls.Load(); got != test.wantAttempts {
				t.Fatalf("backend attempts %d, want %d", got, test.wantAttempts)
			}
		})
	}
}

func TestLivePolishProviderFailureAfterDeltaReplacesThePrivateAttempt(t *testing.T) {
	model := newControlledLiveLLM(false)
	r := newEarlyUnitRun(t, model, nil)
	r.hearLive("Hello there.")
	call := waitLiveCall(t, model, "the speculative request")
	call.sendDelta(t, "partial private attempt")
	call.complete(t, "", errors.New("provider failed"))
	waitLiveFor(t, r, "the failed live request", func(lp *livePolish) bool { return lp.closed })
	result := make(chan string, 1)
	go func() {
		got, _ := liveCleanUp(r, "Hello there.")
		result <- got
	}()
	retry := waitLiveCall(t, model, "the full-text cleanup after failure")
	if got := r.m.Current().Draft; got != "" {
		t.Fatalf("failed partial draft %q remained visible", got)
	}
	retry.complete(t, "complete answer", nil)
	select {
	case got := <-result:
		if got != "complete answer" {
			t.Fatalf("final output %q", got)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("fallback cleanup did not finish")
	}
	if model.requestCount() != 2 {
		t.Fatalf("requests %d, want the failed speculative call and one replacement", model.requestCount())
	}
}

func TestLivePolishAdoptedFailureClearsDraftBeforeFallback(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newControlledLiveLLM(false)
	fixture := newLiveSessionFixture(t, "spoken words.", model, clock, false)
	fallback := &fakeLLM{text: "fallback cleaned answer"}
	fixture.harness.m.mu.Lock()
	fixture.run.svc.Fallback = fallback
	fixture.run.svc.FallbackInfo = Model{ID: "ollama", Name: "Ollama", Model: "qwen3"}
	fixture.harness.m.mu.Unlock()
	fixture.stopAndReleaseFinal(t, 300*time.Millisecond)
	fixture.harness.rec.waitFor(t, "ASR Final", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State == Composing
	})
	waitLiveAdoption(t, fixture.run)
	fixture.call.sendDelta(t, "failed partial answer")
	fixture.harness.rec.waitFor(t, "the adopted partial", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.Draft == "failed partial answer"
	})
	fixture.call.complete(t, "", errors.New("speculative provider failed"))
	primaryRetry := waitLiveCall(t, model, "the whole-text primary retry")
	if draft := fixture.harness.m.Current().Draft; draft != "" {
		t.Fatalf("failed speculative draft %q remained during fallback", draft)
	}
	primaryRetry.complete(t, "", errors.New("whole-text primary failed"))
	end := fixture.harness.rec.waitFor(t, "the fallback result", func(snapshot Snapshot) bool {
		return snapshot.ID == fixture.id && snapshot.State.Ended()
	})
	if end.Result != "fallback cleaned answer" || end.LLM.ID != "ollama" || model.requestCount() != 2 || len(fallback.requests()) != 1 {
		t.Fatalf("fallback result %+v, primary requests %d, fallback requests %d", end, model.requestCount(), len(fallback.requests()))
	}
}

func TestLivePolishUsesDictionaryNormalizedFlightSource(t *testing.T) {
	model := newControlledLiveLLM(false)
	r := newEarlyUnitRun(t, model, []store.Term{{Term: "VoxtNew", Aliases: []string{"aura link"}, Enabled: true}})
	r.hearLive("aura link is ready.")
	call := waitLiveCall(t, model, "the dictionary-normalized request")
	if got := inputOf(call.request); got != "VoxtNew is ready." {
		t.Fatalf("flight source %q", got)
	}
	call.complete(t, "VoxtNew is ready.", nil)
	waitLive(t, r, "VoxtNew is ready.")
	got, err := liveCleanUp(r, "VoxtNew is ready.")
	if err != nil || got != "VoxtNew is ready." || model.requestCount() != 1 {
		t.Fatalf("cleanup %q, error %v, requests %d", got, err, model.requestCount())
	}
}
