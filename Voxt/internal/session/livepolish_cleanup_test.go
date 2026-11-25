package session

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
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
			tailCompletionAt := clock.Now()
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
					t.Logf("scenario=completed_prefix_slow_tail stop_to_first_cleaned_draft_ms=%d asr_final_to_first_cleaned_draft_ms=0 stop_to_cleanup_complete_ms=%d completed_prefix_published=true llm_request_count=%d cleanup_outcome=cleaned scheduled_tail_completion_ms=%d", firstDraftAt.Sub(stopAt).Milliseconds(), clock.Now().Sub(stopAt).Milliseconds(), model.requestCount(), tailCompletionAt.Sub(stopAt).Milliseconds())
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

type retrySessionBackend struct {
	calls         atomic.Int32
	firstDelta    string
	firstRelease  <-chan struct{}
	firstStarted  chan struct{}
	secondDelta   string
	secondRelease <-chan struct{}
	secondStarted chan struct{}
	secondText    string
}

func (backend *retrySessionBackend) Info() llm.BackendInfo {
	return llm.BackendInfo{ID: "retry-session-test"}
}

func (backend *retrySessionBackend) Chat(ctx context.Context, request llm.Request) (llm.Response, error) {
	switch backend.calls.Add(1) {
	case 1:
		if request.OnDelta != nil {
			request.OnDelta(backend.firstDelta)
		}
		close(backend.firstStarted)
		select {
		case <-backend.firstRelease:
			return llm.Response{}, &llm.Error{Code: llm.CodeServer, Retryable: true}
		case <-ctx.Done():
			return llm.Response{}, ctx.Err()
		}
	case 2:
		if request.OnDelta != nil && backend.secondDelta != "" {
			request.OnDelta(backend.secondDelta)
		}
		if backend.secondStarted != nil {
			close(backend.secondStarted)
		}
		if backend.secondRelease != nil {
			select {
			case <-backend.secondRelease:
			case <-ctx.Done():
				return llm.Response{}, ctx.Err()
			}
		}
		return llm.Response{Text: backend.secondText}, nil
	default:
		return llm.Response{}, errors.New("unexpected backend request")
	}
}

func TestLivePolishStreamingCallbackRetrySemantics(t *testing.T) {
	for _, test := range []struct {
		name        string
		emitOnError bool
	}{
		{name: "retry before first delta"},
		{name: "retry after private delta before adoption", emitOnError: true},
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
			waitLive(t, r, "Hello there.")
			if got, err := liveCleanUp(r, "Hello there."); err != nil || got != "complete answer" {
				t.Fatalf("cleanup %q, %v", got, err)
			}
			if got := backend.calls.Load(); got != 2 {
				t.Fatalf("backend attempts %d, want speculative retry and no replacement request", got)
			}
		})
	}
}

type multipartRetryBackend struct {
	calls atomic.Int32
}

func (backend *multipartRetryBackend) Info() llm.BackendInfo {
	return llm.BackendInfo{ID: "multipart-retry-test"}
}

func (backend *multipartRetryBackend) Chat(_ context.Context, request llm.Request) (llm.Response, error) {
	switch backend.calls.Add(1) {
	case 1:
		if request.OnDelta != nil {
			request.OnDelta("failed attempt text")
		}
		return llm.Response{}, &llm.Error{Code: llm.CodeServer, Retryable: true}
	case 2:
		if request.OnDelta != nil {
			request.OnDelta("First cleaned.")
		}
		return llm.Response{Text: "First cleaned."}, nil
	case 3:
		if request.OnDelta != nil {
			request.OnDelta("Second cleaned.")
		}
		return llm.Response{Text: "Second cleaned."}, nil
	default:
		return llm.Response{}, errors.New("unexpected multipart request")
	}
}

func TestLivePolishMultipartRequestRetriesOnePrivatePart(t *testing.T) {
	backend := &multipartRetryBackend{}
	client := &llm.Client{
		Backend: backend,
		Retry:   llm.RetryPolicy{MaxAttempts: 2},
		Sleep:   func(context.Context, time.Duration) error { return nil },
	}
	r := newEarlyUnitRun(t, client, nil)
	r.settings.Features.Transcribe.RecleanSentences = 0
	firstSentence := strings.Repeat("a", 180) + "."
	secondSentence := strings.Repeat("b", 180) + "."
	transcript := firstSentence + " " + secondSentence
	r.hearLive(transcript)
	waitLiveFor(t, r, "the committed first part and completed second part", func(polish *livePolish) bool {
		return len(polish.blocks) == 1 && len(polish.done) == 1 && polish.flight == nil
	})
	result, err := liveCleanUp(r, transcript)
	if err != nil || result != "First cleaned. Second cleaned." || backend.calls.Load() != 3 {
		t.Fatalf("result %q, error %v, backend attempts %d", result, err, backend.calls.Load())
	}
}

func TestLivePolishRetryAuthorizationWinsBeforeAdoptionDuringBackoff(t *testing.T) {
	releaseFirst := make(chan struct{})
	releaseBackoff := make(chan struct{})
	releaseSecond := make(chan struct{})
	backoffStarted := make(chan struct{})
	backend := &retrySessionBackend{
		firstDelta: "old attempt text", firstRelease: releaseFirst, firstStarted: make(chan struct{}),
		secondDelta: "new attempt text", secondRelease: releaseSecond, secondStarted: make(chan struct{}), secondText: "new attempt text",
	}
	client := &llm.Client{
		Backend: backend, Retry: llm.RetryPolicy{MaxAttempts: 2, Base: time.Nanosecond, Max: time.Nanosecond},
		Sleep: func(ctx context.Context, _ time.Duration) error {
			close(backoffStarted)
			select {
			case <-releaseBackoff:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		},
	}
	h, r := newEarlyUnitHarnessRun(t, client, nil)
	r.hearLive("Hello there.")
	select {
	case <-backend.firstStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("the first speculative attempt did not start")
	}
	close(releaseFirst)
	select {
	case <-backoffStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("retry authorization did not enter backoff")
	}
	r.live.mu.Lock()
	if got := r.live.flight.delta.String(); got != "" {
		r.live.mu.Unlock()
		t.Fatalf("failed attempt buffer %q was not cleared before retry", got)
	}
	r.live.mu.Unlock()
	result := make(chan string, 1)
	go func() {
		got, _ := liveCleanUp(r, "Hello there.")
		result <- got
	}()
	waitLiveFor(t, r, "adoption during retry backoff", func(lp *livePolish) bool {
		return lp.flight != nil && lp.flight.adopted
	})
	close(releaseBackoff)
	select {
	case <-backend.secondStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("the authorized retry did not start")
	}
	h.rec.waitFor(t, "the second attempt's draft", func(snapshot Snapshot) bool {
		return snapshot.ID == r.id && snapshot.Draft == "new attempt text"
	})
	if draft := h.m.Current().Draft; strings.Contains(draft, "old attempt") {
		t.Fatalf("failed attempt contaminated Draft: %q", draft)
	}
	close(releaseSecond)
	select {
	case got := <-result:
		if got != "new attempt text" {
			t.Fatalf("cleanup output %q", got)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("cleanup did not consume the retried response")
	}
	if backend.calls.Load() != 2 {
		t.Fatalf("backend attempts %d, want one retry and no replacement request", backend.calls.Load())
	}
}

func TestLivePolishAdoptionPreventsRetryAfterExposure(t *testing.T) {
	releaseFirst := make(chan struct{})
	releaseFallback := make(chan struct{})
	backend := &retrySessionBackend{
		firstDelta: "visible partial", firstRelease: releaseFirst, firstStarted: make(chan struct{}),
		secondRelease: releaseFallback, secondStarted: make(chan struct{}), secondText: "fallback cleanup",
	}
	var sleeps atomic.Int32
	client := &llm.Client{
		Backend: backend, Retry: llm.RetryPolicy{MaxAttempts: 2, Base: time.Nanosecond, Max: time.Nanosecond},
		Sleep: func(context.Context, time.Duration) error { sleeps.Add(1); return nil },
	}
	h, r := newEarlyUnitHarnessRun(t, client, nil)
	r.hearLive("Hello there.")
	select {
	case <-backend.firstStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("the speculative request did not start")
	}
	result := make(chan string, 1)
	go func() {
		got, _ := liveCleanUp(r, "Hello there.")
		result <- got
	}()
	h.rec.waitFor(t, "the adopted speculative partial", func(snapshot Snapshot) bool {
		return snapshot.ID == r.id && snapshot.Draft == "visible partial"
	})
	close(releaseFirst)
	select {
	case <-backend.secondStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("the normal full-text fallback did not start")
	}
	if draft := h.m.Current().Draft; draft != "" {
		t.Fatalf("failed adopted Draft %q remained during fallback", draft)
	}
	close(releaseFallback)
	select {
	case got := <-result:
		if got != "fallback cleanup" {
			t.Fatalf("cleanup output %q", got)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("foreground cleanup did not finish")
	}
	if backend.calls.Load() != 2 || sleeps.Load() != 0 {
		t.Fatalf("backend calls %d, retry sleeps %d; want failed adopted request and one replacement", backend.calls.Load(), sleeps.Load())
	}
}

func TestLivePolishRetryDuringBackoffStopsOnSkip(t *testing.T) {
	clock := newSoftDeadlineClock()
	firstRelease := make(chan struct{})
	firstStarted := make(chan struct{})
	secondStarted := make(chan struct{})
	backoffStarted := make(chan struct{})
	backoffEnded := make(chan struct{})
	backend := &retrySessionBackend{
		firstDelta: "private failed attempt", firstRelease: firstRelease, firstStarted: firstStarted,
		secondStarted: secondStarted, secondText: "must not run",
	}
	client := &llm.Client{
		Backend: backend, Retry: llm.RetryPolicy{MaxAttempts: 2, Base: time.Nanosecond, Max: time.Nanosecond},
		Sleep: func(ctx context.Context, _ time.Duration) error {
			close(backoffStarted)
			<-ctx.Done()
			close(backoffEnded)
			return ctx.Err()
		},
	}
	finalGate := make(chan struct{})
	streamer := &fakeStreamer{
		final: "spoken words.", partials: []string{"spoken words."}, finishGate: finalGate,
		finishStarted: make(chan struct{}, 1), finishEmitted: make(chan bool, 1),
	}
	h := newHarness(t, streamer, client, func(config *Config, currentSettings *settings.Settings) {
		config.Now = clock.Now
		config.AfterFunc = clock.AfterFunc
		currentSettings.Audio.VAD.Off = true
		currentSettings.Features.Transcribe.Enhance = true
	})
	h.audio.notifyAt = 3
	h.audio.notified = make(chan struct{})
	id := h.start()
	h.rec.waitFor(t, "recording", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State == Recording })
	select {
	case <-h.audio.notified:
	case <-time.After(5 * time.Second):
		t.Fatal("recording did not start the speculative request")
	}
	select {
	case <-firstStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("the first speculative attempt did not start")
	}
	close(firstRelease)
	select {
	case <-backoffStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("the privately streamed attempt did not enter retry backoff")
	}
	h.m.Stop(id)
	select {
	case <-streamer.finishStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("ASR did not start finalization")
	}
	clock.Advance(300 * time.Millisecond)
	close(finalGate)
	select {
	case emitted := <-streamer.finishEmitted:
		if !emitted {
			t.Fatal("ASR Finish did not emit its Final event")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("ASR Final did not arrive")
	}
	h.m.mu.Lock()
	run := h.m.cur
	h.m.mu.Unlock()
	waitLiveFor(t, run, "adoption during retry backoff", func(polish *livePolish) bool {
		return polish.flight != nil && polish.flight.adopted
	})
	h.m.SkipLLM(id)
	select {
	case <-backoffEnded:
	case <-time.After(5 * time.Second):
		t.Fatal("Skip did not cancel retry backoff")
	}
	end := h.rec.waitFor(t, "the raw Skip result", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.State.Ended()
	})
	if end.Result != "spoken words." || end.Draft != "" || backend.calls.Load() != 1 || len(h.text.pastes()) != 1 || h.text.pastes()[0] != "spoken words." {
		t.Fatalf("Skip result %+v, attempts %d, pasted %q", end, backend.calls.Load(), h.text.pastes())
	}
	select {
	case <-secondStarted:
		t.Fatal("retry started after Skip")
	default:
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
