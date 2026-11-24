package session

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type softDeadlineTimer struct {
	delay    time.Duration
	callback func()
	fired    bool
	stopped  bool
}

type softDeadlineClock struct {
	mu     sync.Mutex
	now    time.Time
	timers []softDeadlineTimer
}

func newSoftDeadlineClock() *softDeadlineClock {
	return &softDeadlineClock{now: time.Date(2026, time.November, 23, 12, 0, 0, 0, time.UTC)}
}

func (clock *softDeadlineClock) Now() time.Time {
	clock.mu.Lock()
	defer clock.mu.Unlock()
	return clock.now
}

func (clock *softDeadlineClock) Advance(duration time.Duration) {
	clock.mu.Lock()
	clock.now = clock.now.Add(duration)
	clock.mu.Unlock()
}

func (clock *softDeadlineClock) AfterFunc(delay time.Duration, callback func()) func() bool {
	clock.mu.Lock()
	index := len(clock.timers)
	clock.timers = append(clock.timers, softDeadlineTimer{delay: delay, callback: callback})
	clock.mu.Unlock()
	return func() bool {
		clock.mu.Lock()
		defer clock.mu.Unlock()
		if clock.timers[index].stopped || clock.timers[index].fired {
			return false
		}
		clock.timers[index].stopped = true
		return true
	}
}

func (clock *softDeadlineClock) timerCount() int {
	clock.mu.Lock()
	defer clock.mu.Unlock()
	return len(clock.timers)
}

func (clock *softDeadlineClock) timerDelay(index int) time.Duration {
	clock.mu.Lock()
	defer clock.mu.Unlock()
	return clock.timers[index].delay
}

func (clock *softDeadlineClock) Fire(t *testing.T, index int) {
	t.Helper()
	clock.mu.Lock()
	if index >= len(clock.timers) {
		clock.mu.Unlock()
		t.Fatalf("timer %d was not scheduled", index)
	}
	if clock.timers[index].fired || clock.timers[index].stopped {
		clock.mu.Unlock()
		t.Fatalf("timer %d is already finished", index)
	}
	clock.timers[index].fired = true
	callback := clock.timers[index].callback
	clock.mu.Unlock()
	callback()
}

type cleanupGateLLM struct {
	started    chan llm.Request
	release    chan struct{}
	returned   chan struct{}
	beforeText string
	afterText  string
	answer     string
}

func newCleanupGateLLM(beforeText, afterText, answer string) *cleanupGateLLM {
	return &cleanupGateLLM{
		started: make(chan llm.Request, 8), release: make(chan struct{}),
		returned: make(chan struct{}, 8), beforeText: beforeText, afterText: afterText, answer: answer,
	}
}

func (model *cleanupGateLLM) Chat(_ context.Context, request llm.Request) (llm.Response, error) {
	if request.OnDelta != nil && model.beforeText != "" {
		request.OnDelta(model.beforeText)
	}
	model.started <- request
	<-model.release
	if request.OnDelta != nil && model.afterText != "" {
		request.OnDelta(model.afterText)
	}
	model.returned <- struct{}{}
	return llm.Response{Text: model.answer}, nil
}

func (model *cleanupGateLLM) Release() { close(model.release) }

func waitCleanupSignal(t *testing.T, signal <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
	}
}

func waitCleanupRequest(t *testing.T, model *cleanupGateLLM, what string) llm.Request {
	t.Helper()
	select {
	case request := <-model.started:
		return request
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
		return llm.Request{}
	}
}

func cleanupLatencyConfig(clock *softDeadlineClock, autoFallback bool) func(*Config, *settings.Settings) {
	return func(config *Config, settings *settings.Settings) {
		config.Now = clock.Now
		config.AfterFunc = clock.AfterFunc
		if autoFallback {
			config.CleanupLatency = CleanupLatencyPolicy{AutoFallbackAfter: cleanupSoftDeadlineTestDuration}
		}
		settings.Features.Transcribe.Enhance = true
	}
}

func TestCleanupSoftDeadlineKeepsCleanupCompletedBeforeDeadline(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newCleanupGateLLM("", "cleaned draft", "cleaned result")
	h := newHarness(t, &fakeFile{text: "spoken words"}, model, cleanupLatencyConfig(clock, true))
	id := h.start()
	h.speak(id, 3)
	waitCleanupRequest(t, model, "cleanup request")
	if clock.timerCount() != 1 || clock.timerDelay(0) != cleanupSoftDeadlineTestDuration {
		t.Fatalf("timers = %d, delay = %v", clock.timerCount(), clock.timerDelay(0))
	}
	clock.Advance(cleanupSoftDeadlineTestDuration - time.Nanosecond)
	model.Release()
	waitCleanupSignal(t, model.returned, "cleanup response")
	end := h.rec.wait(t)
	if end.Result != "cleaned result" || len(h.text.pastes()) != 1 || h.text.pastes()[0] != "cleaned result" {
		t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
	}
}

func TestCleanupSoftDeadlineSelectsRawForNoTokenAndLatePartialStream(t *testing.T) {
	for _, test := range []struct {
		name       string
		beforeText string
	}{
		{name: "no token"},
		{name: "partial stream", beforeText: "partial draft"},
	} {
		t.Run(test.name, func(t *testing.T) {
			clock := newSoftDeadlineClock()
			model := newCleanupGateLLM(test.beforeText, "late cleaned draft", "cleaned result")
			h := newHarness(t, &fakeFile{text: "spoken words"}, model, cleanupLatencyConfig(clock, true))
			id := h.start()
			h.speak(id, 3)
			waitCleanupRequest(t, model, "cleanup request")
			if test.beforeText != "" {
				h.rec.waitFor(t, "partial cleanup draft", func(snapshot Snapshot) bool {
					return snapshot.ID == id && snapshot.State == Composing && snapshot.Draft == test.beforeText
				})
			}
			clock.Advance(cleanupSoftDeadlineTestDuration)
			clock.Fire(t, 0)
			end := h.rec.waitFor(t, "raw fallback", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
			if end.Result != "spoken words" || len(h.text.pastes()) != 1 || h.text.pastes()[0] != "spoken words" {
				t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
			}
			beforeLateResponse := len(h.rec.all())
			model.Release()
			waitCleanupSignal(t, model.returned, "late cleanup response")
			for _, snapshot := range h.rec.all()[beforeLateResponse:] {
				if snapshot.ID == id && snapshot.Draft == "late cleaned draft" {
					t.Fatalf("late response restored draft: %+v", snapshot)
				}
			}
			if end := h.rec.waitFor(t, "terminal snapshot", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() }); end.Draft != "" {
				t.Fatalf("terminal draft %q", end.Draft)
			}
			if pastes := h.text.pastes(); len(pastes) != 1 {
				t.Fatalf("late response caused another delivery: %q", pastes)
			}
		})
	}
}

func TestCleanupSoftDeadlineExactCompletionSelectsRaw(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newCleanupGateLLM("", "at deadline draft", "cleaned result")
	h := newHarness(t, &fakeFile{text: "spoken words"}, model, cleanupLatencyConfig(clock, true))
	id := h.start()
	h.speak(id, 3)
	waitCleanupRequest(t, model, "cleanup request")
	clock.Advance(cleanupSoftDeadlineTestDuration)
	model.Release()
	waitCleanupSignal(t, model.returned, "cleanup response")
	end := h.rec.waitFor(t, "raw fallback at the deadline", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
	if end.Result != "spoken words" || len(h.text.pastes()) != 1 || h.text.pastes()[0] != "spoken words" || end.Draft != "" {
		t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
	}
}

func TestCleanupSoftDeadlineStartsAtStopAcrossASRDelay(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newCleanupGateLLM("", "", "cleaned result")
	provider := &clockAdvancingFile{fakeFile: fakeFile{text: "spoken words"}, clock: clock, delay: 500 * time.Millisecond}
	h := newHarness(t, provider, model, cleanupLatencyConfig(clock, true))
	id := h.start()
	h.speak(id, 3)
	waitCleanupRequest(t, model, "cleanup request")
	if clock.timerCount() != 1 || clock.timerDelay(0) != time.Second {
		t.Fatalf("timers = %d, delay = %v; want one second left from Stop", clock.timerCount(), clock.timerDelay(0))
	}
	clock.Advance(time.Second)
	clock.Fire(t, 0)
	end := h.rec.waitFor(t, "raw fallback", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
	if end.Result != "spoken words" || len(h.text.pastes()) != 1 {
		t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
	}
	model.Release()
	waitCleanupSignal(t, model.returned, "cancelled cleanup response")
}

func TestCleanupSoftDeadlineExpiredDuringASRDoesNotStartCleanup(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := &fakeLLM{text: "cleanup must not start"}
	provider := &clockAdvancingFile{fakeFile: fakeFile{text: "spoken words"}, clock: clock, delay: cleanupSoftDeadlineTestDuration + time.Millisecond}
	h := newHarness(t, provider, model, cleanupLatencyConfig(clock, true))
	id := h.start()
	h.speak(id, 3)
	end := h.rec.waitFor(t, "raw result after late ASR", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
	if end.Result != "spoken words" || len(model.requests()) != 0 || clock.timerCount() != 0 || len(h.text.pastes()) != 1 || h.text.pastes()[0] != "spoken words" {
		t.Fatalf("end %+v, requests %d, timers %d, pasted %q", end, len(model.requests()), clock.timerCount(), h.text.pastes())
	}
}

type clockAdvancingFile struct {
	fakeFile
	clock *softDeadlineClock
	delay time.Duration
}

func (provider *clockAdvancingFile) Transcribe(ctx context.Context, request asr.FileRequest) (asr.Result, error) {
	provider.clock.Advance(provider.delay)
	return provider.fakeFile.Transcribe(ctx, request)
}

func TestCleanupSoftDeadlineDoesNotResetForProviderFallback(t *testing.T) {
	clock := newSoftDeadlineClock()
	primary := &fakeLLM{err: errors.New("primary failed")}
	fallback := newCleanupGateLLM("", "late fallback draft", "fallback result")
	h := newHarness(t, &fakeFile{text: "spoken words"}, primary, cleanupLatencyConfig(clock, true))
	h.fallback = fallback
	id := h.start()
	h.speak(id, 3)
	waitCleanupRequest(t, fallback, "fallback provider request")
	if clock.timerCount() != 1 || len(primary.requests()) != 1 {
		t.Fatalf("soft timers = %d, primary requests = %d", clock.timerCount(), len(primary.requests()))
	}
	clock.Advance(cleanupSoftDeadlineTestDuration)
	clock.Fire(t, 0)
	end := h.rec.waitFor(t, "raw fallback", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
	if end.Result != "spoken words" || len(h.text.pastes()) != 1 || h.text.pastes()[0] != "spoken words" {
		t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
	}
	fallback.Release()
	waitCleanupSignal(t, fallback.returned, "late provider fallback response")
	if clock.timerCount() != 1 {
		t.Fatalf("provider fallback reset the deadline: %d timers", clock.timerCount())
	}
}

type lateFailureCleanupLLM struct {
	started  chan struct{}
	release  chan struct{}
	returned chan struct{}
}

func (model *lateFailureCleanupLLM) Chat(context.Context, llm.Request) (llm.Response, error) {
	close(model.started)
	<-model.release
	close(model.returned)
	return llm.Response{}, errors.New("primary failed after raw was selected")
}

func TestCleanupSoftDeadlineDoesNotStartProviderFallbackAfterRawSelection(t *testing.T) {
	clock := newSoftDeadlineClock()
	primary := &lateFailureCleanupLLM{started: make(chan struct{}), release: make(chan struct{}), returned: make(chan struct{})}
	fallback := &fakeLLM{text: "must not replace raw"}
	h := newHarness(t, &fakeFile{text: "spoken words"}, primary, cleanupLatencyConfig(clock, true))
	h.fallback = fallback
	id := h.start()
	h.speak(id, 3)
	waitCleanupSignal(t, primary.started, "primary cleanup request")
	clock.Advance(cleanupSoftDeadlineTestDuration)
	clock.Fire(t, 0)
	end := h.rec.waitFor(t, "raw result", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
	beforeLateFailure := len(h.rec.all())
	close(primary.release)
	waitCleanupSignal(t, primary.returned, "late primary failure")
	if end.Result != "spoken words" || end.LLM.ID != "deepseek" || len(fallback.requests()) != 0 || len(h.text.pastes()) != 1 {
		t.Fatalf("end %+v, fallback requests %d, pasted %q", end, len(fallback.requests()), h.text.pastes())
	}
	if got := len(h.rec.all()); got != beforeLateFailure {
		t.Fatalf("late primary failure published another snapshot: %d before, %d after", beforeLateFailure, got)
	}
}

type retryingCleanupBackend struct {
	attempts atomic.Int32
	started  chan struct{}
	release  chan struct{}
	returned chan struct{}
}

func (backend *retryingCleanupBackend) Info() llm.BackendInfo {
	return llm.BackendInfo{ID: "retry-test"}
}

func (backend *retryingCleanupBackend) Chat(_ context.Context, request llm.Request) (llm.Response, error) {
	if backend.attempts.Add(1) == 1 {
		return llm.Response{}, &llm.Error{Code: llm.CodeServer, Retryable: true}
	}
	close(backend.started)
	<-backend.release
	if request.OnDelta != nil {
		request.OnDelta("late retry draft")
	}
	close(backend.returned)
	return llm.Response{Text: "cleaned after retry"}, nil
}

func TestCleanupSoftDeadlineIncludesProviderRetries(t *testing.T) {
	clock := newSoftDeadlineClock()
	backend := &retryingCleanupBackend{started: make(chan struct{}), release: make(chan struct{}), returned: make(chan struct{})}
	model := &llm.Client{
		Backend: backend,
		Retry:   llm.RetryPolicy{MaxAttempts: 2, Base: time.Nanosecond},
		Sleep:   func(context.Context, time.Duration) error { return nil },
	}
	h := newHarness(t, &fakeFile{text: "spoken words"}, model, cleanupLatencyConfig(clock, true))
	id := h.start()
	h.speak(id, 3)
	waitCleanupSignal(t, backend.started, "provider retry")
	if backend.attempts.Load() != 2 || clock.timerCount() != 1 {
		t.Fatalf("attempts = %d, soft timers = %d", backend.attempts.Load(), clock.timerCount())
	}
	clock.Advance(cleanupSoftDeadlineTestDuration)
	clock.Fire(t, 0)
	end := h.rec.waitFor(t, "raw fallback", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
	if end.Result != "spoken words" || len(h.text.pastes()) != 1 {
		t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
	}
	close(backend.release)
	waitCleanupSignal(t, backend.returned, "late retry result")
}

func TestCleanupSoftDeadlineManualSkipAndDisabledPolicy(t *testing.T) {
	t.Run("manual Use transcript", func(t *testing.T) {
		clock := newSoftDeadlineClock()
		model := newCleanupGateLLM("", "late cleaned draft", "cleaned result")
		h := newHarness(t, &fakeFile{text: "spoken words"}, model, cleanupLatencyConfig(clock, true))
		id := h.start()
		h.speak(id, 3)
		waitCleanupRequest(t, model, "cleanup request")
		h.m.SkipLLM(id)
		end := h.rec.waitFor(t, "manual raw result", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
		if end.Result != "spoken words" || len(h.text.pastes()) != 1 || h.text.pastes()[0] != "spoken words" {
			t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
		}
		model.Release()
		waitCleanupSignal(t, model.returned, "cancelled cleanup response")
	})

	t.Run("automatic fallback disabled", func(t *testing.T) {
		clock := newSoftDeadlineClock()
		model := &fakeLLM{hang: true}
		h := newHarness(t, &fakeFile{text: "spoken words"}, model, cleanupLatencyConfig(clock, false))
		id := h.start()
		h.speak(id, 3)
		h.rec.waitFor(t, "Use transcript action", func(snapshot Snapshot) bool {
			return snapshot.ID == id && snapshot.State == Composing && snapshot.CanSkip
		})
		if clock.timerCount() != 0 {
			t.Fatalf("disabled policy scheduled %d timers", clock.timerCount())
		}
		h.m.SkipLLM(id)
		end := h.rec.waitFor(t, "manual raw result", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
		if end.Result != "spoken words" || len(h.text.pastes()) != 1 || h.text.pastes()[0] != "spoken words" {
			t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
		}
	})

	t.Run("zero value preserves waiting", func(t *testing.T) {
		clock := newSoftDeadlineClock()
		model := newCleanupGateLLM("", "", "cleaned result")
		h := newHarness(t, &fakeFile{text: "spoken words"}, model, func(config *Config, settings *settings.Settings) {
			config.Now, config.AfterFunc = clock.Now, clock.AfterFunc
			settings.Features.Transcribe.Enhance = true
		})
		id := h.start()
		h.speak(id, 3)
		waitCleanupRequest(t, model, "cleanup request")
		clock.Advance(cleanupSoftDeadlineTestDuration + time.Second)
		if clock.timerCount() != 0 {
			t.Fatalf("zero-value policy scheduled %d timers", clock.timerCount())
		}
		model.Release()
		waitCleanupSignal(t, model.returned, "cleanup response")
		end := h.rec.wait(t)
		if end.Result != "cleaned result" || len(h.text.pastes()) != 1 || h.text.pastes()[0] != "cleaned result" {
			t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
		}
	})

	t.Run("non-positive value disables automatic fallback", func(t *testing.T) {
		clock := newSoftDeadlineClock()
		model := newCleanupGateLLM("", "", "cleaned result")
		h := newHarness(t, &fakeFile{text: "spoken words"}, model, func(config *Config, settings *settings.Settings) {
			config.Now, config.AfterFunc = clock.Now, clock.AfterFunc
			config.CleanupLatency = CleanupLatencyPolicy{AutoFallbackAfter: -time.Second}
			settings.Features.Transcribe.Enhance = true
		})
		id := h.start()
		h.speak(id, 3)
		waitCleanupRequest(t, model, "cleanup request")
		clock.Advance(cleanupSoftDeadlineTestDuration + time.Second)
		if clock.timerCount() != 0 {
			t.Fatalf("non-positive policy scheduled %d timers", clock.timerCount())
		}
		model.Release()
		waitCleanupSignal(t, model.returned, "cleanup response")
		end := h.rec.wait(t)
		if end.Result != "cleaned result" || len(h.text.pastes()) != 1 || h.text.pastes()[0] != "cleaned result" {
			t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
		}
	})
}

func TestCleanupSoftDeadlineNoRawTranscriptDoesNotFabricateResult(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newCleanupGateLLM("", "", "should not run")
	h := newHarness(t, &fakeFile{text: "  "}, model, cleanupLatencyConfig(clock, true))
	h.speak(h.start(), 3)
	end := h.rec.wait(t)
	if end.Result != "" || end.Problem == nil || end.Problem.Code != ProblemNoSpeech || clock.timerCount() != 0 || len(h.text.pastes()) != 0 {
		t.Fatalf("end %+v, timers %d, pasted %q", end, clock.timerCount(), h.text.pastes())
	}
}

func TestCleanupSoftDeadlineDeliveryFailureKeepsRawRecoverable(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newCleanupGateLLM("", "", "cleaned result")
	h := newHarness(t, &fakeFile{text: "spoken words"}, model, cleanupLatencyConfig(clock, true))
	h.text.err = errors.New("focus failed")
	id := h.start()
	h.speak(id, 3)
	waitCleanupRequest(t, model, "cleanup request")
	clock.Advance(cleanupSoftDeadlineTestDuration)
	clock.Fire(t, 0)
	end := h.rec.waitFor(t, "raw result after delivery failure", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
	entries := h.hist.all()
	if end.Result != "spoken words" || len(entries) != 1 || entries[0].Result != "spoken words" || len(h.text.pastes()) != 0 {
		t.Fatalf("end %+v, entries %+v, pasted %q", end, entries, h.text.pastes())
	}
	model.Release()
	waitCleanupSignal(t, model.returned, "cancelled cleanup response")
}

func TestCleanupSoftDeadlineCancellationSuppressesLateResponse(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newCleanupGateLLM("partial draft", "late cleaned draft", "cleaned result")
	h := newHarness(t, &fakeFile{text: "spoken words"}, model, cleanupLatencyConfig(clock, true))
	id := h.start()
	h.speak(id, 3)
	waitCleanupRequest(t, model, "cleanup request")
	h.m.Cancel(id)
	end := h.rec.waitFor(t, "cancelled cleanup", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
	if end.State != Cancelled || end.Result != "" || len(h.text.pastes()) != 0 {
		t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
	}
	model.Release()
	waitCleanupSignal(t, model.returned, "late cleanup response")
	for _, snapshot := range h.rec.all() {
		if snapshot.ID == id && snapshot.State == Cancelled && snapshot.Result != "" {
			t.Fatalf("cancelled session acquired a result: %+v", snapshot)
		}
	}
}

func TestCleanupCancellationAfterChoiceBeforeDelivery(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := &fakeLLM{text: "cleaned words"}
	h := newHarness(t, &fakeFile{text: "spoken words"}, model, cleanupLatencyConfig(clock, false))
	choiceCommitted := make(chan struct{})
	resumeOwner := make(chan struct{})
	var once sync.Once
	var cancelledOutcomes atomic.Int32
	h.m.cfg.Log = func(_, message string) {
		if strings.Contains(message, "cleanup outcome=cancelled") {
			cancelledOutcomes.Add(1)
		}
		if strings.Contains(message, "cleanup outcome=cleaned") {
			once.Do(func() {
				close(choiceCommitted)
				<-resumeOwner
			})
		}
	}
	id := h.start()
	h.speak(id, 3)
	select {
	case <-choiceCommitted:
	case <-time.After(5 * time.Second):
		t.Fatal("Cleanup outcome was not committed")
	}
	h.m.Cancel(id)
	close(resumeOwner)
	end := h.rec.waitFor(t, "cancellation before delivery", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
	entries := h.hist.all()
	if end.State != Cancelled || end.Result != "" || end.Profile != "" || len(h.text.pastes()) != 0 || len(entries) != 1 || entries[0].Status != store.StatusCancelled || entries[0].Result != "" {
		t.Fatalf("end %+v, pasted %q, history %+v", end, h.text.pastes(), h.hist.all())
	}
	if cancelledOutcomes.Load() != 0 {
		t.Fatalf("Cleanup emitted %d contradictory cancellation outcomes after commitment", cancelledOutcomes.Load())
	}
	for _, snapshot := range h.rec.all() {
		if snapshot.ID == id && snapshot.State == Delivering {
			t.Fatalf("cancelled operation reserved delivery: %+v", snapshot)
		}
	}
}

func TestCleanupSoftDeadlineAppliesOnlyToOrdinaryTranscribe(t *testing.T) {
	for _, kind := range []store.Kind{store.KindTranslate, store.KindRewrite} {
		t.Run(string(kind), func(t *testing.T) {
			clock := newSoftDeadlineClock()
			h := newHarness(t, &fakeFile{text: "spoken words"}, &fakeLLM{hang: true}, func(config *Config, settings *settings.Settings) {
				config.Now, config.AfterFunc = clock.Now, clock.AfterFunc
				config.CleanupLatency = CleanupLatencyPolicy{AutoFallbackAfter: cleanupSoftDeadlineTestDuration}
				config.Select = &fakeSelector{text: "selected text"}
			})
			id, err := h.m.Start(StartRequest{Kind: kind, Target: slack})
			if err != nil {
				t.Fatal(err)
			}
			if kind == store.KindRewrite {
				h.speak(id, 3)
			}
			h.rec.waitFor(t, "composing request", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State == Composing })
			if clock.timerCount() != 0 {
				t.Fatalf("%s scheduled %d cleanup timers", kind, clock.timerCount())
			}
			h.m.Cancel(id)
			h.rec.waitFor(t, "cancelled request", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
		})
	}
}

func TestEarlyCleanupCompletedBeforeStopUsesCleanedText(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := &fakeLLM{text: "cleaned while recording"}
	h := newHarness(t, &fakeStreamer{final: "spoken words", pieces: []string{"spoken words"}}, model, cleanupLatencyConfig(clock, true))
	id := h.start()
	waitEarly(t, model, 1)
	h.speak(id, 5)
	end := h.rec.wait(t)
	if end.Result != "cleaned while recording" || len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
		t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
	}
	if clock.timerCount() != 1 {
		t.Fatalf("soft deadline timers = %d", clock.timerCount())
	}
}

func TestEarlyCleanupCompletedWithinPostStopBudgetUsesCleanedText(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newCleanupGateLLM("", "", "cleaned shortly after Stop")
	h := newHarness(t, &fakeStreamer{final: "spoken words", pieces: []string{"spoken words"}}, model, cleanupLatencyConfig(clock, true))
	id := h.start()
	waitCleanupRequest(t, model, "early cleanup request")
	h.speak(id, 5)
	h.rec.waitFor(t, "composing request", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State == Composing })
	clock.Advance(time.Second)
	model.Release()
	waitCleanupSignal(t, model.returned, "early cleanup response")
	end := h.rec.wait(t)
	if end.Result != "cleaned shortly after Stop" || len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
		t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
	}
}

func TestEarlyCleanupExceedingSoftDeadlineUsesRaw(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newCleanupGateLLM("", "", "late early cleanup")
	h := newHarness(t, &fakeStreamer{final: "spoken words", pieces: []string{"spoken words"}}, model, cleanupLatencyConfig(clock, true))
	id := h.start()
	waitCleanupRequest(t, model, "early cleanup request")
	h.speak(id, 5)
	h.rec.waitFor(t, "composing request", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State == Composing })
	clock.Advance(cleanupSoftDeadlineTestDuration)
	clock.Fire(t, 0)
	end := h.rec.waitFor(t, "raw fallback", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
	if end.Result != "spoken words" || len(h.text.pastes()) != 1 || h.text.pastes()[0] != "spoken words" {
		t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
	}
	model.Release()
	waitCleanupSignal(t, model.returned, "late early cleanup response")
	select {
	case request := <-model.started:
		t.Fatalf("cleanup started after Raw was committed: %+v", request)
	default:
	}
}

func TestEarlyCleanupPrefixMismatchUsesSameSoftDeadline(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := &earlyThenGateCleanupLLM{
		firstReturned: make(chan struct{}), secondStarted: make(chan struct{}), secondRelease: make(chan struct{}), secondReturned: make(chan struct{}),
	}
	h := newHarness(t, &fakeStreamer{final: "changed words", pieces: []string{"spoken words"}}, model, cleanupLatencyConfig(clock, true))
	id := h.start()
	waitCleanupSignal(t, model.firstReturned, "early cleanup")
	h.speak(id, 5)
	waitCleanupSignal(t, model.secondStarted, "full-text cleanup after prefix mismatch")
	if clock.timerCount() != 1 {
		t.Fatalf("soft deadline timers = %d", clock.timerCount())
	}
	clock.Advance(cleanupSoftDeadlineTestDuration)
	clock.Fire(t, 0)
	end := h.rec.waitFor(t, "raw fallback", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
	if end.Result != "changed words" || len(h.text.pastes()) != 1 || h.text.pastes()[0] != "changed words" {
		t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
	}
	close(model.secondRelease)
	waitCleanupSignal(t, model.secondReturned, "late full-text cleanup")
}

type earlyThenGateCleanupLLM struct {
	calls          atomic.Int32
	firstReturned  chan struct{}
	secondStarted  chan struct{}
	secondRelease  chan struct{}
	secondReturned chan struct{}
}

func (model *earlyThenGateCleanupLLM) Chat(_ context.Context, _ llm.Request) (llm.Response, error) {
	switch model.calls.Add(1) {
	case 1:
		close(model.firstReturned)
		return llm.Response{Text: "early cleanup"}, nil
	default:
		close(model.secondStarted)
		<-model.secondRelease
		close(model.secondReturned)
		return llm.Response{Text: "full cleanup"}, nil
	}
}

func TestEarlyCleanupCancelledBeforeDeadlineCannotDeliver(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newCleanupGateLLM("", "late early draft", "late early cleanup")
	h := newHarness(t, &fakeStreamer{final: "spoken words", pieces: []string{"spoken words"}}, model, cleanupLatencyConfig(clock, true))
	id := h.start()
	waitCleanupRequest(t, model, "early cleanup request")
	h.speak(id, 5)
	h.rec.waitFor(t, "composing request", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State == Composing })
	h.m.Cancel(id)
	end := h.rec.waitFor(t, "cancelled dictation", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
	if end.State != Cancelled || end.Result != "" || len(h.text.pastes()) != 0 {
		t.Fatalf("end %+v, pasted %q", end, h.text.pastes())
	}
	model.Release()
	waitCleanupSignal(t, model.returned, "cancelled early cleanup")
	select {
	case request := <-model.started:
		t.Fatalf("cleanup started after Cancel: %+v", request)
	default:
	}
	for _, snapshot := range h.rec.all() {
		if snapshot.ID == id && snapshot.State == Cancelled && snapshot.Result != "" {
			t.Fatalf("cancelled early cleanup acquired a result: %+v", snapshot)
		}
	}
}

type sequencedCleanupFile struct {
	mu    sync.Mutex
	texts []string
	calls int
}

func (provider *sequencedCleanupFile) Caps() asr.Caps { return asr.Caps{File: true} }

func (provider *sequencedCleanupFile) Transcribe(context.Context, asr.FileRequest) (asr.Result, error) {
	provider.mu.Lock()
	defer provider.mu.Unlock()
	text := provider.texts[min(provider.calls, len(provider.texts)-1)]
	provider.calls++
	return asr.Result{Text: text, Model: "sequenced"}, nil
}
