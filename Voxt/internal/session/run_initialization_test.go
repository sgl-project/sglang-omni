package session

import (
	"context"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type initializationSignals struct {
	asrOpenStarted   <-chan struct{}
	asrOpenDone      <-chan struct{}
	firstPCMCaptured <-chan struct{}
	firstPCM         <-chan struct{}
	firstPartial     <-chan struct{}
	asrFinal         <-chan struct{}
	llmRequest       <-chan struct{}
}

func observeSessionSignals(h *harness) initializationSignals {
	h.mu.Lock()
	provider, model := h.asr, h.llm
	h.mu.Unlock()
	var signals initializationSignals
	if streamer, ok := provider.(*fakeStreamer); ok {
		streamer.ensureSignals()
		signals.asrOpenStarted, signals.asrOpenDone = streamer.openStarted, streamer.openDone
		signals.firstPCM, signals.firstPartial, signals.asrFinal = streamer.firstSend, streamer.firstPartial, streamer.finalEvent
	}
	signals.firstPCMCaptured = h.audio.firstPCM
	if fake, ok := model.(*fakeLLM); ok {
		fake.ensureSignals()
		signals.llmRequest = fake.firstRequest
	}
	return signals
}

func waitInitializationSignal[T any](t testing.TB, signal <-chan T, what string) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
	}
}

func setIndependentResolvers(h *harness, provider asr.Provider, resolveLLM LLMResolver) {
	h.m.cfg.ResolveASR = func(context.Context, Kind, settings.Settings) ASRReadiness {
		return ASRReadiness{ASR: provider, ASRInfo: Model{ID: "fake-asr", Name: "Fake ASR", Model: "fake"}}
	}
	h.m.cfg.ResolveLLM = resolveLLM
}

func TestSceneReadinessSignalIsConsumedOnce(t *testing.T) {
	sceneDone := make(chan struct{})
	close(sceneDone)
	r := &run{init: runInitialization{split: true}, sc: sceneState{done: sceneDone}}
	selectable := r.sceneReadySignal()
	if selectable == nil {
		t.Fatal("closed scene readiness was lost before the owner consumed it")
	}
	<-selectable
	r.markSceneReady()
	if selectable = r.sceneReadySignal(); selectable != nil || !r.init.sceneReady {
		t.Fatalf("scene readiness stayed selectable: channel=%v, state=%+v", selectable, r.init)
	}
}

func gatedLLMResolver(model LLM, started chan<- struct{}, release <-chan struct{}, returned chan<- struct{}) LLMResolver {
	return func(context.Context, Kind, settings.Settings) LLMReadiness {
		started <- struct{}{}
		<-release // Deliberately ignores cancellation to exercise a late initializer.
		if returned != nil {
			returned <- struct{}{}
		}
		if model == nil {
			return LLMReadiness{Outcome: LLMUnavailable}
		}
		return LLMReadiness{LLM: model, LLMInfo: Model{ID: "fake-llm", Name: "Fake LLM", Model: "fake"}, Outcome: LLMAvailable}
	}
}

func controlledRelease(t testing.TB) (chan struct{}, func()) {
	t.Helper()
	release := make(chan struct{})
	var once atomic.Bool
	open := func() {
		if once.CompareAndSwap(false, true) {
			close(release)
		}
	}
	t.Cleanup(open)
	return release, open
}

func TestIndependentInitializationOpensASRBeforeLLMAndUsesResolvedLLM(t *testing.T) {
	provider := &fakeStreamer{final: "hello aura link"}
	model := &fakeLLM{text: "Hello, VoxtNew."}
	h := newHarness(t, provider, model, func(_ *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
	})
	signals := observeSessionSignals(h)
	started := make(chan struct{}, 1)
	release, openRelease := controlledRelease(t)
	setIndependentResolvers(h, provider, gatedLLMResolver(model, started, release, nil))

	id := h.start()
	waitInitializationSignal(t, started, "LLM resolver to start")
	waitInitializationSignal(t, signals.asrOpenStarted, "ASR Open to start while LLM initialization is pending")
	waitInitializationSignal(t, signals.asrOpenDone, "ASR Open to complete while LLM initialization is pending")
	waitInitializationSignal(t, signals.firstPCM, "PCM to reach ASR while LLM initialization is pending")
	waitInitializationSignal(t, signals.firstPartial, "an ASR partial while LLM initialization is pending")
	if len(model.requests()) != 0 {
		t.Fatal("cleanup request started before its LLM resolver returned")
	}

	openRelease()
	h.speak(id, 5)
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "Hello, VoxtNew." || len(model.requests()) != 1 {
		t.Fatalf("end %+v, LLM requests %d", end, len(model.requests()))
	}
}

func TestCleanupDisabledSkipsBlockedLLMResolver(t *testing.T) {
	provider := &fakeStreamer{final: "hello aura link"}
	var llmCalls atomic.Int32
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	h := newHarness(t, provider, &fakeLLM{text: "must not run"}, func(_ *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = false
		s.Dictionary.ASRHints = false
	})
	setIndependentResolvers(h, provider, func(context.Context, Kind, settings.Settings) LLMReadiness {
		llmCalls.Add(1)
		<-release // It would remain blocked if cleanup-off started this resolver.
		return LLMReadiness{Outcome: LLMUnavailable}
	})
	signals := observeSessionSignals(h)
	id := h.start()
	waitInitializationSignal(t, signals.asrOpenDone, "ASR to open with cleanup disabled")
	h.speak(id, 5)
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "hello VoxtNew" || llmCalls.Load() != 0 {
		t.Fatalf("end %+v, LLM resolver calls %d", end, llmCalls.Load())
	}
}

func TestASRFinalCompletesBeforePendingLLMInitialization(t *testing.T) {
	provider := &fakeStreamer{final: "hello aura link"}
	model := &fakeLLM{text: "Hello, VoxtNew."}
	h := newHarness(t, provider, model, func(_ *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
	})
	signals := observeSessionSignals(h)
	started := make(chan struct{}, 1)
	release, openRelease := controlledRelease(t)
	setIndependentResolvers(h, provider, gatedLLMResolver(model, started, release, nil))

	id := h.start()
	waitInitializationSignal(t, started, "LLM resolver to start")
	waitInitializationSignal(t, signals.firstPartial, "ASR partial")
	h.speak(id, 5)
	waitInitializationSignal(t, signals.asrFinal, "ASR Final while LLM initialization is still pending")
	if len(model.requests()) != 0 {
		t.Fatal("LLM chat started before LLM initialization was released")
	}
	if end := h.m.Current(); end.ID != id || end.State.Ended() {
		t.Fatalf("session ended before cleanup readiness: %+v", end)
	}

	openRelease()
	end := h.rec.wait(t)
	if end.Result != "Hello, VoxtNew." || len(model.requests()) != 1 || len(h.text.pastes()) != 1 {
		t.Fatalf("end %+v, LLM requests %d, deliveries %q", end, len(model.requests()), h.text.pastes())
	}
}

func TestPendingLLMInitializationConsumesStopRelativeFallbackBudget(t *testing.T) {
	clock := newSoftDeadlineClock()
	deadlineScheduled := make(chan struct{}, 1)
	model := &fakeLLM{text: "must not run"}
	provider := &fakeStreamer{final: "hello aura link"}
	h := newHarness(t, provider, model, func(config *Config, s *settings.Settings) {
		cleanupLatencyConfig(clock, true)(config, s)
		underlying := config.AfterFunc
		config.AfterFunc = func(delay time.Duration, callback func()) func() bool {
			stop := underlying(delay, callback)
			select {
			case deadlineScheduled <- struct{}{}:
			default:
			}
			return stop
		}
	})
	signals := observeSessionSignals(h)
	started := make(chan struct{}, 1)
	release, openRelease := controlledRelease(t)
	returned := make(chan struct{}, 1)
	setIndependentResolvers(h, provider, gatedLLMResolver(model, started, release, returned))

	id := h.start()
	waitInitializationSignal(t, started, "LLM resolver to start")
	waitInitializationSignal(t, signals.firstPartial, "ASR partial")
	h.speak(id, 5)
	waitInitializationSignal(t, signals.asrFinal, "ASR Final")
	waitInitializationSignal(t, deadlineScheduled, "Stop-relative fallback timer")
	clock.Advance(cleanupSoftDeadlineTestDuration)
	clock.Fire(t, 0)
	end := h.rec.waitFor(t, "raw fallback at the original Stop deadline", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.State.Ended()
	})
	if end.Result != "hello VoxtNew" || end.Raw != "hello aura link" || end.LLM.ID != "" {
		t.Fatalf("end %+v; expected the normalized Raw result without stale LLM metadata", end)
	}
	if got := h.text.pastes(); len(got) != 1 || got[0] != "hello VoxtNew" {
		t.Fatalf("deliveries %q, want one normalized Raw result", got)
	}
	if len(model.requests()) != 0 || !h.logged("cleanup outcome=soft_deadline") {
		t.Fatalf("LLM requests %d, soft-deadline outcome logged %v", len(model.requests()), h.logged("cleanup outcome=soft_deadline"))
	}

	openRelease()
	waitInitializationSignal(t, returned, "late LLM resolver to return")
	if snapshot := h.m.Current(); snapshot.ID != id || snapshot.Result != end.Result || snapshot.LLM.ID != "" || len(h.text.pastes()) != 1 {
		t.Fatalf("late resolver changed the completed session: %+v, deliveries %q", snapshot, h.text.pastes())
	}
}

func TestSlowDictionaryAndSnippetsDoNotBlockASRWhenHintsAreOff(t *testing.T) {
	provider := &fakeStreamer{final: "hello aura link and my email address"}
	h := newHarness(t, provider, nil, func(config *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
		s.Dictionary.ASRHints = false
	})
	signals := observeSessionSignals(h)
	started := make(chan struct{}, 1)
	noLLM := func(context.Context, Kind, settings.Settings) LLMReadiness {
		started <- struct{}{}
		return LLMReadiness{Outcome: LLMUnavailable}
	}
	setIndependentResolvers(h, provider, noLLM)
	termsStarted, snippetsStarted := make(chan struct{}, 1), make(chan struct{}, 1)
	termsRelease, openTerms := controlledRelease(t)
	snippetsRelease, openSnippets := controlledRelease(t)
	var termsLoads, snippetLoads atomic.Int32
	h.m.cfg.Glossary = func(context.Context) ([]store.Term, error) {
		termsLoads.Add(1)
		termsStarted <- struct{}{}
		<-termsRelease
		return []store.Term{{Term: "VoxtNew", Aliases: []string{"aura link"}, Enabled: true}}, nil
	}
	h.m.cfg.Snippets = func(context.Context) ([]snippet.Snippet, error) {
		snippetLoads.Add(1)
		snippetsStarted <- struct{}{}
		<-snippetsRelease
		return []snippet.Snippet{{ID: "email", Trigger: "my email address", Text: "ruiling@example.com"}}, nil
	}

	id := h.start()
	waitInitializationSignal(t, started, "LLM resolver")
	waitInitializationSignal(t, termsStarted, "dictionary load")
	waitInitializationSignal(t, snippetsStarted, "snippet load")
	waitInitializationSignal(t, signals.firstPartial, "ASR partial with dictionary and snippets pending")
	h.speak(id, 5)
	waitInitializationSignal(t, signals.asrFinal, "ASR Final with dictionary and snippets pending")
	if end := h.m.Current(); end.ID != id || end.State.Ended() {
		t.Fatalf("ASR did not finish independently: %+v", end)
	}

	openTerms()
	openSnippets()
	end := h.rec.wait(t)
	want := "hello VoxtNew and ruiling@example.com"
	if end.Result != want || termsLoads.Load() != 1 || snippetLoads.Load() != 1 {
		t.Fatalf("result %q, dictionary loads %d, snippet loads %d; want %q and one load each", end.Result, termsLoads.Load(), snippetLoads.Load(), want)
	}
}

func TestRequiredDictionaryHintsGateASROpen(t *testing.T) {
	provider := &fakeStreamer{final: "hello"}
	h := newHarness(t, provider, nil, func(_ *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
		s.Dictionary.ASRHints = true
	})
	signals := observeSessionSignals(h)
	started, termsStarted := make(chan struct{}, 1), make(chan struct{}, 1)
	release, openRelease := controlledRelease(t)
	setIndependentResolvers(h, provider, func(context.Context, Kind, settings.Settings) LLMReadiness {
		started <- struct{}{}
		return LLMReadiness{Outcome: LLMUnavailable}
	})
	h.m.cfg.Glossary = func(context.Context) ([]store.Term, error) {
		termsStarted <- struct{}{}
		<-release
		return []store.Term{{Term: "VoxtNew", Aliases: []string{"aura link"}, Enabled: true}}, nil
	}

	id := h.start()
	waitInitializationSignal(t, started, "LLM resolver")
	waitInitializationSignal(t, termsStarted, "dictionary load")
	h.rec.waitFor(t, "ASR resolver readiness", func(snapshot Snapshot) bool { return snapshot.ASR.ID == "fake-asr" })
	select {
	case <-signals.asrOpenStarted:
		t.Fatal("ASR Open started before required dictionary hints were ready")
	default:
	}

	openRelease()
	waitInitializationSignal(t, signals.asrOpenDone, "ASR Open after dictionary readiness")
	waitInitializationSignal(t, signals.firstPartial, "ASR partial after dictionary readiness")
	provider.mu.Lock()
	hints := append([]string(nil), provider.hints...)
	provider.mu.Unlock()
	if len(hints) != 1 || hints[0] != "VoxtNew" {
		t.Fatalf("ASR hints %q, want [VoxtNew]", hints)
	}
	h.speak(id, 5)
	if end := h.rec.wait(t); end.Result != "hello" {
		t.Fatalf("end %+v", end)
	}
}

type blockedPrefillLLM struct {
	*fakeLLM
	started chan struct{}
	release <-chan struct{}
	calls   atomic.Int32
}

func (m *blockedPrefillLLM) Prefill(ctx context.Context, _ llm.Request) error {
	m.calls.Add(1)
	select {
	case m.started <- struct{}{}:
	default:
	}
	select {
	case <-m.release:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func TestLLMPrefillStartsAfterASROpenAndDoesNotBlockPCM(t *testing.T) {
	provider := &fakeStreamer{final: "hello aura link"}
	release, openRelease := controlledRelease(t)
	model := &blockedPrefillLLM{
		fakeLLM: &fakeLLM{text: "Hello, VoxtNew."}, started: make(chan struct{}, 1), release: release,
	}
	h := newHarness(t, provider, model, func(_ *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
	})
	signals := observeSessionSignals(h)
	setIndependentResolvers(h, provider, func(context.Context, Kind, settings.Settings) LLMReadiness {
		return LLMReadiness{LLM: model, LLMInfo: Model{ID: "fake-llm"}, Outcome: LLMAvailable}
	})

	id := h.start()
	waitInitializationSignal(t, signals.asrOpenDone, "ASR Open to complete")
	waitInitializationSignal(t, model.started, "LLM prefill to start")
	waitInitializationSignal(t, signals.firstPCM, "PCM to reach ASR while prefill is blocked")
	waitInitializationSignal(t, signals.firstPartial, "ASR partial while prefill is blocked")
	openRelease()
	h.speak(id, 3)
	if end := h.rec.wait(t); end.State != Done || end.Result != "Hello, VoxtNew." {
		t.Fatalf("end %+v", end)
	}
	if got := model.calls.Load(); got != 1 {
		t.Fatalf("prefill calls = %d, want 1", got)
	}
}

func TestCancelReturnsWhileLLMInitializationIgnoresContext(t *testing.T) {
	provider := &fakeStreamer{final: "unused"}
	model := &fakeLLM{text: "unused"}
	h := newHarness(t, provider, model, func(_ *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
	})
	signals := observeSessionSignals(h)
	started := make(chan struct{}, 1)
	release, openRelease := controlledRelease(t)
	setIndependentResolvers(h, provider, gatedLLMResolver(model, started, release, nil))

	id := h.start()
	waitInitializationSignal(t, started, "LLM resolver to start")
	waitInitializationSignal(t, signals.firstPartial, "ASR partial")
	h.m.Cancel(id)
	end := h.rec.wait(t)
	if end.State != Cancelled || len(h.text.pastes()) != 0 || len(model.requests()) != 0 {
		t.Fatalf("cancelled session %+v, deliveries %q, LLM requests %d", end, h.text.pastes(), len(model.requests()))
	}
	openRelease()
}

func TestCancelWhileASRResolutionIsPendingDropsLateProvider(t *testing.T) {
	provider := &fakeStreamer{final: "hello"}
	h := newHarness(t, provider, nil, func(_ *Config, s *settings.Settings) {
		s.Dictionary.ASRHints = false
	})
	started, returned := make(chan struct{}, 1), make(chan struct{}, 1)
	release, openRelease := controlledRelease(t)
	setIndependentResolvers(h, provider, func(context.Context, Kind, settings.Settings) LLMReadiness {
		return LLMReadiness{Outcome: LLMUnavailable}
	})
	h.m.cfg.ResolveASR = func(context.Context, Kind, settings.Settings) ASRReadiness {
		started <- struct{}{}
		<-release // Deliberately ignores cancellation.
		returned <- struct{}{}
		return ASRReadiness{ASR: provider, ASRInfo: Model{ID: "late-asr"}}
	}

	id := h.start()
	waitInitializationSignal(t, started, "ASR resolver to start")
	h.m.Cancel(id)
	if end := h.rec.wait(t); end.State != Cancelled {
		t.Fatalf("end %+v", end)
	}
	before := h.m.Current()
	openRelease()
	waitInitializationSignal(t, returned, "late ASR resolver to return")
	after := h.m.Current()
	provider.mu.Lock()
	streamCount := len(provider.streams)
	provider.mu.Unlock()
	if after.Rev != before.Rev || after.ASR.ID != "" || streamCount != 0 || len(h.text.pastes()) != 0 {
		t.Fatalf("late ASR resolution changed ended session: before %+v, after %+v", before, after)
	}
}

type closeSignalStream struct {
	asr.Stream
	closed chan struct{}
	once   sync.Once
}

func (s *closeSignalStream) Close() {
	s.once.Do(func() {
		s.Stream.Close()
		close(s.closed)
	})
}

type lateOpenStreamer struct {
	*fakeStreamer
	started  chan struct{}
	release  <-chan struct{}
	returned chan *closeSignalStream
}

func (s *lateOpenStreamer) Open(_ context.Context, req asr.StreamRequest) (asr.Stream, error) {
	s.started <- struct{}{}
	<-s.release // Deliberately ignores cancellation and returns a live stream late.
	stream, err := s.fakeStreamer.Open(context.Background(), req)
	if err != nil {
		return stream, err
	}
	result := &closeSignalStream{Stream: stream, closed: make(chan struct{})}
	s.returned <- result
	return result, nil
}

func TestLateASROpenAfterCancelClosesTheStream(t *testing.T) {
	release, openRelease := controlledRelease(t)
	provider := &lateOpenStreamer{
		fakeStreamer: &fakeStreamer{final: "hello"}, started: make(chan struct{}, 1),
		release: release, returned: make(chan *closeSignalStream, 1),
	}
	h := newHarness(t, provider, nil, func(_ *Config, s *settings.Settings) {
		s.Dictionary.ASRHints = false
	})
	setIndependentResolvers(h, provider, func(context.Context, Kind, settings.Settings) LLMReadiness {
		return LLMReadiness{Outcome: LLMUnavailable}
	})
	id := h.start()
	waitInitializationSignal(t, provider.started, "ASR Open to start")
	h.m.Cancel(id)
	if end := h.rec.wait(t); end.State != Cancelled {
		t.Fatalf("end %+v", end)
	}
	before := h.m.Current()
	openRelease()
	var stream *closeSignalStream
	select {
	case stream = <-provider.returned:
	case <-time.After(5 * time.Second):
		t.Fatal("late ASR Open did not return")
	}
	waitInitializationSignal(t, stream.closed, "late ASR stream to close")
	after := h.m.Current()
	if after.Rev != before.Rev || after.ID != id || len(h.text.pastes()) != 0 {
		t.Fatalf("late ASR Open changed ended session: before %+v, after %+v", before, after)
	}
}

func TestLateDictionaryAndSnippetResultsAfterCancelAreIgnored(t *testing.T) {
	provider := &fakeStreamer{final: "hello aura link"}
	termsStarted, snippetsStarted := make(chan struct{}, 1), make(chan struct{}, 1)
	termsReturned, snippetsReturned := make(chan struct{}, 1), make(chan struct{}, 1)
	termsRelease, releaseTerms := controlledRelease(t)
	snippetsRelease, releaseSnippets := controlledRelease(t)
	h := newHarness(t, provider, nil, func(c *Config, s *settings.Settings) {
		s.Dictionary.ASRHints = false
		c.Glossary = func(context.Context) ([]store.Term, error) {
			termsStarted <- struct{}{}
			<-termsRelease // Deliberately ignores cancellation.
			termsReturned <- struct{}{}
			return []store.Term{{Term: "VoxtNew", Aliases: []string{"aura link"}, Enabled: true}}, nil
		}
		c.Snippets = func(context.Context) ([]snippet.Snippet, error) {
			snippetsStarted <- struct{}{}
			<-snippetsRelease // Deliberately ignores cancellation.
			snippetsReturned <- struct{}{}
			return []snippet.Snippet{{Trigger: "sig", Text: "late"}}, nil
		}
	})
	signals := observeSessionSignals(h)
	setIndependentResolvers(h, provider, func(context.Context, Kind, settings.Settings) LLMReadiness {
		return LLMReadiness{Outcome: LLMUnavailable}
	})
	id := h.start()
	waitInitializationSignal(t, termsStarted, "dictionary load to start")
	waitInitializationSignal(t, snippetsStarted, "snippet load to start")
	waitInitializationSignal(t, signals.firstPartial, "ASR partial while dictionary and snippets are pending")
	h.m.Cancel(id)
	if end := h.rec.wait(t); end.State != Cancelled {
		t.Fatalf("end %+v", end)
	}
	before := h.m.Current()
	releaseTerms()
	releaseSnippets()
	waitInitializationSignal(t, termsReturned, "late dictionary load to return")
	waitInitializationSignal(t, snippetsReturned, "late snippet load to return")
	after := h.m.Current()
	if after.Rev != before.Rev || len(h.text.pastes()) != 0 {
		t.Fatalf("late data loads changed ended session: before %+v, after %+v", before, after)
	}
}

func TestSupersededLLMInitializationCannotMutateNextSession(t *testing.T) {
	provider := &fakeStreamer{final: "hello aura link"}
	model := &fakeLLM{text: "Cleaned."}
	h := newHarness(t, provider, model, func(_ *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
	})
	firstStarted := make(chan struct{}, 1)
	firstRelease, releaseFirst := controlledRelease(t)
	var calls atomic.Int32
	setIndependentResolvers(h, provider, func(context.Context, Kind, settings.Settings) LLMReadiness {
		if calls.Add(1) == 1 {
			firstStarted <- struct{}{}
			<-firstRelease // Deliberately ignores cancellation.
			return LLMReadiness{LLM: model, LLMInfo: Model{ID: "stale-llm"}, Outcome: LLMAvailable}
		}
		return LLMReadiness{LLM: model, LLMInfo: Model{ID: "current-llm"}, Outcome: LLMAvailable}
	})

	first, err := h.m.Start(StartRequest{Kind: store.KindTranscribe, Target: slack})
	if err != nil {
		t.Fatal(err)
	}
	waitInitializationSignal(t, firstStarted, "first LLM resolver to block")
	h.m.Cancel(first)
	if end := h.rec.wait(t); end.State != Cancelled {
		t.Fatalf("first session end %+v", end)
	}

	second, err := h.m.Start(StartRequest{Kind: store.KindTranscribe, Target: slack})
	if err != nil {
		t.Fatal(err)
	}
	current := h.rec.waitFor(t, "second session ASR and LLM readiness", func(snapshot Snapshot) bool {
		return snapshot.ID == second && snapshot.Partial != "" && snapshot.LLM.ID == "current-llm"
	})
	secondReady := h.rec.waitFor(t, "second LLM resolver readiness", func(snapshot Snapshot) bool { return snapshot.ID == second && snapshot.LLM.ID == "current-llm" })
	releaseFirst()
	after := h.m.Current()
	if after.ID != second || after.LLM.ID != "current-llm" || secondReady.ID != second {
		t.Fatalf("stale LLM result mutated the next session: before %+v, after %+v", current, after)
	}
	h.speak(second, 3)
	if end := h.rec.wait(t); end.State != Done || end.Result != "Cleaned." {
		t.Fatalf("second session end %+v", end)
	}
}

func TestEarlyPieceBeforeCleanupReadinessDoesNotBlockASREvents(t *testing.T) {
	provider := &fakeStreamer{
		final: "the complete final transcript", pieces: []string{"early cumulative piece"},
		partials: []string{"part 1", "part 2", "part 3", "part 4"},
	}
	model := &fakeLLM{text: "cleaned transcript"}
	h := newHarness(t, provider, model, func(_ *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
	})
	signals := observeSessionSignals(h)
	started := make(chan struct{}, 1)
	release, openRelease := controlledRelease(t)
	setIndependentResolvers(h, provider, gatedLLMResolver(model, started, release, nil))

	id := h.start()
	waitInitializationSignal(t, started, "LLM resolver to start")
	waitInitializationSignal(t, signals.firstPartial, "first ASR partial")
	h.rec.waitFor(t, "ASR events after the early EventPiece", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.Partial == "part 4"
	})
	if len(model.requests()) != 0 {
		t.Fatal("Early Cleanup started before its dependencies were ready")
	}
	h.speak(id, 5)
	waitInitializationSignal(t, signals.asrFinal, "ASR Final")
	h.rec.waitFor(t, "ASR Final consumed before LLM readiness", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.Raw == "the complete final transcript"
	})
	openRelease()
	end := h.rec.wait(t)
	if end.Result != "cleaned transcript" || len(model.requests()) != 1 {
		t.Fatalf("end %+v, LLM requests %d", end, len(model.requests()))
	}
	request := model.requests()[0]
	input := request.Messages[len(request.Messages)-1].Content
	if !strings.Contains(input, "the complete final transcript") || strings.Contains(input, "early cumulative piece") {
		t.Fatalf("cleanup used the wrong transcript: %q", input)
	}
}

func TestPendingBrowserURLDoesNotBlockASRAndKeepsCleanupOverride(t *testing.T) {
	provider := &fakeStreamer{final: "hello aura link"}
	model := &fakeLLM{text: "must not run"}
	h := newHarness(t, provider, model, func(config *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
		config.Scene = &SceneConfig{
			Rules: func() (appctx.Rules, bool) {
				return appctx.Rules{Groups: []appctx.Group{
					{ID: "url", Enabled: true, URLs: []string{"example.com"}, Features: appctx.DefaultFeatures, TranscribeCleanup: appctx.CleanupOff},
					{ID: "browser", Enabled: true, Apps: []appctx.App{{BundleID: chrome.BundleID}}, Features: appctx.DefaultFeatures, TranscribeCleanup: appctx.CleanupOn},
				}}, true
			},
			IsBrowser: func(string) bool { return true }, URLTimeout: time.Hour,
		}
	})
	signals := observeSessionSignals(h)
	urlStarted, releaseURL := make(chan struct{}, 1), make(chan struct{})
	var releaseURLOnce sync.Once
	openURL := func() { releaseURLOnce.Do(func() { close(releaseURL) }) }
	t.Cleanup(openURL)
	h.m.cfg.Scene.URL = func(context.Context, proto.App) (string, error) {
		urlStarted <- struct{}{}
		<-releaseURL
		return "https://example.com/chat", nil
	}
	effectiveCleanup := make(chan bool, 1)
	setIndependentResolvers(h, provider, func(_ context.Context, _ Kind, effective settings.Settings) LLMReadiness {
		effectiveCleanup <- effective.Features.Transcribe.Enhance
		if effective.Features.Transcribe.Enhance {
			return LLMReadiness{LLM: model, LLMInfo: Model{ID: "fake-llm"}, Outcome: LLMAvailable}
		}
		return LLMReadiness{Outcome: LLMNotRequested}
	})

	id, err := h.m.Start(StartRequest{Kind: store.KindTranscribe, Target: chrome})
	if err != nil {
		t.Fatal(err)
	}
	waitInitializationSignal(t, urlStarted, "browser URL lookup to start")
	waitInitializationSignal(t, signals.firstPartial, "ASR partial while browser URL is pending")
	h.speak(id, 5)
	waitInitializationSignal(t, signals.asrFinal, "ASR Final while browser URL is pending")
	select {
	case <-effectiveCleanup:
		t.Fatal("Cleanup override was resolved before the URL lookup completed")
	default:
	}

	openURL()
	end := h.rec.wait(t)
	select {
	case got := <-effectiveCleanup:
		t.Fatalf("LLM resolver ran for URL-specific Cleanup-off override: %t", got)
	default:
	}
	if end.Result != "hello VoxtNew" || len(model.requests()) != 0 || end.Scene == nil || end.Scene.GroupID() != "url" {
		t.Fatalf("end %+v, LLM requests %d", end, len(model.requests()))
	}
}
