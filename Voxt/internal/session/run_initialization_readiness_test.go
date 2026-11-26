package session

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func readinessScene(urlStarted chan<- struct{}, release <-chan struct{}, urlCleanup, browserCleanup appctx.Cleanup) *SceneConfig {
	return &SceneConfig{
		Rules: func() (appctx.Rules, bool) {
			return appctx.Rules{Groups: []appctx.Group{
				{ID: "url", Enabled: true, URLs: []string{"example.com"}, Features: appctx.DefaultFeatures, TranscribeCleanup: urlCleanup},
				{ID: "browser", Enabled: true, Apps: []appctx.App{{BundleID: chrome.BundleID}}, Features: appctx.DefaultFeatures, TranscribeCleanup: browserCleanup},
			}}, true
		},
		IsBrowser:  func(string) bool { return true },
		URLTimeout: 5 * time.Second,
		URL: func(ctx context.Context, _ proto.App) (string, error) {
			urlStarted <- struct{}{}
			select {
			case <-release:
				return "https://example.com/chat", nil
			case <-ctx.Done():
				return "", ctx.Err()
			}
		},
	}
}

func readinessRecording(audio *fakeAudio, frames int) <-chan struct{} {
	audio.file = make([]byte, frames*len(frame))
	for i := range frames {
		copy(audio.file[i*len(frame):], frame)
	}
	audio.drained = make(chan struct{})
	return audio.drained
}

func stopReadinessRecording(t testing.TB, h *harness, id string, drained <-chan struct{}) {
	t.Helper()
	waitInitializationSignal(t, drained, "fixed recording to drain")
	h.m.Stop(id)
	h.rec.waitFor(t, "recognizing after Stop", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.State == Recognizing
	})
}

func useReadinessResolvers(h *harness, provider asr.Provider, model LLM, started chan<- struct{}) *atomic.Int32 {
	llmCalls := &atomic.Int32{}
	setIndependentResolvers(h, provider, func(context.Context, Kind, settings.Settings) LLMReadiness {
		llmCalls.Add(1)
		select {
		case started <- struct{}{}:
		default:
		}
		return LLMReadiness{LLM: model, LLMInfo: Model{ID: "fake-llm", Name: "Fake LLM"}, Outcome: LLMAvailable}
	})
	return llmCalls
}

func TestSceneReadinessStartsLLMWhileASRFinalIsPending(t *testing.T) {
	urlStarted := make(chan struct{}, 1)
	urlRelease, releaseURL := controlledRelease(t)
	finalStarted := make(chan struct{}, 1)
	finalRelease, releaseFinal := controlledRelease(t)
	provider := &fakeStreamer{final: "hello aura link", finishGate: finalRelease, finishStarted: finalStarted}
	model := &fakeLLM{text: "Cleaned text."}
	h := newHarness(t, provider, model, func(config *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = false
		config.Scene = readinessScene(urlStarted, urlRelease, appctx.CleanupOn, appctx.CleanupOff)
	})
	drained := readinessRecording(h.audio, 4)
	signals := observeSessionSignals(h)
	llmStarted := make(chan struct{}, 1)
	useReadinessResolvers(h, provider, model, llmStarted)
	id := h.start()
	waitInitializationSignal(t, urlStarted, "browser URL lookup")
	waitInitializationSignal(t, signals.asrOpenDone, "ASR Open")
	waitInitializationSignal(t, signals.firstPartial, "ASR audio consumption")
	stopReadinessRecording(t, h, id, drained)
	waitInitializationSignal(t, finalStarted, "blocked ASR Final")

	releaseURL()
	waitInitializationSignal(t, llmStarted, "LLM resolver before ASR Final is released")
	select {
	case <-signals.asrFinal:
		t.Fatal("ASR Final completed while its gate was closed")
	default:
	}
	releaseFinal()
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "Cleaned text." || end.Scene == nil || end.Scene.GroupID() != "url" {
		t.Fatalf("session result or resolved scene: %+v", end)
	}
	if len(model.requests()) != 1 || len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
		t.Fatalf("LLM requests=%d, deliveries=%q", len(model.requests()), h.text.pastes())
	}
}

func TestLateRemoteLLMReadinessStartsHeadStartWhileASRFinalIsPending(t *testing.T) {
	const partial = "we ship the new build on friday"
	finalStarted := make(chan struct{}, 1)
	finalRelease, releaseFinal := controlledRelease(t)
	provider := &fakeStreamer{final: partial, silent: true, finishGate: finalRelease, finishStarted: finalStarted}
	model := &fakeLLM{text: "We ship the new build on Friday."}
	h := newHarness(t, provider, model, func(_ *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
		s.Dictionary.ASRHints = false
	})
	drained := readinessRecording(h.audio, 4)
	signals := observeSessionSignals(h)
	started, releaseLLM := setBlockedRemoteResolvers(h, provider, model)
	id := h.start()
	waitInitializationSignal(t, started, "blocked remote LLM resolver")
	waitInitializationSignal(t, signals.asrOpenDone, "ASR Open while LLM resolution is blocked")
	stream := initializationStream(t, provider)
	stream.emit(asr.Event{Kind: asr.EventPartial, Text: partial})
	h.rec.waitFor(t, "usable ASR partial", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.Partial == partial
	})
	stopReadinessRecording(t, h, id, drained)
	waitInitializationSignal(t, finalStarted, "ASR Final held after Stop")
	if len(model.requests()) != 0 {
		t.Fatal("Head Start ran before remote LLM readiness")
	}

	releaseLLM()
	waitInitializationSignal(t, signals.llmRequest, "Head Start request before ASR Final")
	select {
	case <-signals.asrFinal:
		t.Fatal("ASR Final completed while its gate was closed")
	default:
	}
	releaseFinal()

	end := h.rec.wait(t)
	requests := model.requests()
	if end.State != Done || end.Raw != partial || end.Result != "We ship the new build on Friday." ||
		len(requests) != 1 || inputOf(requests[0]) != partial || len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
		t.Fatalf("end=%+v, requests=%+v, deliveries=%q", end, requests, h.text.pastes())
	}
}

func TestLateRemoteLLMReadinessDoesNotHeadStartAfterStopDeadline(t *testing.T) {
	const (
		partial = "we ship the new build on friday"
		budget  = 200 * time.Millisecond
	)
	clock := newSoftDeadlineClock()
	finalStarted := make(chan struct{}, 1)
	finalRelease, releaseFinal := controlledRelease(t)
	provider := &fakeStreamer{final: partial, silent: true, finishGate: finalRelease, finishStarted: finalStarted}
	model := &fakeLLM{text: "We ship the new build on Friday."}
	h := newHarness(t, provider, model, func(config *Config, s *settings.Settings) {
		config.Now, config.AfterFunc = clock.Now, clock.AfterFunc
		config.CleanupLatency = CleanupLatencyPolicy{AutoFallbackAfter: budget}
		s.Features.Transcribe.Enhance = true
		s.Dictionary.ASRHints = false
	})
	drained := readinessRecording(h.audio, 4)
	signals := observeSessionSignals(h)
	started, releaseLLM := setBlockedRemoteResolvers(h, provider, model)
	id := h.start()
	waitInitializationSignal(t, started, "blocked remote LLM resolver")
	waitInitializationSignal(t, signals.asrOpenDone, "ASR Open while LLM resolution is blocked")
	stream := initializationStream(t, provider)
	stream.emit(asr.Event{Kind: asr.EventPartial, Text: partial})
	h.rec.waitFor(t, "usable ASR partial", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.Partial == partial
	})
	stopReadinessRecording(t, h, id, drained)
	waitInitializationSignal(t, finalStarted, "ASR Final held after Stop")
	clock.Advance(budget)

	releaseLLM()
	h.rec.waitFor(t, "late LLM readiness", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.LLM.ID == "fake-llm"
	})
	if len(model.requests()) != 0 {
		t.Fatal("Head Start began after the Stop-relative cleanup deadline")
	}
	releaseFinal()

	end := h.rec.wait(t)
	if end.State != Done || end.Raw != partial || end.Result != partial || len(model.requests()) != 0 || len(h.text.pastes()) != 1 || h.text.pastes()[0] != partial {
		t.Fatalf("end=%+v, requests=%d, deliveries=%q", end, len(model.requests()), h.text.pastes())
	}
}

func TestSceneReadinessStartsLLMWhileASRResolverOrOpenIsPending(t *testing.T) {
	t.Run("ASR resolver", func(t *testing.T) {
		urlStarted := make(chan struct{}, 1)
		urlRelease, releaseURL := controlledRelease(t)
		asrStarted := make(chan struct{}, 1)
		asrRelease, releaseASR := controlledRelease(t)
		provider := &fakeStreamer{final: "hello aura link"}
		model := &fakeLLM{text: "Cleaned text."}
		h := newHarness(t, provider, model, func(config *Config, s *settings.Settings) {
			s.Features.Transcribe.Enhance = false
			config.Scene = readinessScene(urlStarted, urlRelease, appctx.CleanupOn, appctx.CleanupOff)
		})
		drained := readinessRecording(h.audio, 4)
		llmStarted := make(chan struct{}, 1)
		h.m.cfg.ResolveASR = func(ctx context.Context, _ Kind, _ settings.Settings) ASRReadiness {
			asrStarted <- struct{}{}
			select {
			case <-asrRelease:
				return ASRReadiness{ASR: provider, ASRInfo: Model{ID: "fake-asr", Name: "Fake ASR"}}
			case <-ctx.Done():
				return ASRReadiness{Err: ctx.Err()}
			}
		}
		h.m.cfg.ResolveLLM = func(context.Context, Kind, settings.Settings) LLMReadiness {
			llmStarted <- struct{}{}
			return LLMReadiness{LLM: model, LLMInfo: Model{ID: "fake-llm", Name: "Fake LLM"}, Outcome: LLMAvailable}
		}
		id := h.start()
		waitInitializationSignal(t, urlStarted, "browser URL lookup")
		waitInitializationSignal(t, asrStarted, "blocked ASR resolver")
		stopReadinessRecording(t, h, id, drained)
		releaseURL()
		waitInitializationSignal(t, llmStarted, "LLM resolver while ASR resolution is pending")
		releaseASR()
		if end := h.rec.wait(t); end.State != Done || end.Result != "Cleaned text." {
			t.Fatalf("end %+v", end)
		}
	})

	t.Run("ASR Open", func(t *testing.T) {
		urlStarted := make(chan struct{}, 1)
		urlRelease, releaseURL := controlledRelease(t)
		openRelease, releaseOpen := controlledRelease(t)
		provider := &fakeStreamer{final: "hello aura link", openGate: openRelease}
		model := &fakeLLM{text: "Cleaned text."}
		h := newHarness(t, provider, model, func(config *Config, s *settings.Settings) {
			s.Features.Transcribe.Enhance = false
			config.Scene = readinessScene(urlStarted, urlRelease, appctx.CleanupOn, appctx.CleanupOff)
		})
		drained := readinessRecording(h.audio, 4)
		signals := observeSessionSignals(h)
		llmStarted := make(chan struct{}, 1)
		useReadinessResolvers(h, provider, model, llmStarted)
		id := h.start()
		waitInitializationSignal(t, urlStarted, "browser URL lookup")
		waitInitializationSignal(t, signals.asrOpenStarted, "blocked ASR Open")
		stopReadinessRecording(t, h, id, drained)
		releaseURL()
		waitInitializationSignal(t, llmStarted, "LLM resolver while ASR Open is pending")
		releaseOpen()
		if end := h.rec.wait(t); end.State != Done || end.Result != "Cleaned text." {
			t.Fatalf("end %+v", end)
		}
	})
}

func TestSceneReadinessStartsLLMWhileFileTranscriptionIsPending(t *testing.T) {
	urlStarted := make(chan struct{}, 1)
	urlRelease, releaseURL := controlledRelease(t)
	fileStarted := make(chan struct{}, 1)
	fileRelease, releaseFile := controlledRelease(t)
	provider := &fakeFile{text: "hello aura link", transcribeStarted: fileStarted, transcribeGate: fileRelease}
	model := &fakeLLM{text: "Cleaned text."}
	h := newHarness(t, provider, model, func(config *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = false
		config.Scene = readinessScene(urlStarted, urlRelease, appctx.CleanupOn, appctx.CleanupOff)
	})
	drained := readinessRecording(h.audio, 4)
	llmStarted := make(chan struct{}, 1)
	useReadinessResolvers(h, provider, model, llmStarted)
	id := h.start()
	waitInitializationSignal(t, urlStarted, "browser URL lookup")
	stopReadinessRecording(t, h, id, drained)
	waitInitializationSignal(t, fileStarted, "blocked file transcription")
	releaseURL()
	waitInitializationSignal(t, llmStarted, "LLM resolver while file transcription is pending")
	releaseFile()
	if end := h.rec.wait(t); end.State != Done || end.Result != "Cleaned text." || len(h.text.pastes()) != 1 {
		t.Fatalf("end %+v, deliveries %q", end, h.text.pastes())
	}
}

func TestSceneCleanupOffOverrideSkipsLLMWhileASRFinalIsPending(t *testing.T) {
	urlStarted := make(chan struct{}, 1)
	urlRelease, releaseURL := controlledRelease(t)
	finalStarted := make(chan struct{}, 1)
	finalRelease, releaseFinal := controlledRelease(t)
	provider := &fakeStreamer{final: "hello aura link", finishGate: finalRelease, finishStarted: finalStarted}
	model := &fakeLLM{text: "Must not run."}
	h := newHarness(t, provider, model, func(config *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
		config.Scene = readinessScene(urlStarted, urlRelease, appctx.CleanupOff, appctx.CleanupOn)
	})
	drained := readinessRecording(h.audio, 4)
	llmCalls := atomic.Int32{}
	h.m.cfg.ResolveASR = func(context.Context, Kind, settings.Settings) ASRReadiness {
		return ASRReadiness{ASR: provider, ASRInfo: Model{ID: "fake-asr", Name: "Fake ASR"}}
	}
	h.m.cfg.ResolveLLM = func(context.Context, Kind, settings.Settings) LLMReadiness {
		llmCalls.Add(1)
		return LLMReadiness{LLM: model, LLMInfo: Model{ID: "fake-llm", Name: "Fake LLM"}, Outcome: LLMAvailable}
	}
	id := h.start()
	waitInitializationSignal(t, urlStarted, "browser URL lookup")
	stopReadinessRecording(t, h, id, drained)
	waitInitializationSignal(t, finalStarted, "blocked ASR Final")
	releaseURL()
	releaseFinal()
	end := h.rec.wait(t)
	if end.State != Done || end.Raw != "hello aura link" || end.Result != "hello VoxtNew" {
		t.Fatalf("end %+v", end)
	}
	if llmCalls.Load() != 0 || len(model.requests()) != 0 || len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
		t.Fatalf("LLM resolvers=%d, chats=%d, deliveries=%q", llmCalls.Load(), len(model.requests()), h.text.pastes())
	}
}

func TestSceneReadinessUsesOriginalStopDeadline(t *testing.T) {
	for _, test := range []struct {
		name       string
		advance    time.Duration
		wantChat   bool
		wantRemain time.Duration
	}{
		{name: "scene before deadline", advance: cleanupSoftDeadlineTestDuration / 2, wantChat: true, wantRemain: cleanupSoftDeadlineTestDuration / 2},
		{name: "scene after deadline", advance: cleanupSoftDeadlineTestDuration + time.Second},
	} {
		t.Run(test.name, func(t *testing.T) {
			clock := newSoftDeadlineClock()
			urlStarted := make(chan struct{}, 1)
			urlRelease, releaseURL := controlledRelease(t)
			finalStarted := make(chan struct{}, 1)
			finalRelease, releaseFinal := controlledRelease(t)
			deadlineScheduled := make(chan time.Duration, 2)
			provider := &fakeStreamer{final: "hello aura link", finishGate: finalRelease, finishStarted: finalStarted}
			model := &fakeLLM{hang: true}
			h := newHarness(t, provider, model, func(config *Config, s *settings.Settings) {
				cleanupLatencyConfig(clock, true)(config, s)
				config.Scene = readinessScene(urlStarted, urlRelease, appctx.CleanupOn, appctx.CleanupOff)
				underlying := config.AfterFunc
				config.AfterFunc = func(delay time.Duration, callback func()) func() bool {
					stop := underlying(delay, callback)
					deadlineScheduled <- delay
					return stop
				}
			})
			drained := readinessRecording(h.audio, 4)
			llmStarted := make(chan struct{}, 1)
			useReadinessResolvers(h, provider, model, llmStarted)
			signals := observeSessionSignals(h)
			id := h.start()
			waitInitializationSignal(t, urlStarted, "browser URL lookup")
			waitInitializationSignal(t, signals.firstPartial, "ASR partial")
			stopReadinessRecording(t, h, id, drained)
			waitInitializationSignal(t, finalStarted, "blocked ASR Final")
			clock.Advance(test.advance)
			releaseURL()
			waitInitializationSignal(t, llmStarted, "LLM resolver after scene decision")
			releaseFinal()
			if test.wantChat {
				waitInitializationSignal(t, signals.llmRequest, "Cleanup Chat before the original deadline")
				scheduled := 0
			forDrain:
				for {
					select {
					case remaining := <-deadlineScheduled:
						scheduled++
						if remaining != test.wantRemain {
							t.Fatalf("deadline timer remaining %v, want %v", remaining, test.wantRemain)
						}
					default:
						break forDrain
					}
				}
				if scheduled == 0 || scheduled > 2 {
					t.Fatalf("Stop-relative cleanup deadline scheduled %d timers, want one active timer and at most one stopped readiness timer", scheduled)
				}
				clock.Advance(test.wantRemain)
				clock.FireNext(t)
			} else {
				end := h.rec.wait(t)
				if end.Result != "hello VoxtNew" || len(model.requests()) != 0 || !h.logged("cleanup outcome=soft_deadline") {
					t.Fatalf("after-deadline scene produced end=%+v chats=%d soft-deadline=%v", end, len(model.requests()), h.logged("cleanup outcome=soft_deadline"))
				}
				if len(h.text.pastes()) != 1 {
					t.Fatalf("deliveries %q", h.text.pastes())
				}
				return
			}
			end := h.rec.wait(t)
			if end.Result != "hello VoxtNew" || len(model.requests()) != 1 || !h.logged("cleanup outcome=soft_deadline") || len(h.text.pastes()) != 1 {
				t.Fatalf("end=%+v chats=%d deliveries=%q", end, len(model.requests()), h.text.pastes())
			}
		})
	}
}

func TestCancelFencesSceneAndLateLLMWhileASRFinalIsPending(t *testing.T) {
	urlStarted := make(chan struct{}, 1)
	urlRelease, releaseURL := controlledRelease(t)
	finalStarted := make(chan struct{}, 1)
	finalRelease, releaseFinal := controlledRelease(t)
	llmStarted := make(chan struct{}, 1)
	llmRelease, releaseLLM := controlledRelease(t)
	llmReturned := make(chan struct{}, 1)
	provider := &fakeStreamer{final: "hello aura link", finishGate: finalRelease, finishStarted: finalStarted}
	model := &fakeLLM{text: "Must not run."}
	h := newHarness(t, provider, model, func(config *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = false
		config.Scene = readinessScene(urlStarted, urlRelease, appctx.CleanupOn, appctx.CleanupOff)
	})
	drained := readinessRecording(h.audio, 4)
	h.m.cfg.ResolveASR = func(context.Context, Kind, settings.Settings) ASRReadiness {
		return ASRReadiness{ASR: provider, ASRInfo: Model{ID: "fake-asr", Name: "Fake ASR"}}
	}
	h.m.cfg.ResolveLLM = gatedLLMResolver(model, llmStarted, llmRelease, llmReturned)
	id := h.start()
	waitInitializationSignal(t, urlStarted, "browser URL lookup")
	stopReadinessRecording(t, h, id, drained)
	waitInitializationSignal(t, finalStarted, "blocked ASR Final")
	releaseURL()
	waitInitializationSignal(t, llmStarted, "blocked LLM resolver")
	h.m.Cancel(id)
	end := h.rec.waitFor(t, "cancelled session", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.State == Cancelled
	})
	releaseLLM()
	waitInitializationSignal(t, llmReturned, "late LLM resolver return")
	releaseFinal()
	if end.Result != "" || len(model.requests()) != 0 || len(h.text.pastes()) != 0 {
		t.Fatalf("cancelled result=%+v chats=%d deliveries=%q", end, len(model.requests()), h.text.pastes())
	}
	if snapshot := h.m.Current(); snapshot.ID != id || snapshot.State != Cancelled || snapshot.Result != "" {
		t.Fatalf("late initialization changed the session: %+v", snapshot)
	}
}

func TestSceneReadinessStartsLLMWhileDictionaryIsPendingAfterASRFinal(t *testing.T) {
	urlStarted := make(chan struct{}, 1)
	urlRelease, releaseURL := controlledRelease(t)
	termsStarted := make(chan struct{}, 1)
	termsRelease, releaseTerms := controlledRelease(t)
	provider := &fakeStreamer{final: "hello aura link"}
	model := &fakeLLM{text: "Hello VoxtNew."}
	h := newHarness(t, provider, model, func(config *Config, currentSettings *settings.Settings) {
		currentSettings.Features.Transcribe.Enhance = false
		currentSettings.Dictionary.Enabled = true
		currentSettings.Dictionary.ASRHints = false
		config.Scene = readinessScene(urlStarted, urlRelease, appctx.CleanupOn, appctx.CleanupOff)
	})
	h.m.cfg.Glossary = func(context.Context) ([]store.Term, error) {
		termsStarted <- struct{}{}
		<-termsRelease
		return []store.Term{{Term: "VoxtNew", Aliases: []string{"aura link"}, Enabled: true}}, nil
	}
	h.m.cfg.Snippets = func(context.Context) ([]snippet.Snippet, error) { return nil, nil }
	drained := readinessRecording(h.audio, 4)
	signals := observeSessionSignals(h)
	llmStarted := make(chan struct{}, 1)
	llmCalls := useReadinessResolvers(h, provider, model, llmStarted)
	id := h.start()
	waitInitializationSignal(t, urlStarted, "browser URL lookup")
	waitInitializationSignal(t, termsStarted, "blocked dictionary load")
	waitInitializationSignal(t, signals.asrOpenDone, "ASR Open")
	waitInitializationSignal(t, signals.firstPartial, "ASR audio consumption")
	stopReadinessRecording(t, h, id, drained)
	waitInitializationSignal(t, signals.asrFinal, "ASR Final with dictionary loading pending")

	releaseURL()
	waitInitializationSignal(t, llmStarted, "LLM resolver while dictionary loading is pending")
	if llmCalls.Load() != 1 || len(model.requests()) != 0 {
		t.Fatalf("LLM resolver calls=%d, Chat requests=%d before dictionary readiness", llmCalls.Load(), len(model.requests()))
	}
	if snapshot := h.m.Current(); snapshot.State.Ended() {
		t.Fatalf("session ended before mandatory dictionary readiness: %+v", snapshot)
	}
	releaseTerms()
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "Hello VoxtNew." || llmCalls.Load() != 1 || len(model.requests()) != 1 {
		t.Fatalf("end=%+v, resolver calls=%d, Chat requests=%d", end, llmCalls.Load(), len(model.requests()))
	}
	request := model.requests()[0]
	if len(request.Messages) == 0 || !strings.Contains(request.Messages[len(request.Messages)-1].Content, "hello VoxtNew") || strings.Contains(request.Messages[len(request.Messages)-1].Content, "aura link") {
		t.Fatalf("cleanup input was not dictionary-normalized: %+v", request.Messages)
	}
	if len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
		t.Fatalf("deliveries=%q, want one normalized cleanup result", h.text.pastes())
	}
}

func TestSceneReadinessStartsLLMWhileSnippetsArePendingAfterASRFinal(t *testing.T) {
	urlStarted := make(chan struct{}, 1)
	urlRelease, releaseURL := controlledRelease(t)
	snippetsStarted := make(chan struct{}, 1)
	snippetsRelease, releaseSnippets := controlledRelease(t)
	provider := &fakeStreamer{final: "send it to my email address please"}
	model := &fakeLLM{text: "Send ⟦my email address⟧, please."}
	h := newHarness(t, provider, model, func(config *Config, currentSettings *settings.Settings) {
		currentSettings.Features.Transcribe.Enhance = false
		currentSettings.Dictionary.Enabled = false
		config.Scene = readinessScene(urlStarted, urlRelease, appctx.CleanupOn, appctx.CleanupOff)
	})
	h.m.cfg.Glossary = func(context.Context) ([]store.Term, error) { return nil, nil }
	h.m.cfg.Snippets = func(context.Context) ([]snippet.Snippet, error) {
		snippetsStarted <- struct{}{}
		<-snippetsRelease
		return []snippet.Snippet{{ID: "email", Trigger: "my email address", Text: "ruiling@example.com"}}, nil
	}
	drained := readinessRecording(h.audio, 4)
	signals := observeSessionSignals(h)
	llmStarted := make(chan struct{}, 1)
	llmCalls := useReadinessResolvers(h, provider, model, llmStarted)
	id := h.start()
	waitInitializationSignal(t, urlStarted, "browser URL lookup")
	waitInitializationSignal(t, snippetsStarted, "blocked snippet load")
	waitInitializationSignal(t, signals.asrOpenDone, "ASR Open")
	waitInitializationSignal(t, signals.firstPartial, "ASR audio consumption")
	stopReadinessRecording(t, h, id, drained)
	waitInitializationSignal(t, signals.asrFinal, "ASR Final with snippet loading pending")

	releaseURL()
	waitInitializationSignal(t, llmStarted, "LLM resolver while snippet loading is pending")
	if llmCalls.Load() != 1 || len(model.requests()) != 0 {
		t.Fatalf("LLM resolver calls=%d, Chat requests=%d before snippet readiness", llmCalls.Load(), len(model.requests()))
	}
	releaseSnippets()
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "Send ruiling@example.com, please." || llmCalls.Load() != 1 || len(model.requests()) != 1 {
		t.Fatalf("end=%+v, resolver calls=%d, Chat requests=%d", end, llmCalls.Load(), len(model.requests()))
	}
	request := model.requests()[0]
	if len(request.Messages) == 0 || !strings.Contains(request.Messages[len(request.Messages)-1].Content, "⟦my email address⟧") {
		t.Fatalf("cleanup input did not preserve snippet marking: %+v", request.Messages)
	}
	if len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
		t.Fatalf("deliveries=%q, want one snippet-expanded result", h.text.pastes())
	}
}

func TestSceneCleanupOffOverrideSkipsLLMWhileDictionaryIsPendingAfterASRFinal(t *testing.T) {
	urlStarted := make(chan struct{}, 1)
	urlRelease, releaseURL := controlledRelease(t)
	termsStarted := make(chan struct{}, 1)
	termsRelease, releaseTerms := controlledRelease(t)
	provider := &fakeStreamer{final: "hello aura link"}
	model := &fakeLLM{text: "must not run"}
	h := newHarness(t, provider, model, func(config *Config, currentSettings *settings.Settings) {
		currentSettings.Features.Transcribe.Enhance = true
		currentSettings.Dictionary.ASRHints = false
		config.Scene = readinessScene(urlStarted, urlRelease, appctx.CleanupOff, appctx.CleanupOn)
	})
	h.m.cfg.Glossary = func(context.Context) ([]store.Term, error) {
		termsStarted <- struct{}{}
		<-termsRelease
		return []store.Term{{Term: "VoxtNew", Aliases: []string{"aura link"}, Enabled: true}}, nil
	}
	h.m.cfg.Snippets = func(context.Context) ([]snippet.Snippet, error) { return nil, nil }
	drained := readinessRecording(h.audio, 4)
	signals := observeSessionSignals(h)
	llmStarted := make(chan struct{}, 1)
	llmCalls := useReadinessResolvers(h, provider, model, llmStarted)
	id := h.start()
	waitInitializationSignal(t, urlStarted, "browser URL lookup")
	waitInitializationSignal(t, termsStarted, "blocked dictionary load")
	waitInitializationSignal(t, signals.asrOpenDone, "ASR Open")
	waitInitializationSignal(t, signals.firstPartial, "ASR audio consumption")
	stopReadinessRecording(t, h, id, drained)
	waitInitializationSignal(t, signals.asrFinal, "ASR Final before scene cleanup-off decision")
	releaseURL()
	h.rec.waitFor(t, "resolved URL scene while dictionary loading is pending", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.Scene != nil
	})
	if llmCalls.Load() != 0 || len(model.requests()) != 0 {
		t.Fatalf("cleanup-off scene invoked LLM: resolver calls=%d, Chat requests=%d", llmCalls.Load(), len(model.requests()))
	}
	releaseTerms()
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "hello VoxtNew" || llmCalls.Load() != 0 || len(model.requests()) != 0 {
		t.Fatalf("end=%+v, resolver calls=%d, Chat requests=%d", end, llmCalls.Load(), len(model.requests()))
	}
	if len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
		t.Fatalf("deliveries=%q, want one normalized raw result", h.text.pastes())
	}
}

func TestCancelFencesSceneWhileTranscribeDataIsPending(t *testing.T) {
	urlStarted := make(chan struct{}, 1)
	urlRelease, releaseURL := controlledRelease(t)
	termsStarted, snippetsStarted := make(chan struct{}, 1), make(chan struct{}, 1)
	termsRelease, releaseTerms := controlledRelease(t)
	snippetsRelease, releaseSnippets := controlledRelease(t)
	provider := &fakeStreamer{final: "hello aura link"}
	model := &fakeLLM{text: "must not run"}
	h := newHarness(t, provider, model, func(config *Config, currentSettings *settings.Settings) {
		currentSettings.Features.Transcribe.Enhance = false
		currentSettings.Dictionary.ASRHints = false
		config.Scene = readinessScene(urlStarted, urlRelease, appctx.CleanupOn, appctx.CleanupOff)
	})
	h.m.cfg.Glossary = func(context.Context) ([]store.Term, error) {
		termsStarted <- struct{}{}
		<-termsRelease
		return []store.Term{{Term: "VoxtNew", Aliases: []string{"aura link"}, Enabled: true}}, nil
	}
	h.m.cfg.Snippets = func(context.Context) ([]snippet.Snippet, error) {
		snippetsStarted <- struct{}{}
		<-snippetsRelease
		return nil, nil
	}
	drained := readinessRecording(h.audio, 4)
	signals := observeSessionSignals(h)
	llmStarted := make(chan struct{}, 1)
	llmCalls := useReadinessResolvers(h, provider, model, llmStarted)
	id := h.start()
	waitInitializationSignal(t, urlStarted, "browser URL lookup")
	waitInitializationSignal(t, termsStarted, "blocked dictionary load")
	waitInitializationSignal(t, snippetsStarted, "blocked snippet load")
	waitInitializationSignal(t, signals.asrOpenDone, "ASR Open")
	waitInitializationSignal(t, signals.firstPartial, "ASR audio consumption")
	stopReadinessRecording(t, h, id, drained)
	waitInitializationSignal(t, signals.asrFinal, "ASR Final before cancellation")
	h.m.Cancel(id)
	end := h.rec.waitFor(t, "cancelled session awaiting scene and transcript data", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.State == Cancelled
	})
	releaseURL()
	releaseTerms()
	releaseSnippets()
	if end.Result != "" || llmCalls.Load() != 0 || len(model.requests()) != 0 || len(h.text.pastes()) != 0 {
		t.Fatalf("cancelled end=%+v, resolver calls=%d, Chats=%d, deliveries=%q", end, llmCalls.Load(), len(model.requests()), h.text.pastes())
	}
}

func TestSceneReadinessAfterOriginalStopDeadlineSkipsCleanupChat(t *testing.T) {
	clock := newSoftDeadlineClock()
	urlStarted := make(chan struct{}, 1)
	urlRelease, releaseURL := controlledRelease(t)
	termsStarted := make(chan struct{}, 1)
	termsRelease, releaseTerms := controlledRelease(t)
	provider := &fakeStreamer{final: "hello aura link"}
	model := &fakeLLM{text: "must not run after deadline"}
	h := newHarness(t, provider, model, func(config *Config, currentSettings *settings.Settings) {
		cleanupLatencyConfig(clock, true)(config, currentSettings)
		currentSettings.Features.Transcribe.Enhance = false
		currentSettings.Dictionary.ASRHints = false
		config.Scene = readinessScene(urlStarted, urlRelease, appctx.CleanupOn, appctx.CleanupOff)
	})
	h.m.cfg.Glossary = func(context.Context) ([]store.Term, error) {
		termsStarted <- struct{}{}
		<-termsRelease
		return []store.Term{{Term: "VoxtNew", Aliases: []string{"aura link"}, Enabled: true}}, nil
	}
	h.m.cfg.Snippets = func(context.Context) ([]snippet.Snippet, error) { return nil, nil }
	drained := readinessRecording(h.audio, 4)
	signals := observeSessionSignals(h)
	llmStarted := make(chan struct{}, 1)
	llmCalls := useReadinessResolvers(h, provider, model, llmStarted)
	id := h.start()
	waitInitializationSignal(t, urlStarted, "browser URL lookup")
	waitInitializationSignal(t, termsStarted, "blocked dictionary load")
	waitInitializationSignal(t, signals.asrOpenDone, "ASR Open")
	waitInitializationSignal(t, signals.firstPartial, "ASR audio consumption")
	stopReadinessRecording(t, h, id, drained)
	waitInitializationSignal(t, signals.asrFinal, "ASR Final while scene and dictionary are pending")
	clock.Advance(cleanupSoftDeadlineTestDuration + time.Second)
	releaseURL()
	waitInitializationSignal(t, llmStarted, "LLM resolver after scene resolves")
	if llmCalls.Load() != 1 || len(model.requests()) != 0 {
		t.Fatalf("before dictionary readiness: resolver calls=%d, Chat requests=%d", llmCalls.Load(), len(model.requests()))
	}
	releaseTerms()
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "hello VoxtNew" || len(model.requests()) != 0 || !h.logged("cleanup outcome=soft_deadline") {
		t.Fatalf("after-deadline readiness produced end=%+v, Chats=%d, soft-deadline=%v", end, len(model.requests()), h.logged("cleanup outcome=soft_deadline"))
	}
	if llmCalls.Load() != 1 || len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
		t.Fatalf("resolver calls=%d deliveries=%q", llmCalls.Load(), h.text.pastes())
	}
}
