package session

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/appctx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
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

func useReadinessResolvers(h *harness, provider asr.Provider, model LLM, started chan<- struct{}) {
	setIndependentResolvers(h, provider, func(context.Context, Kind, settings.Settings) LLMReadiness {
		started <- struct{}{}
		return LLMReadiness{LLM: model, LLMInfo: Model{ID: "fake-llm", Name: "Fake LLM"}, Outcome: LLMAvailable}
	})
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
	signals := observeInitializationTimings(h)
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
		signals := observeInitializationTimings(h)
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
			signals := observeInitializationTimings(h)
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
