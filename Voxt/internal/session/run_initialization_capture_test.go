package session

import (
	"bytes"
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func blockASRResolver(t *testing.T, h *harness, provider asr.Provider) (chan struct{}, func()) {
	t.Helper()
	started := make(chan struct{}, 1)
	release, openRelease := controlledRelease(t)
	h.m.cfg.ResolveASR = func(context.Context, Kind, settings.Settings) ASRReadiness {
		started <- struct{}{}
		<-release
		return ASRReadiness{ASR: provider, ASRInfo: Model{ID: "fake-asr"}}
	}
	return started, openRelease
}

func TestMicrophoneFailureEndsBeforeBlockedInitialization(t *testing.T) {
	provider := &fakeStreamer{final: "unused"}
	h := newHarness(t, provider, nil, func(_ *Config, currentSettings *settings.Settings) {
		currentSettings.Features.Transcribe.Enhance = true
		currentSettings.Dictionary.ASRHints = true
	})
	h.audio.startErr = &proto.Error{Code: proto.CodePermissionDenied, Message: "permission denied"}
	asrStarted, releaseASR := blockASRResolver(t, h, provider)
	llmStarted, termsStarted, snippetsStarted := make(chan struct{}, 1), make(chan struct{}, 1), make(chan struct{}, 1)
	llmRelease, releaseLLM := controlledRelease(t)
	termsRelease, releaseTerms := controlledRelease(t)
	snippetsRelease, releaseSnippets := controlledRelease(t)
	h.m.cfg.ResolveLLM = func(context.Context, Kind, settings.Settings) LLMReadiness {
		llmStarted <- struct{}{}
		<-llmRelease
		return LLMReadiness{Outcome: LLMUnavailable}
	}
	h.m.cfg.Glossary = func(context.Context) ([]store.Term, error) {
		termsStarted <- struct{}{}
		<-termsRelease
		return nil, nil
	}
	h.m.cfg.Snippets = func(context.Context) ([]snippet.Snippet, error) {
		snippetsStarted <- struct{}{}
		<-snippetsRelease
		return nil, nil
	}

	h.start()
	waitInitializationSignal(t, asrStarted, "blocked ASR resolver")
	waitInitializationSignal(t, llmStarted, "blocked LLM resolver")
	waitInitializationSignal(t, termsStarted, "blocked dictionary resolver")
	waitInitializationSignal(t, snippetsStarted, "blocked snippets resolver")
	end := h.rec.wait(t)
	if end.State != Failed || end.Problem == nil || end.Problem.Code != ProblemMicrophone || h.audio.recorded() != 0 {
		t.Fatalf("microphone failure end=%+v, PCM frames=%d", end, h.audio.recorded())
	}
	releaseASR()
	releaseLLM()
	releaseTerms()
	releaseSnippets()
	if len(provider.streams) != 0 || h.m.Current().State != Failed {
		t.Fatalf("late initialization opened ASR or changed the ended session: %+v", h.m.Current())
	}
}

func TestCaptureFailureAfterPCMDoesNotWaitForRequiredHints(t *testing.T) {
	provider := &fakeStreamer{final: "unused"}
	h := newHarness(t, provider, nil, func(_ *Config, currentSettings *settings.Settings) {
		currentSettings.Features.Transcribe.Enhance = false
		currentSettings.Dictionary.ASRHints = true
	})
	h.audio.failAt = 1
	h.audio.failErr = fmt.Errorf("the helper exited: %w", errHelperForTest)
	termsStarted := make(chan struct{}, 1)
	termsRelease, releaseTerms := controlledRelease(t)
	h.m.cfg.Glossary = func(context.Context) ([]store.Term, error) {
		termsStarted <- struct{}{}
		<-termsRelease
		return []store.Term{{Term: "VoxtNew", Aliases: []string{"aura link"}, Enabled: true}}, nil
	}
	setIndependentResolvers(h, provider, func(context.Context, Kind, settings.Settings) LLMReadiness {
		return LLMReadiness{Outcome: LLMNotRequested}
	})
	started, releaseASR := blockASRResolver(t, h, provider)

	h.start()
	waitInitializationSignal(t, started, "blocked ASR resolver")
	waitInitializationSignal(t, termsStarted, "blocked required dictionary hints")
	end := h.rec.wait(t)
	if end.State != Failed || end.Problem == nil || end.Problem.Code != ProblemHelper || h.audio.recorded() != 1 {
		t.Fatalf("capture failure end=%+v, PCM frames=%d", end, h.audio.recorded())
	}
	if len(provider.streams) != 0 {
		t.Fatal("ASR opened after capture had already failed")
	}
	releaseASR()
	releaseTerms()
}

func TestTooShortCaptureEndsBeforeBlockedASRReadiness(t *testing.T) {
	for _, testCase := range []struct {
		name      string
		stop      bool
		maxFrames int
	}{
		{name: "natural capture end", maxFrames: 2},
		{name: "user Stop", stop: true},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			provider := &fakeStreamer{final: "unused"}
			h := newHarness(t, provider, nil, func(_ *Config, currentSettings *settings.Settings) {
				currentSettings.Features.Transcribe.Enhance = false
				currentSettings.Audio.VAD.Off = true
				currentSettings.Audio.Clean.Off = true
			})
			if testCase.stop {
				h.audio.tick = 50 * time.Millisecond
			} else {
				h.audio.maxFrame = testCase.maxFrames
				h.audio.maxDone = make(chan struct{})
			}
			signals := observeSessionSignals(h)
			setIndependentResolvers(h, provider, func(context.Context, Kind, settings.Settings) LLMReadiness {
				return LLMReadiness{Outcome: LLMNotRequested}
			})
			started, releaseASR := blockASRResolver(t, h, provider)
			id := h.start()
			waitInitializationSignal(t, started, "blocked ASR resolver")
			if testCase.stop {
				waitInitializationSignal(t, signals.firstPCMCaptured, "first captured PCM")
				h.m.Stop(id)
			} else {
				select {
				case <-h.audio.maxDone:
				case <-time.After(5 * time.Second):
					t.Fatal("short recording did not finish")
				}
			}
			end := h.rec.wait(t)
			if end.State != Cancelled || end.Problem == nil || end.Problem.Code != ProblemTooShort || len(provider.streams) != 0 {
				t.Fatalf("short capture end=%+v, opened streams=%d", end, len(provider.streams))
			}
			releaseASR()
		})
	}
}

func TestValidCaptureWaitsForASRAndSendsAllCapturedPCM(t *testing.T) {
	provider := &fakeStreamer{final: "valid transcript"}
	h := newHarness(t, provider, nil, func(_ *Config, currentSettings *settings.Settings) {
		currentSettings.Features.Transcribe.Enhance = false
		currentSettings.Audio.VAD.Off = true
		currentSettings.Audio.Clean.Off = true
	})
	h.audio.maxFrame = 4
	h.audio.maxDone = make(chan struct{})
	setIndependentResolvers(h, provider, func(context.Context, Kind, settings.Settings) LLMReadiness {
		return LLMReadiness{Outcome: LLMNotRequested}
	})
	started, releaseASR := blockASRResolver(t, h, provider)
	h.start()
	waitInitializationSignal(t, started, "blocked ASR resolver")
	select {
	case <-h.audio.maxDone:
	case <-time.After(5 * time.Second):
		t.Fatal("valid recording did not finish naturally")
	}
	if snapshot := h.m.Current(); snapshot.State.Ended() {
		t.Fatalf("valid capture ended while ASR was pending: %+v", snapshot)
	}
	releaseASR()
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "valid transcript" {
		t.Fatalf("valid capture end=%+v", end)
	}
	provider.mu.Lock()
	if len(provider.streams) != 1 {
		provider.mu.Unlock()
		t.Fatalf("opened streams=%d, want one", len(provider.streams))
	}
	stream := provider.streams[0]
	provider.mu.Unlock()
	stream.mu.Lock()
	got := append([]byte(nil), stream.data...)
	stream.mu.Unlock()
	want := bytes.Repeat(frame, 4)
	if !bytes.Equal(got, want) {
		t.Fatalf("submitted PCM bytes=%d, captured bytes=%d", len(got), len(want))
	}
}
