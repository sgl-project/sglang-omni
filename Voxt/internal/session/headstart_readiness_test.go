package session

import (
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestLateRemoteLLMReadinessUsesCurrentPauseAndReusesRequest(t *testing.T) {
	provider := &fakeStreamer{partials: []string{"we ship the new build on friday"}, final: "we ship the new build on friday"}
	provider.supportsEarlyPieces = true
	model := &fakeLLM{text: "We ship the new build on Friday."}
	h := newHarness(t, provider, model, func(_ *Config, s *settings.Settings) { s.Features.Transcribe.Enhance = true })
	h.audio.speaks = func(n int) bool { return n < 8 }
	signals := observeSessionSignals(h)
	started, releaseLLM := setBlockedRemoteResolvers(h, provider, model)
	id := h.start()
	waitInitializationSignal(t, started, "remote LLM resolver")
	waitInitializationSignal(t, signals.asrOpenDone, "ASR Open while LLM resolution is blocked")
	waitInitializationSignal(t, signals.firstPCM, "PCM sent while LLM resolution is blocked")
	waitInitializationSignal(t, signals.firstPartial, "partial delivered while LLM resolution is blocked")
	r := sessionRun(t, h)
	waitPauseState(t, r, true)
	provider.mu.Lock()
	requestedEarlyPieces := provider.earlyPieces
	stream := provider.streams[0]
	provider.mu.Unlock()
	if requestedEarlyPieces {
		t.Fatal("ASR Open optimistically requested EarlyPieces before LLM readiness")
	}
	if len(model.requests()) != 0 {
		t.Fatal("speculative Cleanup started before remote LLM readiness")
	}
	releaseLLM()
	waitInitializationSignal(t, signals.llmRequest, "Head Start request after readiness while already paused")
	stream.mu.Lock()
	lateEarlyPieces := stream.earlyPieces
	stream.mu.Unlock()
	if !lateEarlyPieces {
		t.Fatal("eligible late cleanup did not enable EarlyPieces on the existing stream")
	}
	h.speak(id, 4)
	end := h.rec.wait(t)
	if end.State != Done || len(model.requests()) != 1 || len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
		t.Fatalf("end=%+v, requests=%d, deliveries=%q", end, len(model.requests()), h.text.pastes())
	}
}

func TestHeadStartRequestsEarlyPiecesAtOpenWhenEligibilityIsKnown(t *testing.T) {
	provider := &fakeStreamer{partials: []string{"we ship"}, final: "we ship"}
	model := &fakeLLM{text: "We ship."}
	h := newHarness(t, provider, model, func(_ *Config, s *settings.Settings) { s.Features.Transcribe.Enhance = true })
	h.remote = true
	signals := observeSessionSignals(h)
	id := h.start()
	waitInitializationSignal(t, signals.asrOpenDone, "ASR Open")
	provider.mu.Lock()
	requested := provider.earlyPieces
	provider.mu.Unlock()
	if !requested {
		t.Fatal("eligible remote cleanup did not request EarlyPieces at Open")
	}
	h.speak(id, 4)
	end := h.rec.wait(t)
	if end.State != Done || len(h.text.pastes()) != 1 {
		t.Fatalf("end=%+v deliveries=%q", end, h.text.pastes())
	}
}

func TestLateRemoteLLMReadinessDoesNotReplayPauseAfterSpeechResumes(t *testing.T) {
	provider := &fakeStreamer{partials: []string{"we ship the new build on friday"}, final: "we ship the new build on friday"}
	model := &fakeLLM{text: "We ship the new build on Friday."}
	h := newHarness(t, provider, model, func(_ *Config, s *settings.Settings) { s.Features.Transcribe.Enhance = true })
	h.audio.tick = 2 * time.Millisecond
	h.audio.speaks = func(n int) bool { return n < 8 || n >= 45 }
	signals := observeSessionSignals(h)
	started, releaseLLM := setBlockedRemoteResolvers(h, provider, model)
	id := h.start()
	waitInitializationSignal(t, started, "remote LLM resolver")
	waitInitializationSignal(t, signals.firstPartial, "partial delivered while LLM resolution is blocked")
	r := sessionRun(t, h)
	waitPauseState(t, r, true)
	waitPauseState(t, r, false)
	if len(model.requests()) != 0 {
		t.Fatal("speculative Cleanup started before remote LLM readiness")
	}
	releaseLLM()
	if got := h.rec.waitFor(t, "remote LLM readiness", func(s Snapshot) bool { return s.LLM.ID == "fake-llm" }); got.LLM.ID != "fake-llm" {
		t.Fatalf("LLM readiness snapshot %+v", got)
	}
	waitHeadEnabled(t, r)
	if len(model.requests()) != 0 {
		t.Fatal("Head Start replayed a pause after the user resumed speaking")
	}
	h.m.Stop(id)
	end := h.rec.wait(t)
	if end.State != Done || len(model.requests()) != 1 || len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
		t.Fatalf("end=%+v, requests=%d, deliveries=%q", end, len(model.requests()), h.text.pastes())
	}
}

func TestHeadStartEligibilityIncludesRemoteCleanupAndImageRules(t *testing.T) {
	for _, tc := range []struct {
		name string
		want bool
		set  func(*run)
	}{
		{name: "remote cleanup", want: true},
		{name: "local model", set: func(r *run) { r.svc.LLMInfo.Remote = false }},
		{name: "cleanup off", set: func(r *run) { r.settings.Features.Transcribe.Enhance = false }},
		{name: "clean after stop", set: func(r *run) { r.settings.Features.Transcribe.CleanAfterStop = true }},
		{name: "window image", set: func(r *run) { r.shot = func() []byte { return []byte("jpeg") } }},
		{name: "request image", set: func(r *run) { r.req.Images = [][]byte{{1}} }},
		{name: "rate limited", set: func(r *run) { r.m.noteRate("deepseek", llmRateLimited(), nil) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			model := &fakeLLM{text: "cleaned"}
			r := newEarlyUnitRun(t, model, nil)
			r.svc.LLMInfo.Remote = true
			if tc.set != nil {
				tc.set(r)
			}
			r.startHead()
			if got := r.head.enabled.Load(); got != tc.want {
				t.Fatalf("Head Start enabled=%v, want %v", got, tc.want)
			}
		})
	}
}

func llmRateLimited() llm.Response { return llm.Response{RateLimited: true} }
