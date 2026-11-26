package session

import (
	"bytes"
	"context"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func startGatedStream(t *testing.T, provider *fakeStreamer, model LLM, configure func(*Config, *settings.Settings)) (*harness, string, *fakeStream, func()) {
	t.Helper()
	h := newHarness(t, provider, model, configure)
	h.audio.file = bytes.Repeat(frame, 4)
	h.audio.drained = make(chan struct{})
	started := make(chan struct{}, 1)
	release, openRelease := controlledRelease(t)
	signals := observeSessionSignals(h)
	setIndependentResolvers(h, provider, func(context.Context, Kind, settings.Settings) LLMReadiness {
		started <- struct{}{}
		<-release
		return LLMReadiness{LLM: model, LLMInfo: Model{ID: "deepseek", Name: "DeepSeek", Model: "deepseek-chat"}, Outcome: LLMAvailable}
	})
	id := h.start()
	waitInitializationSignal(t, started, "blocked LLM resolver")
	waitInitializationSignal(t, signals.asrOpenDone, "ASR stream to open")
	stream := initializationStream(t, provider)
	waitInitializationSignal(t, h.audio.drained, "fixed PCM recording to finish")
	return h, id, stream, openRelease
}

func initializationStream(t *testing.T, provider *fakeStreamer) *fakeStream {
	t.Helper()
	provider.mu.Lock()
	defer provider.mu.Unlock()
	if len(provider.streams) == 0 {
		t.Fatal("ASR Open completed without registering its fake stream")
	}
	return provider.streams[0]
}

func currentInitializationRun(t *testing.T, h *harness, id string) *run {
	t.Helper()
	h.m.mu.Lock()
	defer h.m.mu.Unlock()
	if h.m.cur == nil || h.m.cur.id != id {
		t.Fatalf("session %q is no longer current", id)
	}
	return h.m.cur
}

func waitEarlyCompatible(t *testing.T, r *run, text string) {
	t.Helper()
	r.earlyMu.Lock()
	early := r.early
	r.earlyMu.Unlock()
	if early == nil {
		t.Fatal("buffered EventPiece did not create early cleanup")
	}
	completed := make(chan bool, 1)
	go func() {
		early.mu.Lock()
		defer early.mu.Unlock()
		ready := func() bool {
			return !early.broken && early.failed == nil && early.covered == text && len(early.segs) > 0 && len(early.segs) == len(early.jobs)
		}
		for !ready() && early.ctx.Err() == nil {
			early.cond.Wait()
		}
		completed <- ready()
	}()
	select {
	case ok := <-completed:
		if !ok {
			t.Fatal("buffered EventPiece cleanup was cancelled before completion")
		}
	case <-time.After(5 * time.Second):
		t.Fatal("buffered EventPiece cleanup did not complete while recording")
	}
}

func TestPendingLivePartialReplaysLatestAfterCleanupReadiness(t *testing.T) {
	const final = "Hello there."
	provider := &fakeStreamer{final: final, silent: true}
	model := newLiveLLM()
	h, id, stream, release := startGatedStream(t, provider, model, func(_ *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
		s.Dictionary.ASRHints = false
	})
	stream.emit(asr.Event{Kind: asr.EventPartial, Text: "Hello"})
	stream.emit(asr.Event{Kind: asr.EventPartial, Text: final})
	h.rec.waitFor(t, "latest ASR partial before readiness", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.Partial == final
	})
	if got := len(model.requests()); got != 0 {
		t.Fatalf("Live Polish started before cleanup readiness: %d requests", got)
	}

	release()
	model.waitAnswered(t, final)
	if got := len(model.requests()); got != 1 {
		t.Fatalf("Live Polish requests after readiness = %d, want one", got)
	}
	h.m.Stop(id)
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "["+final+"]" || end.Raw != final {
		t.Fatalf("end %+v, want the completed Live Polish result reused", end)
	}
	if got := len(model.requests()); got != 1 {
		t.Fatalf("Stop started duplicate cleanup work: %d total requests", got)
	}
}

func TestStopBeforeCleanupReadinessDoesNotReplayPendingLivePartial(t *testing.T) {
	const (
		partial = "Hello there."
		final   = "Hello there. How are you?"
	)
	provider := &fakeStreamer{final: final, silent: true}
	model := newLiveLLM()
	h, id, stream, release := startGatedStream(t, provider, model, func(_ *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
		s.Dictionary.ASRHints = false
	})
	stream.emit(asr.Event{Kind: asr.EventPartial, Text: partial})
	h.rec.waitFor(t, "ASR partial before Stop", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.Partial == partial
	})
	h.m.Stop(id)
	h.rec.waitFor(t, "ASR Final while LLM initialization remains blocked", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.Raw == final && !snapshot.StoppedAt.IsZero()
	})
	if got := len(model.requests()); got != 0 {
		t.Fatalf("cleanup ran before readiness: %d requests", got)
	}

	release()
	model.waitAnswered(t, final)
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "["+final+"]" || end.Raw != final {
		t.Fatalf("end %+v", end)
	}
	reqs := model.requests()
	if len(reqs) != 1 || inputOf(reqs[0]) != final {
		t.Fatalf("requests after Stop %+v, want one whole-transcript cleanup and no replay of %q", reqs, partial)
	}
}

func TestCompatibleEventPiecesReplayAfterReadinessAndSupersedeLivePolish(t *testing.T) {
	const (
		first  = "We need aura link."
		second = "We need aura link. Bring milk."
		final  = "We need aura link. Bring milk. Later."
	)
	provider := &fakeStreamer{final: final, silent: true}
	model := newLiveLLM()
	h, id, stream, release := startGatedStream(t, provider, model, func(_ *Config, s *settings.Settings) {
		s.Features.Transcribe.Enhance = true
		s.Dictionary.ASRHints = false
	})
	stream.emit(asr.Event{Kind: asr.EventPartial, Text: second})
	stream.emit(asr.Event{Kind: asr.EventPiece, Text: first})
	stream.emit(asr.Event{Kind: asr.EventPiece, Text: second})
	stream.emit(asr.Event{Kind: asr.EventPartial, Text: "piece handoff consumed"})
	h.rec.waitFor(t, "ASR reader to consume the pieces before readiness", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.Partial == "piece handoff consumed"
	})
	if got := len(model.requests()); got != 0 {
		t.Fatalf("cleanup started before readiness: %d requests", got)
	}

	r := currentInitializationRun(t, h, id)
	release()
	model.waitAnswered(t, "We need VoxtNew.")
	model.waitAnswered(t, "Bring milk.")
	wantPrefix := "We need VoxtNew. Bring milk."
	waitEarlyCompatible(t, r, wantPrefix)
	if got := len(model.requests()); got != 2 {
		t.Fatalf("pre-Stop requests = %d, want two early-clean pieces and no Live Polish", got)
	}
	for _, req := range model.requests() {
		if input := inputOf(req); strings.Contains(input, "Bring milk") && strings.Contains(input, "We need") {
			t.Fatalf("EventPiece handoff revived Live Polish for cumulative text %q", input)
		}
	}

	h.m.Stop(id)
	end := h.rec.wait(t)
	if end.State != Done || end.Raw != final || end.Result != "[We need VoxtNew.] [Bring milk.] [Later.]" {
		t.Fatalf("end %+v", end)
	}
	if got := len(model.requests()); got != 3 {
		t.Fatalf("total cleanup requests = %d, want two reused early pieces and one tail", got)
	}
	if got := h.text.pastes(); len(got) != 1 || got[0] != end.Result {
		t.Fatalf("deliveries %q, want exactly one final result", got)
	}
}
