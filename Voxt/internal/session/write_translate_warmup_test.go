package session

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type blockingPrefillLLM struct {
	*fakeLLM
	started  chan llm.Request
	release  <-chan struct{}
	returned chan struct{}
	calls    atomic.Int32
}

func (p *blockingPrefillLLM) Prefill(_ context.Context, request llm.Request) error {
	p.calls.Add(1)
	p.started <- request
	<-p.release
	p.returned <- struct{}{}
	return nil
}

func (p *blockingPrefillLLM) CanPrefill() bool { return true }

func TestTranslateAutoTargetWarmupDoesNotBlockRecognition(t *testing.T) {
	release, releasePrefill := controlledRelease(t)
	openRelease, releaseOpen := controlledRelease(t)
	model := &blockingPrefillLLM{
		fakeLLM: &fakeLLM{text: "早上好。"}, started: make(chan llm.Request, 1),
		release: release, returned: make(chan struct{}, 1),
	}
	provider := &fakeStreamer{final: "おはようございます", openGate: openRelease}
	h := newHarness(t, provider, model, func(_ *Config, s *settings.Settings) {
		s.General.MainLanguage, s.Features.Translate.Target = "zh", "ja"
	})
	drained := readinessRecording(h.audio, 4)
	signals := observeSessionSignals(h)
	id, err := h.m.Start(StartRequest{Kind: store.KindTranslate, Target: slack})
	if err != nil {
		t.Fatal(err)
	}
	waitInitializationSignal(t, signals.asrOpenStarted, "Translate ASR Open to start")
	var warmup llm.Request
	select {
	case warmup = <-model.started:
	case <-time.After(5 * time.Second):
		t.Fatal("Translate warmup did not start")
	}
	select {
	case <-signals.asrOpenDone:
		t.Fatal("Translate ASR Open returned before the Prefill request was prepared")
	default:
	}
	releaseOpen()
	waitInitializationSignal(t, signals.asrOpenDone, "Translate ASR Open to complete")
	waitInitializationSignal(t, signals.firstPCM, "PCM send while Prefill is blocked")
	waitInitializationSignal(t, signals.firstPartial, "ASR partial while Prefill is blocked")
	waitInitializationSignal(t, drained, "fixed recording to drain")
	h.m.Stop(id)
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "早上好。" || end.TargetLanguage != "zh-Hans" || len(h.text.pastes()) != 1 {
		t.Fatalf("Translate result=%+v deliveries=%q", end, h.text.pastes())
	}
	if warmup.Skill != llm.SkillTranslateText || len(warmup.Messages) == 0 || !strings.Contains(warmup.Messages[0].Content, "翻译成日语") {
		t.Fatalf("warmup request was not prepared from the initial automatic target: %+v", warmup)
	}
	if model.calls.Load() != 1 {
		t.Fatalf("Translate warmup calls = %d, want one", model.calls.Load())
	}
	chats := model.requests()
	if len(chats) != 1 || !strings.Contains(chats[0].Messages[0].Content, "简体中文") {
		t.Fatalf("Translate did not select the source-dependent target: %+v", chats)
	}
	select {
	case <-model.returned:
		t.Fatal("blocked Prefill delayed its gate until after session delivery")
	default:
	}
	releasePrefill()
	waitInitializationSignal(t, model.returned, "blocked Prefill to return")
	if model.calls.Load() != 1 {
		t.Fatalf("Translate Prefill calls = %d, want one", model.calls.Load())
	}
}
