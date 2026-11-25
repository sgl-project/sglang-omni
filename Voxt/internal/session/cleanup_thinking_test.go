package session

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestTranscribeFirstCleanupUsesProviderThinkingDefault(t *testing.T) {
	thinking := &llm.Thinking{Mode: llm.ThinkOff}
	model := &fakeLLM{text: "Cleaned text."}
	h := newHarness(t, &fakeFile{text: "spoken words."}, model, func(_ *Config, currentSettings *settings.Settings) {
		currentSettings.Features.Transcribe.Enhance = true
	})
	request := StartRequest{Kind: store.KindTranscribe, Target: slack, Thinking: thinking}
	id, err := h.m.Start(request)
	if err != nil {
		t.Fatal(err)
	}
	h.speak(id, 5)
	end := h.rec.wait(t)
	requests := model.requests()
	if end.State != Done || end.Result != "Cleaned text." || len(requests) != 1 || requests[0].Thinking != nil {
		t.Fatalf("end %+v, cleanup requests %+v; first cleanup must use the provider default", end, requests)
	}
}

func TestTranscribeCompletedPrefixTailUsesProviderThinkingDefault(t *testing.T) {
	prefix := "First sentence. Second sentence. Third sentence."
	finalText := prefix + " Fourth sentence."
	thinking := &llm.Thinking{Mode: llm.ThinkOff}
	clock := newSoftDeadlineClock()
	model := newControlledLiveLLM(false)
	fixture := newLiveSessionFixture(t, prefix, model, clock, false, thinking)
	fixture.streamer.final = finalText
	fixture.call.complete(t, "<output>"+prefix+"</output>", nil)
	waitLive(t, fixture.run, prefix)
	fixture.beginStop(t)
	fixture.releaseFinalAndWait(t)
	tailCall := waitLiveCall(t, model, "the first-cleanup tail request")
	if tailCall.request.Thinking != nil {
		t.Fatalf("tail cleanup Thinking = %+v, want provider default", tailCall.request.Thinking)
	}
	tailCall.complete(t, "<output>Second sentence. Third sentence. Fourth sentence.</output>", nil)
	end := fixture.harness.rec.wait(t)
	if end.State != Done || end.Result != finalText || model.requestCount() != 2 {
		t.Fatalf("end %+v, requests %d", end, model.requestCount())
	}
}
