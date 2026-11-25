package session

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
)

type tagGateLLM struct {
	started chan struct{}
	release <-chan struct{}
}

func (m *tagGateLLM) Chat(ctx context.Context, request llm.Request) (llm.Response, error) {
	request.OnDelta("<output>")
	close(m.started)
	select {
	case <-m.release:
	case <-ctx.Done():
		return llm.Response{}, ctx.Err()
	}
	request.OnDelta("cleaned output</output>")
	return llm.Response{Text: "<output>cleaned output</output>"}, nil
}

func TestCleanedPublicationTimingNamesAndTagOnlyDeltas(t *testing.T) {
	release, releaseLLM := controlledRelease(t)
	model := &tagGateLLM{started: make(chan struct{}), release: release}
	h := newHarness(t, &fakeFile{text: "spoken words"}, model, nil)
	signals := observeInitializationTimings(h)
	id := h.start()
	h.speak(id, 4)
	select {
	case <-model.started:
	case <-time.After(5 * time.Second):
		t.Fatal("Cleanup did not emit its tag-only prefix")
	}
	if draft := h.m.Current().Draft; draft != "" {
		t.Fatalf("tag-only delta published a draft: %q", draft)
	}
	releaseLLM()
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "cleaned output" || len(h.text.pastes()) != 1 {
		t.Fatalf("end=%+v deliveries=%q", end, h.text.pastes())
	}
	for name, signal := range map[string]<-chan time.Time{
		"cleaned draft": signals.cleanedDraft, "cleaned result": signals.cleanedResult,
		"common cleaned publication": signals.cleanedPublished, "paste": signals.pasteComplete,
	} {
		waitInitializationSignal(t, signal, name)
	}
	if h.logged("timing event=cleanup_first_visible_text") {
		t.Fatal("the imprecise cleanup_first_visible_text metric was emitted")
	}
}

func TestFailedCleanupDoesNotReportCleanedPublication(t *testing.T) {
	text := strings.Repeat("First sentence should be cleaned. Second sentence should also be cleaned. ", 10)
	model := &fakeLLM{
		texts: []string{"<output>First sentence cleaned.</output>", "<output>Second sentence cleaned.</output>"},
		err:   errors.New("second piece failed"), failOn: 2,
	}
	h := newHarness(t, &fakeFile{text: text}, model, nil)
	id := h.start()
	h.speak(id, 4)
	end := h.rec.wait(t)
	if end.State != Done || end.Problem == nil || end.Problem.Code != ProblemLLM || len(h.text.pastes()) != 1 {
		t.Fatalf("failed Cleanup result=%+v deliveries=%q", end, h.text.pastes())
	}
	for _, event := range []string{
		"first_cleaned_draft_published", "first_cleaned_result_published", "first_cleaned_publication",
	} {
		if h.logged("timing event=" + event + " ") {
			t.Errorf("failed Cleanup emitted %s", event)
		}
	}
	if !h.logged("timing event=paste_completed ") {
		t.Fatal("the raw fallback's successful paste was not measured separately")
	}
}

func TestRawEchoIsNotReportedAsCleanedDraft(t *testing.T) {
	const text = "spoken words"
	h := newHarness(t, &fakeFile{text: text}, &fakeLLM{text: text}, nil)
	signals := observeInitializationTimings(h)
	id := h.start()
	h.speak(id, 4)
	end := h.rec.wait(t)
	if end.State != Done || end.Result != text {
		t.Fatalf("end=%+v", end)
	}
	if _, ok := optionalBenchmarkTime(signals.cleanedDraft); ok {
		t.Fatal("raw transcript echo was reported as cleaned draft text")
	}
	waitInitializationSignal(t, signals.cleanedResult, "successful result publication")
}
