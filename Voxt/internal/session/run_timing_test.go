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

type timingControlLLM struct {
	started  chan llm.Request
	finish   <-chan struct{}
	response string
	err      error
}

func (model *timingControlLLM) Chat(ctx context.Context, request llm.Request) (llm.Response, error) {
	model.started <- request
	select {
	case <-model.finish:
		return llm.Response{Text: model.response}, model.err
	case <-ctx.Done():
		return llm.Response{}, ctx.Err()
	}
}

func receiveCleanupTiming(t *testing.T, signal <-chan time.Time, what string) time.Time {
	t.Helper()
	select {
	case at := <-signal:
		return at
	case <-time.After(5 * time.Second):
		t.Fatalf("timed out waiting for %s", what)
		return time.Time{}
	}
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

func TestFailedAttemptDeltaDoesNotReportCleanedPublication(t *testing.T) {
	clock := newSoftDeadlineClock()
	finish := make(chan struct{})
	model := &timingControlLLM{
		started: make(chan llm.Request, 1), finish: finish,
		response: "failed model output", err: errors.New("provider failed"),
	}
	h := newHarness(t, &fakeFile{text: "spoken words"}, model, cleanupLatencyConfig(clock, false))
	signals := observeInitializationTimings(h)
	id := h.start()
	h.speak(id, 4)
	request := <-model.started
	request.OnDelta("failed model output")
	if snapshot := h.m.Current(); snapshot.Draft != "failed model output" {
		t.Fatalf("failed provider draft did not retain the current UI behavior: %q", snapshot.Draft)
	}
	close(finish)
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "spoken words" || end.Problem == nil || end.Problem.Code != ProblemLLM || len(h.text.pastes()) != 1 {
		t.Fatalf("failed attempt end=%+v deliveries=%q", end, h.text.pastes())
	}
	for _, event := range []string{"first_cleaned_draft_published", "first_cleaned_result_published", "first_cleaned_publication"} {
		if h.logged("timing event=" + event + " ") {
			t.Errorf("failed attempt emitted %s", event)
		}
	}
	if _, ok := optionalBenchmarkTime(signals.cleanedDraft); ok || !h.logged("timing event=paste_completed ") {
		t.Fatal("failed attempt publication was measured as cleaned output or its paste was missed")
	}
}

func TestFallbackCleanupTimingUsesWinningProviderAndFencesLatePrimaryDraft(t *testing.T) {
	clock := newSoftDeadlineClock()
	primaryFinish, fallbackFinish := make(chan struct{}), make(chan struct{})
	primary := &timingControlLLM{
		started: make(chan llm.Request, 1), finish: primaryFinish,
		response: "primary answer", err: errors.New("primary failed after streaming"),
	}
	fallback := &timingControlLLM{started: make(chan llm.Request, 1), finish: fallbackFinish, response: "Fallback result."}
	h := newHarness(t, &fakeFile{text: "spoken words"}, primary, cleanupLatencyConfig(clock, false))
	h.fallback = fallback
	signals := observeInitializationTimings(h)
	id := h.start()
	h.speak(id, 4)
	primaryRequest := <-primary.started
	clock.Advance(200 * time.Millisecond)
	primaryRequest.OnDelta("Primary draft.")
	if snapshot := h.m.Current(); snapshot.Draft != "Primary draft." {
		t.Fatalf("primary draft was not published: %q", snapshot.Draft)
	}
	close(primaryFinish)
	fallbackRequest := <-fallback.started
	if snapshot := h.m.Current(); snapshot.Draft != "" {
		t.Fatalf("fallback did not clear the failed attempt's draft: %q", snapshot.Draft)
	}
	clock.Advance(300 * time.Millisecond)
	primaryRequest.OnDelta(" stale primary callback")
	if snapshot := h.m.Current(); snapshot.Draft != "" {
		t.Fatalf("late primary callback modified the snapshot: %q", snapshot.Draft)
	}
	clock.Advance(300 * time.Millisecond)
	fallbackRequest.OnDelta("Fallback cleaned text.")
	fallbackDraftAt := clock.Now()
	if snapshot := h.m.Current(); snapshot.Draft != "Fallback cleaned text." {
		t.Fatalf("fallback draft was not published: %q", snapshot.Draft)
	}
	clock.Advance(200 * time.Millisecond)
	close(fallbackFinish)
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "Fallback result." || len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
		t.Fatalf("end=%+v deliveries=%q", end, h.text.pastes())
	}
	if got := receiveCleanupTiming(t, signals.cleanedDraft, "winning fallback draft timing"); !got.Equal(fallbackDraftAt) {
		t.Fatalf("cleaned draft timestamp %v, want winning fallback publication %v", got, fallbackDraftAt)
	}
	if !h.logged("timing event=first_cleaned_result_published ") || !h.logged("timing event=first_cleaned_publication ") {
		t.Fatal("successful fallback result publication timings were not recorded")
	}
	if snapshot := h.m.Current(); snapshot.Result != "Fallback result." || snapshot.Draft != "Fallback result." {
		t.Fatalf("late primary callback changed the final published output: %+v", snapshot)
	}
}

func TestCleanedDraftTimingAcceptsUnchangedASRPrefix(t *testing.T) {
	const raw = "Please review this pull request"
	clock := newSoftDeadlineClock()
	finish := make(chan struct{})
	model := &timingControlLLM{started: make(chan llm.Request, 1), finish: finish, response: raw + "."}
	h := newHarness(t, &fakeFile{text: raw}, model, cleanupLatencyConfig(clock, false))
	signals := observeInitializationTimings(h)
	id := h.start()
	h.speak(id, 4)
	request := <-model.started
	clock.Advance(100 * time.Millisecond)
	request.OnDelta(raw)
	prefixPublishedAt := clock.Now()
	if snapshot := h.m.Current(); snapshot.Draft != raw {
		t.Fatalf("unchanged ASR prefix was not accepted as cleaned text: %q", snapshot.Draft)
	}
	clock.Advance(250 * time.Millisecond)
	request.OnDelta(".")
	close(finish)
	end := h.rec.wait(t)
	if end.State != Done || end.Result != raw+"." || len(h.text.pastes()) != 1 {
		t.Fatalf("end=%+v deliveries=%q", end, h.text.pastes())
	}
	if got := receiveCleanupTiming(t, signals.cleanedDraft, "first generated prefix timing"); !got.Equal(prefixPublishedAt) {
		t.Fatalf("cleaned draft timestamp %v, want initial matching prefix %v", got, prefixPublishedAt)
	}
	if got := receiveCleanupTiming(t, signals.cleanedPublished, "first cleaned publication timing"); !got.Equal(prefixPublishedAt) {
		t.Fatalf("first cleaned publication %v, want initial matching prefix %v", got, prefixPublishedAt)
	}
	if got := receiveCleanupTiming(t, signals.cleanedResult, "cleaned result timing"); !got.Equal(clock.Now()) {
		t.Fatalf("cleaned result timestamp %v, want accepted final result time %v", got, clock.Now())
	}
}

func TestSuccessfulUnchangedLLMOutputReportsCleanedPublication(t *testing.T) {
	const raw = "spoken words"
	clock := newSoftDeadlineClock()
	finish := make(chan struct{})
	model := &timingControlLLM{started: make(chan llm.Request, 1), finish: finish, response: raw}
	h := newHarness(t, &fakeFile{text: raw}, model, cleanupLatencyConfig(clock, false))
	signals := observeInitializationTimings(h)
	id := h.start()
	h.speak(id, 4)
	request := <-model.started
	clock.Advance(100 * time.Millisecond)
	request.OnDelta(raw)
	draftPublishedAt := clock.Now()
	clock.Advance(400 * time.Millisecond)
	close(finish)
	end := h.rec.wait(t)
	if end.State != Done || end.Result != raw || len(h.text.pastes()) != 1 || h.text.pastes()[0] != raw {
		t.Fatalf("successful unchanged output end=%+v deliveries=%q", end, h.text.pastes())
	}
	if got := receiveCleanupTiming(t, signals.cleanedDraft, "unchanged model output draft timing"); !got.Equal(draftPublishedAt) {
		t.Fatalf("cleaned draft timestamp %v, want %v", got, draftPublishedAt)
	}
	if got := receiveCleanupTiming(t, signals.cleanedResult, "unchanged model output result timing"); !got.Equal(clock.Now()) {
		t.Fatalf("cleaned result timestamp %v, want %v", got, clock.Now())
	}
	if _, ok := optionalBenchmarkTime(signals.cleanedPublished); !ok {
		t.Fatal("successful unchanged model output was not reported as cleaned publication")
	}
}

func TestManualSkipDoesNotReportCleanedPublication(t *testing.T) {
	model := &fakeLLM{hang: true, hangDelta: "partial model output"}
	h := newHarness(t, &fakeFile{text: "spoken words"}, model, nil)
	signals := observeInitializationTimings(h)
	id := h.start()
	h.speak(id, 4)
	h.rec.waitFor(t, "streamed draft before Skip", func(snapshot Snapshot) bool {
		return snapshot.ID == id && snapshot.State == Composing && snapshot.Draft != ""
	})
	h.m.SkipLLM(id)
	end := h.rec.wait(t)
	if end.State != Done || end.Result != "spoken words" || len(h.text.pastes()) != 1 {
		t.Fatalf("manual Skip end=%+v deliveries=%q", end, h.text.pastes())
	}
	if _, ok := optionalBenchmarkTime(signals.cleanedDraft); ok || h.logged("timing event=first_cleaned_result_published ") || h.logged("timing event=first_cleaned_publication ") {
		t.Fatal("manual Skip emitted successful-cleanup publication timings")
	}
	if !h.logged("timing event=paste_completed ") {
		t.Fatal("manual Skip delivery was not measured independently")
	}
}

func TestSoftDeadlineRawFallbackDoesNotReportCleanedPublication(t *testing.T) {
	clock := newSoftDeadlineClock()
	model := newCleanupGateLLM("partial model output", "late output", "cleaned result")
	h := newHarness(t, &fakeFile{text: "spoken words"}, model, cleanupLatencyConfig(clock, true))
	signals := observeInitializationTimings(h)
	id := h.start()
	h.speak(id, 4)
	waitCleanupRequest(t, model, "cleanup request before soft deadline")
	clock.Advance(cleanupSoftDeadlineTestDuration)
	clock.Fire(t, 0)
	end := h.rec.waitFor(t, "soft-deadline Raw result", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State.Ended() })
	if end.Result != "spoken words" || len(h.text.pastes()) != 1 || h.text.pastes()[0] != end.Result {
		t.Fatalf("soft-deadline result=%+v deliveries=%q", end, h.text.pastes())
	}
	model.Release()
	waitCleanupSignal(t, model.returned, "late cleanup response after Raw selection")
	if _, ok := optionalBenchmarkTime(signals.cleanedDraft); ok || h.logged("timing event=first_cleaned_result_published ") || h.logged("timing event=first_cleaned_publication ") {
		t.Fatal("soft-deadline Raw fallback emitted successful-cleanup publication timings")
	}
	if !h.logged("timing event=paste_completed ") {
		t.Fatal("soft-deadline Raw delivery was not measured independently")
	}
}
