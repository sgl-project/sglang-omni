package session

import (
	"context"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type pairedBenchmarkCall struct {
	request  llm.Request
	result   chan string
	returned chan struct{}
}

func (call *pairedBenchmarkCall) emitDelta(text string) {
	if call.request.OnDelta != nil {
		call.request.OnDelta(text)
	}
}

func (call *pairedBenchmarkCall) complete(t *testing.T, answer string) {
	t.Helper()
	call.result <- answer
	select {
	case <-call.returned:
	case <-time.After(5 * time.Second):
		t.Fatal("the controlled cleanup request did not return")
	}
}

type pairedBenchmarkLLM struct {
	calls    chan *pairedBenchmarkCall
	requests atomic.Int32
}

func (model *pairedBenchmarkLLM) Chat(requestContext context.Context, request llm.Request) (llm.Response, error) {
	call := &pairedBenchmarkCall{request: request, result: make(chan string, 1), returned: make(chan struct{})}
	model.requests.Add(1)
	model.calls <- call
	defer close(call.returned)
	select {
	case answer := <-call.result:
		return llm.Response{Text: answer}, nil
	case <-requestContext.Done():
		return llm.Response{}, requestContext.Err()
	}
}

type pairedBenchmarkASR struct {
	finishGate    <-chan struct{}
	finishStarted chan struct{}
	finishEmitted chan struct{}
	partialText   string
	finalText     string
}

func (*pairedBenchmarkASR) Caps() asr.Caps { return asr.Caps{Stream: true} }

func (provider *pairedBenchmarkASR) Open(requestContext context.Context, request asr.StreamRequest) (asr.Stream, error) {
	stream, err := (&fakeStreamer{final: provider.finalText, partials: []string{provider.partialText}}).Open(requestContext, request)
	if err != nil {
		return nil, err
	}
	return &pairedBenchmarkStream{
		Stream: stream, requestContext: requestContext, finalGate: provider.finishGate,
		finishStarted: provider.finishStarted, finishEmitted: provider.finishEmitted,
	}, nil
}

type pairedBenchmarkStream struct {
	asr.Stream
	requestContext context.Context
	finalGate      <-chan struct{}
	finishStarted  chan struct{}
	finishEmitted  chan struct{}
}

func (stream *pairedBenchmarkStream) Finish() error {
	select {
	case stream.finishStarted <- struct{}{}:
	default:
	}
	select {
	case <-stream.finalGate:
		if err := stream.Stream.Finish(); err != nil {
			return err
		}
		stream.finishEmitted <- struct{}{}
		return nil
	case <-stream.requestContext.Done():
		return stream.requestContext.Err()
	}
}

type pairedBenchmarkText struct {
	*fakeText
	clock   *softDeadlineClock
	pasteAt chan time.Time
}

func (textIO *pairedBenchmarkText) Paste(requestContext context.Context, request proto.PasteParams) (proto.PasteResult, error) {
	result, err := textIO.fakeText.Paste(requestContext, request)
	textIO.pasteAt <- textIO.clock.Now()
	return result, err
}

type pairedBenchmarkObservation struct {
	firstDraft        chan time.Time
	firstPresentation chan time.Time
	draftUpdates      chan time.Time
	pasteAt           chan time.Time
}

func newPairedBenchmarkHarness(t *testing.T, streamer asr.Provider, model *pairedBenchmarkLLM, clock *softDeadlineClock, liveCleanupEnabled bool) (*harness, *pairedBenchmarkObservation) {
	t.Helper()
	observations := &pairedBenchmarkObservation{
		firstDraft: make(chan time.Time, 1), firstPresentation: make(chan time.Time, 1), draftUpdates: make(chan time.Time, 16), pasteAt: make(chan time.Time, 4),
	}
	var firstDraftOnce sync.Once
	var firstPresentationOnce sync.Once
	fixture := &harness{
		t:     t,
		audio: newFakeAudio(),
		text:  &fakeText{focus: proto.FocusResult{PID: 812, Role: "AXTextArea", Editable: true}},
		hist:  &memHistory{dir: t.TempDir()},
		rec:   newRecorder(),
		asr:   streamer,
		llm:   model,
	}
	state := settings.Defaults()
	state.Features.Transcribe.Enhance = true
	state.Features.Transcribe.CleanAfterStop = !liveCleanupEnabled
	state.Dictionary.Enabled = true
	config := Config{
		Audio: fixture.audio,
		Text:  &pairedBenchmarkText{fakeText: fixture.text, clock: clock, pasteAt: observations.pasteAt},
		Resolve: func(_ context.Context, _ Kind, _ settings.Settings, speech bool) (Services, error) {
			fixture.mu.Lock()
			defer fixture.mu.Unlock()
			fixture.speech = append(fixture.speech, speech)
			var services Services
			if speech {
				services = Services{ASR: fixture.asr, ASRInfo: Model{ID: "doubao", Name: "Doubao", Model: "seed"}, VAD: fixture.vadModel}
			}
			services.LLM, services.LLMInfo = fixture.llm, Model{ID: "deepseek", Name: "DeepSeek", Model: "deepseek-chat"}
			return services, nil
		},
		Glossary: func(context.Context) ([]store.Term, error) {
			return []store.Term{{Term: "VoxtNew", Aliases: []string{"aura link"}, Enabled: true}, {Term: "Off", Enabled: false}}, nil
		},
		Settings: func() settings.Settings { return state },
		OnChange: func(snapshot Snapshot) {
			fixture.rec.add(snapshot)
			if snapshot.State == Composing && pairedBenchmarkEligibleText(snapshot.Draft) {
				firstDraftOnce.Do(func() { observations.firstDraft <- clock.Now() })
				firstPresentationOnce.Do(func() { observations.firstPresentation <- clock.Now() })
				select {
				case observations.draftUpdates <- clock.Now():
				default:
				}
			} else if snapshot.State.Ended() && snapshot.Result != snapshot.Raw && pairedBenchmarkEligibleText(snapshot.Result) {
				firstPresentationOnce.Do(func() { observations.firstPresentation <- clock.Now() })
			}
		},
		Log:       func(string, string) {},
		Now:       clock.Now,
		AfterFunc: clock.AfterFunc,
		LLMBase:   time.Second, LLMStall: 500 * time.Millisecond, LLMPerChar: time.Millisecond, LLMTimeout: 2 * time.Second,
		ASRTimeout: 2 * time.Second, StopGrace: 200 * time.Millisecond,
	}
	state.Audio.VAD.Off = true
	fixture.m = New(config)
	t.Cleanup(fixture.m.Close)
	return fixture, observations
}

func pairedBenchmarkEligibleText(text string) bool {
	text = strings.TrimSpace(text)
	return text != "" && !strings.Contains(text, "<output") && !strings.Contains(text, "</") && !strings.Contains(text, "<think")
}

func TestLivePolishPairedBufferedStreamLatency(t *testing.T) {
	runPairedBenchmarkScenario(t, "matching_buffered")
}

func TestLivePolishPairedFirstDeltaLatency(t *testing.T) {
	runPairedBenchmarkScenario(t, "matching_first_delta")
}

func TestLivePolishPairedCompletedPrefixLatency(t *testing.T) {
	runPairedBenchmarkScenario(t, "completed_prefix_slow_tail")
}

func TestLivePolishPairedLiveCleanupDisabled(t *testing.T) {
	runPairedBenchmarkScenario(t, "live_cleanup_disabled")
}

func runPairedBenchmarkScenario(t *testing.T, scenario string) {
	t.Helper()
	partialText, finalText := "spoken words.", "spoken words."
	speculativeAnswer, foregroundAnswer := "<output>Cleaned text.</output>", "<output>Cleaned final.</output>"
	firstDelta := "<output>Cleaned"
	liveCleanupEnabled, deltaBeforeFinal, completeBeforeFinal := true, false, false
	expectedRequests, expectedResult := int32(1), "Cleaned text."
	switch scenario {
	case "matching_buffered":
		deltaBeforeFinal = true
	case "matching_first_delta":
		firstDelta = "<output>Cleaned"
	case "completed_prefix_slow_tail":
		partialText = "First sentence. Second sentence. Third sentence."
		finalText = partialText + " Fourth sentence."
		speculativeAnswer = "<output>First sentence. Second sentence. Third sentence.</output>"
		foregroundAnswer = "<output>Second sentence. Third sentence. Fourth sentence.</output>"
		expectedRequests, expectedResult = 2, finalText
		completeBeforeFinal = true
	case "live_cleanup_disabled":
		liveCleanupEnabled = false
		expectedRequests, expectedResult = 1, "Cleaned final."
	default:
		t.Fatalf("unknown paired benchmark scenario %q", scenario)
	}
	clock := newSoftDeadlineClock()
	model := &pairedBenchmarkLLM{calls: make(chan *pairedBenchmarkCall, 4)}
	finalGate := make(chan struct{})
	streamer := &pairedBenchmarkASR{
		finishGate: finalGate, finishStarted: make(chan struct{}, 1), finishEmitted: make(chan struct{}, 1),
		partialText: partialText, finalText: finalText,
	}
	fixture, observations := newPairedBenchmarkHarness(t, streamer, model, clock, liveCleanupEnabled)
	fixture.audio.notifyAt = 3
	fixture.audio.notified = make(chan struct{})
	id := fixture.start()
	fixture.rec.waitFor(t, "recording", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State == Recording })
	fixture.m.mu.Lock()
	currentRun := fixture.m.cur
	fixture.m.mu.Unlock()
	select {
	case <-fixture.audio.notified:
	case <-time.After(5 * time.Second):
		t.Fatal("recording did not reach the Stop fixture")
	}
	var speculativeCall *pairedBenchmarkCall
	if liveCleanupEnabled {
		speculativeCall = waitPairedBenchmarkCall(t, model, "the speculative cleanup request did not start")
	}
	stopAt := clock.Now()
	fixture.m.Stop(id)
	select {
	case <-streamer.finishStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("ASR did not start finalization")
	}
	if completeBeforeFinal {
		clock.Advance(100 * time.Millisecond)
		speculativeCall.complete(t, speculativeAnswer)
		waitLive(t, currentRun, partialText)
		clock.Advance(200 * time.Millisecond)
	} else if deltaBeforeFinal {
		clock.Advance(100 * time.Millisecond)
		speculativeCall.emitDelta(firstDelta)
		if fixture.m.Current().Draft != "" {
			t.Fatalf("speculative text escaped before ASR Final in %s: %q", scenario, fixture.m.Current().Draft)
		}
		clock.Advance(200 * time.Millisecond)
	} else {
		clock.Advance(300 * time.Millisecond)
	}
	close(finalGate)
	select {
	case <-streamer.finishEmitted:
	case <-time.After(5 * time.Second):
		t.Fatal("ASR Final did not arrive")
	}
	asrFinalAt := clock.Now()
	fixture.rec.waitFor(t, "the composing state", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State == Composing })
	var foregroundCall *pairedBenchmarkCall
	switch scenario {
	case "completed_prefix_slow_tail", "live_cleanup_disabled":
		foregroundCall = waitPairedBenchmarkCall(t, model, "the foreground cleanup request did not start")
	case "matching_first_delta":
		foregroundCall = speculativeCall
	}
	prefixPublishedBeforeTailDelta := false
	if scenario == "completed_prefix_slow_tail" {
		waitForPairedDraftOrDeadline(observations, 150*time.Millisecond)
		draft := fixture.m.Current().Draft
		if strings.Contains(draft, "Second sentence.") || strings.Contains(draft, "Third sentence.") {
			t.Fatalf("sentences selected for reclean were published before tail progress: %q", draft)
		}
		drainPairedBenchmarkDraftUpdates(observations)
		clock.Advance(100 * time.Millisecond)
		foregroundCall.emitDelta("<output>Second sentence.")
		waitForPairedDraft(t, observations, "tail progress did not publish a cleaned draft")
		prefixPublishedBeforeTailDelta = pairedBenchmarkEligibleText(draft)
	} else if scenario == "matching_buffered" {
		waitForPairedDraftOrDeadline(observations, 150*time.Millisecond)
	} else if scenario == "matching_first_delta" {
		clock.Advance(100 * time.Millisecond)
		foregroundCall.emitDelta("<output>Cleaned")
		waitForPairedDraftOrDeadline(observations, 150*time.Millisecond)
	} else if scenario == "live_cleanup_disabled" {
		clock.Advance(100 * time.Millisecond)
		foregroundCall.emitDelta("<output>Cleaned")
		waitForPairedDraftOrDeadline(observations, 150*time.Millisecond)
	}
	if liveCleanupEnabled && !completeBeforeFinal {
		clock.Advance(time.Second - clock.Now().Sub(stopAt))
		speculativeCall.complete(t, speculativeAnswer)
	} else {
		clock.Advance(time.Second - clock.Now().Sub(stopAt))
		foregroundCall.complete(t, foregroundAnswer)
	}
	end := fixture.rec.wait(t)
	completionAt := clock.Now()
	var firstDraftAt *time.Time
	select {
	case observedAt := <-observations.firstDraft:
		firstDraftAt = &observedAt
	default:
	}
	var firstPresentationAt *time.Time
	select {
	case observedAt := <-observations.firstPresentation:
		firstPresentationAt = &observedAt
	default:
	}
	var pasteAt *time.Time
	select {
	case observedAt := <-observations.pasteAt:
		pasteAt = &observedAt
	case <-time.After(5 * time.Second):
		t.Fatal("the harness did not observe fake paste delivery")
	}
	if end.Result != expectedResult || model.requests.Load() != expectedRequests || len(fixture.text.pastes()) != 1 || fixture.text.pastes()[0] != end.Result {
		t.Fatalf("scenario %s: result %+v, requests %d, pasted %q; want result %q and %d requests", scenario, end, model.requests.Load(), fixture.text.pastes(), expectedResult, expectedRequests)
	}
	fixture.m.mu.Lock()
	outcome := currentRun.cleanupOutcome
	fixture.m.mu.Unlock()
	if firstPresentationAt == nil {
		t.Fatalf("scenario %s completed without a cleaned presentation: %+v", scenario, end)
	}
	t.Logf("paired_benchmark scenario=%s stop_to_first_cleaned_draft_ms=%s stop_to_first_cleaned_presentation_ms=%s asr_final_to_first_cleaned_presentation_ms=%s stop_to_session_complete_ms=%d stop_to_paste_ms=%s llm_request_count=%d cleanup_outcome=%s completed_prefix_published_before_tail_delta=%t final_output=%q", scenario, pairedBenchmarkLatency(firstDraftAt, stopAt), pairedBenchmarkLatency(firstPresentationAt, stopAt), pairedBenchmarkLatency(firstPresentationAt, asrFinalAt), completionAt.Sub(stopAt).Milliseconds(), pairedBenchmarkLatency(pasteAt, stopAt), model.requests.Load(), outcome, prefixPublishedBeforeTailDelta, end.Result)
}

func waitPairedBenchmarkCall(t *testing.T, model *pairedBenchmarkLLM, failure string) *pairedBenchmarkCall {
	t.Helper()
	select {
	case call := <-model.calls:
		return call
	case <-time.After(5 * time.Second):
		t.Fatal(failure)
		return nil
	}
}

func waitForPairedDraftOrDeadline(observations *pairedBenchmarkObservation, deadline time.Duration) {
	select {
	case <-observations.draftUpdates:
	case <-time.After(deadline):
	}
}

func waitForPairedDraft(t *testing.T, observations *pairedBenchmarkObservation, failure string) {
	t.Helper()
	select {
	case <-observations.draftUpdates:
	case <-time.After(5 * time.Second):
		t.Fatal(failure)
	}
}

func drainPairedBenchmarkDraftUpdates(observations *pairedBenchmarkObservation) {
	for {
		select {
		case <-observations.draftUpdates:
		default:
			return
		}
	}
}

func pairedBenchmarkLatency(observedAt *time.Time, origin time.Time) string {
	if observedAt == nil {
		return "null"
	}
	return strconv.FormatInt(observedAt.Sub(origin).Milliseconds(), 10)
}
