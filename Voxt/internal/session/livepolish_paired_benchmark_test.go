package session

import (
	"context"
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
}

func (provider *pairedBenchmarkASR) Caps() asr.Caps {
	return asr.Caps{Stream: true}
}

func (provider *pairedBenchmarkASR) Open(requestContext context.Context, _ asr.StreamRequest) (asr.Stream, error) {
	return &pairedBenchmarkStream{
		context: requestContext, finalGate: provider.finishGate, finishStarted: provider.finishStarted,
		finishEmitted: provider.finishEmitted, events: make(chan asr.Event, 4),
	}, nil
}

type pairedBenchmarkStream struct {
	context       context.Context
	finalGate     <-chan struct{}
	finishStarted chan struct{}
	finishEmitted chan struct{}
	events        chan asr.Event
	partialOnce   sync.Once
}

func (stream *pairedBenchmarkStream) Send([]byte) error {
	stream.partialOnce.Do(func() {
		stream.events <- asr.Event{Kind: asr.EventPartial, Text: "spoken words."}
	})
	return nil
}

func (stream *pairedBenchmarkStream) Finish() error {
	close(stream.finishStarted)
	select {
	case <-stream.finalGate:
		stream.events <- asr.Event{Kind: asr.EventFinal, Text: "spoken words.", Result: asr.Result{Text: "spoken words.", Model: "paired-benchmark"}}
		close(stream.events)
		close(stream.finishEmitted)
		return nil
	case <-stream.context.Done():
		return stream.context.Err()
	}
}

func (*pairedBenchmarkStream) Close() {}

func (stream *pairedBenchmarkStream) Events() <-chan asr.Event {
	return stream.events
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
	pasteAt           chan time.Time
}

func newPairedBenchmarkHarness(t *testing.T, streamer asr.Provider, model *pairedBenchmarkLLM, clock *softDeadlineClock) (*harness, *pairedBenchmarkObservation) {
	t.Helper()
	observations := &pairedBenchmarkObservation{
		firstDraft: make(chan time.Time, 1), firstPresentation: make(chan time.Time, 1), pasteAt: make(chan time.Time, 4),
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
			if snapshot.State == Composing && snapshot.Draft != "" {
				firstDraftOnce.Do(func() { observations.firstDraft <- clock.Now() })
				firstPresentationOnce.Do(func() { observations.firstPresentation <- clock.Now() })
			} else if snapshot.State.Ended() && snapshot.Result != "" {
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
	state.Features.Transcribe.Enhance = true
	fixture.m = New(config)
	t.Cleanup(fixture.m.Close)
	return fixture, observations
}

func TestLivePolishPairedBaselineBufferedStreamLatency(t *testing.T) {
	runPairedBenchmarkScenario(t, "matching_buffered", true)
}

func TestLivePolishPairedBaselineFirstDeltaLatency(t *testing.T) {
	runPairedBenchmarkScenario(t, "matching_first_delta", false)
}

func runPairedBenchmarkScenario(t *testing.T, scenario string, deltaBeforeFinal bool) {
	t.Helper()
	clock := newSoftDeadlineClock()
	model := &pairedBenchmarkLLM{calls: make(chan *pairedBenchmarkCall, 4)}
	finalGate := make(chan struct{})
	streamer := &pairedBenchmarkASR{
		finishGate: finalGate, finishStarted: make(chan struct{}), finishEmitted: make(chan struct{}),
	}
	fixture, observations := newPairedBenchmarkHarness(t, streamer, model, clock)
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
	select {
	case speculativeCall = <-model.calls:
	case <-time.After(5 * time.Second):
		t.Fatal("the speculative cleanup request did not start")
	}
	stopAt := clock.Now()
	fixture.m.Stop(id)
	select {
	case <-streamer.finishStarted:
	case <-time.After(5 * time.Second):
		t.Fatal("ASR did not start finalization")
	}
	clock.Advance(100 * time.Millisecond)
	if deltaBeforeFinal {
		speculativeCall.emitDelta("<output>Cleaned")
		if fixture.m.Current().Draft != "" {
			t.Fatalf("speculative text escaped before ASR Final: %q", fixture.m.Current().Draft)
		}
	}
	clock.Advance(200 * time.Millisecond)
	close(finalGate)
	select {
	case <-streamer.finishEmitted:
	case <-time.After(5 * time.Second):
		t.Fatal("ASR Final did not arrive")
	}
	asrFinalAt := clock.Now()
	fixture.rec.waitFor(t, "the composing state", func(snapshot Snapshot) bool { return snapshot.ID == id && snapshot.State == Composing })
	if !deltaBeforeFinal {
		clock.Advance(100 * time.Millisecond)
		speculativeCall.emitDelta("<output>Cleaned")
	}
	var firstDraftAt *time.Time
	select {
	case observedAt := <-observations.firstDraft:
		firstDraftAt = &observedAt
	case <-time.After(500 * time.Millisecond):
	}
	clock.Advance(time.Second - clock.Now().Sub(stopAt))
	speculativeCall.complete(t, "<output>Cleaned text.</output>")
	end := fixture.rec.wait(t)
	completionAt := clock.Now()
	var firstPresentationAt time.Time
	select {
	case firstPresentationAt = <-observations.firstPresentation:
	case <-time.After(5 * time.Second):
		t.Fatal("the harness did not observe cleaned text")
	}
	var pasteAt time.Time
	select {
	case pasteAt = <-observations.pasteAt:
	case <-time.After(5 * time.Second):
		t.Fatal("the harness did not observe paste delivery")
	}
	if end.Result != "Cleaned text." || model.requests.Load() != 1 || len(fixture.text.pastes()) != 1 || fixture.text.pastes()[0] != end.Result {
		t.Fatalf("result %+v, requests %d, pasted %q", end, model.requests.Load(), fixture.text.pastes())
	}
	fixture.m.mu.Lock()
	outcome := currentRun.cleanupOutcome
	fixture.m.mu.Unlock()
	var stopToFirstDraft any
	if firstDraftAt != nil {
		stopToFirstDraft = firstDraftAt.Sub(stopAt).Milliseconds()
	}
	bufferNonemptyAtAdoption := firstDraftAt != nil && deltaBeforeFinal
	t.Logf("paired_benchmark scenario=%s stop_to_first_cleaned_draft_ms=%v stop_to_first_cleaned_presentation_ms=%d asr_final_to_first_cleaned_presentation_ms=%d stop_to_cleanup_complete_ms=%d stop_to_paste_ms=%d llm_request_count=%d llm_attempt_count=%d cleanup_outcome=%s speculative_flight_adopted=%t buffer_nonempty_at_adoption=%t completed_prefix_published=false final_text=%q", scenario, stopToFirstDraft, firstPresentationAt.Sub(stopAt).Milliseconds(), firstPresentationAt.Sub(asrFinalAt).Milliseconds(), completionAt.Sub(stopAt).Milliseconds(), pasteAt.Sub(stopAt).Milliseconds(), model.requests.Load(), model.requests.Load(), outcome, firstDraftAt != nil, bufferNonemptyAtAdoption, end.Result)
}
