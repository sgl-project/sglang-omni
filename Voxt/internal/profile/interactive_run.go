package profile

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

const interactiveArrivalDelay = 100 * time.Millisecond

type interactiveOptions struct {
	now                     func() time.Time
	waitForInteractive      func(context.Context) error
	afterInteractiveArrival func()
	arrivalCallbackMu       *sync.Mutex
	detector                vad.Model
}

type interactiveArrivalState struct {
	at         time.Time
	duringLong bool
	err        error
}

type interactiveLongOutcome struct {
	finished time.Time
	err      error
}

type interactiveOutcome struct {
	text     string
	started  time.Time
	finished time.Time
	err      error
}

type interactiveMethodRun struct {
	config           InteractiveConfig
	method           string
	speech           asr.FileTranscriber
	longPCM          []byte
	interactivePCM   []byte
	options          interactiveOptions
	started          time.Time
	chunks           []InteractiveChunk
	longParts        []string
	longSegments     []InteractiveSegment
	longCalls        int
	interactiveCalls int
	longStarted      time.Time
	longFinished     time.Time
	longComplete     bool
	resume           *InteractiveResume
	errors           []string
}

func defaultInteractiveOptions() interactiveOptions {
	return interactiveOptions{
		now: time.Now,
		waitForInteractive: func(ctx context.Context) error {
			return sleepContext(ctx, interactiveArrivalDelay)
		},
	}
}

// RunInteractive compares whole-track queueing with contiguous and VAD-planned
// cooperative priority over one shared ASR client.
func RunInteractive(ctx context.Context, config InteractiveConfig, longPCM, interactivePCM []byte, speech asr.FileTranscriber, detector vad.Model) InteractiveReport {
	options := defaultInteractiveOptions()
	options.detector = detector
	return runInteractiveWithOptions(ctx, config, longPCM, interactivePCM, speech, options)
}

func runInteractiveWithOptions(ctx context.Context, config InteractiveConfig, longPCM, interactivePCM []byte, speech asr.FileTranscriber, options interactiveOptions) InteractiveReport {
	if options.arrivalCallbackMu == nil {
		options.arrivalCallbackMu = &sync.Mutex{}
	}
	started := options.now()
	report := InteractiveReport{
		SchemaVersion: 2, Name: config.Name, Status: "invalid", StartedAt: started, Config: config,
		Baseline:         runInteractiveMethod(ctx, config, longPCM, interactivePCM, speech, "baseline", options),
		SpeakerStitching: Unavailable[bool](interactiveSpeakerStitchingReason),
		Resources:        Unavailable[ResourceReport]("interactive scheduling does not sample process or host resources"),
		Limitations:      append([]string(nil), interactiveLimitations...), Errors: []string{},
	}
	report.Priority = runInteractiveMethod(ctx, config, longPCM, interactivePCM, speech, "priority", options)
	report.PriorityVAD = runInteractiveMethod(ctx, config, longPCM, interactivePCM, speech, "priority_vad", options)
	report.Priority.ComparisonToBaseline = compareInteractiveToBaseline(config, report.Baseline, report.Priority)
	report.PriorityVAD.ComparisonToBaseline = compareInteractiveToBaseline(config, report.Baseline, report.PriorityVAD)
	report.LongTextEqual = report.Priority.ComparisonToBaseline.LongTextEqual
	report.EndedAt = options.now()
	for _, method := range []InteractiveMethodReport{report.Baseline, report.Priority, report.PriorityVAD} {
		for _, err := range method.Errors {
			report.Errors = append(report.Errors, method.Method+": "+err)
		}
	}
	if report.Baseline.Status == "success" && report.Priority.Status == "success" && report.PriorityVAD.Status == "success" {
		report.Status = "success"
	}
	return report
}

func runInteractiveMethod(ctx context.Context, config InteractiveConfig, longPCM, interactivePCM []byte, speech asr.FileTranscriber, method string, options interactiveOptions) InteractiveMethodReport {
	run := &interactiveMethodRun{
		config: config, method: method, speech: speech, longPCM: longPCM, interactivePCM: interactivePCM,
		options: options, started: options.now(), chunks: []InteractiveChunk{}, errors: []string{},
	}
	if len(longPCM) == 0 || len(interactivePCM) == 0 || len(longPCM)%2 != 0 || len(interactivePCM)%2 != 0 {
		run.errors = append(run.errors, "interactive: long and interactive audio must be non-empty PCM16")
		return run.finish(interactiveArrivalState{}, interactiveOutcome{})
	}
	switch method {
	case "baseline":
		return run.runBaseline(ctx)
	case "priority":
		return run.runPriority(ctx)
	case "priority_vad":
		return run.runPriorityVAD(ctx)
	default:
		run.errors = append(run.errors, "interactive: unknown method "+method)
		return run.finish(interactiveArrivalState{}, interactiveOutcome{})
	}
}

func (r *interactiveMethodRun) chunkCursors() [][2]int {
	totalSamples := len(r.longPCM) / 2
	chunkSamples := int(r.config.SegmentLengthSeconds * asr.SampleRate)
	if chunkSamples <= 0 {
		return nil
	}
	var cursors [][2]int
	for start := 0; start < totalSamples; start += chunkSamples {
		end := min(start+chunkSamples, totalSamples)
		cursors = append(cursors, [2]int{start, end})
	}
	return cursors
}

func (r *interactiveMethodRun) startArrival(ctx context.Context, longStarted <-chan struct{}, longDone <-chan struct{}, arrivals chan<- interactiveArrivalState, baselineInteractive chan<- interactiveOutcome) {
	if r.options.arrivalCallbackMu == nil {
		r.options.arrivalCallbackMu = &sync.Mutex{}
	}
	go func() {
		<-longStarted
		waitErr := r.options.waitForInteractive(ctx)
		state := interactiveArrivalState{at: r.options.now(), duringLong: true, err: waitErr}
		select {
		case <-longDone:
			state.duringLong = false
		default:
		}
		arrivals <- state
		if r.options.afterInteractiveArrival != nil {
			r.options.arrivalCallbackMu.Lock()
			r.options.afterInteractiveArrival()
			r.options.arrivalCallbackMu.Unlock()
		}
		if baselineInteractive != nil {
			baselineInteractive <- r.callInteractive(ctx)
		}
	}()
}

func (r *interactiveMethodRun) runBaseline(ctx context.Context) InteractiveMethodReport {
	longStarted := make(chan struct{})
	longDone := make(chan struct{})
	arrivals := make(chan interactiveArrivalState, 1)
	longResults := make(chan interactiveLongOutcome, 1)
	interactiveResults := make(chan interactiveOutcome, 1)
	go func() {
		close(longStarted)
		outcome := r.callLong(ctx, 0, 0, len(r.longPCM)/2)
		close(longDone)
		longResults <- outcome
	}()
	r.startArrival(ctx, longStarted, longDone, arrivals, interactiveResults)
	arrival := <-arrivals
	interactive := <-interactiveResults
	long := <-longResults
	r.longComplete = long.err == nil
	r.appendError("interactive arrival", arrival.err)
	r.appendError("long whole-track request", long.err)
	r.appendError("interactive request", interactive.err)
	if !arrival.duringLong {
		r.errors = append(r.errors, "baseline: interactive arrived after the whole-track request completed")
	}
	return r.finish(arrival, interactive)
}

func (r *interactiveMethodRun) runPriority(ctx context.Context) InteractiveMethodReport {
	return r.runPriorityChunks(ctx, r.chunkCursors())
}

func (r *interactiveMethodRun) runPriorityVAD(ctx context.Context) InteractiveMethodReport {
	if r.options.detector == nil {
		r.errors = append(r.errors, "priority_vad: VAD model is required")
		return r.finish(interactiveArrivalState{}, interactiveOutcome{})
	}
	regions, err := r.options.detector.Speech(ctx, r.longPCM, asr.SampleRate)
	if err != nil {
		r.errors = append(r.errors, "priority_vad: VAD: "+err.Error())
		return r.finish(interactiveArrivalState{}, interactiveOutcome{})
	}
	if len(regions) == 0 {
		r.errors = append(r.errors, "priority_vad: VAD returned no speech regions")
		return r.finish(interactiveArrivalState{}, interactiveOutcome{})
	}
	params := meeting.DefaultChunkParams()
	params.MaxDur = time.Duration(r.config.SegmentLengthSeconds * float64(time.Second))
	planned := meeting.PlanChunks(regions, len(r.longPCM)/2, params)
	cursors := make([][2]int, 0, len(planned))
	for _, chunk := range planned {
		cursors = append(cursors, [2]int{chunk.Start, chunk.End})
	}
	return r.runPriorityChunks(ctx, cursors)
}

func (r *interactiveMethodRun) runPriorityChunks(ctx context.Context, cursors [][2]int) InteractiveMethodReport {
	if len(cursors) == 0 {
		r.errors = append(r.errors, r.method+": segment plan produced no long chunks")
		return r.finish(interactiveArrivalState{}, interactiveOutcome{})
	}
	longStarted := make(chan struct{})
	longDone := make(chan struct{})
	arrivals := make(chan interactiveArrivalState, 1)
	close(longStarted)
	r.startArrival(ctx, longStarted, longDone, arrivals, nil)
	var arrival *interactiveArrivalState
	var interactive interactiveOutcome
	interactiveHandled := false
	pollArrival := func() {
		if arrival != nil {
			return
		}
		select {
		case state := <-arrivals:
			arrival = &state
		default:
		}
	}
	runPending := func(afterChunk, nextChunk int, stopped time.Time, canResume bool) {
		if arrival == nil || interactiveHandled {
			return
		}
		interactive = r.callInteractive(ctx)
		interactiveHandled = true
		r.appendError("interactive request", interactive.err)
		if canResume && interactive.err == nil && nextChunk < len(cursors) {
			resumed := r.options.now()
			r.resume = &InteractiveResume{
				AfterChunk: afterChunk, NextChunk: nextChunk,
				StoppedMS: milliseconds(stopped.Sub(r.started)), InteractiveStartedMS: milliseconds(interactive.started.Sub(r.started)), ResumedMS: milliseconds(resumed.Sub(r.started)),
			}
		}
	}
	longFailed := false
	for index, cursor := range cursors {
		pollArrival()
		if index > 0 && arrival != nil && !interactiveHandled {
			runPending(index-1, index, r.longFinished, true)
		}
		if interactiveHandled && interactive.err != nil {
			break
		}
		outcome := r.callLong(ctx, index, cursor[0], cursor[1])
		pollArrival()
		runPending(index, index+1, outcome.finished, outcome.err == nil)
		if outcome.err != nil {
			longFailed = true
			r.appendError(fmt.Sprintf("long chunk %d", index), outcome.err)
			break
		}
		if interactiveHandled && interactive.err != nil {
			break
		}
	}
	close(longDone)
	if arrival == nil {
		state := <-arrivals
		arrival = &state
	}
	if !interactiveHandled {
		stopped := r.options.now()
		if !r.longFinished.IsZero() {
			stopped = r.longFinished
		}
		runPending(len(r.chunks)-1, len(cursors), stopped, false)
	}
	r.longComplete = !longFailed && len(r.chunks) == len(cursors)
	r.appendError("interactive arrival", arrival.err)
	if !arrival.duringLong {
		r.errors = append(r.errors, r.method+": interactive arrived after the long chunks completed")
	}
	if len(r.errors) == 0 && r.resume == nil {
		r.errors = append(r.errors, r.method+": interactive did not run before a remaining long chunk")
	}
	return r.finish(*arrival, interactive)
}

func (r *interactiveMethodRun) callLong(ctx context.Context, index, startSample, endSample int) interactiveLongOutcome {
	started := r.options.now()
	if r.longStarted.IsZero() {
		r.longStarted = started
	}
	r.longCalls++
	request := asr.FileRequest{Audio: r.longPCM[startSample*2 : endSample*2], Format: "pcm", SampleRate: asr.SampleRate, Lang: r.config.Language}
	var result asr.Result
	var err error
	if r.config.Operation == "diarize" {
		diarizer, ok := r.speech.(asr.Diarizer)
		if !ok {
			err = fmt.Errorf("interactive: configured speech client cannot diarize")
		} else {
			result, err = diarizer.Diarize(ctx, request)
		}
	} else {
		result, err = r.speech.Transcribe(ctx, request)
	}
	finished := r.options.now()
	r.longFinished = finished
	chunk := InteractiveChunk{
		Index: index, CursorStartSample: startSample, CursorEndSample: endSample,
		StartedMS: milliseconds(started.Sub(r.started)), FinishedMS: milliseconds(finished.Sub(r.started)),
	}
	text := interactiveResultText(result)
	if err == nil {
		if r.method == "priority_vad" && len(r.longParts) > 0 {
			text = meeting.DedupeOverlapText(r.longParts[len(r.longParts)-1], text)
		}
		hash := HashText(text)
		chunk.TextSHA256 = &hash
		r.longParts = append(r.longParts, text)
		chunkOffset := time.Duration(startSample) * time.Second / asr.SampleRate
		for _, segment := range result.Segments {
			r.longSegments = append(r.longSegments, InteractiveSegment{
				ChunkIndex: index, StartMS: milliseconds(chunkOffset + segment.Start), EndMS: milliseconds(chunkOffset + segment.End),
				Speaker: segment.Speaker, Text: segment.Text, TextSHA256: HashText(segment.Text),
			})
		}
	} else {
		chunk.Error = err.Error()
	}
	r.chunks = append(r.chunks, chunk)
	return interactiveLongOutcome{finished: finished, err: err}
}

func (r *interactiveMethodRun) callInteractive(ctx context.Context) interactiveOutcome {
	started := r.options.now()
	r.interactiveCalls++
	result, err := r.speech.Transcribe(ctx, asr.FileRequest{Audio: r.interactivePCM, Format: "pcm", SampleRate: asr.SampleRate, Lang: r.config.Language})
	return interactiveOutcome{text: interactiveResultText(result), started: started, finished: r.options.now(), err: err}
}

func interactiveResultText(result asr.Result) string {
	if strings.TrimSpace(result.Text) != "" || len(result.Segments) == 0 {
		return result.Text
	}
	parts := make([]string, 0, len(result.Segments))
	for _, segment := range result.Segments {
		parts = append(parts, segment.Text)
	}
	return strings.Join(parts, "\n")
}

func joinInteractiveParts(parts []string) string {
	nonempty := make([]string, 0, len(parts))
	for _, part := range parts {
		if text := strings.TrimSpace(part); text != "" {
			nonempty = append(nonempty, text)
		}
	}
	return strings.Join(nonempty, "\n")
}

func (r *interactiveMethodRun) appendError(operation string, err error) {
	if err != nil {
		r.errors = append(r.errors, operation+": "+err.Error())
	}
}

func (r *interactiveMethodRun) finish(arrival interactiveArrivalState, interactive interactiveOutcome) InteractiveMethodReport {
	report := InteractiveMethodReport{
		Method: r.method, Status: "invalid", InteractiveArrivedDuringLong: arrival.duringLong,
		LongCalls: r.longCalls, InteractiveCalls: r.interactiveCalls,
		Chunks: append([]InteractiveChunk(nil), r.chunks...), LongSegments: append([]InteractiveSegment(nil), r.longSegments...),
		Resume: r.resume, Errors: append([]string(nil), r.errors...),
	}
	if !r.longStarted.IsZero() && !r.longFinished.IsZero() {
		value := milliseconds(r.longFinished.Sub(r.longStarted))
		report.LongTotalMS = &value
	}
	if !arrival.at.IsZero() && !interactive.finished.IsZero() {
		value := milliseconds(interactive.finished.Sub(arrival.at))
		report.InteractiveWaitMS = &value
	}
	if !interactive.started.IsZero() && !interactive.finished.IsZero() {
		value := milliseconds(interactive.finished.Sub(interactive.started))
		report.InteractiveTotalMS = &value
	}
	if r.longComplete {
		text := joinInteractiveParts(r.longParts)
		hash := HashText(text)
		report.LongText, report.LongTextHash = &text, &hash
	}
	if !interactive.finished.IsZero() && interactive.err == nil {
		text := interactive.text
		hash := HashText(text)
		report.InteractiveText, report.InteractiveTextHash = &text, &hash
	}
	if len(r.errors) == 0 && r.longComplete && !interactive.finished.IsZero() && interactive.err == nil && arrival.duringLong && (r.method == "baseline" || r.resume != nil) {
		report.Status = "success"
	}
	return report
}
