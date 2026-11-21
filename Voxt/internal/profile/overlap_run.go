package profile

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

type overlapOptions struct {
	now       func() time.Time
	waitUntil func(context.Context, time.Time) error
	afterFeed func(context.Context, int) error
	newLLM    func() overlapLLM
}

type overlapRun struct {
	config           OverlapConfig
	method           string
	speech           asr.FileTranscriber
	pcm              []byte
	chunks           []meeting.Chunk
	llm              overlapLLM
	options          overlapOptions
	started          time.Time
	recordingStarted time.Time
	stopped          time.Time
	segmentCount     int
	rawParts         []string
	cleanedParts     []string
	summaryMap       []string
	summaryReduce    []string
	finalSummary     string
	previousCleaned  string

	mu           sync.Mutex
	stages       []OverlapStage
	stageIndexes map[string]int
	errors       []string
	asrTotal     time.Duration
	asrAfterStop time.Duration
	llmTotal     time.Duration
	llmAfterStop time.Duration
	asrCalls     int
	llmCalls     int
	asrCompleted int
	llmCompleted int
	asrFailed    bool
	firstASRText time.Time
	firstLLMText time.Time
	backlog      OverlapBacklog
}

func defaultOverlapOptions(config OverlapConfig) overlapOptions {
	options := overlapOptions{
		now: time.Now,
		newLLM: func() overlapLLM {
			return newSimulatedLLM(time.Duration(config.SimulatedLLM.LatencyMS) * time.Millisecond)
		},
	}
	options.waitUntil = func(ctx context.Context, deadline time.Time) error {
		return sleepContext(ctx, time.Until(deadline))
	}
	return options
}

// RunOverlap runs comparable one_shot, serial, and overlap experiments over
// the same PCM and one shared speech-model client.
func RunOverlap(ctx context.Context, config OverlapConfig, pcm []byte, speech asr.FileTranscriber, detector vad.Model) []OverlapReport {
	return runOverlapWithOptions(ctx, config, pcm, speech, detector, defaultOverlapOptions(config))
}

func runOverlapWithOptions(ctx context.Context, config OverlapConfig, pcm []byte, speech asr.FileTranscriber, detector vad.Model, options overlapOptions) []OverlapReport {
	reports := make([]OverlapReport, 0, 3)
	for _, method := range []string{"one_shot", "serial", "overlap"} {
		reports = append(reports, runOverlapMethod(ctx, config, pcm, speech, detector, method, options))
	}
	addOverlapComparisons(reports)
	return reports
}

func newOverlapRun(config OverlapConfig, pcm []byte, speech asr.FileTranscriber, method string, options overlapOptions) *overlapRun {
	started := options.now()
	return &overlapRun{
		config: config, method: method, speech: speech, pcm: pcm, options: options,
		started: started, llm: options.newLLM(), stageIndexes: map[string]int{},
	}
}

func runOverlapMethod(ctx context.Context, config OverlapConfig, pcm []byte, speech asr.FileTranscriber, detector vad.Model, method string, options overlapOptions) OverlapReport {
	run := newOverlapRun(config, pcm, speech, method, options)
	if err := run.plan(ctx, detector); err != nil {
		run.fail(err)
		return run.finish()
	}
	run.recordingStarted = run.options.now()
	switch method {
	case "one_shot":
		run.waitForRecording(ctx, asr.Duration(pcm, asr.SampleRate))
		run.markStop()
		if run.transcribe(ctx, 0, pcm) {
			run.processSegment(ctx, 0)
		}
	case "serial":
		run.waitForRecording(ctx, asr.Duration(pcm, asr.SampleRate))
		run.markStop()
		for index, chunk := range run.chunks {
			if run.transcribe(ctx, index, pcmForChunk(pcm, chunk)) {
				run.processSegment(ctx, index)
			}
		}
	case "overlap":
		run.runConcurrent(ctx)
	}
	if !run.hasFailed() && config.Mode == "meeting" {
		run.reduceSummary(ctx)
	}
	return run.finish()
}

func (r *overlapRun) plan(ctx context.Context, detector vad.Model) error {
	started := r.options.now()
	regions := []vad.Segment{{Start: 0, End: len(r.pcm) / 2}}
	var err error
	if r.config.VAD.Enabled {
		regions, err = detector.Speech(ctx, r.pcm, asr.SampleRate)
		if err == nil && len(regions) == 0 {
			err = fmt.Errorf("overlap: VAD returned no speech regions")
		}
	}
	if err == nil {
		params := meeting.DefaultChunkParams()
		if r.config.SegmentLengthSeconds > 0 {
			params.MaxDur = time.Duration(r.config.SegmentLengthSeconds * float64(time.Second))
		}
		r.chunks = meeting.PlanChunks(regions, len(r.pcm)/2, params)
		if len(r.chunks) == 0 {
			err = fmt.Errorf("overlap: audio produced no ASR chunks")
		}
	}
	if r.method == "one_shot" {
		r.segmentCount = 1
	} else {
		r.segmentCount = len(r.chunks)
	}
	r.rawParts = make([]string, r.segmentCount)
	r.cleanedParts = make([]string, r.segmentCount)
	r.finishStage("segment_plan", started, asr.Duration(r.pcm, asr.SampleRate), "", "", err)
	return err
}

func pcmForChunk(pcm []byte, chunk meeting.Chunk) []byte {
	return pcm[chunk.Start*2 : chunk.End*2]
}

func (r *overlapRun) waitForRecording(ctx context.Context, duration time.Duration) {
	if err := r.options.waitUntil(ctx, r.recordingStarted.Add(duration)); err != nil {
		r.fail(err)
	}
}

func (r *overlapRun) markStop() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.finishStageLocked("recording", r.recordingStarted, asr.Duration(r.pcm, asr.SampleRate), "", "", nil)
	r.stopped = r.options.now()
	asrRemaining := int64(r.segmentCount - r.asrCompleted)
	llmRemaining := int64(r.segmentCount - r.llmCompleted)
	r.backlog = OverlapBacklog{ASRSegments: Available(asrRemaining), LLMSegments: Available(llmRemaining)}
}

func (r *overlapRun) transcribe(ctx context.Context, index int, pcm []byte) bool {
	started := r.options.now()
	result, err := r.speech.Transcribe(ctx, asr.FileRequest{Audio: pcm, Format: "pcm", SampleRate: asr.SampleRate, Lang: r.config.Language})
	finished := r.options.now()
	r.mu.Lock()
	r.asrCalls++
	r.asrTotal += finished.Sub(started)
	r.asrAfterStop += afterStopDuration(started, finished, r.stopped)
	if err == nil {
		text := result.Text
		if index > 0 {
			text = meeting.DedupeOverlapText(r.rawParts[index-1], text)
		}
		r.rawParts[index] = text
		r.asrCompleted++
		if strings.TrimSpace(text) != "" && r.firstASRText.IsZero() {
			r.firstASRText = finished
		}
	} else {
		r.asrFailed = true
	}
	r.finishStageLocked("asr_chunk", started, asr.Duration(pcm, asr.SampleRate), "", result.Text, err)
	r.mu.Unlock()
	if err != nil {
		r.fail(err)
		return false
	}
	return true
}

func (r *overlapRun) processSegment(ctx context.Context, index int) {
	cleaned, err := r.callLLM(ctx, simulatedCleanup, []SimulatedPart{{Index: index, Text: r.rawParts[index]}}, r.previousCleaned)
	if err != nil {
		return
	}
	r.cleanedParts[index] = cleaned
	r.previousCleaned = cleaned
	if r.config.Mode == "meeting" {
		windows := meeting.SummaryWindows(cleaned, 0)
		if len(windows) == 0 {
			r.fail(fmt.Errorf("overlap: summary cannot map an empty transcript segment %d", index))
			return
		}
		for _, window := range windows {
			mapIndex := len(r.summaryMap)
			raw, mapErr := r.callLLM(ctx, simulatedSummaryMap, []SimulatedPart{{Index: mapIndex, Text: window}}, "")
			if mapErr != nil {
				return
			}
			r.summaryMap = append(r.summaryMap, raw)
		}
	}
	r.mu.Lock()
	r.llmCompleted++
	r.mu.Unlock()
}

func (r *overlapRun) reduceSummary(ctx context.Context) {
	if len(r.summaryMap) == 0 {
		r.fail(fmt.Errorf("overlap: meeting produced no summary map output"))
		return
	}
	current := append([]string(nil), r.summaryMap...)
	for {
		windows := meeting.SummaryWindows(strings.Join(current, "\n"), 0)
		next := make([]string, 0, len(windows))
		for index, window := range windows {
			raw, err := r.callLLM(ctx, simulatedSummaryReduce, []SimulatedPart{{Index: index, Text: window}}, "")
			if err != nil {
				return
			}
			r.summaryReduce = append(r.summaryReduce, raw)
			next = append(next, raw)
		}
		if len(next) == 1 {
			r.finalSummary = next[0]
			return
		}
		current = next
	}
}

func (r *overlapRun) callLLM(ctx context.Context, operation string, parts []SimulatedPart, previous string) (string, error) {
	request := simulatedLLMRequest{Operation: operation, Parts: parts, PreviousContext: previous}
	input, _ := json.Marshal(request)
	started := r.options.now()
	raw, err := r.llm.Call(ctx, request)
	finished := r.options.now()
	text := raw
	if err == nil {
		switch operation {
		case simulatedCleanup:
			var response []struct {
				Index *int    `json:"i"`
				Text  *string `json:"text"`
			}
			if json.Unmarshal([]byte(raw), &response) != nil || len(response) != 1 || response[0].Index == nil || response[0].Text == nil || *response[0].Index != parts[0].Index {
				err = fmt.Errorf("overlap: simulated cleanup returned an invalid ordered structure")
			} else {
				text = *response[0].Text
			}
		case simulatedSummaryMap, simulatedSummaryReduce:
			if _, parseErr := meeting.ParseSummary(raw); parseErr != nil {
				err = fmt.Errorf("overlap: simulated summary returned an invalid structure: %w", parseErr)
			}
		}
	}
	r.mu.Lock()
	r.llmCalls++
	r.llmTotal += finished.Sub(started)
	r.llmAfterStop += afterStopDuration(started, finished, r.stopped)
	if err == nil && strings.TrimSpace(text) != "" && r.firstLLMText.IsZero() {
		r.firstLLMText = finished
	}
	r.finishStageLocked("llm_"+operation, started, 0, HashText(string(input)), raw, err)
	if previous != "" && len(r.stages) > 0 {
		r.stages[len(r.stages)-1].PreviousContextSHA256 = HashText(previous)
	}
	r.mu.Unlock()
	if err != nil {
		r.fail(err)
		return "", err
	}
	return text, nil
}

func afterStopDuration(started, finished, stopped time.Time) time.Duration {
	if stopped.IsZero() || !finished.After(stopped) {
		return 0
	}
	if started.Before(stopped) {
		started = stopped
	}
	return finished.Sub(started)
}

func (r *overlapRun) finishStage(name string, started time.Time, audio time.Duration, inputSHA, text string, err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.finishStageLocked(name, started, audio, inputSHA, text, err)
}

func (r *overlapRun) finishStageLocked(name string, started time.Time, audio time.Duration, inputSHA, text string, err error) {
	finished := r.options.now()
	index := r.stageIndexes[name]
	r.stageIndexes[name] = index + 1
	afterStop := Available(milliseconds(afterStopDuration(started, finished, r.stopped)))
	if r.stopped.IsZero() && err != nil {
		afterStop = Unavailable[float64]("recording did not stop")
	}
	stage := OverlapStage{
		Stage:     Stage{Name: name, Index: index, WallMS: milliseconds(finished.Sub(started)), Text: text, TextSHA256: HashText(text)},
		StartedMS: milliseconds(started.Sub(r.started)), AfterStopMS: afterStop, InputSHA256: inputSHA,
	}
	if audio > 0 {
		audioMS := audio.Milliseconds()
		rtf := stage.WallMS / float64(audioMS)
		stage.AudioMS = &audioMS
		stage.RTF = &rtf
	}
	if err != nil {
		stage.Error = err.Error()
	}
	r.stages = append(r.stages, stage)
}

func (r *overlapRun) fail(err error) {
	if err == nil {
		return
	}
	r.mu.Lock()
	r.errors = append(r.errors, err.Error())
	r.mu.Unlock()
}

func (r *overlapRun) hasFailed() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.errors) > 0
}

func joinOverlapParts(parts []string) string {
	nonempty := make([]string, 0, len(parts))
	for _, part := range parts {
		if text := strings.TrimSpace(part); text != "" {
			nonempty = append(nonempty, text)
		}
	}
	return strings.Join(nonempty, "\n")
}
