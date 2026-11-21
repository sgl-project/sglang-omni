package profile

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

type fakeInteractiveSpeech struct {
	shortBytes     int
	gates          map[int]<-chan struct{}
	failLongCall   int
	longError      error
	interactiveErr error
	longResults    map[int]asr.Result

	mu               sync.Mutex
	longCalls        int
	interactiveCalls int
	events           []string
}

func (f *fakeInteractiveSpeech) Caps() asr.Caps { return asr.Caps{File: true} }

func (f *fakeInteractiveSpeech) Transcribe(ctx context.Context, request asr.FileRequest) (asr.Result, error) {
	if len(request.Audio) == f.shortBytes {
		return f.transcribeInteractive(ctx)
	}
	return f.transcribeLong(ctx)
}

func (f *fakeInteractiveSpeech) Diarize(ctx context.Context, _ asr.FileRequest) (asr.Result, error) {
	return f.transcribeLong(ctx)
}

func (f *fakeInteractiveSpeech) transcribeLong(ctx context.Context) (asr.Result, error) {
	f.mu.Lock()
	call := f.longCalls
	f.longCalls++
	f.events = append(f.events, fmt.Sprintf("long:start:%d", call))
	f.mu.Unlock()
	if gate := f.gates[call]; gate != nil {
		select {
		case <-gate:
		case <-ctx.Done():
			return asr.Result{}, ctx.Err()
		}
	}
	f.mu.Lock()
	f.events = append(f.events, fmt.Sprintf("long:end:%d", call))
	f.mu.Unlock()
	if call == f.failLongCall {
		return asr.Result{}, f.longError
	}
	if result, ok := f.longResults[call]; ok {
		return result, nil
	}
	return asr.Result{Text: fmt.Sprintf("part-%d", call)}, nil
}

func (f *fakeInteractiveSpeech) transcribeInteractive(ctx context.Context) (asr.Result, error) {
	f.mu.Lock()
	f.interactiveCalls++
	f.events = append(f.events, "interactive:start")
	f.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return asr.Result{}, err
	}
	f.mu.Lock()
	f.events = append(f.events, "interactive:end")
	f.mu.Unlock()
	if f.interactiveErr != nil {
		return asr.Result{}, f.interactiveErr
	}
	return asr.Result{Text: "short-dictation"}, nil
}

func (f *fakeInteractiveSpeech) eventOrder() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.events...)
}

type interactiveVADFunc func(context.Context, []byte, int) ([]vad.Segment, error)

func (f interactiveVADFunc) Speech(ctx context.Context, pcm []byte, rate int) ([]vad.Segment, error) {
	return f(ctx, pcm, rate)
}

func (f interactiveVADFunc) Open(context.Context) (vad.ModelStream, error) { return nil, nil }

func fakeInteractiveConfig(operation string) InteractiveConfig {
	return InteractiveConfig{
		Name: "fake", Operation: operation, LongAudio: "long.wav", InteractiveAudio: "short.wav", OutputDir: "unused",
		Model: ModelConfig{ID: "asr.fast", Dir: "fake-model"}, SegmentLengthSeconds: 20,
		VAD: VADConfig{Enabled: true, ModelDir: "fake-vad"}, Language: "zh",
	}
}

func fakeInteractivePCM(seconds int) []byte {
	return make([]byte, seconds*asr.SampleRate*2)
}

func arrivedInteractiveOptions(arrived chan struct{}) interactiveOptions {
	return interactiveOptions{
		now:                     time.Now,
		waitForInteractive:      func(context.Context) error { return nil },
		afterInteractiveArrival: func() { close(arrived) },
	}
}

func TestMeetingInteractiveFakePriorityRunsShortAfterCurrentChunk(t *testing.T) {
	for _, operation := range []string{"transcribe", "diarize"} {
		t.Run(operation, func(t *testing.T) {
			arrived := make(chan struct{})
			speech := &fakeInteractiveSpeech{
				shortBytes: len(fakeInteractivePCM(1)), gates: map[int]<-chan struct{}{0: arrived}, failLongCall: -1,
			}
			config := fakeInteractiveConfig(operation)
			report := runInteractiveMethod(context.Background(), config, fakeInteractivePCM(65), fakeInteractivePCM(1), speech, "priority", arrivedInteractiveOptions(arrived))
			if report.Status != "success" {
				t.Fatalf("priority is invalid: %+v", report.Errors)
			}
			wantEvents := []string{"long:start:0", "long:end:0", "interactive:start", "interactive:end"}
			for index := 1; index < 4; index++ {
				wantEvents = append(wantEvents, fmt.Sprintf("long:start:%d", index), fmt.Sprintf("long:end:%d", index))
			}
			if got := speech.eventOrder(); strings.Join(got, ",") != strings.Join(wantEvents, ",") {
				t.Fatalf("wrong execution order:\ngot  %v\nwant %v", got, wantEvents)
			}
			wantText := "part-0\npart-1\npart-2\npart-3"
			if report.LongText == nil || *report.LongText != wantText || report.LongTextHash == nil || *report.LongTextHash != HashText(wantText) {
				t.Errorf("ordered long text is wrong: %+v", report)
			}
			if report.LongCalls != 4 || report.InteractiveCalls != 1 || len(report.Chunks) != 4 {
				t.Errorf("calls or chunks are wrong: %+v", report)
			}
			totalSamples := 65 * asr.SampleRate
			for index, chunk := range report.Chunks {
				if chunk.Index != index || chunk.CursorEndSample <= chunk.CursorStartSample {
					t.Errorf("bad chunk cursor: %+v", chunk)
				}
				if index > 0 && chunk.CursorStartSample != report.Chunks[index-1].CursorEndSample {
					t.Errorf("chunk %d dropped or repeated samples: %+v", index, chunk)
				}
			}
			if report.Chunks[0].CursorStartSample != 0 || report.Chunks[3].CursorEndSample != totalSamples {
				t.Errorf("chunk cursors do not cover the input: %+v", report.Chunks)
			}
			if report.Resume == nil || report.Resume.AfterChunk != 0 || report.Resume.NextChunk != 1 || report.Resume.StoppedMS > report.Resume.InteractiveStartedMS || report.Resume.InteractiveStartedMS > report.Resume.ResumedMS {
				t.Errorf("stop/resume was not recorded at the safe point: %+v", report.Resume)
			}
			if !report.InteractiveArrivedDuringLong || report.InteractiveWaitMS == nil || report.InteractiveTotalMS == nil || report.LongTotalMS == nil {
				t.Errorf("interactive timing is unavailable: %+v", report)
			}
		})
	}
}

func TestMeetingInteractiveFakeBaselineComparisonAndJSON(t *testing.T) {
	gates := []chan struct{}{make(chan struct{}), make(chan struct{}), make(chan struct{})}
	speech := &fakeInteractiveSpeech{
		shortBytes: len(fakeInteractivePCM(1)), gates: map[int]<-chan struct{}{0: gates[0], 1: gates[1], 5: gates[2]}, failLongCall: -1,
		longResults: map[int]asr.Result{
			0: {Text: "part-0", Segments: []asr.SpeakerSegment{{Start: 100 * time.Millisecond, End: 200 * time.Millisecond, Speaker: "S01", Text: "baseline"}}},
			5: {Text: "part-5", Segments: []asr.SpeakerSegment{{Start: 150 * time.Millisecond, End: 250 * time.Millisecond, Speaker: "S02", Text: "chunked"}}},
		},
	}
	options := interactiveOptions{now: time.Now, waitForInteractive: func(context.Context) error { return nil }}
	options.detector = interactiveVADFunc(func(_ context.Context, pcm []byte, _ int) ([]vad.Segment, error) {
		return []vad.Segment{{Start: 0, End: len(pcm) / 2}}, nil
	})
	arrival := 0
	options.afterInteractiveArrival = func() {
		close(gates[arrival])
		arrival++
	}
	report := runInteractiveWithOptions(context.Background(), fakeInteractiveConfig("diarize"), fakeInteractivePCM(65), fakeInteractivePCM(1), speech, options)
	if report.Status != "success" || report.Baseline.Status != "success" || report.Priority.Status != "success" || report.PriorityVAD.Status != "success" {
		t.Fatalf("comparison is invalid: baseline=%v priority=%v priority_vad=%v all=%v", report.Baseline.Errors, report.Priority.Errors, report.PriorityVAD.Errors, report.Errors)
	}
	if report.Baseline.LongCalls != 1 || len(report.Baseline.Chunks) != 1 || report.Baseline.Resume != nil {
		t.Errorf("baseline was not one whole-track call: %+v", report.Baseline)
	}
	if report.Baseline.Chunks[0].CursorEndSample != 65*asr.SampleRate {
		t.Errorf("baseline did not cover the whole track: %+v", report.Baseline.Chunks[0])
	}
	if report.LongTextEqual == nil || *report.LongTextEqual {
		t.Errorf("different whole/chunked texts were not compared honestly: %+v", report.LongTextEqual)
	}
	if report.Baseline.LongText == nil || *report.Baseline.LongText != "part-0" || report.Priority.LongText == nil || *report.Priority.LongText != "part-1\npart-2\npart-3\npart-4" || report.PriorityVAD.LongText == nil || *report.PriorityVAD.LongText != "part-5\npart-6\npart-7\npart-8" {
		t.Errorf("comparison texts are wrong: baseline=%v priority=%v priority_vad=%v", report.Baseline.LongText, report.Priority.LongText, report.PriorityVAD.LongText)
	}
	comparison := report.PriorityVAD.ComparisonToBaseline
	if comparison == nil || comparison.LongTextEqual == nil || *comparison.LongTextEqual || comparison.BaselineLongTextHash == nil || *comparison.BaselineLongTextHash != *report.Baseline.LongTextHash || comparison.LongTextHash == nil || *comparison.LongTextHash != *report.PriorityVAD.LongTextHash {
		t.Errorf("priority_vad text comparison is wrong: %+v", comparison)
	}
	if comparison == nil || comparison.SegmentTimestampsEqual == nil || *comparison.SegmentTimestampsEqual || comparison.SpeakerLabelsEqual == nil || *comparison.SpeakerLabelsEqual || comparison.BaselineSegmentCount != 1 || comparison.SegmentCount != 1 {
		t.Errorf("segment timestamps or raw speaker labels were not compared: %+v", comparison)
	}
	if report.SpeakerStitching.Value != nil || !strings.Contains(report.SpeakerStitching.UnavailableReason, "voice embeddings") || !strings.Contains(report.SpeakerStitching.UnavailableReason, "global clustering") {
		t.Errorf("speaker stitching was claimed without embeddings: %+v", report.SpeakerStitching)
	}
	output := t.TempDir()
	if err := WriteInteractiveReport(output, report); err != nil {
		t.Fatal(err)
	}
	content, err := os.ReadFile(filepath.Join(output, "result.json"))
	if err != nil {
		t.Fatal(err)
	}
	var decoded struct {
		Baseline         map[string]any `json:"baseline"`
		Priority         map[string]any `json:"priority"`
		PriorityVAD      map[string]any `json:"priority_vad"`
		LongTextEqual    *bool          `json:"long_text_equal"`
		SpeakerStitching struct {
			Value             *bool  `json:"value"`
			UnavailableReason string `json:"unavailable_reason"`
		} `json:"speaker_stitching"`
		Resources struct {
			Value             *ResourceReport `json:"value"`
			UnavailableReason string          `json:"unavailable_reason"`
		} `json:"resources"`
	}
	if err := json.Unmarshal(content, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Baseline["method"] != "baseline" || decoded.Priority["method"] != "priority" || decoded.PriorityVAD["method"] != "priority_vad" || decoded.LongTextEqual == nil || *decoded.LongTextEqual {
		t.Errorf("JSON does not distinguish methods and comparison: %s", content)
	}
	if decoded.SpeakerStitching.Value != nil || decoded.SpeakerStitching.UnavailableReason == "" {
		t.Errorf("JSON does not mark speaker stitching unavailable: %s", content)
	}
	if decoded.Resources.Value != nil || decoded.Resources.UnavailableReason == "" {
		t.Errorf("unmeasured resources became a plausible zero: %s", content)
	}
}

func TestMeetingInteractiveFakePriorityVADPlansDedupesAndUsesAbsoluteTime(t *testing.T) {
	arrived := make(chan struct{})
	speech := &fakeInteractiveSpeech{
		shortBytes: len(fakeInteractivePCM(1)), gates: map[int]<-chan struct{}{0: arrived}, failLongCall: -1,
		longResults: map[int]asr.Result{
			0: {Text: "we discussed the plan", Segments: []asr.SpeakerSegment{{Start: time.Second, End: 2 * time.Second, Speaker: "S01", Text: "first"}}},
			1: {Text: "the plan for next week", Segments: []asr.SpeakerSegment{{Start: 500 * time.Millisecond, End: 1500 * time.Millisecond, Speaker: "S01", Text: "second"}}},
			2: {Text: "last part", Segments: []asr.SpeakerSegment{{Start: 0, End: time.Second, Speaker: "S02", Text: "third"}}},
		},
	}
	options := arrivedInteractiveOptions(arrived)
	vadCalls := 0
	options.detector = interactiveVADFunc(func(_ context.Context, pcm []byte, rate int) ([]vad.Segment, error) {
		vadCalls++
		if rate != asr.SampleRate || len(pcm) != len(fakeInteractivePCM(45)) {
			t.Errorf("VAD received the wrong input: rate=%d bytes=%d", rate, len(pcm))
		}
		return []vad.Segment{{Start: 0, End: len(pcm) / 2}}, nil
	})
	report := runInteractiveMethod(context.Background(), fakeInteractiveConfig("diarize"), fakeInteractivePCM(45), fakeInteractivePCM(1), speech, "priority_vad", options)
	if report.Status != "success" || vadCalls != 1 || report.LongCalls != 3 || len(report.Chunks) != 3 {
		t.Fatalf("priority_vad did not use the VAD plan: calls=%d report=%+v", vadCalls, report)
	}
	wantText := "we discussed the plan\nfor next week\nlast part"
	if report.LongText == nil || *report.LongText != wantText || report.LongTextHash == nil || *report.LongTextHash != HashText(wantText) {
		t.Errorf("overlap text was not deduplicated: %+v", report)
	}
	if report.Chunks[1].TextSHA256 == nil || *report.Chunks[1].TextSHA256 != HashText("for next week") {
		t.Errorf("chunk hash does not match the deduplicated contribution: %+v", report.Chunks[1])
	}
	if report.Chunks[0].CursorEndSample-report.Chunks[1].CursorStartSample < asr.SampleRate || report.Chunks[1].CursorStartSample != 302080 {
		t.Errorf("default overlap or padding was not preserved: %+v", report.Chunks)
	}
	if len(report.LongSegments) != 3 || report.LongSegments[1].ChunkIndex != 1 || report.LongSegments[1].StartMS != 19380 || report.LongSegments[1].EndMS != 20380 {
		t.Errorf("MTD timestamps were not shifted to absolute track time: %+v", report.LongSegments)
	}
	if report.LongSegments[0].Speaker != "S01" || report.LongSegments[1].Speaker != "S01" || report.LongSegments[2].Speaker != "S02" {
		t.Errorf("chunk-local speaker labels were rewritten: %+v", report.LongSegments)
	}
}

func TestMeetingInteractiveFakePriorityVADFailuresAreInvalid(t *testing.T) {
	for _, test := range []struct {
		name      string
		detector  vad.Model
		wantError string
	}{
		{name: "missing model", wantError: "VAD model is required"},
		{name: "model error", detector: interactiveVADFunc(func(context.Context, []byte, int) ([]vad.Segment, error) {
			return nil, errors.New("fake VAD failure")
		}), wantError: "fake VAD failure"},
		{name: "no speech", detector: interactiveVADFunc(func(context.Context, []byte, int) ([]vad.Segment, error) {
			return nil, nil
		}), wantError: "no speech regions"},
	} {
		t.Run(test.name, func(t *testing.T) {
			speech := &fakeInteractiveSpeech{shortBytes: len(fakeInteractivePCM(1)), failLongCall: -1}
			options := interactiveOptions{now: time.Now, waitForInteractive: func(context.Context) error { return nil }, detector: test.detector}
			report := runInteractiveMethod(context.Background(), fakeInteractiveConfig("diarize"), fakeInteractivePCM(45), fakeInteractivePCM(1), speech, "priority_vad", options)
			if report.Status != "invalid" || len(report.Errors) == 0 || !strings.Contains(strings.Join(report.Errors, ";"), test.wantError) {
				t.Fatalf("VAD failure became a plausible result: %+v", report)
			}
			if report.LongText != nil || report.LongTextHash != nil || speech.longCalls != 0 || speech.interactiveCalls != 0 {
				t.Errorf("VAD failure fell back to silent ASR output: %+v", report)
			}
		})
	}
}

func TestMeetingInteractiveFakeFailuresAreInvalid(t *testing.T) {
	for _, test := range []struct {
		name           string
		failLongCall   int
		longError      error
		interactiveErr error
		wantChunks     int
	}{
		{name: "long chunk", failLongCall: 1, longError: errors.New("fake long failure"), wantChunks: 2},
		{name: "interactive", failLongCall: -1, interactiveErr: errors.New("fake interactive failure"), wantChunks: 1},
	} {
		t.Run(test.name, func(t *testing.T) {
			arrived := make(chan struct{})
			speech := &fakeInteractiveSpeech{
				shortBytes: len(fakeInteractivePCM(1)), gates: map[int]<-chan struct{}{0: arrived}, failLongCall: test.failLongCall,
				longError: test.longError, interactiveErr: test.interactiveErr,
			}
			report := runInteractiveMethod(context.Background(), fakeInteractiveConfig("transcribe"), fakeInteractivePCM(65), fakeInteractivePCM(1), speech, "priority", arrivedInteractiveOptions(arrived))
			if report.Status != "invalid" || len(report.Errors) == 0 || len(report.Chunks) != test.wantChunks {
				t.Fatalf("failure became a plausible result: %+v", report)
			}
			if report.LongText != nil || report.LongTextHash != nil {
				t.Errorf("incomplete long text was published: %+v", report)
			}
			encoded, err := json.Marshal(report)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(encoded), `"long_text_hash":null`) {
				t.Errorf("unavailable text hash was not null: %s", encoded)
			}
		})
	}
}

func TestInteractiveConfigValidationAndStrictJSON(t *testing.T) {
	config := fakeInteractiveConfig("transcribe")
	config.SegmentLengthSeconds = 0
	config.Draft = DraftConfig{Enabled: true, Dir: "draft"}
	if err := config.Validate(); err != nil {
		t.Fatal(err)
	}
	if config.SegmentLengthSeconds != 25 || config.Draft.Bits != 8 {
		t.Errorf("defaults were not applied: %+v", config)
	}
	for _, change := range []func(*InteractiveConfig){
		func(c *InteractiveConfig) { c.Operation = "stream" },
		func(c *InteractiveConfig) { c.SegmentLengthSeconds = 19 },
		func(c *InteractiveConfig) { c.SegmentLengthSeconds = 31 },
		func(c *InteractiveConfig) { c.InteractiveAudio = "short.mp3" },
		func(c *InteractiveConfig) { c.Model.Dir = "" },
		func(c *InteractiveConfig) { c.VAD.Enabled = false },
		func(c *InteractiveConfig) { c.VAD.ModelDir = "" },
	} {
		invalid := fakeInteractiveConfig("transcribe")
		change(&invalid)
		if err := invalid.Validate(); err == nil {
			t.Errorf("invalid config was accepted: %+v", invalid)
		}
	}
	path := filepath.Join(t.TempDir(), "interactive.json")
	if err := os.WriteFile(path, []byte(`{"operation":"transcribe","long_audio":"long.wav","interactive_audio":"short.wav","output_dir":"out","model":{"id":"asr.fast","dir":"m"},"language_model":{"provider":"openai"}}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadInteractiveConfig(path); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("unknown or remote model config was accepted: %v", err)
	}
}
