package profile

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

type overlapASRFunc func(context.Context, asr.FileRequest) (asr.Result, error)

func (f overlapASRFunc) Caps() asr.Caps { return asr.Caps{File: true} }
func (f overlapASRFunc) Transcribe(ctx context.Context, request asr.FileRequest) (asr.Result, error) {
	return f(ctx, request)
}

type overlapVADFunc func(context.Context, []byte, int) ([]vad.Segment, error)

func (f overlapVADFunc) Speech(ctx context.Context, pcm []byte, rate int) ([]vad.Segment, error) {
	return f(ctx, pcm, rate)
}
func (f overlapVADFunc) Open(context.Context) (vad.ModelStream, error) { return nil, nil }

type overlapLLMFunc func(context.Context, simulatedLLMRequest) (string, error)

func (f overlapLLMFunc) Call(ctx context.Context, request simulatedLLMRequest) (string, error) {
	return f(ctx, request)
}

func fakeOverlapConfig(mode string) OverlapConfig {
	return OverlapConfig{
		Mode: mode, Audio: "fake.wav", OutputDir: "unused", Model: ModelConfig{ID: "asr.fast", Dir: "fake-model"},
		SegmentLengthSeconds: 2.1, SimulatedLLM: SimulatedLLMConfig{LatencyMS: 3},
	}
}

func fakeOverlapPCM() []byte {
	pcm := make([]byte, asr.SampleRate*2*6)
	for offset := 0; offset < len(pcm); offset += 2 {
		binary.LittleEndian.PutUint16(pcm[offset:], uint16(offset/2))
	}
	return pcm
}

func fakeOverlapOptions(config OverlapConfig) overlapOptions {
	options := defaultOverlapOptions(config)
	options.waitUntil = func(context.Context, time.Time) error { return nil }
	return options
}

func fakeOverlapText(pcm []byte) string {
	return "text-" + HashText(string(pcm))[:12]
}

func fakeOrderedASR() overlapASRFunc {
	return func(_ context.Context, request asr.FileRequest) (asr.Result, error) {
		return asr.Result{Text: fakeOverlapText(request.Audio)}, nil
	}
}

func TestMeetingOverlapFakeThreeMethodsAndStructuredOutput(t *testing.T) {
	config := fakeOverlapConfig("meeting")
	reports := runOverlapWithOptions(context.Background(), config, fakeOverlapPCM(), fakeOrderedASR(), nil, fakeOverlapOptions(config))
	if len(reports) != 3 {
		t.Fatalf("got %d reports", len(reports))
	}
	for _, report := range reports {
		if report.Status != "success" {
			t.Fatalf("%s failed: %+v", report.Method, report.Errors)
		}
		if !report.SimulatedLLM || report.StopToFinalMS.Value == nil || report.Timing.ASRAfterStopMS.Value == nil || report.Timing.LLMAfterStopMS.Value == nil {
			t.Errorf("%s is missing scheduling measurements: %+v", report.Method, report)
		}
		if report.Summary == nil || report.Structure.Document == nil || report.ComparisonToOneShot == nil {
			t.Errorf("%s is missing summary, structure, or baseline comparison", report.Method)
		}
	}
	oneShot, serial, overlap := &reports[0], &reports[1], &reports[2]
	if oneShot.Timing.ASRCalls != 1 || serial.Timing.ASRCalls <= 1 || overlap.Timing.ASRCalls != serial.Timing.ASRCalls {
		t.Fatalf("ASR calls are not comparable: one_shot=%d serial=%d overlap=%d", oneShot.Timing.ASRCalls, serial.Timing.ASRCalls, overlap.Timing.ASRCalls)
	}
	segments := serial.Timing.ASRCalls
	if serial.Timing.LLMCalls != 2*segments+1 || overlap.Timing.LLMCalls != serial.Timing.LLMCalls || oneShot.Timing.LLMCalls != 3 {
		t.Errorf("cleanup/map/reduce calls: one_shot=%d serial=%d overlap=%d", oneShot.Timing.LLMCalls, serial.Timing.LLMCalls, overlap.Timing.LLMCalls)
	}
	if serial.Transcript.Text != overlap.Transcript.Text || serial.Transcript.SHA256 != overlap.Transcript.SHA256 {
		t.Errorf("ordered overlap text differs from serial:\nserial=%q\noverlap=%q", serial.Transcript.Text, overlap.Transcript.Text)
	}
	for index, part := range overlap.Structure.Document.Cleanup {
		if part.Index != index || part.Text != serial.Structure.Document.Cleanup[index].Text {
			t.Errorf("cleanup part %d is reordered or changed: %+v", index, part)
		}
	}
	if overlap.ComparisonToOneShot.StructureEqual || overlap.ComparisonToOneShot.CleanupPartCountDelta != segments-1 {
		t.Errorf("one-shot structure comparison is wrong: %+v", overlap.ComparisonToOneShot)
	}
	if overlap.ComparisonToOneShot.SummaryEqual == nil || oneShot.ComparisonToOneShot.SummaryEqual == nil || !oneShot.ComparisonToOneShot.TextEqual {
		t.Error("summary or self comparison is unavailable")
	}
	if serial.SchedulingEffects == nil || serial.SchedulingEffects.ASRChunkingAfterStopSavedMS.Value == nil || overlap.SchedulingEffects == nil || overlap.SchedulingEffects.LLMOverlapAfterStopSavedMS.Value == nil {
		t.Error("ASR chunking and LLM overlap deltas are not separate")
	}
	contextStages := 0
	for _, stage := range overlap.Stages {
		if stage.Name == "llm_cleanup" && stage.PreviousContextSHA256 != "" {
			contextStages++
		}
	}
	if contextStages != segments-1 {
		t.Errorf("got %d previous-context requests, want %d", contextStages, segments-1)
	}
}

func TestMeetingOverlapFakeDictationCleanupOnlyAndJSON(t *testing.T) {
	config := fakeOverlapConfig("dictation")
	report := runOverlapMethod(context.Background(), config, fakeOverlapPCM(), fakeOrderedASR(), nil, "one_shot", fakeOverlapOptions(config))
	if report.Status != "success" || report.Timing.LLMCalls != 1 || report.Summary != nil || report.SummaryUnavailableReason == "" {
		t.Fatalf("dictation did more than cleanup: %+v", report)
	}
	for _, stage := range report.Stages {
		if strings.Contains(stage.Name, "summary") {
			t.Errorf("unexpected dictation summary stage: %+v", stage)
		}
	}
	output := t.TempDir()
	if err := WriteOverlapReports(output, []OverlapReport{report}); err != nil {
		t.Fatal(err)
	}
	content, err := os.ReadFile(filepath.Join(output, "one_shot", "result.json"))
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(content, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded["simulated_llm"] != true || decoded["summary"] != nil || decoded["transcript"] == nil {
		t.Errorf("JSON does not expose the simulated boundary and artifacts: %s", content)
	}
}

func TestMeetingOverlapFakeVADPlansAndRejectsFailure(t *testing.T) {
	config := fakeOverlapConfig("dictation")
	config.VAD = VADConfig{Enabled: true, ModelDir: "fake-vad"}
	calls := 0
	detector := overlapVADFunc(func(_ context.Context, pcm []byte, _ int) ([]vad.Segment, error) {
		calls++
		return []vad.Segment{{Start: 0, End: len(pcm) / 2}}, nil
	})
	report := runOverlapMethod(context.Background(), config, fakeOverlapPCM(), fakeOrderedASR(), detector, "serial", fakeOverlapOptions(config))
	if report.Status != "success" || calls != 1 {
		t.Fatalf("VAD was not used in segment planning: status=%s calls=%d errors=%v", report.Status, calls, report.Errors)
	}
	failed := runOverlapMethod(context.Background(), config, fakeOverlapPCM(), fakeOrderedASR(), overlapVADFunc(func(context.Context, []byte, int) ([]vad.Segment, error) {
		return nil, errors.New("fake VAD failure")
	}), "serial", fakeOverlapOptions(config))
	if failed.Status != "invalid" || failed.StoppedAt != nil || failed.BacklogAtStop.ASRSegments.Value != nil {
		t.Fatalf("VAD failure did not invalidate before recording: %+v", failed)
	}
}

func TestMeetingOverlapFakeFailuresAreInvalidAndNull(t *testing.T) {
	config := fakeOverlapConfig("dictation")
	for _, test := range []struct {
		name   string
		speech asr.FileTranscriber
		llm    overlapLLM
	}{
		{"ASR", overlapASRFunc(func(context.Context, asr.FileRequest) (asr.Result, error) {
			return asr.Result{}, errors.New("fake ASR failure")
		}), nil},
		{"LLM", fakeOrderedASR(), overlapLLMFunc(func(context.Context, simulatedLLMRequest) (string, error) {
			return "", errors.New("fake LLM failure")
		})},
		{"malformed LLM", fakeOrderedASR(), overlapLLMFunc(func(context.Context, simulatedLLMRequest) (string, error) {
			return `[{"i":0}]`, nil
		})},
	} {
		t.Run(test.name, func(t *testing.T) {
			options := fakeOverlapOptions(config)
			if test.llm != nil {
				options.newLLM = func() overlapLLM { return test.llm }
			}
			report := runOverlapMethod(context.Background(), config, fakeOverlapPCM(), test.speech, nil, "one_shot", options)
			if report.Status != "invalid" || len(report.Errors) == 0 || report.StopToFinalMS.Value != nil || report.Transcript != nil || report.Structure.Document != nil || report.Structure.SHA256 != nil {
				t.Fatalf("failure became a plausible result: %+v", report)
			}
			encoded, err := json.Marshal(report)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(encoded), `"stop_to_final_ms":{"value":null`) {
				t.Errorf("unavailable stop-to-final was not null: %s", encoded)
			}
		})
	}
}

func TestSimulatedLLMDeterministicLatencyCallsAndText(t *testing.T) {
	model := newSimulatedLLM(25 * time.Millisecond)
	var sleeps []time.Duration
	model.sleep = func(_ context.Context, duration time.Duration) error {
		sleeps = append(sleeps, duration)
		return nil
	}
	request := simulatedLLMRequest{Operation: simulatedCleanup, Parts: []SimulatedPart{{Index: 7, Text: "  ordered text  "}}}
	first, err := model.Call(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	second, err := model.Call(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if first != second || first != `[{"i":7,"text":"ordered text"}]` || len(sleeps) != 2 || sleeps[0] != 25*time.Millisecond || sleeps[1] != 25*time.Millisecond {
		t.Errorf("responses %q/%q, sleeps %v", first, second, sleeps)
	}
}
