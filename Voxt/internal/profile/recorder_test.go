package profile

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

type fakeClock struct{ now time.Time }

func (c *fakeClock) Now() time.Time                 { return c.now }
func (c *fakeClock) Advance(duration time.Duration) { c.now = c.now.Add(duration) }

type fileFunc struct {
	transcribe func(context.Context, asr.FileRequest) (asr.Result, error)
}

func (f fileFunc) Caps() asr.Caps { return asr.Caps{File: true} }
func (f fileFunc) Transcribe(ctx context.Context, request asr.FileRequest) (asr.Result, error) {
	return f.transcribe(ctx, request)
}

func TestRecorderASRChunkOrderAndTiming(t *testing.T) {
	clock := &fakeClock{now: time.Unix(100, 0)}
	recorder := NewRecorder(validConfig())
	recorder.now = clock.Now
	calls := 0
	file := fileFunc{transcribe: func(context.Context, asr.FileRequest) (asr.Result, error) {
		calls++
		clock.Advance(40 * time.Millisecond)
		return asr.Result{Text: string(rune('a' + calls - 1)), Acquire: 5 * time.Millisecond, Latency: 30 * time.Millisecond}, nil
	}}
	wrapped := recorder.WrapFile("system", file)
	err := recorder.Measure("final_pass", 2*time.Second, func() error {
		request := asr.FileRequest{Audio: make([]byte, asr.SampleRate*2), Format: "pcm", SampleRate: asr.SampleRate}
		if _, err := wrapped.Transcribe(context.Background(), request); err != nil {
			return err
		}
		_, err := wrapped.Transcribe(context.Background(), request)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	report := recorder.Finish(nil)
	if len(report.Stages) != 3 {
		t.Fatalf("stages %+v", report.Stages)
	}
	for index, name := range []string{"asr_chunk", "asr_chunk", "final_pass"} {
		if report.Stages[index].Name != name {
			t.Errorf("stage %d is %s", index, report.Stages[index].Name)
		}
	}
	for index := 0; index < 2; index++ {
		stage := report.Stages[index]
		if stage.Index != index || stage.WallMS != 40 || stage.QueueAcquire.Value == nil || *stage.QueueAcquire.Value != 5 || stage.ResultLatency.Value == nil || *stage.ResultLatency.Value != 30 {
			t.Errorf("chunk %d: %+v", index, stage)
		}
		if stage.RTF == nil || *stage.RTF != 0.04 {
			t.Errorf("chunk RTF %+v", stage.RTF)
		}
		if stage.ASRFirstToken.Value != nil || stage.ASRFirstToken.UnavailableReason == "" {
			t.Errorf("first token must be unavailable: %+v", stage.ASRFirstToken)
		}
	}
	if report.FirstFinalText.Value == nil || *report.FirstFinalText.Value != 40 {
		t.Errorf("first final text %+v", report.FirstFinalText)
	}
}

type diarizerFunc struct {
	diarize func(context.Context, asr.FileRequest) (asr.Result, error)
}

func (d diarizerFunc) Caps() asr.Caps { return asr.Caps{File: true} }
func (d diarizerFunc) Diarize(ctx context.Context, request asr.FileRequest) (asr.Result, error) {
	return d.diarize(ctx, request)
}

func TestRecorderDiarizationDoesNotInventSegmentTimes(t *testing.T) {
	recorder := NewRecorder(validConfig())
	diarizer := diarizerFunc{diarize: func(context.Context, asr.FileRequest) (asr.Result, error) {
		return asr.Result{Segments: []asr.SpeakerSegment{
			{Start: time.Second, End: 2 * time.Second, Speaker: "S01", Text: "one"},
			{Start: 2 * time.Second, End: 3 * time.Second, Speaker: "S02", Text: "two"},
		}, Acquire: time.Millisecond, Latency: 20 * time.Millisecond}, nil
	}}
	if _, err := recorder.WrapDiarizer("system", diarizer).Diarize(context.Background(), asr.FileRequest{Audio: make([]byte, 32000), SampleRate: 16000}); err != nil {
		t.Fatal(err)
	}
	report := recorder.Finish(nil)
	stage := report.Stages[0]
	if stage.Name != "diarization" || stage.Text != "one\ntwo" || len(stage.Segments) != 2 {
		t.Fatalf("%+v", stage)
	}
	for _, segment := range stage.Segments {
		if segment.InferenceDuration.Value != nil || segment.InferenceDuration.UnavailableReason == "" {
			t.Errorf("segment inference was fabricated: %+v", segment)
		}
	}
}

type failingVAD struct{}

func (failingVAD) Speech(context.Context, []byte, int) ([]vad.Segment, error) {
	return nil, errors.New("VAD unavailable")
}
func (failingVAD) Open(context.Context) (vad.ModelStream, error) { return nil, nil }

func TestVADErrorTimingAndWarningStatus(t *testing.T) {
	recorder := NewRecorder(validConfig())
	_, err := recorder.WrapVAD(failingVAD{}).Speech(context.Background(), nil, 16000)
	if err == nil {
		t.Fatal("the model error was hidden")
	}
	report := recorder.Finish(nil)
	if report.Status != "success" || len(report.Warnings) != 0 || report.Stages[0].Error != "" {
		t.Fatalf("a retryable VAD error failed the run: %+v", report)
	}
	recorder.Warn("VAD used energy fallback")
	if report := recorder.Finish(nil); report.Status != "failed" || len(report.Warnings) != 1 {
		t.Fatalf("energy fallback warning was accepted: %+v", report)
	}
}

func TestStageErrorCannotBeSuccessfulRun(t *testing.T) {
	recorder := NewRecorder(validConfig())
	_ = recorder.Measure("decode", 0, func() error { return errors.New("decode failed") })
	if report := recorder.Finish(nil); report.Status != "failed" {
		t.Fatalf("stage error was accepted: %+v", report)
	}
}

type chatFunc struct {
	chat func(context.Context, llm.Request) (llm.Response, error)
}

func (c chatFunc) Chat(ctx context.Context, request llm.Request) (llm.Response, error) {
	return c.chat(ctx, request)
}

func TestRecorderCorrectionAndSummaryRequests(t *testing.T) {
	clock := &fakeClock{now: time.Unix(100, 0)}
	recorder := NewRecorder(validConfig())
	recorder.now = clock.Now
	const answer = `[{"i":0,"text":"answer"}]`
	chat := chatFunc{chat: func(context.Context, llm.Request) (llm.Response, error) {
		clock.Advance(7 * time.Millisecond)
		return llm.Response{Text: answer, Latency: 6 * time.Millisecond, Attempts: 2}, nil
	}}
	for _, kind := range []string{"correction", "correction", "summary", "summary"} {
		if _, err := recorder.WrapChat(kind, chat).Chat(context.Background(), llm.Request{}); err != nil {
			t.Fatal(err)
		}
	}
	report := recorder.Finish(nil)
	if len(report.Stages) != 4 {
		t.Fatalf("%+v", report.Stages)
	}
	for index, stage := range report.Stages {
		wantIndex := index % 2
		if stage.Index != wantIndex || stage.WallMS != 7 || stage.Attempts != 2 || stage.TextSHA256 != HashText(answer) {
			t.Errorf("stage %d: %+v", index, stage)
		}
	}
}

func TestCorrectionRejectsMalformedJSON(t *testing.T) {
	for _, raw := range []string{"not json", "null", `[{"i":1}]`} {
		recorder := NewRecorder(validConfig())
		chat := chatFunc{chat: func(context.Context, llm.Request) (llm.Response, error) {
			return llm.Response{Text: raw}, nil
		}}
		if _, err := recorder.WrapChat("correction", chat).Chat(context.Background(), llm.Request{}); err == nil {
			t.Errorf("malformed correction %q was accepted", raw)
		}
		if report := recorder.Finish(nil); report.Status != "failed" {
			t.Errorf("malformed correction %q produced %s", raw, report.Status)
		}
	}
}

func TestOptionalJSONAndNativeWarning(t *testing.T) {
	encoded, err := json.Marshal(Unavailable[int64]("not measurable"))
	if err != nil {
		t.Fatal(err)
	}
	if string(encoded) != `{"value":null,"unavailable_reason":"not measurable"}` {
		t.Fatalf("%s", encoded)
	}
	recorder := NewRecorder(validConfig())
	path := filepath.Join(t.TempDir(), "native.log")
	log, err := OpenNativeLog(path, recorder)
	if err != nil {
		t.Fatal(err)
	}
	log.Write("warning", "request timing wall_ms=12")
	if report := recorder.Finish(nil); report.Status != "success" {
		t.Fatalf("routine native timing was rejected: %+v", report)
	}
	log.Write("warning", "model did not start with its draft: starting without it")
	if err := log.Close(); err != nil {
		t.Fatal(err)
	}
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(content), "starting without it") {
		t.Fatalf("native log %q", content)
	}
	if report := recorder.Finish(nil); report.Status != "failed" {
		t.Fatalf("native warning was accepted: %+v", report)
	}
}

func TestReportDoesNotPersistEnvironmentSecret(t *testing.T) {
	const secret = "profile-test-secret-value"
	t.Setenv("PROFILE_TEST_KEY", secret)
	recorder := NewRecorder(validConfig())
	path := filepath.Join(t.TempDir(), "result.json")
	if err := WriteReport(path, recorder.Finish(nil)); err != nil {
		t.Fatal(err)
	}
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(content), secret) {
		t.Fatal("the environment secret reached result.json")
	}
	if !strings.Contains(string(content), `"value": null`) {
		t.Fatal("unavailable metrics were not written as null")
	}
}

func TestCorrectionResponseCategoriesAndBoundedRaw(t *testing.T) {
	tests := []struct {
		name, raw, want string
	}{
		{"bare", `[{"i":0,"text":"answer"}]`, "successful_edit"},
		{"fenced", "```json\n[{\"i\":0,\"text\":\"answer\"}]\n```", "successful_edit"},
		{"embedded", "before [{\"i\":0,\"text\":\"answer\"}] after", "successful_edit"},
		{"empty", `[]`, "valid_noop"},
		{"malformed", "not json", "malformed"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			recorder := NewRecorder(validConfig())
			chat := chatFunc{chat: func(context.Context, llm.Request) (llm.Response, error) {
				return llm.Response{Text: test.raw}, nil
			}}
			_, err := recorder.WrapChat("correction", chat).Chat(context.Background(), llm.Request{})
			if test.want == "malformed" && err == nil {
				t.Fatal("malformed response was accepted")
			}
			if test.want != "malformed" && err != nil {
				t.Fatal(err)
			}
			report := recorder.Finish(nil)
			if got := report.Stages[0].ParserCategory; got != test.want {
				t.Fatalf("category %q, want %q", got, test.want)
			}
		})
	}
	recorder := NewRecorder(validConfig())
	const secret = "profile-test-secret-value"
	raw := `[{"i":0,"text":"answer"}]` + strings.Repeat(" ", 7990) + secret + "tail"
	chat := chatFunc{chat: func(context.Context, llm.Request) (llm.Response, error) {
		return llm.Response{Text: raw}, nil
	}}
	if _, err := recorder.WrapChat("correction", chat).Chat(context.Background(), llm.Request{}); err != nil {
		t.Fatal(err)
	}
	stage := recorder.Finish(nil).Stages[0]
	if len(stage.Text) > 8000 || strings.Contains(stage.Text, secret) {
		t.Fatalf("raw response was not safely bounded: length=%d", len(stage.Text))
	}
	if stage.TextSHA256 != HashText(stage.Text) {
		t.Fatalf("stored text hash does not match the bounded raw text: %+v", stage)
	}
}

func TestCorrectionTransportErrorCategory(t *testing.T) {
	recorder := NewRecorder(validConfig())
	chat := chatFunc{chat: func(context.Context, llm.Request) (llm.Response, error) {
		return llm.Response{}, errors.New("timeout")
	}}
	if _, err := recorder.WrapChat("correction", chat).Chat(context.Background(), llm.Request{}); err == nil {
		t.Fatal("transport error was accepted")
	}
	if got := recorder.Finish(nil).Stages[0].ParserCategory; got != "transport_error" {
		t.Fatalf("category %q, want transport_error", got)
	}
}

func TestSummaryContentHashIgnoresGeneratedAt(t *testing.T) {
	first := NewRecorder(validConfig())
	first.SetTranscripts("asr", "final", `{"title":"title","body":"body","todos":["one"],"generated_at":1}`)
	second := NewRecorder(validConfig())
	second.SetTranscripts("asr", "final", `{"title":"title","body":"body","todos":["one"],"generated_at":2}`)
	if first.report.SummaryContentSHA256 != second.report.SummaryContentSHA256 {
		t.Fatal("generated_at changed the summary content hash")
	}
	if first.report.Summary.SHA256 == second.report.Summary.SHA256 {
		t.Fatal("the persisted artifact hash did not retain generated_at")
	}
	second.SetTranscripts("asr", "final", `{"title":"title","body":"changed","todos":["one"],"generated_at":2}`)
	if first.report.SummaryContentSHA256 == second.report.SummaryContentSHA256 {
		t.Fatal("changed body did not change the summary content hash")
	}
}

func TestFirstASRHookRunsOnce(t *testing.T) {
	recorder := NewRecorder(validConfig())
	fires := 0
	recorder.SetFirstASRHook(func() { fires++ })
	file := fileFunc{transcribe: func(context.Context, asr.FileRequest) (asr.Result, error) {
		return asr.Result{Text: "hello"}, nil
	}}
	wrapped := recorder.WrapFile("system", file)
	request := asr.FileRequest{Audio: make([]byte, asr.SampleRate), Format: "pcm", SampleRate: asr.SampleRate}
	if _, err := wrapped.Transcribe(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	if _, err := wrapped.Transcribe(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	if fires != 1 {
		t.Fatalf("first-ASR hook fired %d times, want 1", fires)
	}
}
