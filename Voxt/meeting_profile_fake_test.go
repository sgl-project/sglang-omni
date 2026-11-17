package main

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
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/profile"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
)

type fakeProfileASR struct {
	transcribeCalls int
	diarizeCalls    int
	segments        int
	releases        *int
}

func (f *fakeProfileASR) Caps() asr.Caps { return asr.Caps{File: true} }
func (f *fakeProfileASR) Hold(context.Context) (func(), error) {
	return func() { *f.releases++ }, nil
}
func (f *fakeProfileASR) Transcribe(context.Context, asr.FileRequest) (asr.Result, error) {
	f.transcribeCalls++
	return asr.Result{Text: "plain", Acquire: time.Millisecond, Latency: 2 * time.Millisecond}, nil
}
func (f *fakeProfileASR) Diarize(context.Context, asr.FileRequest) (asr.Result, error) {
	f.diarizeCalls++
	result := asr.Result{Acquire: time.Millisecond, Latency: 2 * time.Millisecond}
	for index := 0; index < f.segments; index++ {
		result.Segments = append(result.Segments, asr.SpeakerSegment{
			Start: time.Duration(index) * time.Millisecond, End: time.Duration(index+1) * time.Millisecond,
			Speaker: "S01", Text: strings.Repeat("x", 200),
		})
	}
	return result, nil
}

type fakeProfileVAD struct {
	calls        int
	releases     *int
	speechError  error
	succeedAfter int
}

func (f *fakeProfileVAD) Speech(_ context.Context, pcm []byte, rate int) ([]vad.Segment, error) {
	f.calls++
	if f.speechError != nil && (f.succeedAfter == 0 || f.calls < f.succeedAfter) {
		return nil, f.speechError
	}
	return []vad.Segment{{Start: 0, End: len(pcm) / 2}}, nil
}
func (f *fakeProfileVAD) Open(context.Context) (vad.ModelStream, error) { return nil, nil }
func (f *fakeProfileVAD) Hold(context.Context) (func(), error) {
	return func() { *f.releases++ }, nil
}

type fakeProfileChat struct {
	correctionError bool
}

func (f fakeProfileChat) Chat(_ context.Context, request llm.Request) (llm.Response, error) {
	if len(request.Messages) > 0 && request.Messages[0].Role == llm.System {
		if f.correctionError {
			return llm.Response{}, errors.New("correction timeout")
		}
		var lines []struct {
			Index int `json:"i"`
		}
		if err := json.Unmarshal([]byte(request.Messages[len(request.Messages)-1].Content), &lines); err != nil {
			return llm.Response{}, err
		}
		var response strings.Builder
		response.WriteByte('[')
		for index, line := range lines {
			if index > 0 {
				response.WriteByte(',')
			}
			encoded, _ := json.Marshal(map[string]any{"i": line.Index, "text": strings.Repeat("y", 200)})
			response.Write(encoded)
		}
		response.WriteByte(']')
		return llm.Response{Text: response.String(), Latency: time.Millisecond, Attempts: 1}, nil
	}
	return llm.Response{Text: `{"transcript_summary":{"title":"title","content":"body"},"todo_list":[]}`, Latency: time.Millisecond, Attempts: 1}, nil
}

func fakeProfileConfig(mode string) profile.Config {
	return profile.Config{
		Mode: mode, Audio: "fake.wav", OutputDir: "unused",
		Model:         profile.ModelConfig{ID: "asr.moss", Dir: "fake-model"},
		VAD:           profile.VADConfig{Enabled: true, ModelDir: "fake-vad"},
		LanguageModel: profile.LLMConfig{Provider: "fake"},
	}
}

func TestMeetingProfileArtifacts(t *testing.T) {
	dir := t.TempDir()
	stop, err := startMeetingProfiles(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := stop(); err != nil {
		t.Fatal(err)
	}
	if err := stop(); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"cpu.pprof", "heap.pprof"} {
		info, err := os.Stat(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		if info.Size() == 0 {
			t.Errorf("%s is empty", name)
		}
	}
}

func TestMeetingProfileFakeImportStagesAndRequests(t *testing.T) {
	dir := t.TempDir()
	reference := filepath.Join(dir, "reference.txt")
	if err := os.WriteFile(reference, []byte("reference"), 0o600); err != nil {
		t.Fatal(err)
	}
	config := fakeProfileConfig("import")
	config.Reference = reference
	config.Cleanup = true
	config.Summary = true
	releases := 0
	speech := &fakeProfileASR{segments: 65, releases: &releases}
	detector := &fakeProfileVAD{releases: &releases}
	dependencies := defaultProfileDependencies()
	dependencies.decode = func(string) ([]byte, error) {
		return make([]byte, asr.SampleRate*2), nil
	}
	dependencies.newASR = func(profile.Config) (profileASR, error) { return speech, nil }
	dependencies.newVAD = func(profile.Config) (profileVAD, error) { return detector, nil }
	dependencies.newLLM = func(context.Context, profile.LLMConfig) (meeting.SummaryChat, error) {
		return fakeProfileChat{}, nil
	}
	databaseDir := ""
	dependencies.openStore = func(ctx context.Context, dir string) (store.Store, error) {
		databaseDir = dir
		return sqlitestore.Open(ctx, dir)
	}
	closed := false
	dependencies.closeRuntime = func() error { closed = true; return nil }
	recorder := profile.NewRecorder(config)
	report, runErr := runMeetingProfile(context.Background(), config, dependencies, recorder, nil, func() error { return nil })
	if runErr != nil {
		t.Fatal(runErr)
	}
	// MTD diarizes in pieces planned from speech regions, so the import
	// runs VAD once before the single diarization request.
	if report.Status != "success" || speech.diarizeCalls != 1 || speech.transcribeCalls != 0 || detector.calls != 1 || releases != 2 || !closed {
		t.Fatalf("status %s, ASR %+v, VAD %+v, releases %d, closed %v", report.Status, speech, detector, releases, closed)
	}
	counts := map[string]int{}
	for _, stage := range report.Stages {
		counts[stage.Name]++
	}
	if counts["correction"] != 5 || counts["summary"] != 3 || counts["diarization"] != 1 || counts["teardown"] != 1 || counts["vad"] != 1 {
		t.Errorf("stage counts %+v", counts)
	}
	if report.Evaluation.Value == nil || report.ASRTranscript.SHA256 == report.Transcript.SHA256 || !strings.Contains(report.Transcript.Text, "yyy") {
		t.Errorf("evaluation or corrected text missing: %+v", report)
	}
	if _, err := os.Stat(databaseDir); !os.IsNotExist(err) {
		t.Errorf("temporary database remains at %s", databaseDir)
	}
}

func TestMeetingProfileFakeSubtitlesWithoutVAD(t *testing.T) {
	config := fakeProfileConfig("subtitles")
	config.Model.ID = "asr.fast"
	config.VAD.Enabled = false
	config.SegmentLengthSeconds = 1.1
	releases := 0
	speech := &fakeProfileASR{releases: &releases}
	dependencies := defaultProfileDependencies()
	dependencies.decode = func(string) ([]byte, error) { return make([]byte, asr.SampleRate*4), nil }
	dependencies.newASR = func(profile.Config) (profileASR, error) { return speech, nil }
	recorder := profile.NewRecorder(config)
	report, runErr := runMeetingProfile(context.Background(), config, dependencies, recorder, nil, nil)
	if runErr != nil {
		t.Fatal(runErr)
	}
	if speech.transcribeCalls < 4 || speech.diarizeCalls != 0 {
		t.Errorf("ASR calls %+v", speech)
	}
	for _, stage := range report.Stages {
		if stage.Name == "vad" || stage.Name == "diarization" {
			t.Errorf("unexpected stage %+v", stage)
		}
	}
	if report.Evaluation.Value != nil || report.Evaluation.UnavailableReason == "" {
		t.Errorf("evaluation without reference %+v", report.Evaluation)
	}
}

type echoProfileChat struct{ text string }

func (c echoProfileChat) Chat(context.Context, llm.Request) (llm.Response, error) {
	return llm.Response{Text: c.text, Model: "actual-model-sentinel"}, errors.New(c.text)
}

func TestMeetingProfileRedactsLLMDetailsEverywhere(t *testing.T) {
	const (
		endpoint = "http://127.0.0.1:1/private-endpoint-sentinel"
		model    = "configured-model-sentinel"
		key      = "secret-key-sentinel"
		actual   = "actual-model-sentinel"
	)
	t.Setenv("PROFILE_REDACTION_KEY", key)
	config := fakeProfileConfig("import")
	config.LanguageModel = profile.LLMConfig{Provider: "custom", Endpoint: endpoint, Model: model, KeyEnv: "PROFILE_REDACTION_KEY"}
	recorder := profile.NewRecorder(config)
	if _, err := openProfileLLM(config.LanguageModel, recorder); err != nil {
		t.Fatal(err)
	}
	echo := endpoint + " " + model + " " + key + " " + actual
	_, runErr := recorder.WrapChat("summary", echoProfileChat{text: echo}).Chat(context.Background(), llm.Request{})
	report := recorder.Finish(runErr)
	report.Errors = append(report.Errors, echo)
	if report.Config.LanguageModel.Endpoint != "" || report.Config.LanguageModel.Model != "" {
		t.Fatalf("report config retained LLM details: %+v", report.Config.LanguageModel)
	}
	if report.Config.Model.ID != "asr.moss" {
		t.Fatalf("public ASR id was redacted: %q", report.Config.Model.ID)
	}
	path := filepath.Join(t.TempDir(), "result.json")
	if err := profile.WriteReport(path, report); err != nil {
		t.Fatal(err)
	}
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, sentinel := range []string{endpoint, model, key, actual} {
		if strings.Contains(string(content), sentinel) || strings.Contains(recorder.Clean(echo), sentinel) {
			t.Errorf("diagnostics retained %q", sentinel)
		}
	}
}

func fakeProfileVADDependencies(speech profileASR, detector profileVAD) profileDependencies {
	dependencies := defaultProfileDependencies()
	dependencies.decode = func(string) ([]byte, error) { return make([]byte, asr.SampleRate*2), nil }
	dependencies.newASR = func(profile.Config) (profileASR, error) { return speech, nil }
	dependencies.newVAD = func(profile.Config) (profileVAD, error) { return detector, nil }
	return dependencies
}

func TestMeetingProfileFakeVADBatchSuccessAndEnergyFallback(t *testing.T) {
	for _, test := range []struct {
		name         string
		succeedAfter int
		wantStatus   string
	}{
		{"whole track error then neural batch success", 2, "success"},
		{"energy fallback", 0, "failed"},
	} {
		t.Run(test.name, func(t *testing.T) {
			config := fakeProfileConfig("subtitles")
			config.Model.ID = "asr.fast"
			releases := 0
			speech := &fakeProfileASR{segments: 1, releases: &releases}
			detector := &fakeProfileVAD{releases: &releases, speechError: errors.New("neural VAD failed"), succeedAfter: test.succeedAfter}
			recorder := profile.NewRecorder(config)
			report, _ := runMeetingProfile(context.Background(), config, fakeProfileVADDependencies(speech, detector), recorder, nil, nil)
			if report.Status != test.wantStatus {
				t.Fatalf("status %s, warnings %+v", report.Status, report.Warnings)
			}
			if detector.calls == 0 {
				t.Fatal("ordinary final pass did not call VAD")
			}
		})
	}
}

func TestMeetingProfileFakeCorrectionFailureStillSummarizes(t *testing.T) {
	config := fakeProfileConfig("import")
	config.Cleanup = true
	config.Summary = true
	releases := 0
	speech := &fakeProfileASR{segments: 1, releases: &releases}
	dependencies := defaultProfileDependencies()
	dependencies.decode = func(string) ([]byte, error) { return make([]byte, asr.SampleRate*2), nil }
	dependencies.newASR = func(profile.Config) (profileASR, error) { return speech, nil }
	dependencies.newVAD = func(profile.Config) (profileVAD, error) {
		return &fakeProfileVAD{releases: &releases}, nil
	}
	dependencies.newLLM = func(context.Context, profile.LLMConfig) (meeting.SummaryChat, error) {
		return fakeProfileChat{correctionError: true}, nil
	}
	recorder := profile.NewRecorder(config)
	report, runErr := runMeetingProfile(context.Background(), config, dependencies, recorder, nil, nil)
	if runErr == nil || report.Status != "failed" {
		t.Fatalf("run error %v, report %+v", runErr, report)
	}
	summaryStages := 0
	for _, stage := range report.Stages {
		if stage.Name == "summary" {
			summaryStages++
		}
	}
	if summaryStages == 0 || report.Summary == nil {
		t.Error("summary did not continue after correction failed")
	}
}
