package main

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/profile"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

type leaseTestASR struct {
	*fakeProfileASR
}

func (f *leaseTestASR) Hold(ctx context.Context) (func(), error) {
	*f.events = append(*f.events, "asr hold")
	return f.fakeProfileASR.Hold(ctx)
}

func TestPerTrackLeaseOrder(t *testing.T) {
	events := []string{}
	releases := 0
	speech := &leaseTestASR{fakeProfileASR: &fakeProfileASR{releases: &releases, events: &events}}
	detector := &fakeProfileVAD{releases: &releases, events: &events}
	monitor := profile.StartResourceMonitor(context.Background(), profile.SelfAndNative(func() []int { return nil }))
	defer monitor.Stop(context.Background())
	var release func()
	vad := &perTrackVAD{
		model: detector, speech: speech, monitor: monitor, release: &release,
		unloadASR: func(context.Context) error {
			events = append(events, "asr unload")
			return nil
		},
	}
	if _, err := vad.Speech(context.Background(), make([]byte, 32000), 16000); err != nil {
		t.Fatal(err)
	}
	if _, err := vad.Speech(context.Background(), make([]byte, 32000), 16000); err != nil {
		t.Fatal(err)
	}
	want := []string{"vad speech", "asr hold", "asr release", "asr unload", "vad speech", "asr hold"}
	if !reflect.DeepEqual(events, want) {
		t.Fatalf("events %v, want %v", events, want)
	}
}

func TestMeetingProfileDefersEvaluation(t *testing.T) {
	reference := filepath.Join(t.TempDir(), "reference.txt")
	if err := os.WriteFile(reference, []byte("reference"), 0o600); err != nil {
		t.Fatal(err)
	}
	config := fakeProfileConfig("import")
	config.Reference = reference
	config.DeferEvaluation = true
	releases := 0
	speech := &fakeProfileASR{segments: 1, releases: &releases}
	detector := &fakeProfileVAD{releases: &releases}
	dependencies := defaultProfileDependencies()
	dependencies.decode = func(context.Context, string) ([]byte, error) {
		return make([]byte, asr.SampleRate*2), nil
	}
	dependencies.newASR = func(profile.Config) (profileASR, error) { return speech, nil }
	dependencies.newVAD = func(profile.Config) (profileVAD, error) { return detector, nil }
	dependencies.openStore = func(ctx context.Context, dir string) (store.Store, error) {
		return sqlitestore.Open(ctx, dir)
	}
	recorder := profile.NewRecorder(config)
	report, runErr := runMeetingProfile(context.Background(), config, dependencies, recorder, nil, func() error { return nil })
	if runErr != nil {
		t.Fatal(runErr)
	}
	if report.Evaluation.Value != nil {
		t.Fatalf("evaluation ran inside the timed stages: %+v", report.Evaluation)
	}
	evaluation := profile.Evaluate("reference", report.ASRTranscript.Text, report.Transcript.Text)
	if evaluation.Value == nil {
		t.Fatal("evaluation was not computed after the run")
	}
}

func TestProfileFakeLLMIsDeterministic(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "bin"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "bin", "qwen3_asr_server"), []byte("fake"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("VOXTNEW_OMNI_RUNTIME", dir)
	config := fakeProfileConfig("import")
	recorder := profile.NewRecorder(config)
	nativeLog, err := profile.OpenNativeLog(filepath.Join(t.TempDir(), "native.log"), recorder)
	if err != nil {
		t.Fatal(err)
	}
	defer nativeLog.Close()
	dependencies, err := realProfileDependencies(config, nativeLog, recorder)
	if err != nil {
		t.Fatal(err)
	}
	defer dependencies.closeRuntime()
	chat, err := dependencies.newLLM(context.Background(), config.LanguageModel)
	if err != nil {
		t.Fatal(err)
	}
	response, err := chat.Chat(context.Background(), llm.Request{})
	if err != nil {
		t.Fatal(err)
	}
	if response.Text != `{"transcript_summary":{"title":"title","content":"body"},"todo_list":[]}` || response.Latency != time.Millisecond || response.Attempts != 1 {
		t.Fatalf("unexpected fake response: %+v", response)
	}
}
