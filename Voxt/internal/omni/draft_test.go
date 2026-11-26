package omni

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni/omnitest"
)

// TestDraft checks each model's draft flags and that disabling its draft
// uses a separate server.
func TestDraft(t *testing.T) {
	req := asr.FileRequest{Audio: tone(2*time.Second, 0.3), Format: "pcm", SampleRate: 16000}
	for _, model := range []struct {
		id         string
		draftFlags []string
	}{
		{ASRFast, []string{"--draft-bits", "8"}},
		{ASRMoss, nil},
		{ASRWhisperLarge, nil},
	} {
		t.Run(model.id, func(t *testing.T) {
			f := newFake(t, omnitest.Opts{}, withServers(t, "moss_transcribe_diarize_server", "whisper_server"))
			draft := filepath.Join(f.dir, "draft")
			if err := os.MkdirAll(draft, 0o755); err != nil {
				t.Fatal(err)
			}
			a := f.asr(t, model.id)
			if _, err := a.WithDraft(draft).Transcribe(context.Background(), req); err != nil {
				t.Fatal(err)
			}
			launches := f.records("launch")
			if len(launches) != 1 {
				t.Fatalf("launches %+v", launches)
			}
			args := launches[0].Args
			want := append([]string{"--draft-model-path", draft}, model.draftFlags...)
			if start := slices.Index(args, "--draft-model-path"); start < 0 || !slices.Equal(args[start:], want) {
				t.Errorf("launch args %q, want suffix %q", args, want)
			}
			if _, err := a.Transcribe(context.Background(), req); err != nil {
				t.Fatal(err)
			}
			launches = f.records("launch")
			if len(launches) != 2 || slices.Contains(launches[1].Args, "--draft-model-path") {
				t.Errorf("without the draft: %+v", launches)
			}
		})
	}
}

func TestDraftBitsAndTuning(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, nil)
	cacheMB := 0
	base := f.asr(t, ASRFast)
	tuned, err := base.WithTuning(Tuning{DecodeInterval: 250 * time.Millisecond, CacheLimitMB: &cacheMB})
	if err != nil {
		t.Fatal(err)
	}
	draft := filepath.Join(f.dir, "draft")
	if err := os.MkdirAll(draft, 0o755); err != nil {
		t.Fatal(err)
	}
	a, err := tuned.WithDraftBits(draft, 4)
	if err != nil {
		t.Fatal(err)
	}
	res, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: tone(time.Second, 0.3), Format: "pcm"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Acquire <= 0 || res.Latency <= 0 {
		t.Errorf("acquire %v, result latency %v", res.Acquire, res.Latency)
	}
	launches := f.records("launch")
	if len(launches) != 1 {
		t.Fatalf("launches %+v", launches)
	}
	args := strings.Join(launches[0].Args, " ")
	for _, want := range []string{"--decode-interval-ms 250", "--cache-limit-mb 0", "--draft-bits 4"} {
		if !strings.Contains(args, want) {
			t.Errorf("launch args %q lack %q", args, want)
		}
	}
	if got := base.Spec().ASR.ChunkSeconds; got != 60 {
		t.Errorf("the model default changed to %v", got)
	}
	if _, err := base.WithDraftBits(filepath.Join(f.dir, "draft"), 16); err == nil {
		t.Error("16-bit draft was accepted")
	}
	if _, err := f.asr(t, ASRWhisperTurbo).WithTuning(Tuning{DecodeInterval: time.Second}); err == nil {
		t.Error("an unsupported decode interval was ignored")
	}
}

func TestTuningChangesRestartServer(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, nil)
	base := f.asr(t, ASRFast)
	zero, defaultCache := 0, 512
	draft := filepath.Join(f.dir, "draft")
	if err := os.MkdirAll(draft, 0o755); err != nil {
		t.Fatal(err)
	}
	first, err := base.WithTuning(Tuning{DecodeInterval: 250 * time.Millisecond, CacheLimitMB: &zero})
	if err != nil {
		t.Fatal(err)
	}
	first, err = first.WithDraftBits(draft, 4)
	if err != nil {
		t.Fatal(err)
	}
	second, err := base.WithTuning(Tuning{DecodeInterval: 500 * time.Millisecond, CacheLimitMB: &defaultCache})
	if err != nil {
		t.Fatal(err)
	}
	second, err = second.WithDraftBits(draft, 4)
	if err != nil {
		t.Fatal(err)
	}
	request := asr.FileRequest{Audio: tone(time.Second, 0.3), Format: "pcm"}
	if _, err := first.Transcribe(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	if _, err := second.Transcribe(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	if _, err := second.Transcribe(context.Background(), request); err != nil {
		t.Fatal(err)
	}
	launches := f.records("launch")
	if len(launches) != 2 {
		t.Fatalf("two tunings and one reuse made %d launches: %+v", len(launches), launches)
	}
	args := strings.Join(launches[1].Args, " ")
	if !strings.Contains(args, "--decode-interval-ms 500") || !strings.Contains(args, "--cache-limit-mb 512") || !strings.Contains(args, "--draft-bits 4") {
		t.Errorf("second launch args %q", args)
	}
	if !gone(launches[0].PID) {
		t.Errorf("the first tuning's server still runs")
	}
}

func TestDraftDeletedOldProvider(t *testing.T) {
	req := asr.FileRequest{Audio: tone(2*time.Second, 0.3), Format: "pcm", SampleRate: 16000}
	for _, model := range []string{ASRFast, ASRMoss, ASRWhisperLarge} {
		t.Run(model, func(t *testing.T) {
			f := newFake(t, omnitest.Opts{}, withServers(t, "moss_transcribe_diarize_server", "whisper_server"))
			a := f.asr(t, model)
			draft := filepath.Join(f.dir, "draft")
			if err := os.MkdirAll(draft, 0o755); err != nil {
				t.Fatal(err)
			}
			a = a.WithDraft(draft)
			if _, err := a.Transcribe(context.Background(), req); err != nil {
				t.Fatal(err)
			}
			if err := os.RemoveAll(draft); err != nil {
				t.Fatal(err)
			}
			f.rt.Unload(model)
			if _, err := a.Transcribe(context.Background(), req); err != nil {
				t.Fatal(err)
			}
			launches := f.records("launch")
			if len(launches) != 2 || slices.Contains(launches[1].Args, "--draft-model-path") {
				t.Errorf("launches after the draft was deleted: %+v", launches)
			}
		})
	}
}

// TestDraftRefused: a server from before --draft-model-path refuses the
// flag; the model starts again without it and the log says so.
func TestDraftRefused(t *testing.T) {
	f := newFake(t, omnitest.Opts{NoDraft: true}, nil)
	draft := filepath.Join(f.dir, "draft")
	if err := os.MkdirAll(draft, 0o755); err != nil {
		t.Fatal(err)
	}
	a := f.asr(t, ASRFast).WithDraft(draft)
	if _, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: tone(2*time.Second, 0.3), Format: "pcm"}); err != nil {
		t.Fatal(err)
	}
	launches := f.records("launch") // the refused start recorded nothing
	if len(launches) != 1 || slices.Contains(launches[0].Args, "--draft-model-path") {
		t.Errorf("launches %+v", launches)
	}
	f.waitLogged(t, "did not start with its draft")
}

// TestDraftDeleted: deleting the draft unloads its model's server
// (models_omni.go unloadOmni), and the model without the draft starts
// only once that server's process exited, as after a switch of models:
// the weights are never in memory twice.
func TestDraftDeleted(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, nil)
	ctx := context.Background()
	a := f.asr(t, ASRFast)
	draft := filepath.Join(f.dir, "draft")
	if err := os.MkdirAll(draft, 0o755); err != nil {
		t.Fatal(err)
	}
	l, err := f.rt.acquire(ctx, a.spec, a.dir, draft) // a dictation in progress
	if err != nil {
		t.Fatal(err)
	}
	f.rt.Unload(ASRFast) // the draft is deleted: its server stops after the use
	started := make(chan error, 1)
	go func() {
		l2, err := f.rt.acquire(ctx, a.spec, a.dir, "")
		if err == nil {
			l2.release()
		}
		started <- err
	}()
	time.Sleep(300 * time.Millisecond)
	if n := len(f.records("launch")); n != 1 {
		t.Fatalf("%d launches while the drafted server still runs", n)
	}
	l.release()
	if err := <-started; err != nil {
		t.Fatal(err)
	}
	launches := f.records("launch")
	if len(launches) != 2 || slices.Contains(launches[1].Args, "--draft-model-path") {
		t.Fatalf("launches %+v", launches)
	}
	if !gone(launches[0].PID) {
		t.Errorf("the drafted server (pid %d) still runs", launches[0].PID)
	}
}

func TestTranscribeKeepsDraftCounters(t *testing.T) {
	f := newFake(t, omnitest.Opts{DraftCounters: true}, nil)
	res, err := f.asr(t, ASRFast).WithDraft(filepath.Join(f.dir, "draft")).Transcribe(context.Background(), asr.FileRequest{Audio: tone(2*time.Second, 0.3), Format: "pcm"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Draft == nil {
		t.Fatal("the draft counters were dropped")
	}
	if *res.Draft != (asr.DraftStats{VerifyRounds: 3, Proposed: 12, Accepted: 8}) {
		t.Errorf("draft counters: %+v", *res.Draft)
	}
}
