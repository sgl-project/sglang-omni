package omni

import (
	"context"
	"os"
	"path/filepath"
	"slices"
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

// TestDraftRefused: a server from before --draft-model-path refuses the
// flag; the model starts again without it and the log says so.
func TestDraftRefused(t *testing.T) {
	f := newFake(t, omnitest.Opts{NoDraft: true}, nil)
	a := f.asr(t, ASRFast).WithDraft(filepath.Join(f.dir, "draft"))
	if _, err := a.Transcribe(context.Background(), asr.FileRequest{Audio: tone(2*time.Second, 0.3), Format: "pcm"}); err != nil {
		t.Fatal(err)
	}
	launches := f.records("launch") // the refused start recorded nothing
	if len(launches) != 1 || slices.Contains(launches[0].Args, "--draft-model-path") {
		t.Errorf("launches %+v", launches)
	}
	f.waitLogged(t, "starting without it")
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
