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

// TestDraft: a speech model with a draft passes its folder to the server
// after the model's flags; the same model without one is another server.
func TestDraft(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, nil)
	draft := filepath.Join(f.dir, "draft")
	if err := os.MkdirAll(draft, 0o755); err != nil {
		t.Fatal(err)
	}
	req := asr.FileRequest{Audio: tone(2*time.Second, 0.3), Format: "pcm", SampleRate: 16000}
	if _, err := f.asr(t, ASRFast).WithDraft(draft).Transcribe(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	launches := f.records("launch")
	if len(launches) != 1 || !strings.HasSuffix(strings.Join(launches[0].Args, " "), "--draft-model-path "+draft) {
		t.Fatalf("launches %+v", launches)
	}
	if _, err := f.asr(t, ASRFast).Transcribe(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	launches = f.records("launch")
	if len(launches) != 2 || slices.Contains(launches[1].Args, "--draft-model-path") {
		t.Errorf("without the draft: %+v", launches)
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
