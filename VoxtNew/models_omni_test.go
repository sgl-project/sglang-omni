package main

import (
	"context"
	"errors"
	fammodels "github.com/hehehai/actnowkit/models"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/omni/omnitest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// fakeOmni makes this test binary the app's Omni runtime (omnitest) and
// returns the fake's log.
func fakeOmni(t *testing.T, o omnitest.Opts) string {
	t.Helper()
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	o.Log = filepath.Join(t.TempDir(), "omni.jsonl")
	old := omniLocate
	omniLocate = func() (string, error) { return self, nil }
	t.Cleanup(func() { omniLocate = old })
	k, v, _ := strings.Cut(omnitest.Env(o)[0], "=")
	t.Setenv(k, v)
	return o.Log
}

// TestSessionOmni: a Transcribe session through the app's own local
// models (the Library and localOpeners) over the fake Omni server: the
// live preview while recording, then the Final; Silero (installed with
// the speech model) gates the stream; deleting the model unloads it;
// quitting stops the servers.
func TestSessionOmni(t *testing.T) {
	log := fakeOmni(t, omnitest.Opts{Text: "来自 Omni 的文本"})
	ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(2500, "transcribe/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		s.Features.Transcribe.Enhance = false
		s.General.MainLanguage = "zh"
	}, realLocal{installed: []string{models.ASRFast, models.VADSilero}})
	partial := false
	a.pump("the end", func() bool {
		partial = partial || a.sess.State == session.Recording && strings.HasPrefix(a.sess.Partial, "partial ")
		return a.sess.State.Ended()
	})
	end := a.sess
	if end.State != session.Done || end.Raw != "来自 Omni 的文本" || end.ASR.ID != models.ASRFast {
		t.Fatalf("end %s %+v %q", end.State, end.Problem, end.Raw)
	}
	if !partial {
		t.Error("no live partial")
	}
	rs := omnitest.Records(log)
	finals := omnitest.Of(rs, "final")
	if len(omnitest.Of(rs, "launch")) != 2 || len(finals) != 1 || finals[0].Fields["language"] != "zh" || len(omnitest.Of(rs, "vad-stream")) != 1 {
		t.Errorf("omni records %+v", rs)
	}

	// Deleting the model stops its server first.
	if _, err := a.local.(*local.Library).Delete(context.Background(), models.ASRFast, fammodels.ThisApp); err != nil {
		t.Fatal(err)
	}
	eventually(t, "the unload", func() bool { return len(omnitest.Of(omnitest.Records(log), "shutdown")) == 1 })
	a.quit()
	if n := len(omnitest.Of(omnitest.Records(log), "shutdown")); n != 2 {
		t.Errorf("%d servers shut down", n)
	}
	if _, err := a.openOmniASR(context.Background(), models.ASRFast, t.TempDir()); !errors.Is(err, local.ErrNoRuntime) {
		t.Errorf("an open after quit: %v", err)
	}
}

// TestOmniOpener: the opener is cheap after the first call, carries
// Silero only when it is installed, and says "no runtime" without one.
func TestOmniOpener(t *testing.T) {
	fakeOmni(t, omnitest.Opts{})
	a := newSessionApp(t, "", func(*settings.Settings) {}, realLocal{installed: []string{models.ASRFast}})
	dir := a.models.Dir(models.ASRFast)
	p1, err := a.openOmniASR(context.Background(), models.ASRFast, dir)
	if err != nil {
		t.Fatal(err)
	}
	p2, _ := a.openOmniASR(context.Background(), models.ASRFast, dir)
	if p1 != p2 {
		t.Error("a second open made another provider")
	}
	if va, ok := p1.(local.VoiceActivity); !ok || va.VAD() != nil {
		t.Error("a VAD that is not installed")
	}
	if _, err := a.openOmniASR(context.Background(), "llm.fast", dir); err == nil {
		t.Error("a language model opened as speech")
	}

	old := omniLocate
	omniLocate = func() (string, error) { return "", errors.New("no runtime here") }
	defer func() { omniLocate = old }()
	b := &app{}
	if _, err := b.openOmniASR(context.Background(), models.ASRFast, dir); !errors.Is(err, local.ErrNoRuntime) {
		t.Errorf("without a runtime: %v", err)
	}
}

// eventually waits for cond, at most 10 s.
func eventually(t *testing.T, what string, cond func() bool) {
	t.Helper()
	for end := time.Now().Add(10 * time.Second); time.Now().Before(end); time.Sleep(20 * time.Millisecond) {
		if cond() {
			return
		}
	}
	t.Fatalf("timed out: %s", what)
}
