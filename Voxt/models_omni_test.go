package main

import (
	"context"
	"errors"
	"github.com/hehehai/actnowkit/conformance"
	fammodels "github.com/hehehai/actnowkit/models"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models/modelstest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni/omnitest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/resident"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
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
	events := omniEvents(t)
	ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		s.Features.Transcribe.Enhance = false
		s.General.MainLanguage = "zh"
	}, realLocal{installed: []string{models.ASRFast, models.VADSilero}})
	// The live text comes while recording: stop once the first partial
	// showed and the recording is longer than the shortest one kept. The
	// fake helper sends a level before each 100 ms frame, so five levels
	// are at least 400 ms recorded, whether or not the preview goes on
	// (it may stop on a busy Mac: no later partial comes, and the shown
	// one may go).
	partial := false
	a.pump("a live partial", func() bool {
		partial = partial || a.sess.State == session.Recording && strings.HasPrefix(a.sess.Partial, "partial ")
		return partial && a.sess.State == session.Recording && len(a.sess.Levels) >= 5
	})
	a.stopSession()
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	end := a.sess
	if end.State != session.Done || end.Raw != "来自 Omni 的文本" || end.ASR.ID != models.ASRFast {
		t.Fatalf("end %s %+v %q", end.State, end.Problem, end.Raw)
	}
	rs := omnitest.Records(log)
	finals := omnitest.Of(rs, "final")
	if len(omnitest.Of(rs, "launch")) != 2 || len(finals) != 1 || finals[0].Fields["language"] != "zh" || len(omnitest.Of(rs, "vad-stream")) != 1 {
		t.Errorf("omni records %+v", rs)
	}

	// Deleting the model stops its server first.
	if _, err := a.localLib.Delete(context.Background(), models.ASRFast, fammodels.ThisApp); err != nil {
		t.Fatal(err)
	}
	waitUnloaded(t, events, models.ASRFast)
	if n := len(omnitest.Of(omnitest.Records(log), "shutdown")); n != 1 {
		t.Errorf("%d servers shut down after the delete", n)
	}
	a.quit()
	if n := len(omnitest.Of(omnitest.Records(log), "shutdown")); n != 2 {
		t.Errorf("%d servers shut down", n)
	}
	if _, err := a.openOmniASR(context.Background(), models.ASRFast, t.TempDir()); !errors.Is(err, local.ErrNoRuntime) {
		t.Errorf("an open after quit: %v", err)
	}
}

// omniEvents reports the model events (Config.OnModel) of the omni
// runtimes made from now on.
func omniEvents(t *testing.T) chan resident.Event {
	ch := make(chan resident.Event, 256)
	newRT := omniNewRuntime
	omniNewRuntime = func(c omni.Config) *omni.Runtime {
		c.OnModel = reportTo(ch, c.OnModel)
		return newRT(c)
	}
	t.Cleanup(func() { omniNewRuntime = newRT })
	return ch
}

// reportTo is a runtime's Config.OnModel that hands each event to on
// (the app's: the runtime file), then sends it on ch.
func reportTo(ch chan<- resident.Event, on func(resident.Event)) func(resident.Event) {
	return func(ev resident.Event) {
		if on != nil {
			on(ev)
		}
		ch <- ev
	}
}

// waitUnloaded waits until the processes of ids exited.
func waitUnloaded(t *testing.T, events <-chan resident.Event, ids ...string) {
	t.Helper()
	timeout := time.After(10 * time.Second)
	for len(ids) > 0 {
		select {
		case ev := <-events:
			if ev.Kind == resident.Unloaded {
				ids = slices.DeleteFunc(ids, func(id string) bool { return id == ev.ID })
			}
		case <-timeout:
			t.Fatalf("%v did not stop", ids)
		}
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

// TestOmniDraft: an installed draft goes with its speech model (the
// provider changes with it), and deleting the draft unloads the model.
func TestOmniDraft(t *testing.T) {
	fakeOmni(t, omnitest.Opts{})
	a := newSessionApp(t, "", func(*settings.Settings) {}, realLocal{installed: []string{models.ASRFast, models.ASRFastDraft}})
	draft := a.models.Dir(models.ASRFastDraft)
	if draft == "" || a.omniDraft(models.ASRFast) != draft {
		t.Fatalf("draft %q, omniDraft %q", draft, a.omniDraft(models.ASRFast))
	}
	if a.omniDraft(models.ASRAccurate) != "" || a.omniDraft(models.ASRFastDraft) != "" || a.omniDraft(models.VADSilero) != "" {
		t.Error("a draft for a model without one installed")
	}
	dir := a.models.Dir(models.ASRFast)
	p1, err := a.openOmniASR(context.Background(), models.ASRFast, dir)
	if err != nil {
		t.Fatal(err)
	}
	if p2, _ := a.openOmniASR(context.Background(), models.ASRFast, dir); p2 != p1 {
		t.Error("a second open made another provider")
	}
	a.unloadOmni(models.ASRFastDraft)
	if len(a.omniASR) != 0 {
		t.Errorf("%d providers after the draft's unload", len(a.omniASR))
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

// TestOmniSpecsAreTheCatalogue: every omni model of the catalogue (but the
// tokenizers, files of another) has a spec of the same repo and commit, and
// every spec a catalogue entry.
func TestOmniSpecsAreTheCatalogue(t *testing.T) {
	n := 0
	for _, m := range models.Catalog() {
		if m.Runtime != models.Omni || m.Kind == models.Aux || m.Kind == models.Draft { // a draft rides on its model's server
			continue
		}
		n++
		s, ok := omni.Lookup(m.ID)
		if !ok || s.Repo != m.Repo || s.Revision != m.Revision || (s.ASR != nil) != (m.Kind == models.ASR) {
			t.Errorf("%s: spec %+v, catalogue %s@%s", m.ID, s, m.Repo, m.Revision)
		}
	}
	if len(omni.Models()) != n {
		t.Errorf("%d specs, %d omni models in the catalogue", len(omni.Models()), n)
	}
}

// TestOmniFolderLinksTheTokenizer: a Whisper model's server gets a folder
// with its own files and its tokenizer's; a model without one, its own.
func TestOmniFolderLinksTheTokenizer(t *testing.T) {
	m, hub := openTestModels(t, models.ASRWhisperSmall, models.AuxWhisperSmall, models.ASRFast)
	a := &app{models: m}
	a.omniCache = t.TempDir()
	own := m.Dir(models.ASRFast)
	if got, err := a.omniFolder(models.ASRFast, own); err != nil || got != own {
		t.Errorf("qwen: %q, %v", got, err)
	}
	dir, err := a.omniFolder(models.ASRWhisperSmall, m.Dir(models.ASRWhisperSmall))
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"config.json", "weights.npz", "vocab.json", "tokenizer_config.json"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			t.Errorf("%s: %v", f, err)
		}
	}
	if !strings.HasPrefix(dir, a.omniCache) || strings.HasPrefix(dir, hub) {
		t.Errorf("folder %s", dir)
	}
	// Opened again, the links stay as they are: a server may be loading.
	before, _ := os.Lstat(filepath.Join(dir, "vocab.json"))
	if again, err := a.omniFolder(models.ASRWhisperSmall, m.Dir(models.ASRWhisperSmall)); err != nil || again != dir {
		t.Fatalf("again: %q, %v", again, err)
	}
	if after, err := os.Lstat(filepath.Join(dir, "vocab.json")); err != nil || !os.SameFile(before, after) {
		t.Errorf("the links were made again: %v", err)
	}
	b, _ := openTestModels(t, models.ASRWhisperSmall)
	if _, err := (&app{models: b}).omniFolder(models.ASRWhisperSmall, b.Dir(models.ASRWhisperSmall)); !errors.Is(err, local.ErrNotInstalled) {
		t.Errorf("without the tokenizer: %v", err)
	}
}

// openTestModels is a Manager over a temporary model library with the
// models ids installed (sparse files), and its $HUB.
func openTestModels(t *testing.T, ids ...string) (*models.Manager, string) {
	t.Helper()
	fx := conformance.New(t)
	if err := modelstest.Seed(fx.Hub, ids...); err != nil {
		t.Fatal(err)
	}
	m := models.Open(models.Options{Store: fx.Peer("dev.hehehai.voxtnew")})
	t.Cleanup(m.Close)
	return m, fx.Hub
}
