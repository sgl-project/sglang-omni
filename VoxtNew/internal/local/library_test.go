package local

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/hehehai/actnowkit/conformance"
	fammodels "github.com/hehehai/actnowkit/models"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models/modelstest"
)

type fakeProvider struct{ asr.FileTranscriber }

// openModels is a Manager over a temporary family library ($HUB) with
// the models ids installed.
func openModels(t *testing.T, ids ...string) (*models.Manager, string) {
	t.Helper()
	fx := conformance.New(t)
	if err := modelstest.Seed(fx.Hub, ids...); err != nil {
		t.Fatal(err)
	}
	m := models.Open(models.Options{Store: fx.Peer("dev.hehehai.voxtnew")})
	t.Cleanup(m.Close)
	return m, fx.Hub
}

// snapshot is where model id is installed under hub.
func snapshot(hub, id string) string {
	m, _ := models.Lookup(id)
	return models.SnapshotDir(hub, m, "")
}

func TestLibrary(t *testing.T) {
	m, hub := openModels(t, models.ASRFast, models.VADSilero, models.LLMTranslate, models.LLMFast)
	name := func(id string) string { return "name of " + id }
	var s Services = NewLibrary(m, Openers{}, name)

	if l := s.Installed(ASR); len(l) != 1 || l[0].ID != models.ASRFast || l[0].Name != "name of asr.fast" {
		t.Errorf("asr %+v", l)
	}
	llms := s.Installed(LLM)
	if len(llms) != 2 || llms[0].ID != models.LLMFast || llms[1].ID != models.LLMTranslate {
		t.Fatalf("llm %+v", llms)
	}
	if got := ForFeature(llms, "rewrite"); len(got) != 1 || got[0].ID != models.LLMFast {
		t.Errorf("rewrite %+v", got)
	}
	if got := ForFeature(llms, "translate"); len(got) != 2 {
		t.Errorf("translate %+v", got)
	}

	// No runtime: a clear error, not "not installed".
	_, _, err := s.ASR(context.Background(), "")
	if !errors.Is(err, ErrNoRuntime) || errors.Is(err, ErrNotInstalled) {
		t.Errorf("asr without runtime: %v", err)
	}
	if _, _, err := s.LLM(context.Background(), models.LLMBalanced); !errors.Is(err, ErrNoRuntime) {
		t.Errorf("llm without runtime: %v", err)
	}

	// With openers: the installed folder.
	var opened []string
	lib := NewLibrary(m, Openers{
		ASR: func(_ context.Context, id, dir string) (asr.Provider, error) {
			opened = append(opened, id+" "+dir)
			return fakeProvider{}, nil
		},
		LLM: func(_ context.Context, id, dir string) (Chat, error) {
			opened = append(opened, id+" "+dir)
			return nil, errors.New("boom")
		},
	}, nil)
	p, mod, err := lib.ASR(context.Background(), models.ASRAccurate) // not installed: the first
	if err != nil || p == nil || mod.ID != models.ASRFast || mod.Name != models.ASRFast {
		t.Errorf("asr %v %+v %v", p, mod, err)
	}
	if _, _, err := lib.LLM(context.Background(), models.LLMTranslate); err == nil || err.Error() != "boom" {
		t.Errorf("llm %v", err)
	}
	// The runtimes load the snapshot: $HUB/models--<owner>--<name>/snapshots/<commit>.
	want := []string{models.ASRFast + " " + snapshot(hub, models.ASRFast), models.LLMTranslate + " " + snapshot(hub, models.LLMTranslate)}
	if len(opened) != 2 || opened[0] != want[0] || opened[1] != want[1] {
		t.Errorf("opened %q, want %q", opened, want)
	}
	if !strings.Contains(want[0], "/models--mlx-community--Qwen3-ASR-0.6B-4bit/snapshots/313d850181767edf09f00a9c289becca70e58cd0") {
		t.Errorf("snapshot %s", want[0])
	}
	if lib.Dir(models.VADSilero) != snapshot(hub, models.VADSilero) || lib.Dir(models.ASRAccurate) != "" {
		t.Error("Dir")
	}
	// Opening claimed them for VoxtNew (another app's would stay with it).
	for _, id := range []string{models.ASRFast, models.VADSilero, models.LLMTranslate} {
		if !m.Status(id).Mine {
			t.Errorf("%s not claimed", id)
		}
	}

	// Nothing installed.
	none, _ := openModels(t)
	empty := NewLibrary(none, Openers{}, nil)
	if _, _, err := empty.ASR(context.Background(), ""); !errors.Is(err, ErrNotInstalled) {
		t.Errorf("empty: %v", err)
	}
}

// TestLibraryDelete: the runtime unloads a model before its files go.
func TestLibraryDelete(t *testing.T) {
	m, _ := openModels(t, models.ASRFast)
	var unloaded []string
	lib := NewLibrary(m, Openers{Unload: func(id string) {
		if !m.Installed(id) {
			t.Error("unloaded after the delete")
		}
		unloaded = append(unloaded, id)
	}}, nil)
	if _, err := lib.Delete(context.Background(), models.ASRFast, fammodels.ThisApp); err != nil {
		t.Fatal(err)
	}
	if len(unloaded) != 1 || unloaded[0] != models.ASRFast || m.Installed(models.ASRFast) {
		t.Errorf("unloaded %q, installed %v", unloaded, m.Installed(models.ASRFast))
	}
	if _, err := NewLibrary(m, Openers{}, nil).Delete(context.Background(), models.ASRFast, fammodels.ThisApp); err != nil {
		t.Errorf("no Unload: %v", err)
	}
}
