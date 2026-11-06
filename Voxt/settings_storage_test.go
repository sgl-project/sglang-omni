package main

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/datadir"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestStorageModels: Settings → Storage shows the shared model folder and
// VoxtNew's part of it; erasing keeps the models other apps use, and
// removes VoxtNew's files from the family directory.
func TestStorageModels(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedModels(t, a, models.ASRFast, models.VADSilero, models.LLMTranslate)
	useModels(t, a, models.ASRFast, models.VADSilero, models.LLMTranslate)
	claimFor(t, a, audioSmithID, models.ASRFast)
	addPeer(t, a, audioSmithID, nil)
	a.familyFound(a.fam.Discover())
	a.navigate("/settings/storage")
	frames(tt)
	r := a.stReport
	fast, _ := models.Lookup(models.ASRFast)
	vad, _ := models.Lookup(models.VADSilero)
	tr, _ := models.Lookup(models.LLMTranslate)
	if r == nil || r.modelsShared != fast.Size() || r.modelsMine != vad.Size()+tr.Size() {
		t.Fatalf("usage: %+v (want shared %d, mine %d)", r, fast.Size(), vad.Size()+tr.Size())
	}
	for _, s := range []string{"Model folder", "Shared by all ActNow apps", "Only VoxtNew", "Shared with other ActNow apps"} {
		if !has(tt, s) {
			t.Errorf("no %q", s)
		}
	}
	for _, c := range r.cats {
		if c.id == "other" && c.bytes > 100<<20 {
			t.Errorf("the data folder counts the model library: %d", c.bytes)
		}
	}

	relaunched := false
	old := appRelaunch
	appRelaunch = func() { relaunched = true }
	defer func() { appRelaunch = old }()
	tt.SetSize(windowWidth, 1400)
	// Erase… is below the settings dialog's fold since the Time Machine
	// row (spec §7.12): its button's action.
	if !has(tt, "Erase…") {
		t.Fatal("no Erase…")
	}
	a.askErase()
	frames(tt)
	if !has(tt, "These models are also used by other ActNow apps and will be kept: Fast speech recognition (712.8 MB)") {
		t.Fatalf("no kept models: %q", tt.Texts())
	}
	a.resetText = "DELETE"
	click(t, tt, "Erase and restart")
	if !relaunched {
		t.Fatal("no restart")
	}
	if !onDisk(a, models.ASRFast) || onDisk(a, models.LLMTranslate) || onDisk(a, models.VADSilero) {
		t.Errorf("on disk after erase: fast %v, translate %v, vad %v", onDisk(a, models.ASRFast), onDisk(a, models.LLMTranslate), onDisk(a, models.VADSilero))
	}
	if _, err := os.Stat(a.fam.Dir().ClaimsFile(a.fam.BundleID())); !os.IsNotExist(err) {
		t.Errorf("claims file: %v", err)
	}
	if _, err := os.Stat(filepath.Join(a.home(), datadir.ResetFile)); err != nil {
		t.Errorf("no reset: %v", err)
	}
}

// TestEraseStopsWhenRecordingsStay: Erase with the recordings in a folder
// outside the data folder, where one cannot be deleted. Nothing is
// erased and VoxtNew does not restart; a toast says why. It used to
// restart and erase the rest, leaving that file with no entry naming it.
func TestEraseStopsWhenRecordingsStay(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root deletes a read-only folder's files")
	}
	a, _, tt := newTestApp(t)
	seedModels(t, a, models.ASRFast)
	useModels(t, a, models.ASRFast)
	h := a.lib.store.History()
	locked := filepath.Join(t.TempDir(), "Recordings")
	if err := os.Mkdir(locked, 0o755); err != nil {
		t.Fatal(err)
	}
	h.SetAudioDir(locked)
	os.WriteFile(filepath.Join(locked, "e1.wav"), []byte("RIFF"), 0o644)
	if err := h.Put(context.Background(), store.Entry{ID: "e1", Kind: store.KindTranscribe, CreatedAt: time.Now(), Result: "x", Audio: "e1.wav"}); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(locked, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(locked, 0o755) })
	relaunched := false
	old := appRelaunch
	appRelaunch = func() { relaunched = true }
	defer func() { appRelaunch = old }()
	a.navigate("/settings/storage")
	a.askErase()
	frames(tt)
	a.resetText = "DELETE"
	click(t, tt, "Erase and restart")
	if relaunched {
		t.Error("restarted to erase, with a recording left behind")
	}
	if !hasPrefix(tt, "Could not erase:") && !slices.ContainsFunc(a.toasts, func(s string) bool { return strings.HasPrefix(s, "Could not erase:") }) {
		t.Errorf("no toast: %q %q", a.toasts, firstTexts(tt))
	}
	if !onDisk(a, models.ASRFast) {
		t.Error("the models went before the recordings")
	}
	if _, err := os.Stat(filepath.Join(a.home(), datadir.ResetFile)); !os.IsNotExist(err) {
		t.Errorf("reset asked: %v", err)
	}
}
