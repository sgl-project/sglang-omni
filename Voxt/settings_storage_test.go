package main

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"testing"
	"time"

	fammodels "github.com/hehehai/actnowkit/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/actnow"
	"github.com/sgl-project/sglang-omni/Voxt/internal/datadir"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models/modelstest"
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
	for _, s := range []string{"Model folder", "Shared by all apps that use this model library", "Only Voxt", "Shared with other apps that use this model library"} {
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
	if !has(tt, "Other apps also use these models, so they are kept: Qwen3-ASR 0.6B (4bit) (712.8 MB)") {
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
	if runtime.GOOS == "windows" {
		recording, err := os.Open(filepath.Join(locked, "e1.wav"))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { recording.Close() })
	} else {
		if err := os.Chmod(locked, 0o500); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { os.Chmod(locked, 0o755) })
	}
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

// TestModelFolderChange: Change… brings the models into the folder the
// user chose (a "Voxt Models" folder in one that holds other files), the
// models another app claims too, writes the family's models.root and
// restarts to use it; the library stays as it was.
func TestModelFolderChange(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedModels(t, a, models.ASRFast, models.VADSilero, models.LLMTranslate)
	useModels(t, a, models.ASRFast, models.VADSilero)
	claimFor(t, a, audioSmithID, models.LLMTranslate) // retired for VoxtNew, AudioSmith's
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("mine"), 0o644); err != nil {
		t.Fatal(err)
	}
	chosen := a.models.Hub()
	relaunched := false
	oldChoose, oldRelaunch := chooseDir, appRelaunch
	chooseDir = func(_ *app, _, _ string, then func(string)) { then(chosen) }
	appRelaunch = func() { relaunched = true }
	defer func() { chooseDir, appRelaunch = oldChoose, oldRelaunch }()
	a.navigate("/settings/storage")
	click(t, tt, "Change…")
	if relaunched || !toasted(a, tt, "The models are already there") {
		t.Fatalf("the same folder: relaunched %v, toasts %q", relaunched, a.toasts)
	}
	chosen = a.dataDir() // Erase everything would delete the models with it
	click(t, tt, "Change…")
	if relaunched || !toasted(a, tt, "Choose a folder outside Voxt's data and logs.") {
		t.Fatalf("Voxt's data folder: relaunched %v, toasts %q", relaunched, a.toasts)
	}

	chosen = dir
	click(t, tt, "Change…")
	frames(tt)
	if !relaunched {
		t.Fatalf("no restart: %q", a.toasts)
	}
	to := filepath.Join(dir, modelsDirName)
	if root, _ := a.fam.Shared().Get(actnow.KeyModelsRoot); !fammodels.SameHub(root, to) {
		t.Errorf("models.root = %q, want %q", root, to)
	}
	for _, id := range []string{models.ASRFast, models.VADSilero, models.LLMTranslate} {
		m, _ := models.Lookup(id)
		if models.Installed(to, m) == "" {
			t.Errorf("%s not in the new folder", id)
		}
		if !onDisk(a, id) {
			t.Errorf("%s left the library", id)
		}
	}
	if cs, _ := fammodels.ReadClaims(a.fam.Dir(), a.fam.BundleID()); !fammodels.SameHub(cs.Root, to) || len(cs.Claims) != 2 {
		t.Errorf("VoxtNew's claims: %+v", cs)
	}
}

// TestModelFolderBringsOtherLibraries: the models are kept in one place.
// Since 2.0.1, VoxtNew's library is its own folder while AudioSmith keeps
// its models in the family's default library (ActNow/models/hub); Change…
// writes models.root for both, so it brings AudioSmith's models from
// there too, and AudioSmith finds them in the new folder; what no app
// claims there stays.
func TestModelFolderBringsOtherLibraries(t *testing.T) {
	a, _, tt := newTestApp(t)
	seedModels(t, a, models.ASRFast)
	useModels(t, a, models.ASRFast)
	ctx := context.Background()
	old := &fammodels.Store{Family: a.fam.Dir(), Hub: fammodels.DefaultHubIn(a.fam.Dir()), BundleID: audioSmithID, AppID: "audiosmith"}
	vad, _ := models.Lookup(models.VADSilero)
	if err := modelstest.SeedModel(old.Hub, vad); err != nil {
		t.Fatal(err)
	}
	if err := old.Claim(ctx, vad.Claim("", fammodels.OriginDownloaded)); err != nil {
		t.Fatal(err)
	}
	// One VoxtNew no longer has, left in the old library by 2.0.1's hard
	// links and claimed by no one: it stays behind.
	gone, _ := models.Lookup(models.ASRAccurate)
	if err := modelstest.SeedModel(old.Hub, gone); err != nil {
		t.Fatal(err)
	}
	if fammodels.SameHub(old.Hub, a.models.Hub()) {
		t.Fatal("VoxtNew's library is the family's default")
	}
	dir := t.TempDir()
	oldChoose, oldRelaunch := chooseDir, appRelaunch
	chooseDir = func(_ *app, _, _ string, then func(string)) { then(dir) }
	appRelaunch = func() {}
	defer func() { chooseDir, appRelaunch = oldChoose, oldRelaunch }()
	a.navigate("/settings/storage")
	click(t, tt, "Change…")
	frames(tt)
	if root, _ := a.fam.Shared().Get(actnow.KeyModelsRoot); !fammodels.SameHub(root, dir) {
		t.Fatalf("models.root = %q, toasts %q", root, a.toasts)
	}
	fast, _ := models.Lookup(models.ASRFast)
	for _, m := range []models.Model{vad, fast} {
		if models.Installed(dir, m) == "" {
			t.Errorf("%s is not in the new folder", m.ID)
		}
	}
	if models.Installed(dir, gone) != "" {
		t.Errorf("%s, which no app has, came along", gone.ID)
	}
}

// TestTimeMachineSwitch: VoiceOver names Storage's Time Machine switch
// and can turn it on; it had no label, so it read as a bare "switch".
func TestTimeMachineSwitch(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/settings/storage")
	frames(tt)
	if a.fam.Backup() {
		t.Fatal("Time Machine is on by default")
	}
	if err := tt.ClickControl("Back up models with Time Machine"); err != nil {
		t.Fatal(err)
	}
	eventually(t, "the family's models.backup on", func() bool { return a.fam.ReadBackup() })
}
