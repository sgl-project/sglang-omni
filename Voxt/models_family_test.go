package main

// The Speech models page in the ActNow family (models_cards.go,
// models_delete.go): the other apps are played with actnowkit, in the test
// app's family directory and model library.

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/hehehai/actnowkit/family"
	fammodels "github.com/hehehai/actnowkit/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/actnow"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models/modelstest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// Other family apps, as the tests play them.
const (
	audioSmithID = "dev.hehehai.audiosmith"
	dictaID      = "dev.hehehai.dicta"
)

// otherApp is family app bundleID's view of a's model library.
func otherApp(a *app, bundleID string) *fammodels.Store {
	appID, _, _ := strings.Cut(strings.TrimPrefix(bundleID, "dev.hehehai."), ".")
	return &fammodels.Store{Family: a.fam.Dir(), Hub: a.models.Hub(), BundleID: bundleID, AppID: appID}
}

// claimFor makes app bundleID claim the models ids (on disk), then reads
// the family again as a notification would.
func claimFor(t *testing.T, a *app, bundleID string, ids ...string) {
	t.Helper()
	s := otherApp(a, bundleID)
	for _, id := range ids {
		m, _ := models.Lookup(id)
		if err := s.Claim(context.Background(), m.Claim("", fammodels.OriginDownloaded)); err != nil {
			t.Fatal(err)
		}
	}
	a.models.Refresh()
}

// useModels makes VoxtNew claim the models ids.
func useModels(t *testing.T, a *app, ids ...string) {
	t.Helper()
	if err := a.models.Use(context.Background(), ids...); err != nil {
		t.Fatal(err)
	}
}

// onDisk reports whether model id is installed in a's library.
func onDisk(a *app, id string) bool {
	m, _ := models.Lookup(id)
	return models.Installed(a.models.Hub(), m) != ""
}

// toasted reports whether msg was toasted (still queued, or shown).
func toasted(a *app, tt *ui.Tester, msg string) bool {
	return slices.Contains(a.toasts, msg) || has(tt, msg)
}

// TestModelsInOnePlace: with no model folder chosen, the start keeps
// every app's models in VoxtNew's folder (SGLang Omni/models): AudioSmith's
// come over from the family's default library, where they were before
// 2.0.1, and models.root names the folder, so that AudioSmith finds them
// there. A folder the user chose stays.
func TestModelsInOnePlace(t *testing.T) {
	a, _, _ := newTestApp(t)
	ctx := context.Background()
	old := &fammodels.Store{Family: a.fam.Dir(), Hub: fammodels.DefaultHubIn(a.fam.Dir()), BundleID: audioSmithID, AppID: "audiosmith"}
	vad, _ := models.Lookup(models.VADSilero)
	if err := modelstest.SeedModel(old.Hub, vad); err != nil {
		t.Fatal(err)
	}
	if err := old.Claim(ctx, vad.Claim("", fammodels.OriginDownloaded)); err != nil {
		t.Fatal(err)
	}
	if err := gatherModels(ctx, a.fam, a.models, a.prefs); err != nil {
		t.Fatal(err)
	}
	if root, _ := a.fam.Shared().Get(actnow.KeyModelsRoot); !fammodels.SameHub(root, a.models.Hub()) {
		t.Errorf("models.root = %q, not %s", root, a.models.Hub())
	}
	if models.Installed(a.models.Hub(), vad) == "" {
		t.Error("AudioSmith's model did not come over")
	}
	chosen := t.TempDir()
	if err := a.fam.Set(ctx, a.prefs, actnow.KeyModelsRoot, chosen); err != nil {
		t.Fatal(err)
	}
	if err := gatherModels(ctx, a.fam, a.models, a.prefs); err != nil {
		t.Fatal(err)
	}
	if root, _ := a.fam.Shared().Get(actnow.KeyModelsRoot); !fammodels.SameHub(root, chosen) {
		t.Errorf("the folder the user chose became %q", root)
	}
}

// TestGatherSkipsLibraries: a library keep refuses (at the start: one on
// another disk, which would be copied before the window opens) is left to
// its app, and models.root is written all the same.
func TestGatherSkipsLibraries(t *testing.T) {
	a, _, _ := newTestApp(t)
	ctx := context.Background()
	old := &fammodels.Store{Family: a.fam.Dir(), Hub: fammodels.DefaultHubIn(a.fam.Dir()), BundleID: audioSmithID, AppID: "audiosmith"}
	vad, _ := models.Lookup(models.VADSilero)
	if err := modelstest.SeedModel(old.Hub, vad); err != nil {
		t.Fatal(err)
	}
	if err := old.Claim(ctx, vad.Claim("", fammodels.OriginDownloaded)); err != nil {
		t.Fatal(err)
	}
	committed := false
	if err := a.models.Gather(ctx, false, func(string) bool { return false }, func() error { committed = true; return nil }); err != nil || !committed {
		t.Fatalf("gather: %v, committed %v", err, committed)
	}
	if models.Installed(a.models.Hub(), vad) != "" {
		t.Error("brought from a library keep refused")
	}
}

// TestModelsFromTheFamilyLibrary: 2.0.0 kept the models in the family's
// default library (ActNow/models/hub); this version's start brings them
// into its own folder (hard links: no space used) and claims them there.
func TestModelsFromTheFamilyLibrary(t *testing.T) {
	a, _, tt := newTestApp(t)
	ctx := context.Background()
	old := &fammodels.Store{Family: a.fam.Dir(), Hub: fammodels.DefaultHubIn(a.fam.Dir()), BundleID: a.fam.BundleID(), AppID: "voxtnew"}
	fast, _ := models.Lookup(models.ASRFast)
	if err := modelstest.SeedModel(old.Hub, fast); err != nil {
		t.Fatal(err)
	}
	if err := old.Claim(ctx, fast.Claim("", fammodels.OriginDownloaded)); err != nil {
		t.Fatal(err)
	}
	if onDisk(a, models.ASRFast) {
		t.Fatal("in the library before the import")
	}
	if n, _, err := a.models.Import(ctx, false); err != nil || n != 1 || !onDisk(a, models.ASRFast) {
		t.Fatalf("imported %d: %v", n, err)
	}
	files, _ := filepath.Glob(filepath.Join(models.Installed(old.Hub, fast), "*"))
	if len(files) == 0 {
		t.Fatal("no files in 2.0.0's snapshot")
	}
	was, _ := os.Stat(files[0])
	now, err := os.Stat(filepath.Join(models.Installed(a.models.Hub(), fast), filepath.Base(files[0])))
	if err != nil || !os.SameFile(was, now) {
		t.Errorf("%s is not the same file: %v", filepath.Base(files[0]), err)
	}
	if cs, _ := fammodels.ReadClaims(a.fam.Dir(), a.fam.BundleID()); !fammodels.SameHub(cs.Root, a.models.Hub()) || len(cs.Claims) != 1 {
		t.Errorf("VoxtNew's claims: %+v", cs)
	}
	// AudioSmith keeps its models in the family's library, not in VoxtNew's
	// folder: Storage does not call the folder shared.
	audioSmith := &fammodels.Store{Family: a.fam.Dir(), Hub: old.Hub, BundleID: audioSmithID, AppID: "audiosmith"}
	if err := audioSmith.Claim(ctx, fast.Claim("", fammodels.OriginDownloaded)); err != nil {
		t.Fatal(err)
	}
	addPeer(t, a, audioSmithID, nil)
	a.familyFound(a.fam.Discover())
	a.navigate("/settings/storage")
	frames(tt)
	if !has(tt, "Model folder") || has(tt, "Shared by all apps that use this model library") {
		t.Errorf("Storage: %q", firstTexts(tt))
	}
}

func TestModelDeleteShared(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1600)
	seedModels(t, a, models.ASRFast, models.VADSilero)
	useModels(t, a, models.ASRFast)
	claimFor(t, a, audioSmithID, models.ASRFast)
	a.navigate(modelsPath)
	if !has(tt, "Shared with AudioSmith") {
		t.Fatalf("no Shared with: %q", tt.Texts())
	}

	// Remove from VoxtNew: the claim goes, the files stay for AudioSmith.
	click(t, tt, "Delete Qwen3-ASR 0.6B (4bit)…")
	if !has(tt, `Remove "Qwen3-ASR 0.6B (4bit)" from Voxt?`) ||
		!has(tt, "AudioSmith also uses this model, so its files (712.8 MB) stay on this Mac.") {
		t.Fatalf("no family dialog: %q", tt.Texts())
	}
	click(t, tt, "Remove from Voxt")
	if !onDisk(a, models.ASRFast) || a.models.Status(models.ASRFast).Mine {
		t.Errorf("after Remove from VoxtNew: on disk %v, %+v", onDisk(a, models.ASRFast), a.models.Status(models.ASRFast))
	}
	if msg := "Qwen3-ASR 0.6B (4bit) was removed from Voxt. Its files stay on this Mac for AudioSmith."; !toasted(a, tt, msg) {
		t.Errorf("no toast %q: %q", msg, tt.Texts())
	}

	// Delete for all apps…: asks again, then the files go and AudioSmith
	// finds a tombstone.
	useModels(t, a, models.ASRFast)
	click(t, tt, "Delete Qwen3-ASR 0.6B (4bit)…")
	click(t, tt, "Delete for all apps…")
	if !has(tt, `Delete "Qwen3-ASR 0.6B (4bit)" for all apps?`) || !has(tt, "AudioSmith will need to download it again (712.8 MB).") {
		t.Fatalf("no second confirmation: %q", tt.Texts())
	}
	click(t, tt, "Delete Qwen3-ASR 0.6B (4bit) for all apps")
	if onDisk(a, models.ASRFast) || a.modelStatus[models.ASRFast].Installed {
		t.Error("the files stayed")
	}
	if hit, err := otherApp(a, audioSmithID).ApplyTombstones(); err != nil || len(hit) != 1 {
		t.Errorf("AudioSmith's tombstones: %v %v", hit, err)
	}
}

func TestModelDeleteCancelShared(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 4800) // the retired models under every speech model's card
	seedModels(t, a, models.LLMTranslate)
	useModels(t, a, models.LLMTranslate)
	claimFor(t, a, dictaID, models.LLMTranslate)
	claimFor(t, a, audioSmithID, models.LLMTranslate)
	a.navigate(modelsPath)
	if !has(tt, "Shared with AudioSmith and Dicta") {
		t.Fatalf("no Shared with: %q", tt.Texts())
	}
	click(t, tt, "Delete Hy-MT2 1.8B (Q4_K_M)…")
	if !has(tt, "AudioSmith and Dicta also use this model, so its files (1.1 GB) stay on this Mac.") {
		t.Fatalf("dialog: %q", tt.Texts())
	}
	click(t, tt, "Cancel")
	if a.modelShared != nil || !a.models.Status(models.LLMTranslate).Mine {
		t.Error("Cancel removed it")
	}
}

// TestModelDeleteBusy: another app holds the model's lock (it downloads
// it): deleting says so and keeps everything.
func TestModelDeleteBusy(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1600)
	seedModels(t, a, models.ASRFast, models.VADSilero)
	fast, _ := models.Lookup(models.ASRFast)
	lock, err := fammodels.TryLockRepo(a.models.Hub(), fast.Repo, family.Shared, fammodels.Owner{BundleID: audioSmithID, Op: fammodels.OpDownload})
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Release()
	a.navigate(modelsPath)
	click(t, tt, "Delete Qwen3-ASR 0.6B (4bit)…")
	click(t, tt, "Delete Qwen3-ASR 0.6B (4bit)")
	if msg := "AudioSmith is downloading or using this model. Try again later."; !toasted(a, tt, msg) {
		t.Errorf("no toast %q: %q", msg, tt.Texts())
	}
	if !onDisk(a, models.ASRFast) {
		t.Error("deleted")
	}
}

// TestModelElsewhere: another app downloads a model VoxtNew lacks: the
// card says so and Download waits; a model deleted elsewhere says who.
func TestModelElsewhere(t *testing.T) {
	srv := httptest.NewServer(http.NotFoundHandler()) // a download that starts fails at once
	defer srv.Close()
	t.Setenv("HF_ENDPOINT", srv.URL)
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1600)
	reopenModels(a)
	accurate, _ := models.Lookup(models.ASRAccurate)
	lock, err := fammodels.TryLockRepo(a.models.Hub(), accurate.Repo, family.Shared, fammodels.Owner{BundleID: audioSmithID, Op: fammodels.OpDownload})
	if err != nil {
		t.Fatal(err)
	}
	a.models.Refresh()
	a.navigate(modelsPath)
	if !has(tt, "Downloading in AudioSmith…") || a.modelStatus[models.ASRAccurate].Elsewhere != audioSmithID {
		t.Fatalf("elsewhere: %+v %q", a.modelStatus[models.ASRAccurate], tt.Texts())
	}
	click(t, tt, "Download Qwen3-ASR 1.7B (8bit)") // disabled
	if st := a.models.Status(models.ASRAccurate); st.Downloading || st.Err != nil {
		t.Errorf("Download ran: %+v", st)
	}
	// AudioSmith finishes: installed on the next read of the family.
	lock.Release()
	modelstest.Seed(a.models.Hub(), models.ASRAccurate)
	a.models.Refresh()
	if !a.modelStatus[models.ASRAccurate].Installed || has(tt, "Downloading in AudioSmith…") {
		t.Errorf("not installed: %+v", a.modelStatus[models.ASRAccurate])
	}

	// Deleted for all apps in AudioSmith.
	useModels(t, a, models.ASRAccurate)
	if _, err := otherApp(a, audioSmithID).Delete(context.Background(), accurate.Ref(""), fammodels.AllApps); err != nil {
		t.Fatal(err)
	}
	a.models.Refresh()
	if !has(tt, "Deleted in AudioSmith") || !has(tt, "Download again") {
		t.Errorf("deleted in: %+v %q", a.modelStatus[models.ASRAccurate], tt.Texts())
	}
}

// TestModelWaitingFor: VoxtNew's download waits for a file another app
// is downloading.
func TestModelWaitingFor(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rest, ok := strings.CutPrefix(r.URL.Path, "/api/models/"); ok {
			repo, _, _ := strings.Cut(rest, "/revision/")
			fakeModelInfo(w, repo)
			return
		}
		http.Error(w, "no", http.StatusServiceUnavailable)
	}))
	defer srv.Close()
	t.Setenv("HF_ENDPOINT", srv.URL)
	reserve := modelsReserve
	modelsReserve = -1
	defer func() { modelsReserve = reserve }()
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1600)
	drain := queuePosts(a)
	reopenModels(a)
	// A model of one file, the one locked: Hy-MT2 (retired: the page
	// offers no Download, but shows its card while it downloads).
	tr, _ := models.Lookup(models.LLMTranslate)
	lock, err := fammodels.TryLockBlob(a.models.Hub(), tr.Repo, tr.Files[0].SHA256, fammodels.Owner{BundleID: dictaID, Op: fammodels.OpDownload})
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Release()
	a.navigate(modelsPath)
	a.downloadModel(models.LLMTranslate)
	deadline := time.Now().Add(5 * time.Second)
	for !has(tt, "Downloading in Dicta… 0%") && time.Now().Before(deadline) {
		drain()
	}
	if !has(tt, "Downloading in Dicta… 0%") {
		t.Fatalf("waiting: %+v %q", a.models.Status(models.LLMTranslate), tt.Texts())
	}
	click(t, tt, "Cancel downloading Hy-MT2 1.8B (Q4_K_M)")
	drain()
}

func TestModelFamilyText(t *testing.T) {
	asr, _ := models.Lookup(models.ASRFast)
	qwen, _ := models.Lookup(models.LLMQwen3)
	for _, c := range []struct {
		m    models.Model
		st   models.Status
		want string
	}{
		{asr, models.Status{Installed: true, OnDisk: true, Users: []string{"AudioSmith", "Dicta", "Auralink"}}, "Shared with AudioSmith, Dicta and Auralink"},
		{asr, models.Status{Downloading: true, Done: 43, Total: 100, WaitingFor: "unknown"}, "Downloading in another app… 43%"},
		{asr, models.Status{Elsewhere: "dev.hehehai.dicta.dev"}, "Downloading in Dicta (dev)…"},
		{asr, models.Status{DeletedBy: dictaID}, "Deleted in Dicta"},
		{asr, models.Status{Installed: true, OnDisk: true, DeletedBy: dictaID}, ""},
		{qwen, models.Status{OnDisk: true, Users: []string{"Dicta"}}, "Installed by Dicta"},
		{qwen, models.Status{OnDisk: true}, "Already on this Mac"},
		{qwen, models.Status{OnDisk: true, Installed: true, Mine: true, Users: []string{"Dicta"}}, "Shared with Dicta"},
	} {
		if got := modelFamilyText(c.m, c.st); got != c.want {
			t.Errorf("%s %+v: %q, want %q", c.m.ID, c.st, got, c.want)
		}
	}
}

// TestBorrowedModel: a model of VoxtNew's catalogue another app
// installed on this Mac: Use it, then Remove from VoxtNew (its files stay
// for the app). The card used to offer Download, which does nothing for a
// model already on disk, so it never became VoxtNew's. A retired model
// (the borrowed language models) another app installed is not listed.
func TestBorrowedModel(t *testing.T) {
	t.Run("retired", func(t *testing.T) {
		a, _, tt := newTestApp(t)
		tt.SetSize(windowWidth, 1600)
		m, _ := models.Lookup(models.LLMQwen3)
		if err := modelstest.SeedModel(a.models.Hub(), m); err != nil {
			t.Fatal(err)
		}
		claimFor(t, a, dictaID, models.LLMQwen3)
		a.navigate(modelsPath)
		if has(tt, uitext.ModelName(models.LLMQwen3)) || len(a.installed(local.LLM)) != 0 {
			t.Fatalf("a retired model of Dicta's is listed: %q", tt.Texts())
		}
	})
	for _, tc := range []struct {
		id         string
		borrowed   bool
		remove     string // the card's button once used
		needsThere bool   // Dicta installed what it needs too
	}{
		{models.ASRFast, false, "Delete", false},
		{models.ASRFast, false, "Delete", true},
	} {
		t.Run(fmt.Sprintf("%s/needs-installed=%v", tc.id, tc.needsThere), func(t *testing.T) {
			a, _, tt := newTestApp(t)
			tt.SetSize(windowWidth, 1600)
			name := uitext.ModelName(tc.id)
			a.navigate(modelsPath)
			if tc.borrowed && has(tt, name) {
				t.Fatal("a borrowed model not on this Mac is listed")
			}
			m, _ := models.Lookup(tc.id)
			if err := modelstest.SeedModel(a.models.Hub(), m); err != nil {
				t.Fatal(err)
			}
			claimFor(t, a, dictaID, tc.id)
			if tc.needsThere {
				for _, need := range m.Needs {
					n, _ := models.Lookup(need)
					if err := modelstest.SeedModel(a.models.Hub(), n); err != nil {
						t.Fatal(err)
					}
					claimFor(t, a, dictaID, need)
				}
			}
			if has(tt, "Download "+name) {
				t.Fatalf("the card offers Download, which does nothing for a model on disk")
			}
			if !has(tt, name) || !has(tt, "Installed by Dicta") {
				t.Fatalf("installed by Dicta: %q", tt.Texts())
			}
			click(t, tt, "Use "+name)
			if st := a.models.Status(tc.id); !st.Installed || !st.Mine || !has(tt, "Shared with Dicta") {
				t.Fatalf("used: %+v %q", st, tt.Texts())
			}
			// What it needs comes with it: Dicta's voice detection is used,
			// a missing one is downloaded.
			for _, need := range m.Needs {
				st := a.models.Status(need)
				if tc.needsThere && (!st.Installed || !st.Mine) || !st.Downloading && !st.Installed && st.Err == nil {
					t.Errorf("%s needs %s: neither used nor downloading (%+v)", tc.id, need, st)
				}
			}
			click(t, tt, tc.remove+" "+name+"…")
			if !has(tt, fmt.Sprintf("Remove %q from Voxt?", name)) {
				t.Fatalf("dialog: %q", tt.Texts())
			}
			click(t, tt, "Remove from Voxt")
			if st := a.models.Status(tc.id); st.Installed || !st.OnDisk || !has(tt, "Installed by Dicta") {
				t.Errorf("removed: %+v %q", st, tt.Texts())
			}
		})
	}
}

// TestModelDeleteOwnModels: the files of a model stay when another of
// VoxtNew's models uses them (rare: not in the catalogue today).
func TestModelDeleteOwnModels(t *testing.T) {
	res := []fammodels.Result{{OwnModels: []string{models.ASRAccurate}}}
	if got := deleteOutcome(models.ASRFast, res, nil); got != "Qwen3-ASR 0.6B (4bit) was removed. Its files stay: Qwen3-ASR 1.7B (8bit) uses them too." {
		t.Errorf("%q", got)
	}
	if got := deleteOutcome(models.ASRFast, []fammodels.Result{{Deleted: true}}, nil); got != "" {
		t.Errorf("deleted: %q", got)
	}
	if got := deleteOutcome(models.ASRFast, nil, &fammodels.BusyError{}); !slices.Contains([]string{"Another app is downloading or using this model. Try again later."}, got) {
		t.Errorf("busy: %q", got)
	}
}
