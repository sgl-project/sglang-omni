package main

// The Local models page in the ActNow family (models_cards.go,
// models_delete.go): the other apps are played with actnowkit, in the test
// app's family directory and model library.

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/hehehai/actnowkit/family"
	fammodels "github.com/hehehai/actnowkit/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models/modelstest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
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
	click(t, tt, "Delete Fast speech recognition…")
	if !has(tt, `Remove "Fast speech recognition" from VoxtNew?`) ||
		!has(tt, "AudioSmith also uses this model, so its files (712.8 MB) stay on this Mac.") {
		t.Fatalf("no family dialog: %q", tt.Texts())
	}
	click(t, tt, "Remove from VoxtNew")
	if !onDisk(a, models.ASRFast) || a.models.Status(models.ASRFast).Mine {
		t.Errorf("after Remove from VoxtNew: on disk %v, %+v", onDisk(a, models.ASRFast), a.models.Status(models.ASRFast))
	}
	if msg := "Fast speech recognition was removed from VoxtNew. Its files stay on this Mac for AudioSmith."; !toasted(a, tt, msg) {
		t.Errorf("no toast %q: %q", msg, tt.Texts())
	}

	// Delete for all apps…: asks again, then the files go and AudioSmith
	// finds a tombstone.
	useModels(t, a, models.ASRFast)
	click(t, tt, "Delete Fast speech recognition…")
	click(t, tt, "Delete for all apps…")
	if !has(tt, `Delete "Fast speech recognition" for all apps?`) || !has(tt, "AudioSmith will need to download it again (712.8 MB).") {
		t.Fatalf("no second confirmation: %q", tt.Texts())
	}
	click(t, tt, "Delete Fast speech recognition for all apps")
	if onDisk(a, models.ASRFast) || a.modelStatus[models.ASRFast].Installed {
		t.Error("the files stayed")
	}
	if hit, err := otherApp(a, audioSmithID).ApplyTombstones(); err != nil || len(hit) != 1 {
		t.Errorf("AudioSmith's tombstones: %v %v", hit, err)
	}
}

func TestModelDeleteCancelShared(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1600)
	seedModels(t, a, models.LLMTranslate)
	useModels(t, a, models.LLMTranslate)
	claimFor(t, a, dictaID, models.LLMTranslate)
	claimFor(t, a, audioSmithID, models.LLMTranslate)
	a.navigate(modelsPath)
	if !has(tt, "Shared with AudioSmith and Dicta") {
		t.Fatalf("no Shared with: %q", tt.Texts())
	}
	click(t, tt, "Delete Translation model…")
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
	click(t, tt, "Delete Fast speech recognition…")
	click(t, tt, "Delete Fast speech recognition")
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
	fast, _ := models.Lookup(models.LLMFast)
	lock, err := fammodels.TryLockRepo(a.models.Hub(), fast.Repo, family.Shared, fammodels.Owner{BundleID: audioSmithID, Op: fammodels.OpDownload})
	if err != nil {
		t.Fatal(err)
	}
	a.models.Refresh()
	a.navigate(modelsPath)
	if !has(tt, "Downloading in AudioSmith…") || a.modelStatus[models.LLMFast].Elsewhere != audioSmithID {
		t.Fatalf("elsewhere: %+v %q", a.modelStatus[models.LLMFast], tt.Texts())
	}
	click(t, tt, "Download Fast language model") // disabled
	if st := a.models.Status(models.LLMFast); st.Downloading || st.Err != nil {
		t.Errorf("Download ran: %+v", st)
	}
	// AudioSmith finishes: installed on the next read of the family.
	lock.Release()
	modelstest.Seed(a.models.Hub(), models.LLMFast)
	a.models.Refresh()
	if !a.modelStatus[models.LLMFast].Installed || has(tt, "Downloading in AudioSmith…") {
		t.Errorf("not installed: %+v", a.modelStatus[models.LLMFast])
	}

	// Deleted for all apps in AudioSmith.
	useModels(t, a, models.LLMFast)
	if _, err := otherApp(a, audioSmithID).Delete(context.Background(), fast.Ref(""), fammodels.AllApps); err != nil {
		t.Fatal(err)
	}
	a.models.Refresh()
	if !has(tt, "Deleted in AudioSmith") || !has(tt, "Download again") {
		t.Errorf("deleted in: %+v %q", a.modelStatus[models.LLMFast], tt.Texts())
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
	tr, _ := models.Lookup(models.LLMTranslate)
	lock, err := fammodels.TryLockBlob(a.models.Hub(), tr.Repo, tr.Files[0].SHA256, fammodels.Owner{BundleID: dictaID, Op: fammodels.OpDownload})
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Release()
	a.navigate(modelsPath)
	click(t, tt, "Download Translation model")
	deadline := time.Now().Add(5 * time.Second)
	for !has(tt, "Downloading in Dicta… 0%") && time.Now().Before(deadline) {
		drain()
	}
	if !has(tt, "Downloading in Dicta… 0%") {
		t.Fatalf("waiting: %+v %q", a.models.Status(models.LLMTranslate), tt.Texts())
	}
	click(t, tt, "Cancel downloading Translation model")
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

// TestBorrowedModel: a model of another app on this Mac: Use it, then
// Remove from VoxtNew (its files stay for the app).
func TestBorrowedModel(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1600)
	name := uitext.ModelName(models.LLMQwen3)
	a.navigate(modelsPath)
	if has(tt, name) {
		t.Fatal("a borrowed model not on this Mac is listed")
	}
	qwen, _ := models.Lookup(models.LLMQwen3)
	if err := modelstest.SeedModel(a.models.Hub(), qwen); err != nil {
		t.Fatal(err)
	}
	claimFor(t, a, dictaID, models.LLMQwen3)
	if !has(tt, name) || !has(tt, "Installed by Dicta") || has(tt, "Download "+name) {
		t.Fatalf("borrowed: %q", tt.Texts())
	}
	click(t, tt, "Use "+name)
	if st := a.models.Status(models.LLMQwen3); !st.Installed || !st.Mine || !has(tt, "Shared with Dicta") {
		t.Fatalf("used: %+v %q", st, tt.Texts())
	}
	click(t, tt, "Remove "+name+"…")
	if !has(tt, fmt.Sprintf("Remove %q from VoxtNew?", name)) {
		t.Fatalf("dialog: %q", tt.Texts())
	}
	click(t, tt, "Remove from VoxtNew")
	if st := a.models.Status(models.LLMQwen3); st.Installed || !st.OnDisk || !has(tt, "Installed by Dicta") {
		t.Errorf("removed: %+v %q", st, tt.Texts())
	}
}

// TestModelDeleteOwnModels: the files of a model stay when another of
// VoxtNew's models uses them (rare: not in the catalogue today).
func TestModelDeleteOwnModels(t *testing.T) {
	res := []fammodels.Result{{OwnModels: []string{models.ASRAccurate}}}
	if got := deleteOutcome(models.ASRFast, res, nil); got != "Fast speech recognition was removed. Its files stay: Accurate speech recognition uses them too." {
		t.Errorf("%q", got)
	}
	if got := deleteOutcome(models.ASRFast, []fammodels.Result{{Deleted: true}}, nil); got != "" {
		t.Errorf("deleted: %q", got)
	}
	if got := deleteOutcome(models.ASRFast, nil, &fammodels.BusyError{}); !slices.Contains([]string{"Another app is downloading or using this model. Try again later."}, got) {
		t.Errorf("busy: %q", got)
	}
}
