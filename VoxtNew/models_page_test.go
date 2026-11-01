package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/hehehai/actnowkit/hfhub"
	fammodels "github.com/hehehai/actnowkit/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/actnow"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/hotkey"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models/modelstest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// seedModels installs the models ids (files of their sizes, no bytes)
// and opens a's models again over them, as a start would.
func seedModels(t *testing.T, a *app, ids ...string) {
	t.Helper()
	if err := modelstest.Seed(a.models.Hub(), ids...); err != nil {
		t.Fatal(err)
	}
	reopenModels(a)
}

// reopenModels reads the models on disk again, with the environment's
// options (VOXTNEW_HF_ENDPOINT).
func reopenModels(a *app) {
	a.models.Close()
	a.models, a.local, a.modelStatus = nil, nil, nil
	a.openModels()
}

func TestModelsPage(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1600) // every card on screen
	a.navigate(modelsPath)
	for _, s := range []string{"Accurate speech recognition", "Fast speech recognition", "Balanced language model",
		"Fast language model", "Translation model", "Default", "2.5 GB download", "3.3 GB download",
		"30 languages and 22 Chinese dialects, including Chinese, English, Cantonese, Japanese and Korean",
		"Translates between 33 languages, including Chinese, English, Japanese, Korean and Cantonese"} {
		if !has(tt, s) {
			t.Errorf("no %q in %q", s, tt.Texts())
		}
	}
	if has(tt, "Voice detection") || has(tt, "Installed") {
		t.Error("the VAD is listed, or a model is installed")
	}

	// Installed models: the page, the features' choices.
	seedModels(t, a, models.ASRFast, models.VADSilero, models.LLMTranslate)
	if !has(tt, "Installed") || !has(tt, "712.8 MB on disk") {
		t.Errorf("installed: %q", tt.Texts())
	}
	if m, ok := a.featureModel(local.ASR, hotkey.Transcribe); !ok || m.ID != models.ASRFast || m.Name != "Fast speech recognition" {
		t.Errorf("transcribe's speech model %+v", m)
	}
	if m, ok := a.featureModel(local.LLM, hotkey.Translate); !ok || m.ID != models.LLMTranslate {
		t.Errorf("translate's language model %+v", m)
	}
	if m, ok := a.featureModel(local.LLM, hotkey.Rewrite); ok {
		t.Errorf("rewrite uses the translation model: %+v", m)
	}
	a.navigate("/rewrite")
	if !has(tt, "No language model is installed: install one to rewrite.") {
		t.Errorf("rewrite: %q", tt.Texts())
	}
	a.navigate("/translate")
	if !has(tt, "Automatic: Translation model") {
		t.Errorf("translate: %q", tt.Texts())
	}

	// Deleting asks first.
	a.navigate(modelsPath)
	click(t, tt, "Delete Fast speech recognition…")
	if !has(tt, "Delete Fast speech recognition?") || !has(tt, "This frees 712.8 MB. You can download it again later.") {
		t.Fatalf("no confirmation: %q", tt.Texts())
	}
	click(t, tt, "Delete Fast speech recognition")
	if a.models.Installed(models.ASRFast) || a.modelStatus[models.ASRFast].Installed {
		t.Error("still installed")
	}
	if fast, _ := models.Lookup(models.ASRFast); models.Installed(a.models.Hub(), fast) != "" {
		t.Error("the snapshot stayed")
	}
	if len(a.installed(local.ASR)) != 0 || a.notes == nil {
		t.Error("installed list")
	}
}

// TestModelDownloadStates: a download in progress (Cancel), stopped
// (Resume), failed (Try again), against a fake Hub.
func TestModelDownloadStates(t *testing.T) {
	release := make(chan struct{})
	var mode atomic.Value // "hold" | "missing"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if mode.Load() == "missing" {
			http.Error(w, `{"error":"Repository not found"}`, http.StatusNotFound)
			return
		}
		if rest, ok := strings.CutPrefix(r.URL.Path, "/api/models/"); ok {
			repo, _, _ := strings.Cut(rest, "/revision/")
			fakeModelInfo(w, repo)
			return
		}
		if r.Header.Get("Range") != "" && !strings.HasPrefix(r.Header.Get("Range"), "bytes=0-") {
			select { // the other parts never come either
			case <-release:
			case <-r.Context().Done():
			}
			return
		}
		size := fakeFileSize(r.URL.Path)
		if rg := r.Header.Get("Range"); rg != "" { // the first part
			var from, to int64
			fmt.Sscanf(rg, "bytes=%d-%d", &from, &to)
			w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", from, to, size))
			w.Header().Set("Content-Length", fmt.Sprint(to-from+1))
			w.WriteHeader(http.StatusPartialContent)
		} else {
			w.Header().Set("Content-Length", fmt.Sprint(size))
			w.WriteHeader(http.StatusOK)
		}
		w.Write([]byte(strings.Repeat("x", 100))) // less than any file: the rest never comes
		w.(http.Flusher).Flush()
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	defer close(release)
	t.Setenv("HF_ENDPOINT", srv.URL)
	reserve := modelsReserve
	modelsReserve = -1 // no free-space check
	defer func() { modelsReserve = reserve }()
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1600)
	drain := queuePosts(a)
	reopenModels(a)
	a.navigate(modelsPath)

	mode.Store("hold")
	click(t, tt, "Download Fast language model")
	deadline := time.Now().Add(5 * time.Second)
	for !(has(tt, "Downloading: ") && partialBytes(a, models.LLMFast) > 0) && time.Now().Before(deadline) { // bytes came: a cancel keeps them
		drain()
	}
	if !has(tt, "Cancel") || !a.modelStatus[models.LLMFast].Downloading {
		t.Fatalf("downloading: %+v %+v %q", a.modelStatus[models.LLMFast], a.models.Status(models.LLMFast), tt.Texts()[25:])
	}
	click(t, tt, "Cancel downloading Fast language model")
	drain()
	st := a.modelStatus[models.LLMFast]
	if st.Downloading || st.Done == 0 || st.Err != nil || !has(tt, "Resume") || !has(tt, "Stopped: ") {
		t.Fatalf("stopped: %+v %q", st, tt.Texts())
	}

	// A failure: what the card says, and Try again.
	mode.Store("missing")
	click(t, tt, "Download Translation model")
	deadline = time.Now().Add(5 * time.Second)
	for !has(tt, "The download server does not have this model.") && time.Now().Before(deadline) {
		drain()
	}
	if !has(tt, "The download server does not have this model.") || !has(tt, "Try again") {
		t.Fatalf("failed: %q", tt.Texts())
	}

	// Deleting what a stopped download kept.
	click(t, tt, "Delete what was downloaded of Fast language model")
	drain()
	if st := a.modelStatus[models.LLMFast]; st.Done != 0 || has(tt, "Resume") {
		t.Errorf("kept: %+v", st)
	}
}

// partialBytes are the bytes of model id's unfinished downloads on disk.
func partialBytes(a *app, id string) int64 {
	m, _ := models.Lookup(id)
	var n int64
	incomplete, _ := filepath.Glob(filepath.Join(fammodels.RepoDir(a.models.Hub(), m.Repo), "blobs", "*.incomplete"))
	for _, f := range incomplete {
		if fi, err := os.Stat(f); err == nil {
			n += fi.Size()
		}
	}
	return n
}

// fakeModelInfo answers the Hub's model info for a catalogue repo, at
// its pinned commit (hfhub plans a download with it).
func fakeModelInfo(w http.ResponseWriter, repo string) {
	type lfs struct {
		SHA256 string `json:"sha256"`
		Size   int64  `json:"size"`
	}
	type sibling struct {
		Path   string `json:"rfilename"`
		Size   int64  `json:"size"`
		BlobID string `json:"blobId"`
		LFS    *lfs   `json:"lfs,omitempty"`
	}
	for _, m := range models.Catalog() {
		if m.Repo != repo {
			continue
		}
		var sib []sibling
		for i, f := range m.Files {
			x := sibling{Path: f.Path, Size: f.Size, BlobID: fmt.Sprintf("%040x", i+1)}
			if f.LFS {
				x.LFS = &lfs{SHA256: f.SHA256, Size: f.Size}
			}
			sib = append(sib, x)
		}
		json.NewEncoder(w).Encode(map[string]any{"id": repo, "sha": m.Revision, "siblings": sib})
		return
	}
	http.NotFound(w, nil)
}

// fakeFileSize is the size of the catalogue's file at a Hub path
// (/<owner>/<name>/resolve/<rev>/<path>).
func fakeFileSize(p string) int64 {
	parts := strings.SplitN(strings.TrimPrefix(p, "/"), "/", 5)
	if len(parts) < 5 {
		return 0
	}
	for _, m := range models.Catalog() {
		if m.Repo == parts[0]+"/"+parts[1] {
			for _, f := range m.Files {
				if f.Path == parts[4] {
					return f.Size
				}
			}
		}
	}
	return 0
}

// modelShots saves the Local models page with a model of each state
// (TestSnapshots): installed (one shared with another ActNow app),
// downloading (waiting for a file another app downloads), stopped,
// failed, another app's model VoxtNew can use; the family's states
// (downloading elsewhere, deleted elsewhere); the delete dialog of a
// shared model, the Network dialog; and Settings → Storage with the
// model folder and Erase everything keeping a shared model.
func modelShots(t *testing.T, a *app, tt *ui.Tester, save func(string)) {
	seedModels(t, a, models.ASRAccurate, models.VADSilero, models.LLMTranslate)
	qwen, _ := models.Lookup(models.LLMQwen3)
	if err := modelstest.SeedModel(a.models.Hub(), qwen); err != nil {
		t.Fatal(err)
	}
	useModels(t, a, models.ASRAccurate, models.VADSilero, models.LLMTranslate)
	claimFor(t, a, audioSmithID, models.ASRAccurate, models.VADSilero)
	claimFor(t, a, dictaID, models.LLMQwen3)
	// Both installed: the shared settings speak of the family.
	addPeer(t, a, audioSmithID, nil)
	addPeer(t, a, dictaID, nil)
	a.familyFound(a.fam.Discover())
	fast, _ := models.Lookup(models.ASRFast)
	balanced, _ := models.Lookup(models.LLMBalanced)
	a.modelStatus[models.ASRFast] = models.Status{Done: fast.Size() / 3, Total: fast.Size()}
	a.modelStatus[models.LLMBalanced] = models.Status{Downloading: true, Done: balanced.Size() * 2 / 5, Total: balanced.Size(), WaitingFor: dictaID}
	a.modelStatus[models.LLMFast] = models.Status{Total: 1, Err: fmt.Errorf("needs 1.7 GB: %w", hfhub.ErrDiskSpace)}
	a.navigate(modelsPath)
	// Two of them loaded (models_memory.go), as a session leaves them.
	a.modelMem = map[string]int64{models.ASRAccurate: 2_469_606_195, models.LLMTranslate: 1_234_173_952}
	save("models-states")
	scrollToText(tt, i18n.T("Language models"))
	save("models-states-language")
	// The family's states: another app downloads one, another deleted one.
	a.modelStatus[models.ASRFast] = models.Status{Elsewhere: audioSmithID, Done: fast.Size() / 4}
	a.modelStatus[models.LLMFast] = models.Status{DeletedBy: dictaID}
	a.navigate("/home")
	a.navigate(modelsPath)
	save("models-family")
	scrollToText(tt, i18n.T("Fast language model"))
	save("models-family-language") // deleted elsewhere, another app's model to use
	a.askDeleteModel(models.ASRAccurate)
	save("models-delete-shared")
	sd := *a.modelShared
	sd.forAll = true
	a.modelShared = &sd
	save("models-delete-all")
	a.modelShared = nil
	a.updateSettings(func(s *settings.Settings) {
		s.Family.Follow = map[string]bool{actnow.KeyProxy: false}
		s.Family.Conflicts = []string{actnow.KeyProxy}
		s.Network.Proxy = "http://127.0.0.1:7890"
	})
	a.openNetwork()
	save("models-network")
	a.netOpen = false
	a.updateSettings(func(s *settings.Settings) { s.Family = settings.Family{}; s.Network = settings.Network{} })
	a.stReport = nil // measured again
	a.navigate("/settings/storage")
	frames(tt)
	pinStorage(a)
	save("settings-storage-models")
	a.askErase()
	a.resetText = "DELETE"
	save("settings-storage-erase")
	a.resetAsk = ""
	a.settingsOpen = false
	// The Advanced section under the cards, folded and open
	// (models_keep.go).
	a.navigate(modelsPath)
	scrollToText(tt, i18n.T("Advanced"))
	save("models-advanced")
	a.modelAdvanced = true
	scrollToText(tt, i18n.T("Advanced"))
	save("models-advanced-open")
	a.modelAdvanced = false
	// The page as it was: the setup guide's shots show it behind them.
	a.navigate("/home")
	a.navigate(modelsPath)
	frames(tt)
	scrollToText(tt, i18n.T("Fast language model"))
}
