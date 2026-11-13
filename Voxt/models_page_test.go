package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/hehehai/actnowkit/hfhub"
	fammodels "github.com/hehehai/actnowkit/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/actnow"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models/modelstest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
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
	skipWindowsActNow(t)
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 4800) // the retired models are below the speech model cards
	a.navigate(modelsPath)
	frames(tt)
	texts := tt.Texts()
	for _, family := range []string{"Qwen3-ASR", "Whisper"} {
		i := slices.Index(texts, family)
		if i < 0 || i+1 == len(texts) || texts[i+1] != "3 models" {
			t.Errorf("expected a card with three %s models: %q", family, texts)
		}
	}
	for _, s := range []string{"Qwen3-ASR 1.7B (8bit)", "Qwen3-ASR 0.6B (4bit)", "Default", "2.5 GB download",
		"Local", "Accurate", "Fast", "Realtime", "Supports your main language",
		"30 languages and 22 Chinese dialects, including Chinese, English, Cantonese, Japanese and Korean"} {
		if !has(tt, s) {
			t.Errorf("no %q in %q", s, tt.Texts())
		}
	}
	if has(tt, "Silero VAD v6") || has(tt, "Installed") || has(tt, "Qwen3.5 4B OptiQ (4bit)") || has(tt, "Language models no longer used") || has(tt, "Faster transcription") {
		t.Error("the VAD, a language model or a draft is listed, or a model is installed")
	}

	// Installed models: the page (a speech model's draft with it), the
	// features' choices. A language model Voxt 2.0 downloaded is retired:
	// listed, to be deleted, and offered to no feature.
	seedModels(t, a, models.ASRFast, models.VADSilero, models.LLMTranslate)
	if !has(tt, "3 models · 1 installed") || !has(tt, "Installed") || !has(tt, "712.8 MB on disk") || !has(tt, "· Used by Transcribe · Translate · Rewrite") ||
		!has(tt, "Language models no longer used") || !has(tt, "Hy-MT2 1.8B (Q4_K_M)") ||
		!has(tt, "Faster transcription") || !has(tt, "262.5 MB download") {
		t.Errorf("installed: %q", tt.Texts())
	}
	if m, ok := a.featureModel(local.ASR, hotkey.Transcribe); !ok || m.ID != models.ASRFast || m.Name != "Qwen3-ASR 0.6B (4bit)" {
		t.Errorf("transcribe's speech model %+v", m)
	}
	if m, ok := a.featureModel(local.LLM, hotkey.Translate); ok {
		t.Errorf("translate uses a retired model: %+v", m)
	}
	a.navigate("/translate")
	if !has(tt, "No language model is set up. Set one up to translate.") || has(tt, "Automatic (now: Hy-MT2 1.8B (Q4_K_M))") {
		t.Errorf("translate: %q", tt.Texts())
	}

	// Deleting asks first.
	a.navigate(modelsPath)
	click(t, tt, "Delete Qwen3-ASR 0.6B (4bit)…")
	if !has(tt, "Delete Qwen3-ASR 0.6B (4bit)?") || !has(tt, "This frees 712.8 MB. You can download it again later.") {
		t.Fatalf("no confirmation: %q", tt.Texts())
	}
	click(t, tt, "Delete Qwen3-ASR 0.6B (4bit)")
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

// TestModelsPageDraftAlone: a draft on disk without its speech model
// (the model deleted first) keeps its row, so it can be deleted too; so
// does a draft still downloading.
func TestModelsPageDraftAlone(t *testing.T) {
	skipWindowsActNow(t)
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1600)
	seedModels(t, a, models.ASRFastDraft)
	a.navigate(modelsPath)
	if !has(tt, "Faster transcription") || !has(tt, "262.5 MB on disk") || has(tt, "712.8 MB on disk") {
		t.Fatalf("the draft alone: %q", tt.Texts())
	}
	click(t, tt, "Delete Qwen3-ASR 0.6B DFlash draft…")
	click(t, tt, "Delete Qwen3-ASR 0.6B DFlash draft")
	if has(tt, "Faster transcription") || a.models.Installed(models.ASRFastDraft) {
		t.Errorf("after the delete: %q", tt.Texts())
	}
	draft, _ := models.Lookup(models.ASRFastDraft)
	a.modelStatus[models.ASRFastDraft] = models.Status{Downloading: true, Done: draft.Size() / 4, Total: draft.Size()}
	a.navigate("/home")
	a.navigate(modelsPath)
	if !has(tt, "Faster transcription") || !has(tt, "Cancel") {
		t.Errorf("downloading: %q", tt.Texts())
	}
}

// TestModelsPageSystemDictation: System dictation's card says it is
// built in and which features use it; once a speech model is downloaded,
// Automatic moves the features to that one.
func TestModelsPageSystemDictation(t *testing.T) {
	a, fs, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1600)
	a.dictation.SetHelper(fs) // as helperChanged does for a helper with speech.*
	a.navigate(modelsPath)
	used := func() int {
		return strings.Count(strings.Join(tt.Texts(), "\n"), "· Used by Transcribe · Translate · Rewrite")
	}
	if !has(tt, "System dictation") || !has(tt, "Built-in") || used() != 1 {
		t.Fatalf("no dictation card, or not used: %q", tt.Texts())
	}
	seedModels(t, a, models.ASRFast, models.VADSilero)
	a.dictation.SetHelper(fs) // seedModels made the services anew
	if m, _ := a.featureModel(local.ASR, hotkey.Transcribe); m.ID != models.ASRFast || used() != 1 || !has(tt, "System dictation") {
		t.Errorf("after a download: transcribe uses %s, %d cards used, %q", m.ID, used(), tt.Texts())
	}
}

// TestHideModel: a speech model's card hides it, as Voxt 1.x's model
// list did: the features' menus leave it out, its card goes into the
// folded Hidden models section, whose Show puts it back. A model in use
// has no Hide, and one in use since it was hidden is listed again.
func TestHideModel(t *testing.T) {
	a, fs, tt := newTestApp(t)
	tt.SetSize(windowWidth, 4800) // the hidden models under every card
	seedModels(t, a, models.ASRFast, models.VADSilero)
	a.dictation.SetHelper(fs) // System dictation after the speech model: not in use
	fast, systemDictation, accurate := uitext.ModelName(models.ASRFast), uitext.ModelName(dictation.ModelID), uitext.ModelName(models.ASRAccurate)
	menu := func(path, label string) []string {
		t.Helper()
		a.navigate(path)
		click(t, tt, label)
		texts := tt.Texts()
		tt.Key(0, ui.KeyEscape)
		frames(tt)
		return texts
	}
	labels := func() []string { a.navigate(modelsPath); frames(tt); return tt.AccessLabels() }
	if l := labels(); slices.Contains(l, "Hide "+fast) || !slices.Contains(l, "Hide "+systemDictation) || !slices.Contains(l, "Hide "+accurate) {
		t.Fatalf("Hide on the model in use, or not on the others: %q", l)
	}
	if !slices.Contains(menu("/transcribe", "Speech model for Transcribe"), systemDictation) {
		t.Fatalf("System dictation is not offered: %q", tt.Texts())
	}

	a.navigate(modelsPath)
	click(t, tt, "Hide "+systemDictation)
	click(t, tt, "Hide "+accurate)
	if has(tt, systemDictation) || has(tt, accurate) || !has(tt, "Hidden models") {
		t.Fatalf("hidden cards still listed, or no Hidden models: %q", tt.Texts())
	}
	for _, f := range []struct{ path, label string }{{"/transcribe", "Speech model for Transcribe"}, {"/translate", "Speech model for Translate"}} {
		if texts := menu(f.path, f.label); slices.Contains(texts, systemDictation) || !slices.Contains(texts, "Automatic (now: "+fast+")") {
			t.Errorf("%s offers a hidden model: %q", f.path, texts)
		}
	}

	// A hidden model chosen for a feature (Voxt 1.x's settings, or the
	// only one left) is in use: listed, offered, and neither hidden nor
	// shown until the feature leaves it.
	a.updateSettings(func(s *settings.Settings) { s.Features.Translate.ASR = dictation.ModelID })
	if l := labels(); !has(tt, systemDictation) || slices.Contains(l, "Hide "+systemDictation) || slices.Contains(l, "Show "+systemDictation) {
		t.Errorf("the hidden model in use: %q", l)
	}
	if !slices.Contains(menu("/transcribe", "Speech model for Transcribe"), systemDictation) {
		t.Errorf("the model in use is not offered: %q", tt.Texts())
	}
	a.updateSettings(func(s *settings.Settings) { s.Features.Translate.ASR = "" })

	// Show, in the folded section; the section goes with the last one.
	a.navigate(modelsPath)
	click(t, tt, "Hidden models")
	click(t, tt, "Show "+systemDictation)
	if !slices.Contains(menu("/transcribe", "Speech model for Transcribe"), systemDictation) {
		t.Errorf("shown again but not offered: %q", tt.Texts())
	}
	a.navigate(modelsPath)
	click(t, tt, "Show "+accurate)
	if !has(tt, systemDictation) || !has(tt, accurate) || has(tt, "Hidden models") {
		t.Errorf("after Show: %q", tt.Texts())
	}
}

// TestModelDownloadStates: a download in progress (Cancel), stopped
// (Resume), failed (Try again), against a fake Hub.
func TestModelDownloadStates(t *testing.T) {
	skipWindowsActNow(t)
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
	seedModels(t, a, models.VADSilero) // the speech models' voice detection: only the model downloads
	a.navigate(modelsPath)

	mode.Store("hold")
	click(t, tt, "Download Qwen3-ASR 0.6B (4bit)")
	deadline := time.Now().Add(5 * time.Second)
	for !(has(tt, "Downloading: ") && partialBytes(a, models.ASRFast) > 0) && time.Now().Before(deadline) { // bytes came: a cancel keeps them
		drain()
	}
	if !has(tt, "Cancel") || !a.modelStatus[models.ASRFast].Downloading {
		t.Fatalf("downloading: %+v %+v %q", a.modelStatus[models.ASRFast], a.models.Status(models.ASRFast), tt.Texts()[25:])
	}
	click(t, tt, "Cancel downloading Qwen3-ASR 0.6B (4bit)")
	drain()
	st := a.modelStatus[models.ASRFast]
	if st.Downloading || st.Done == 0 || st.Err != nil || !has(tt, "Resume") || !has(tt, "Stopped: ") {
		t.Fatalf("stopped: %+v %q", st, tt.Texts())
	}

	// A failure: what the card says, and Try again.
	mode.Store("missing")
	click(t, tt, "Download Qwen3-ASR 1.7B (8bit)")
	deadline = time.Now().Add(5 * time.Second)
	for !has(tt, "The download server does not have this model.") && time.Now().Before(deadline) {
		drain()
	}
	if !has(tt, "The download server does not have this model.") || !has(tt, "Try again") {
		t.Fatalf("failed: %q", tt.Texts())
	}

	// Deleting what a stopped download kept.
	click(t, tt, "Delete the partial download of Qwen3-ASR 0.6B (4bit)")
	drain()
	if st := a.modelStatus[models.ASRFast]; st.Done != 0 || has(tt, "Resume") {
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

// modelShots saves the Speech models page with a model of each state
// (TestSnapshots): installed (one shared with another ActNow app),
// downloading (waiting for a file another app downloads), a draft
// downloading, stopped, failed, another app's model VoxtNew can use; the family's states
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
	draft, _ := models.Lookup(models.ASRAccurateDraft) // its bar under the row's text
	a.modelStatus[models.ASRAccurateDraft] = models.Status{Downloading: true, Done: draft.Size() / 2, Total: draft.Size()}
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
	scrollToText(tt, uitext.ModelName(models.LLMFast))
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
	// A hidden model, in its section opened (models_cards.go).
	fastName := uitext.ModelName(models.ASRFast)
	a.navigate("/home")
	a.navigate(modelsPath)
	scrollToText(tt, fastName) // on screen: a click off it does nothing (the smallest window)
	click(t, tt, i18n.T("Hide %s", fastName))
	scrollToText(tt, i18n.T("Hidden models"))
	click(t, tt, i18n.T("Hidden models"))
	scrollToText(tt, i18n.T("Hidden models"))
	save("models-hidden")
	click(t, tt, i18n.T("Show %s", fastName))
	// The page as it was: the setup guide's shots show it behind them.
	a.navigate("/home")
	a.navigate(modelsPath)
	frames(tt)
	scrollToText(tt, uitext.ModelName(models.LLMFast))
}
