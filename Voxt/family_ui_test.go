package main

// The other ActNow apps in VoxtNew's views: the setup guide's reuse card
// and alternative line (onboarding_models.go), the shortcut conflicts'
// warnings and prompt (settings_shortcuts_family.go).

import (
	"context"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync/atomic"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/hehehai/actnowkit/conformance"
	"github.com/hehehai/actnowkit/family"
	fammodels "github.com/hehehai/actnowkit/models"
	"github.com/hehehai/actnowkit/registry"
	"github.com/hehehai/actnowkit/shortcuts"
	"github.com/sgl-project/sglang-omni/Voxt/internal/actnow"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// addPeer installs another family app (a fake .app and its manifest,
// with links and shortcuts) in a's family directory, and returns its
// view of the model library, for its claims. Discovery finds it once a
// test runs a.familyFound(a.fam.Discover()).
func addPeer(t *testing.T, a *app, bundleID string, links map[string]string, sc ...shortcuts.Shortcut) *fammodels.Store {
	t.Helper()
	fa, _, err := family.ParseBundleID(bundleID)
	if err != nil {
		t.Fatal(err)
	}
	path, err := conformance.FakeApp(t.TempDir(), fa.Name, bundleID)
	if err != nil {
		t.Fatal(err)
	}
	m := registry.Manifest{BundleID: bundleID, Name: fa.Name, Version: "1.0.0", Path: path, Links: links, Shortcuts: sc,
		Capabilities: []string{registry.CapModelsHF}, Models: &registry.ModelsInfo{Root: a.models.Hub()}}
	if err := registry.Write(a.fam.Dir(), m); err != nil {
		t.Fatal(err)
	}
	return &fammodels.Store{Family: a.fam.Dir(), Hub: a.models.Hub(), BundleID: bundleID, AppID: fa.ID,
		Apps: func() map[string]fammodels.AppState { return registry.States(a.fam.Found()) }}
}

// peerClaims makes peer claim the models ids, as the app that downloaded
// them.
func peerClaims(t *testing.T, peer *fammodels.Store, ids ...string) {
	t.Helper()
	for _, id := range ids {
		m, _ := models.Lookup(id)
		if err := peer.Claim(context.Background(), m.Claim("", fammodels.OriginDownloaded)); err != nil {
			t.Fatal(err)
		}
	}
}

// countingHub is a Hub that answers nothing (404) and counts what was
// asked of it: model steps that must not download.
func countingHub(t *testing.T) *atomic.Int64 {
	t.Helper()
	var n atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n.Add(1)
		http.Error(w, `{"error":"Repository not found"}`, http.StatusNotFound)
	}))
	t.Cleanup(srv.Close)
	t.Setenv("HF_ENDPOINT", srv.URL)
	return &n
}

// voxtnewClaim is VoxtNew's claim of model id, if any.
func voxtnewClaim(t *testing.T, a *app, id string) (fammodels.Claim, bool) {
	t.Helper()
	cs, err := fammodels.ReadClaims(a.fam.Dir(), a.fam.BundleID())
	if err != nil {
		t.Fatal(err)
	}
	m, _ := models.Lookup(id)
	return cs.Find(m.Ref(""))
}

// TestOnboardingReuse: the speech models AudioSmith installed make the
// reuse card; Use them claims them for VoxtNew (origin reused) and
// downloads nothing; the language model is still to set up.
func TestOnboardingReuse(t *testing.T) {
	asked := countingHub(t)
	a, tt := obTester(t)
	reopenModels(a) // the counting Hub
	smith := addPeer(t, a, "dev.hehehai.audiosmith", nil)
	seedModels(t, a, models.ASRAccurate, models.VADSilero)
	peerClaims(t, smith, models.ASRAccurate, models.VADSilero)
	a.familyFound(a.fam.Discover())
	a.models.Refresh()
	a.obGo(obModels)
	for _, s := range []string{"2 models are already on this Mac", "Installed by AudioSmith · 2.5 GB · no download needed",
		"Qwen3-ASR 1.7B (8bit) · Silero VAD v6", "Use them", "Choose models…", "Installed · from AudioSmith",
		"Set up a language model"} {
		if !has(tt, s) {
			t.Errorf("no %q in %q", s, tt.Texts())
		}
	}
	if !a.obReuseShown() {
		t.Error("Continue stays the primary button")
	}
	click(t, tt, "Use them")
	for _, id := range []string{models.ASRAccurate, models.VADSilero} {
		if c, ok := voxtnewClaim(t, a, id); !ok || c.Origin != fammodels.OriginReused {
			t.Errorf("%s: claim %+v %v", id, c, ok)
		}
	}
	if has(tt, "Use them") || !has(tt, "Ready: Qwen3-ASR 1.7B (8bit)") || !has(tt, "Set up a language model") || hasPrefix(tt, "Download (") {
		t.Errorf("after Use them: %q", tt.Texts())
	}
	for _, m := range models.Catalog() {
		if a.models.Status(m.ID).Downloading {
			t.Errorf("%s downloads", m.ID)
		}
	}
	if n := asked.Load(); n != 0 {
		t.Errorf("the Hub was asked %d times", n)
	}

	// Choose models… opens the Speech models page.
	reuseAgain(t, a)
	click(t, tt, "Choose models…")
	if a.ob.open || a.shownPath() != modelsPath {
		t.Errorf("choose models: %v %s", a.ob.open, a.shownPath())
	}
}

// reuseAgain forgets VoxtNew's claims and opens the guide's models step
// again: the card shows again.
func reuseAgain(t *testing.T, a *app) {
	t.Helper()
	var refs []fammodels.Ref
	for _, m := range models.Catalog() {
		refs = append(refs, m.Ref(""))
	}
	if err := a.models.Store().Unclaim(refs...); err != nil {
		t.Fatal(err)
	}
	a.models.Refresh()
	a.openOnboarding()
	a.obGo(obModels)
}

// TestOnboardingAlternative: the accurate speech model missing, the fast
// one AudioSmith installed is offered for now; Download stays the
// default; Use claims it, and the step says it is ready.
func TestOnboardingAlternative(t *testing.T) {
	asked := countingHub(t)
	a, tt := obTester(t)
	drain := queuePosts(a)
	reopenModels(a)
	smith := addPeer(t, a, "dev.hehehai.audiosmith", nil)
	seedModels(t, a, models.ASRFast, models.VADSilero)
	peerClaims(t, smith, models.ASRFast)
	a.familyFound(a.fam.Discover())
	a.models.Refresh()
	drain()
	a.obGo(obModels)
	drain()
	line := "Qwen3-ASR 1.7B (8bit) isn't installed, but Qwen3-ASR 0.6B (4bit) (from AudioSmith) is. Use it for now?"
	for _, s := range []string{line, "Download Qwen3-ASR 1.7B (8bit)", "Use Qwen3-ASR 0.6B (4bit)",
		"1 model is already on this Mac", "Silero VAD v6", "Set up a language model"} {
		if !has(tt, s) {
			t.Errorf("no %q in %q", s, tt.Texts())
		}
	}
	if hasPrefix(tt, "Download and continue") || has(tt, "Ready: Qwen3-ASR 0.6B (4bit)") {
		t.Errorf("the speech model is in the step's Download, or ready: %q", tt.Texts())
	}

	// Download (the default) starts the accurate one.
	obScrollTo(tt, "Download Qwen3-ASR 1.7B (8bit)")
	click(t, tt, "Download Qwen3-ASR 1.7B (8bit)")
	deadline := time.Now().Add(5 * time.Second)
	for asked.Load() == 0 && time.Now().Before(deadline) {
		drain()
	}
	for a.models.Status(models.ASRAccurate).Downloading && time.Now().Before(deadline) {
		drain()
	}
	drain()
	if asked.Load() == 0 {
		t.Fatalf("Download asked nothing of the Hub: %+v", a.models.Status(models.ASRAccurate))
	}

	// Use it for now.
	obScrollTo(tt, "Use Qwen3-ASR 0.6B (4bit)")
	click(t, tt, "Use Qwen3-ASR 0.6B (4bit)")
	drain()
	if c, ok := voxtnewClaim(t, a, models.ASRFast); !ok || c.Origin != fammodels.OriginReused {
		t.Errorf("claim %+v %v", c, ok)
	}
	if has(tt, line) || !has(tt, "Ready: Qwen3-ASR 0.6B (4bit)") {
		t.Errorf("after Use: %q", tt.Texts())
	}
}

// obScrollTo scrolls the guide's step until s is in its middle.
func obScrollTo(tt *ui.Tester, s string) {
	frames(tt)
	if r, ok := tt.Find(s); ok {
		tt.Scroll(windowWidth/2, windowHeight/2, 0, r.Y-windowHeight/2)
		frames(tt)
	}
}

// conflictPeers installs Auralink, whose Transcribe has VoxtNew's
// Transcribe keys and got them before VoxtNew (VoxtNew prompts), and
// AudioSmith, whose Dictation has Translate's keys and got them after
// VoxtNew (a warning only); then discovers them.
func conflictPeers(t *testing.T, a *app) (aura, smith shortcuts.Shortcut) {
	t.Helper()
	a.updateSettings(func(s *settings.Settings) { actnow.StampKeys(s, time.Now()) })
	f := settings.Defaults().Features
	norm := func(keys string) string {
		k, err := shortcuts.Normalize(keys)
		if err != nil {
			t.Fatal(err)
		}
		return k
	}
	before, after := time.Now().Add(-24*time.Hour), time.Now().Add(24*time.Hour)
	aura = shortcuts.Shortcut{ID: "transcribe", Title: "Transcribe", Keys: norm(f.Transcribe.Hotkeys[0].Keys), Trigger: shortcuts.Tap,
		Enabled: true, Mechanism: shortcuts.EventTap, EnabledAt: &before}
	addPeer(t, a, "dev.hehehai.auralink", map[string]string{registry.LinkShortcuts: "auralink://settings/shortcuts"}, aura,
		shortcuts.Shortcut{ID: "rewrite", Title: "Rewrite", Keys: "Control+Option+K", Trigger: shortcuts.Press, Enabled: true, Mechanism: shortcuts.EventTap, EnabledAt: &before})
	smith = shortcuts.Shortcut{ID: "dictation", Title: "Dictation", Keys: norm(f.Translate.Hotkeys[0].Keys), Trigger: shortcuts.Press,
		Enabled: true, Mechanism: shortcuts.Carbon, EnabledAt: &after}
	addPeer(t, a, "dev.hehehai.audiosmith", nil, smith)
	a.familyFound(a.fam.Discover())
	return aura, smith
}

// TestShortcutConflicts: the rows of Settings → Shortcuts say which app
// uses their keys too; the prompt shows once, for the conflict VoxtNew
// is the later of, and is remembered whatever the answer.
func TestShortcutConflicts(t *testing.T) {
	a, _, tt := newTestApp(t)
	conflictPeers(t, a)
	if len(a.famConflicts) != 2 {
		t.Fatalf("conflicts %+v", a.famConflicts)
	}
	keys := uitext.HotkeyText(settings.Defaults().Features.Transcribe.Hotkeys[0].Keys)
	prompt := keys + " is also the Transcribe shortcut in Auralink. Pressing it will trigger both apps."
	if !has(tt, prompt) || !has(tt, "Open Auralink settings") || has(tt, "Dictation shortcut in AudioSmith") {
		t.Fatalf("prompt: %q", tt.Texts())
	}
	click(t, tt, "Open Auralink settings")
	if !slices.Contains(tt.OpenedURLs(), "auralink://settings/shortcuts") || has(tt, prompt) {
		t.Errorf("open: %q %q", tt.OpenedURLs(), tt.Texts())
	}
	key := a.famConflicts[slices.IndexFunc(a.famConflicts, func(c shortcuts.Conflict) bool { return c.Prompt })].Key()
	if got := a.lib.settings.Get().Family.Prompted; !slices.Equal(got, []string{key}) {
		t.Errorf("prompted %q", got)
	}
	// Once only, through the next discoveries.
	a.familyFound(a.fam.Discover())
	if has(tt, prompt) {
		t.Error("prompted again")
	}

	// The rows warn, both conflicts.
	a.navigate("/settings/shortcuts")
	for _, s := range []string{"Also used by Auralink (Transcribe)", "Also used by AudioSmith (Dictation)"} {
		if !has(tt, s) {
			t.Errorf("no %q in %q", s, tt.Texts())
		}
	}
	a.settingsOpen = false

	// Change shortcut: the feature's page; Ignore and Escape record it too.
	for _, answer := range []string{"Change shortcut", "Ignore"} {
		a.updateSettings(func(s *settings.Settings) { s.Family.Prompted = nil })
		a.shortcutPrompt()
		click(t, tt, answer)
		if has(tt, prompt) || len(a.lib.settings.Get().Family.Prompted) != 1 {
			t.Errorf("%s: %q %q", answer, a.lib.settings.Get().Family.Prompted, tt.Texts())
		}
		if answer == "Change shortcut" && a.shownPath() != "/transcribe" {
			t.Errorf("change: %s", a.shownPath())
		}
	}
}

// TestShortcutPromptWaitsForGuide: on the first start the prompt waits
// for the setup guide, whose steps warn in its place.
func TestShortcutPromptWaitsForGuide(t *testing.T) {
	a, tt := obTester(t)
	conflictPeers(t, a)
	if a.scPrompt == nil || has(tt, "Change shortcut") {
		t.Fatalf("prompt %+v over the guide: %q", a.scPrompt, tt.Texts())
	}
	a.obGo(obTranscribe)
	if !has(tt, "Also used by Auralink (Transcribe)") {
		t.Errorf("try step: %q", tt.Texts())
	}
	a.obGo(obMore)
	for _, s := range []string{"Also used by Auralink (Transcribe)", "Also used by AudioSmith (Dictation)"} {
		if !has(tt, s) {
			t.Errorf("more: no %q in %q", s, tt.Texts())
		}
	}
	click(t, tt, "Start using Voxt")
	if !has(tt, "Change shortcut") {
		t.Errorf("no prompt after the guide: %q", tt.Texts())
	}
}

// TestShortcutRecorderUsedBy: keys another app uses say so under the
// recorder once they are pressed (and saved: saving is allowed).
func TestShortcutRecorderUsedBy(t *testing.T) {
	a, _, tt := newTestApp(t)
	conflictPeers(t, a)
	a.famStarted = true // the manifest and conflicts follow the settings (family.go)
	a.scPrompt = nil
	a.navigate("/translate")
	if !has(tt, "Also used by AudioSmith (Dictation)") {
		t.Errorf("translate: %q", tt.Texts())
	}
	a.navigate("/rewrite")
	if has(tt, "Also used by Auralink (Rewrite)") {
		t.Fatal("warned before")
	}
	row := &a.fui(hotkey.Rewrite).rows[0]
	row.sel = "Custom…"
	a.chooseKeys(hotkey.Rewrite, 0, row, nil)
	frames(tt)
	click(t, tt, "Custom keys of shortcut 1")
	tt.Key(ui.Ctrl|ui.Alt, ui.KeyK)
	frames(tt)
	if hs := a.lib.settings.Get().Features.Rewrite.Hotkeys; len(hs) != 1 || hs[0].Keys != "Control+Option+K" || !has(tt, "Also used by Auralink (Rewrite)") {
		t.Errorf("saved %+v: %q", hs, tt.Texts())
	}
	// Saved after the warning: no prompt for it later (the pending one,
	// Transcribe's, still comes).
	key := "rewrite|dev.hehehai.auralink|rewrite"
	if !slices.Contains(a.prefs.Family.Prompted, key) || a.scPrompt == nil || a.scPrompt.Key() == key {
		t.Errorf("prompted %q, prompt %+v", a.prefs.Family.Prompted, a.scPrompt)
	}
}

// obFamilyShots saves the guide's models step with the models other
// ActNow apps installed (onboardingShots): the reuse card (the speech
// models AudioSmith installed, the language model still to download),
// and the alternative lines (Qwen3-ASR 0.6B (4bit) from AudioSmith, the
// all-round language model from Dicta). From memory: the survey's
// findings and the models' statuses.
func obFamilyShots(a *app, save func(string)) {
	local, status, found := a.local, a.modelStatus, a.ob.found
	a.local = &fakeLocal{}
	a.modelStatus = map[string]models.Status{}
	size := func(id string) int64 { m, _ := models.Lookup(id); return m.Size() }
	smith := []string{"AudioSmith"}
	for _, id := range []string{models.ASRAccurate, models.VADSilero} {
		a.modelStatus[id] = models.Status{Installed: true, OnDisk: true, Users: smith}
	}
	a.ob.found = []models.Found{{ID: models.ASRAccurate, Bytes: size(models.ASRAccurate), By: smith}, {ID: models.VADSilero, Bytes: size(models.VADSilero), By: smith}}
	save("onboarding-3-models-reuse")
	a.modelStatus = map[string]models.Status{
		models.ASRFast:  {Installed: true, OnDisk: true, Users: smith},
		models.LLMQwen3: {OnDisk: true, Users: []string{"Dicta"}},
	}
	a.ob.found = []models.Found{{ID: models.ASRFast, Bytes: size(models.ASRFast), By: smith}, {ID: models.LLMQwen3, Bytes: size(models.LLMQwen3), By: []string{"Dicta"}}}
	save("onboarding-3-models-alternative")
	a.local, a.modelStatus, a.ob.found = local, status, found
}

// familyShots saves Settings → Shortcuts with shortcuts other ActNow apps
// use too, and the prompt (snapshotsMain); it leaves no conflict behind.
func familyShots(t *testing.T, a *app, tt *ui.Tester, save func(string)) {
	conflictPeers(t, a)
	a.navigate("/home")
	save("shortcut-prompt")
	a.scPrompt = nil
	a.navigate("/settings/shortcuts")
	save("settings-shortcuts-family")
	a.settingsOpen = false
	a.famConflicts = nil
}
