package main

// The easy default setup: the speech model the guide recommends for this
// Mac's memory, Download and continue as its primary button, and what a
// session and Home do while that download runs.

import (
	"context"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models/modelstest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// guideAt is the guide at its models step, against a Hub that counts
// what it is asked and has nothing.
func guideAt(t *testing.T) (*app, *ui.Tester, *atomic.Int64, func()) {
	t.Helper()
	asked := countingHub(t)
	a, tt := obTester(t)
	drain := queuePosts(a)
	reopenModels(a)
	a.obGo(obModels)
	drain()
	frames(tt)
	return a, tt, asked, drain
}

// downloadAsked waits for model id's download to have been tried (the
// test Hub has nothing, so it fails).
func downloadAsked(t *testing.T, a *app, drain func(), id string) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for a.models.Status(id).Err == nil && time.Now().Before(deadline) {
		drain()
	}
	if a.models.Status(id).Err == nil {
		t.Fatalf("%s: no download was asked for: %+v", id, a.models.Status(id))
	}
}

// TestGuideDownloadAndContinue: the guide recommends MOSS-Transcribe-Diarize
// on every Mac, 8 GB ones included, and downloads only the pack (it, its
// catalogue draft and the VAD).
func TestGuideDownloadAndContinue(t *testing.T) {
	a, tt, asked, drain := guideAt(t)
	label := "Download and continue (" + a.obDownloadLeft() + ")"
	if !has(tt, "MOSS-Transcribe-Diarize") || !has(tt, label) || !has(tt, "Skip this step") || hasPrefix(tt, "Chosen for this Mac") {
		t.Fatalf("%q", tt.Texts())
	}
	if asked.Load() != 0 {
		t.Fatalf("%d requests before the click", asked.Load())
	}
	click(t, tt, findPrefix(t, tt, label))
	if a.ob.step != obTranscribe {
		t.Fatalf("step %d", a.ob.step)
	}
	downloadAsked(t, a, drain, models.ASRMoss)
	for _, id := range []string{models.ASRFast, models.ASRAccurate} {
		if s := a.models.Status(id); s.Err != nil || s.Downloading {
			t.Errorf("%s downloaded too: %+v", id, s)
		}
	}
}

// obTestDraft is a draft the tests pair with the recommended speech
// model, standing in for the catalogue's (MTD's arrives with #3827).
var obTestDraft = models.Model{
	ID: "test.moss.draft", Name: "MTD test draft", Kind: models.Draft, Runtime: models.Omni, Family: "test",
	Repo: "mlx-community/test-moss-draft", Revision: "4444444444444444444444444444444444444444",
	Files: []models.File{
		{Path: "config.json", Size: 1888, SHA256: strings.Repeat("4", 64)},
		{Path: "model.safetensors", Size: 262451248, SHA256: strings.Repeat("5", 64), LFS: true},
	},
}

// obCatalog is the catalogue with the recommended speech model's draft
// as draftID ("": none), plus the draft itself when named.
func obCatalog(draftID string) []models.Model {
	cat := slices.Clone(models.Catalog())
	for i := range cat {
		if cat[i].ID == models.ASRMoss {
			cat[i].Draft = draftID
		}
	}
	if draftID != "" {
		cat = append(cat, obTestDraft)
	}
	return cat
}

// obModelsWith reopens a's models over the catalogue cat.
func obModelsWith(t *testing.T, a *app, cat []models.Model) {
	t.Helper()
	opt := a.managerOptions(func() {})
	opt.Catalog = cat
	a.models.Close()
	a.models = models.Open(opt)
	a.modelStatus = nil
	a.readModelStatus()
}

// TestObPackWithoutDraft: with no draft in the catalogue (today, before
// #3827), the pack is the speech model and the VAD, and the label is the
// speech model's size alone.
func TestObPackWithoutDraft(t *testing.T) {
	a, _ := obTester(t)
	reopenModels(a)
	obModelsWith(t, a, obCatalog(""))
	if got := a.obPack(); !slices.Equal(got, []string{models.ASRMoss, models.VADSilero}) {
		t.Errorf("pack %v", got)
	}
	if got := a.obDownloadLeft(); got != "1.8 GB" {
		t.Errorf("left %q, want the speech model's size", got)
	}
}

// TestObPackWithDraft: with a draft in the catalogue (as #3827 adds
// MTD's), the pack includes it and the label counts it; with the speech
// model on this Mac the download is the draft alone, so that Faster
// transcription is on (#4067, Models.FasterOn's default).
func TestObPackWithDraft(t *testing.T) {
	countingHub(t)
	a, _ := obTester(t)
	drain := queuePosts(a)
	reopenModels(a)
	obModelsWith(t, a, obCatalog(obTestDraft.ID))
	if got := a.obPack(); !slices.Equal(got, []string{models.ASRMoss, obTestDraft.ID, models.VADSilero}) {
		t.Fatalf("pack %v", got)
	}
	if got := a.obDownloadLeft(); got != "2.1 GB" {
		t.Errorf("left %q, want the speech model and the draft", got)
	}
	if err := modelstest.Seed(a.models.Hub(), models.ASRMoss, models.VADSilero); err != nil {
		t.Fatal(err)
	}
	a.models.Refresh()
	a.modelStatus = nil
	a.readModelStatus()
	if got := a.obDownloadLeft(); got != "262.5 MB" {
		t.Errorf("left %q, want the draft's size", got)
	}
	a.obDownload()
	downloadAsked(t, a, drain, obTestDraft.ID)
	if !a.settings.Get().Models.FasterOn(models.ASRMoss, true) {
		t.Error("Faster transcription off with the draft installed and never switched")
	}
}

// TestObPackMainLanguageGate: MOSS-Transcribe-Diarize knows Chinese and
// English only, so a main language it does not know makes the guide's
// pack Qwen3-ASR 1.7B (8bit) and the VAD, with no MTD in it.
func TestObPackMainLanguageGate(t *testing.T) {
	a, _ := obTester(t)
	a.prefs.General.MainLanguage = "ja"
	if got := a.obSpeech(); got != models.ASRAccurate {
		t.Errorf("ja: obSpeech %s, want %s", got, models.ASRAccurate)
	}
	if got := a.obPack(); !slices.Equal(got, []string{models.ASRAccurate, models.ASRAccurateDraft, models.VADSilero}) {
		t.Errorf("ja: pack %v, want Qwen3-ASR 1.7B, its draft and the VAD, no MTD", got)
	}
	a.prefs.General.MainLanguage = "zh"
	if got := a.obSpeech(); got != models.ASRMoss {
		t.Errorf("zh: obSpeech %s, want %s", got, models.ASRMoss)
	}
}

// findPrefix is the text on screen that starts with s.
func findPrefix(t *testing.T, tt *ui.Tester, s string) string {
	t.Helper()
	for _, x := range tt.Texts() {
		if strings.HasPrefix(x, s) {
			return x
		}
	}
	t.Fatalf("nothing starts with %q in %q", s, tt.Texts())
	return ""
}

func TestGuideSkipDownloadsNothing(t *testing.T) {
	a, tt, asked, drain := guideAt(t)
	click(t, tt, "Skip this step")
	drain()
	if a.ob.step != obTranscribe || asked.Load() != 0 {
		t.Fatalf("step %d, %d requests", a.ob.step, asked.Load())
	}
	a.obGo(obModels)
	a.closeOnboarding()
	drain()
	if asked.Load() != 0 || a.models.Status(models.ASRMoss).Downloading {
		t.Fatalf("closing the guide asked %d: %+v", asked.Load(), a.models.Status(models.ASRMoss))
	}
}

// TestGuideOnDiskNoDownload: another app's copy the survey has not
// reported yet is on disk, so the step offers no download of it, and
// Home sends the user to the page that can claim it.
func TestGuideOnDiskNoDownload(t *testing.T) {
	a, tt, _, _ := guideAt(t)
	a.modelStatus[models.ASRMoss] = models.Status{OnDisk: true, Users: []string{"AudioSmith"}}
	frames(tt)
	if hasPrefix(tt, "Download and continue") {
		t.Errorf("offers what is on disk: %q", tt.Texts())
	}
	if r := homeSpeech(a); r.Action != "Open Speech models" {
		t.Errorf("Home: %+v", r)
	}
}

// TestGuideAlternativeNoDownloadButton: an alternative on this Mac, with
// no reuse card, is a choice the step's line offers: the footer does not
// take it for the download.
func TestGuideAlternativeNoDownloadButton(t *testing.T) {
	a, tt, _, _ := guideAt(t)
	a.ob.found = []models.Found{{ID: models.ASRFast, By: []string{"AudioSmith"}}}
	frames(tt)
	if !has(tt, "Download MOSS-Transcribe-Diarize") || !hasExact(tt, "Continue") || hasPrefix(tt, "Download and continue") {
		t.Fatalf("%q", tt.Texts())
	}
}

// TestGuideReuseUnchanged: with the recommended model on this Mac the
// reuse card's Use them stays the step's primary button.
func TestGuideReuseUnchanged(t *testing.T) {
	asked := countingHub(t)
	a, tt := obTester(t)
	reopenModels(a)
	smith := addPeer(t, a, "dev.hehehai.audiosmith", nil)
	seedModels(t, a, models.ASRMoss, models.VADSilero)
	peerClaims(t, smith, models.ASRMoss, models.VADSilero)
	a.familyFound(a.fam.Discover())
	a.models.Refresh()
	a.obGo(obModels)
	if !has(tt, "Use them") || !hasExact(tt, "Continue") || hasPrefix(tt, "Download and continue") || hasPrefix(tt, "Download (") {
		t.Fatalf("%q", tt.Texts())
	}
	click(t, tt, "Continue")
	if a.ob.step != obTranscribe || asked.Load() != 0 {
		t.Fatalf("step %d, %d requests", a.ob.step, asked.Load())
	}
}

// TestGuideOnlyVADOnThisMac: only Silero on this Mac makes the reuse
// card, whose Use them is primary: the speech model's Download stays in
// the body, and once it is claimed the footer offers the download.
func TestGuideOnlyVADOnThisMac(t *testing.T) {
	countingHub(t)
	a, tt := obTester(t)
	reopenModels(a)
	smith := addPeer(t, a, "dev.hehehai.audiosmith", nil)
	seedModels(t, a, models.VADSilero)
	peerClaims(t, smith, models.VADSilero)
	a.familyFound(a.fam.Discover())
	a.models.Refresh()
	a.navigate("/history") // not Home, which offers the download too
	a.obGo(obModels)
	if !has(tt, "Use them") || !has(tt, "Download ("+a.obDownloadLeft()+")") || !hasExact(tt, "Continue") || hasPrefix(tt, "Download and continue") {
		t.Fatalf("reuse card: %q", tt.Texts())
	}
	click(t, tt, "Use them")
	if !has(tt, "Download and continue ("+a.obDownloadLeft()+")") || hasPrefix(tt, "Download (") {
		t.Fatalf("after Use them: %q", tt.Texts())
	}
}

func TestGuideChooseOther(t *testing.T) {
	a, tt, _, _ := guideAt(t)
	click(t, tt, "Choose another model…")
	if a.ob.open || a.shownPath() != modelsPath || !has(tt, "The setup guide is paused at the Models step.") || !has(tt, "Back to the setup guide") {
		t.Fatalf("open %v, %s, %q", a.ob.open, a.shownPath(), tt.Texts())
	}
	// A model in use: the way to the others stays.
	readyASROnly(a)
	click(t, tt, "Back to the setup guide")
	if !has(tt, "Ready: Qwen3-ASR 1.7B (8bit)") || !has(tt, "Choose another model…") || !hasExact(tt, "Continue") || hasPrefix(tt, "Download and continue") {
		t.Fatalf("with a model: %q", tt.Texts())
	}
}

// onlyDictation is a Mac with no speech model but System dictation, and
// a language model.
type onlyDictation struct{ *fakeLocal }

func (o onlyDictation) Installed(kind local.Kind) []local.Model {
	if kind == local.ASR {
		return []local.Model{{Kind: local.ASR, ID: dictation.ModelID, Name: "System dictation", Stream: true}}
	}
	return o.fakeLocal.Installed(kind)
}

func (o onlyDictation) ASR(context.Context, string) (asr.Provider, local.Model, error) {
	return &fakeTranscriptions{}, local.Model{Kind: local.ASR, ID: dictation.ModelID}, nil
}

func dictationApp(t *testing.T) (*app, *ui.Tester) {
	t.Helper()
	a, tt := obTester(t)
	a.local = onlyDictation{&fakeLocal{Chat: &llm.Client{Backend: &fakeChat{}}}}
	a.checkModels()
	return a, tt
}

func downloading(a *app, id string) {
	m, _ := models.Lookup(id)
	a.modelStatus[id] = models.Status{Downloading: true, Done: m.Size() / 3, Total: m.Size()}
}

func TestSpeechWaitsForDownload(t *testing.T) {
	a, tt := dictationApp(t)
	const waiting = "The speech model is still downloading (33%)."
	if n := a.sessionBlocker(store.KindTranscribe); n != nil {
		t.Fatalf("no download, System dictation: %+v", n)
	}
	downloading(a, a.obSpeech())
	for _, kind := range []store.Kind{store.KindTranscribe, store.KindRewrite} {
		if n := a.sessionBlocker(kind); n == nil || n.text != waiting || n.action != actionModels {
			t.Errorf("%s: %+v", kind, n)
		}
	}
	if n := a.sessionBlocker(store.KindTranslate); n != nil {
		t.Errorf("translate is held before it knows of a selection: %+v", n)
	}
	a.navigate("/history") // not Home, which says so too
	a.obGo(obTranscribe)
	if !has(tt, "still downloading (33%)") {
		t.Errorf("the try step: %q", tt.Texts())
	}

	// Another model downloading is not the wait.
	a.modelStatus = map[string]models.Status{}
	downloading(a, models.ASRWhisperSmall)
	if n := a.sessionBlocker(store.KindTranscribe); n != nil {
		t.Errorf("another model's download: %+v", n)
	}

	// System dictation chosen goes on, and is not told about the download.
	downloading(a, a.obSpeech())
	a.updateSettings(func(s *settings.Settings) { s.Features.Transcribe.ASR = dictation.ModelID })
	frames(tt)
	if n := a.sessionBlocker(store.KindTranscribe); n != nil {
		t.Errorf("System dictation chosen: %+v", n)
	}
	if has(tt, "still downloading") {
		t.Errorf("the try step, System dictation chosen: %q", tt.Texts())
	}
	a.updateSettings(func(s *settings.Settings) { s.Features.Transcribe.ASR = "" })

	// With nothing at all to fall back on, the wait still says so.
	a.local = &fakeLocal{}
	if n := a.sessionBlocker(store.KindTranscribe); n == nil || n.text != waiting {
		t.Errorf("no model at all: %+v", n)
	}

	// A model of its own already: the recommended one's download is no
	// reason to wait.
	readyModels(a)
	if n := a.sessionBlocker(store.KindTranscribe); n != nil {
		t.Errorf("a model of its own: %+v", n)
	}

	// Done: the model it downloaded.
	a.modelStatus = map[string]models.Status{}
	if n := a.sessionBlocker(store.KindTranscribe); n != nil {
		t.Errorf("downloaded: %+v", n)
	}
}

// TestTranslateByVoiceWaits: Translate is not held at the start (a
// selection needs no speech model), but speaking to it during the download
// ends with the words of the wait and no recording.
func TestTranslateByVoiceWaits(t *testing.T) {
	a, _ := dictationApp(t)
	drain := queuePosts(a)
	downloading(a, a.obSpeech())
	if n := a.startSession(session.StartRequest{Kind: store.KindTranslate, Try: true}); n != nil {
		t.Fatalf("held at the start: %+v", n)
	}
	deadline := time.Now().Add(5 * time.Second)
	for !a.sess.State.Ended() && time.Now().Before(deadline) {
		drain()
	}
	end := a.sess
	if end.State != session.Failed || end.Problem == nil || end.Problem.Code != session.ProblemSpeechWait {
		t.Fatalf("end %+v", end)
	}
	if text, _, action := sessionProblemText(end.Problem); text != "The speech model is still downloading (33%)." || action != actionModels {
		t.Errorf("notice %q, %q", text, action)
	}
}

func homeSpeech(a *app) homeReadinessViewState {
	for _, r := range a.readinessList() {
		if r.Key == "home-asr" {
			return r
		}
	}
	return homeReadinessViewState{}
}

func TestHomeSpeechRow(t *testing.T) {
	asked := countingHub(t)
	a, tt := homeTester(t)
	drain := queuePosts(a)
	reopenModels(a)
	if r := homeSpeech(a); r.OK || !strings.HasPrefix(r.Action, "Download (") {
		t.Fatalf("no model: %+v", r)
	}
	click(t, tt, "Speech recognition: "+homeSpeech(a).Action)
	if a.shownPath() == modelsPath {
		t.Error("the line left Home instead of downloading")
	}
	downloadAsked(t, a, drain, models.ASRMoss)
	if asked.Load() == 0 {
		t.Error("nothing was asked of the Hub")
	}

	a.local = onlyDictation{&fakeLocal{}}
	a.modelStatus = map[string]models.Status{}
	r := homeSpeech(a)
	if r.OK || r.Kind != "info" || r.Text != "Using System dictation for now." || !strings.HasPrefix(r.Action, "Download the recommended model (") {
		t.Errorf("automatic on System dictation: %+v", r)
	}
	if missing, _ := homeMissing(a.readinessList()); !containsKey(missing, "home-asr") {
		t.Errorf("counted as ready: %+v", missing)
	}

	downloading(a, a.obSpeech())
	if r := homeSpeech(a); r.OK || r.Text != "The speech model is still downloading (33%)." {
		t.Errorf("downloading: %+v", r)
	}

	a.modelStatus = map[string]models.Status{}
	a.updateSettings(func(s *settings.Settings) { s.Features.Transcribe.ASR = dictation.ModelID })
	if r := homeSpeech(a); !r.OK || r.Text != "System dictation" {
		t.Errorf("System dictation chosen: %+v", r)
	}
}

func containsKey(list []homeReadinessViewState, key string) bool {
	for _, r := range list {
		if r.Key == key {
			return true
		}
	}
	return false
}

// obDefaultShots saves the models step with nothing on this Mac.
func obDefaultShots(a *app, save func(string)) {
	loc, status, found := a.local, a.modelStatus, a.ob.found
	a.local, a.modelStatus, a.ob.found = &fakeLocal{}, map[string]models.Status{}, nil
	save("onboarding-3-models-default")
	a.local, a.modelStatus, a.ob.found = loc, status, found
}

// TestAutomaticSpeech: with both Qwen3-ASR models installed, Automatic
// is the accurate one; a feature's explicit choice stays.
func TestAutomaticSpeech(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 4800)
	seedModels(t, a, models.ASRAccurate, models.ASRFast, models.VADSilero)
	if m, ok := a.featureModel(local.ASR, hotkey.Transcribe); !ok || m.ID != models.ASRAccurate {
		t.Errorf("Transcribe uses %+v, want %s", m, models.ASRAccurate)
	}
	a.navigate("/transcribe")
	click(t, tt, "Speech model for Transcribe")
	if want := "Automatic (now: " + uitext.ModelName(models.ASRAccurate) + ")"; !has(tt, want) {
		t.Errorf("no %q: %q", want, tt.Texts())
	}
	tt.Key(0, ui.KeyEscape)
	frames(tt)

	a.updateSettings(func(s *settings.Settings) { s.Features.Transcribe.ASR = models.ASRFast })
	if m, _ := a.featureModel(local.ASR, hotkey.Transcribe); m.ID != models.ASRFast {
		t.Errorf("the explicit choice became %s", m.ID)
	}

	// Automatic skips a model that does not know the main language
	// (#4067): MTD is the catalogue's default and first, but it knows
	// Chinese and English only, so ja lands on Qwen3-ASR — in the pages
	// (featureModel) and in the session's resolver alike. An explicit
	// MTD choice still opens MTD.
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Transcribe.ASR = ""
		s.General.MainLanguage = "ja"
	})
	seedModels(t, a, models.ASRMoss, models.ASRAccurate, models.VADSilero)
	if m, ok := a.featureModel(local.ASR, hotkey.Transcribe); !ok || m.ID != models.ASRAccurate {
		t.Errorf("ja: Transcribe's Automatic uses %+v, want %s", m, models.ASRAccurate)
	}
	lm := &automaticGateLocal{}
	svc, err := resolver(lm)(context.Background(), store.KindTranscribe, a.prefs, true)
	if err != nil {
		t.Fatal(err)
	}
	if svc.ASRInfo.ID != models.ASRAccurate {
		t.Errorf("ja: the resolver opened %s, want %s", svc.ASRInfo.ID, models.ASRAccurate)
	}

	a.updateSettings(func(s *settings.Settings) { s.Features.Transcribe.ASR = models.ASRMoss })
	if m, _ := a.featureModel(local.ASR, hotkey.Transcribe); m.ID != models.ASRMoss {
		t.Errorf("ja: the explicit MTD choice became %s", m.ID)
	}
	lm.opened = ""
	svc, err = resolver(lm)(context.Background(), store.KindTranscribe, a.prefs, true)
	if err != nil {
		t.Fatal(err)
	}
	if svc.ASRInfo.ID != models.ASRMoss || lm.opened != models.ASRMoss {
		t.Errorf("ja: the explicit MTD choice opened %s (ASRInfo %s)", lm.opened, svc.ASRInfo.ID)
	}
}

// automaticGateLocal is MTD and Qwen3-ASR 1.7B (8bit), MTD first as the
// catalogue's default, recording what a session's resolver opens.
type automaticGateLocal struct {
	local.NotInstalled
	opened string
}

func (l *automaticGateLocal) Installed(kind local.Kind) []local.Model {
	if kind == local.ASR {
		return []local.Model{{Kind: local.ASR, ID: models.ASRMoss}, {Kind: local.ASR, ID: models.ASRAccurate}}
	}
	return nil
}

func (l *automaticGateLocal) ASR(_ context.Context, id string) (asr.Provider, local.Model, error) {
	l.opened = id
	for _, m := range l.Installed(local.ASR) {
		if m.ID == id {
			return &fakeTranscriptions{}, m, nil
		}
	}
	return nil, local.Model{}, local.NotInstalledError(local.ASR, id)
}
