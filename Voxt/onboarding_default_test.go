package main

// The easy default setup: the speech model the guide recommends for this
// Mac's memory, Download and continue as its primary button, and what a
// session and Home do while that download runs.

import (
	"context"
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

// TestGuideDownloadAndContinue: the guide recommends Qwen3-ASR 1.7B
// (8bit) on every Mac, 8 GB ones included, and downloads only it.
func TestGuideDownloadAndContinue(t *testing.T) {
	label := "Download and continue (2.5 GB)"
	a, tt, asked, drain := guideAt(t)
	if !has(tt, "Qwen3-ASR 1.7B (8bit)") || !has(tt, label) || !has(tt, "Skip this step") || hasPrefix(tt, "Chosen for this Mac") {
		t.Fatalf("%q", tt.Texts())
	}
	if asked.Load() != 0 {
		t.Fatalf("%d requests before the click", asked.Load())
	}
	click(t, tt, findPrefix(t, tt, label))
	if a.ob.step != obTranscribe {
		t.Fatalf("step %d", a.ob.step)
	}
	downloadAsked(t, a, drain, models.ASRAccurate)
	if s := a.models.Status(models.ASRFast); s.Err != nil || s.Downloading {
		t.Errorf("%s downloaded too: %+v", models.ASRFast, s)
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
	if asked.Load() != 0 || a.models.Status(models.ASRAccurate).Downloading {
		t.Fatalf("closing the guide asked %d: %+v", asked.Load(), a.models.Status(models.ASRAccurate))
	}
}

// TestGuideOnDiskNoDownload: another app's copy the survey has not
// reported yet is on disk, so the step offers no download of it, and
// Home sends the user to the page that can claim it.
func TestGuideOnDiskNoDownload(t *testing.T) {
	a, tt, _, _ := guideAt(t)
	a.modelStatus[models.ASRAccurate] = models.Status{OnDisk: true, Users: []string{"AudioSmith"}}
	frames(tt)
	if hasPrefix(tt, "Download and continue") {
		t.Errorf("offers what is on disk: %q", tt.Texts())
	}
	if r := homeSpeech(a); r.action != "Open Speech models" {
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
	if !has(tt, "Download Qwen3-ASR 1.7B (8bit)") || !hasExact(tt, "Continue") || hasPrefix(tt, "Download and continue") {
		t.Fatalf("%q", tt.Texts())
	}
}

// TestGuideReuseUnchanged: with the recommended model on this Mac the
// reuse card's Use them stays the step's primary button.
func TestGuideReuseUnchanged(t *testing.T) {
	skipWindowsActNow(t)
	asked := countingHub(t)
	a, tt := obTester(t)
	reopenModels(a)
	smith := addPeer(t, a, "dev.hehehai.audiosmith", nil)
	seedModels(t, a, models.ASRAccurate, models.VADSilero)
	peerClaims(t, smith, models.ASRAccurate, models.VADSilero)
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
	skipWindowsActNow(t)
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
	if !has(tt, "Use them") || !has(tt, "Download (2.5 GB)") || !hasExact(tt, "Continue") || hasPrefix(tt, "Download and continue") {
		t.Fatalf("reuse card: %q", tt.Texts())
	}
	click(t, tt, "Use them")
	if !has(tt, "Download and continue (2.5 GB)") || hasPrefix(tt, "Download (") {
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

func homeSpeech(a *app) readiness {
	for _, r := range a.readinessList() {
		if r.key == "home-asr" {
			return r
		}
	}
	return readiness{}
}

func TestHomeSpeechRow(t *testing.T) {
	asked := countingHub(t)
	a, tt := homeTester(t)
	drain := queuePosts(a)
	reopenModels(a)
	if r := homeSpeech(a); r.ok || !strings.HasPrefix(r.action, "Download (") {
		t.Fatalf("no model: %+v", r)
	}
	click(t, tt, "Speech recognition: "+homeSpeech(a).action)
	if a.shownPath() == modelsPath {
		t.Error("the line left Home instead of downloading")
	}
	downloadAsked(t, a, drain, models.ASRAccurate)
	if asked.Load() == 0 {
		t.Error("nothing was asked of the Hub")
	}

	a.local = onlyDictation{&fakeLocal{}}
	a.modelStatus = map[string]models.Status{}
	r := homeSpeech(a)
	if r.ok || r.kind != noticeInfo || r.text != "Using System dictation for now." || !strings.HasPrefix(r.action, "Download the recommended model (") {
		t.Errorf("automatic on System dictation: %+v", r)
	}
	if missing, _ := homeMissing(a.readinessList()); !containsKey(missing, "home-asr") {
		t.Errorf("counted as ready: %+v", missing)
	}

	downloading(a, a.obSpeech())
	if r := homeSpeech(a); r.ok || r.text != "The speech model is still downloading (33%)." {
		t.Errorf("downloading: %+v", r)
	}

	a.modelStatus = map[string]models.Status{}
	a.updateSettings(func(s *settings.Settings) { s.Features.Transcribe.ASR = dictation.ModelID })
	if r := homeSpeech(a); !r.ok || r.text != "System dictation" {
		t.Errorf("System dictation chosen: %+v", r)
	}
}

func containsKey(list []readiness, key string) bool {
	for _, r := range list {
		if r.key == key {
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
}
