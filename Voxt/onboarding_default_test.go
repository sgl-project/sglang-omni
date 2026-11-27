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
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/openai"
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

// TestGuideDownloadAndContinue: the guide recommends Qwen3-ASR 1.7B (8bit)
// on every Mac, 8 GB ones included, and downloads only the pack (it, its
// catalogue draft and the VAD), not MOSS-Transcribe-Diarize.
func TestGuideDownloadAndContinue(t *testing.T) {
	a, tt, asked, drain := guideAt(t)
	label := "Download and continue (" + a.obDownloadLeft() + ")"
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
	for _, id := range []string{models.ASRFast, models.ASRMoss} {
		if s := a.models.Status(id); s.Err != nil || s.Downloading {
			t.Errorf("%s downloaded too: %+v", id, s)
		}
	}
}

// obTestDraft is a draft the tests pair with the recommended speech
// model in place of the catalogue's.
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
		if cat[i].ID == models.ASRAccurate {
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

// TestObPackWithoutDraft: with no draft in the catalogue, the pack is the
// speech model and the VAD, and the label is the speech model's size alone.
func TestObPackWithoutDraft(t *testing.T) {
	a, _ := obTester(t)
	reopenModels(a)
	obModelsWith(t, a, obCatalog(""))
	if got := a.obPack(); !slices.Equal(got, []string{models.ASRAccurate, models.VADSilero}) {
		t.Errorf("pack %v", got)
	}
	if got := a.obDownloadLeft(); got != "2.5 GB" {
		t.Errorf("left %q, want the speech model's size", got)
	}
}

// TestObPackWithDraft: with a draft in the catalogue, the pack includes it and the label counts it; with the speech
// model on this Mac the download is the draft alone, so that Faster
// transcription is on (#4067, Models.FasterOn's default).
func TestObPackWithDraft(t *testing.T) {
	countingHub(t)
	a, _ := obTester(t)
	drain := queuePosts(a)
	reopenModels(a)
	obModelsWith(t, a, obCatalog(obTestDraft.ID))
	if got := a.obPack(); !slices.Equal(got, []string{models.ASRAccurate, obTestDraft.ID, models.VADSilero}) {
		t.Fatalf("pack %v", got)
	}
	if got := a.obDownloadLeft(); got != "2.7 GB" {
		t.Errorf("left %q, want the speech model and the draft", got)
	}
	if err := modelstest.Seed(a.models.Hub(), models.ASRAccurate, models.VADSilero); err != nil {
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
	if !a.settings.Get().Models.FasterOn(models.ASRAccurate, true, false) {
		t.Error("Faster transcription off with the draft installed and never switched")
	}
}

// TestObPackIsQwen: the guide's pack is Qwen3-ASR 1.7B (8bit), its draft
// and the VAD whatever the main language; MOSS-Transcribe-Diarize is not
// the default (#4069 reverted).
func TestObPackIsQwen(t *testing.T) {
	a, _ := obTester(t)
	for _, main := range []string{"zh", "en", "ja"} {
		a.prefs.General.MainLanguage = main
		if got := a.obPack(); !slices.Equal(got, []string{models.ASRAccurate, models.ASRAccurateDraft, models.VADSilero}) {
			t.Errorf("%s: pack %v, want Qwen3-ASR 1.7B, its draft and the VAD", main, got)
		}
	}
	if m, _ := models.Lookup(models.ASRAccurate); !m.Default || models.Catalog()[0].ID != models.ASRAccurate {
		t.Errorf("Qwen3-ASR 1.7B (8bit) is not the catalogue's default and first: %+v", models.Catalog()[0].ID)
	}
}

// TestGuidePackDraftLeft: with the recommended speech model and the VAD
// on this Mac but its catalogue draft missing, the guide offers the
// draft's download — the pack's Faster transcription would never start
// otherwise (#4067) — and once the models are claimed the click
// downloads the draft alone.
func TestGuidePackDraftLeft(t *testing.T) {
	countingHub(t)
	a, tt := obTester(t)
	drain := queuePosts(a)
	reopenModels(a)
	seedModels(t, a, models.ASRAccurate, models.VADSilero)
	a.obGo(obModels)
	drain()
	frames(tt)
	if !has(tt, "Use them") || !has(tt, "Download (725.7 MB)") || hasPrefix(tt, "Download and continue") {
		t.Fatalf("on disk: %q", tt.Texts())
	}
	click(t, tt, "Use them")
	drain()
	frames(tt)
	if !has(tt, "Ready: Qwen3-ASR 1.7B (8bit)") || !hasPrefix(tt, "Download and continue (725.7 MB)") {
		t.Fatalf("claimed: %q", tt.Texts())
	}
	click(t, tt, findPrefix(t, tt, "Download and continue"))
	downloadAsked(t, a, drain, models.ASRAccurateDraft)
	for _, m := range models.Catalog() {
		if m.ID == models.ASRAccurateDraft {
			continue
		}
		if s := a.models.Status(m.ID); s.Err != nil || s.Downloading {
			t.Errorf("%s downloaded too: %+v", m.ID, s)
		}
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
// reported yet is on disk, so the step offers no download of it.
func TestGuideOnDiskNoDownload(t *testing.T) {
	a, tt, _, _ := guideAt(t)
	a.modelStatus[models.ASRAccurate] = models.Status{OnDisk: true, Users: []string{"AudioSmith"}}
	frames(tt)
	if hasPrefix(tt, "Download and continue") {
		t.Errorf("offers what is on disk: %q", tt.Texts())
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
// reuse card's Use them stays the step's primary button. Its catalogue
// draft is not on this Mac, so the body offers the draft's download
// (TestGuidePackDraftLeft drives it).
func TestGuideReuseUnchanged(t *testing.T) {
	asked := countingHub(t)
	a, tt := obTester(t)
	reopenModels(a)
	smith := addPeer(t, a, "dev.hehehai.audiosmith", nil)
	seedModels(t, a, models.ASRAccurate, models.VADSilero)
	peerClaims(t, smith, models.ASRAccurate, models.VADSilero)
	a.familyFound(a.fam.Discover())
	a.models.Refresh()
	a.obGo(obModels)
	if !has(tt, "Use them") || !hasExact(tt, "Continue") || hasPrefix(tt, "Download and continue") || !has(tt, "Download (725.7 MB)") {
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
	// (#4067): MTD knows Chinese and English only, so ja lands on
	// Qwen3-ASR — in the pages (featureModel) and in the session's
	// resolver alike. An explicit MTD choice still opens MTD.
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

// automaticGateLocal is MTD and Qwen3-ASR 1.7B (8bit), MTD first,
// recording what a session's resolver opens.
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

// TestAutomaticSpeechRemoteGuard: Automatic never promotes a remote
// speech service over an installed on-device model the language gate
// removed — a remote service is a feature's explicit choice, or the
// fallback when no local model is installed (providers.Services,
// #4067). With MTD (Chinese and English) installed, a ready remote
// service and the main language ja, the resolver opens nothing, or
// System dictation when it is there; only with no on-device model at
// all does Automatic open the remote service.
func TestAutomaticSpeechRemoteGuard(t *testing.T) {
	for _, tc := range []struct {
		name       string
		system     bool
		wantOpened string // "": the resolver opens no speech model
	}{
		{"local MTD installed, the remote service stays unopened", false, ""},
		{"System dictation is the on-device fallback", true, dictation.ModelID},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, _, _ := newTestApp(t)
			a.prefs.General.MainLanguage = "ja"
			lm := &remoteGateLocal{system: tc.system}
			svc, err := resolver(lm)(context.Background(), store.KindTranscribe, a.prefs, true)
			if err != nil {
				t.Fatal(err)
			}
			if lm.opened != tc.wantOpened {
				t.Errorf("opened %q, want %q", lm.opened, tc.wantOpened)
			}
			if tc.wantOpened == "" && svc.ASR != nil {
				t.Errorf("the session got a speech model: %+v", svc.ASRInfo)
			}
		})
	}
	t.Run("no on-device model: Automatic opens the remote service", func(t *testing.T) {
		a, _, _ := newTestApp(t)
		a.prefs.General.MainLanguage = "ja"
		lm := &remoteGateLocal{remoteOnly: true}
		svc, err := resolver(lm)(context.Background(), store.KindTranscribe, a.prefs, true)
		if err != nil {
			t.Fatal(err)
		}
		if lm.opened != openai.ID || svc.ASRInfo.ID != openai.ID {
			t.Errorf("opened %q (ASRInfo %q), want %s", lm.opened, svc.ASRInfo.ID, openai.ID)
		}
	})
}

// remoteGateLocal is MTD and a ready remote speech service (System
// dictation with system; the remote service alone with remoteOnly),
// recording what a session's resolver opens.
type remoteGateLocal struct {
	local.NotInstalled
	opened     string
	system     bool
	remoteOnly bool
}

func (l *remoteGateLocal) Installed(kind local.Kind) []local.Model {
	if kind != local.ASR {
		return nil
	}
	var out []local.Model
	if !l.remoteOnly {
		out = append(out, local.Model{Kind: local.ASR, ID: models.ASRMoss, Name: "MOSS-Transcribe-Diarize", Stream: true})
	}
	if l.system {
		out = append(out, local.Model{Kind: local.ASR, ID: dictation.ModelID, Name: "System dictation", Stream: true})
	}
	return append(out, local.Model{Kind: local.ASR, ID: openai.ID, Name: "OpenAI Transcribe", Stream: true})
}

func (l *remoteGateLocal) ASR(_ context.Context, id string) (asr.Provider, local.Model, error) {
	m, ok := local.Choose(l.Installed(local.ASR), id)
	if !ok {
		return nil, local.Model{}, local.NotInstalledError(local.ASR, id)
	}
	l.opened = m.ID
	return &fakeTranscriptions{}, m, nil
}
