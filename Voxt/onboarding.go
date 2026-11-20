package main

// The setup guide (AudioSmith's wizard in look, Voxt's
// OnboardingGuide in what it checks): a dialog over the window on the
// first start, and again from Settings → General, Home or ⌘K. Seven
// steps, each of which can be skipped:
//
//	0 welcome and privacy     4 try Translate (the sample text, or speech)
//	1 permissions             5 try Rewrite (write from a spoken request)
//	2 the local models        6 App enhancement and the dictionary
//	3 try Transcribe
//
// The models step offers the models other ActNow apps already put on
// this Mac first (onboarding_models.go); the steps that show a shortcut
// say when another ActNow app uses it too, and the one-time prompt of a
// shortcut conflict waits until the guide closes
// (settings_shortcuts_family.go).
//
// The tries are real sessions (the microphone, the speech model, the
// language model; nothing is inserted or kept: session.StartRequest.Try),
// started with Start or the feature's own shortcut while VoxtNew is in
// front. A step passes only when the session it started (by id) ends
// done with a result: a key press, a selection or an older session's late
// snapshot never does. While its try runs the guide cannot move on, and
// closing the guide cancels it. Closing or finishing marks the guide done
// (settings.Setup), so it does not open again by itself. Closed before its
// end, it opens again at the step it was left on (this run); the models
// step's links to the Speech and Language models pages leave it with a
// way back on that page (obBackBar).

import (
	"context"
	"strings"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/lang"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// The guide's steps.
const (
	obWelcome = iota
	obPermissions
	obModels
	obTranscribe
	obTranslate
	obRewrite
	obMore
	obSteps
)

// onboardingState is the guide's state.
type onboardingState struct {
	ob struct {
		open bool
		step int
		// tries are the sessions each try step started, by step; passed
		// the steps whose try ended with a result.
		tries  map[int]string
		passed map[int]bool
		// block says why a try did not start; hint that another feature's
		// shortcut was pressed.
		block *sessionNotice
		hint  string
		// found are the recommended models (and their alternatives) on
		// this Mac, as the models step's survey found them
		// (onboarding_models.go).
		found []models.Found
		// legacy is the last import from the previous Voxt
		// (legacy_import.go), which the welcome step's card shows, or
		// the dialog legacyOpen (legacy_report.go) after one from
		// elsewhere.
		legacy     legacyReport
		legacyOpen bool
		// legacyAvailable says the welcome survey found a previous Voxt;
		// legacyDone records that an import brought the dictionary over.
		legacyAvailable bool
		legacyDone      bool
		// awayTo is the page one of the guide's links left it for: that
		// page offers the way back ("" none).
		awayTo string
		// view caches the welcome step while its inputs stay unchanged.
		view       onboardingViewState
		viewKey    onboardingWelcomeCacheKey
		viewCached bool
	}
}

// obStepKind is the feature a step tries.
func obStepKind(step int) (store.Kind, bool) {
	switch step {
	case obTranscribe:
		return store.KindTranscribe, true
	case obTranslate:
		return store.KindTranslate, true
	case obRewrite:
		return store.KindRewrite, true
	}
	return "", false
}

// openOnboarding opens the guide at the step it was left on this run,
// with its tries; at its first step when it never opened or was finished.
func (a *app) openOnboarding() {
	a.settingsOpen = false
	a.ob.open, a.ob.awayTo = true, ""
	if a.ob.tries == nil {
		a.ob.step, a.ob.tries, a.ob.passed = obWelcome, map[int]string{}, map[int]bool{}
	}
	a.ob.block, a.ob.hint = nil, ""
	a.ob.found = nil
	a.ob.legacyAvailable = legacyLibrary() != ""
	a.recheckPermissions()
	if a.ob.step == obModels {
		a.obSurvey() // onboarding_models.go
	}
}

// finishOnboarding closes the guide at its end: it opens again from the
// start, then opens the page the user chose.
func (a *app) finishOnboarding(path string) {
	a.closeOnboarding()
	a.ob.tries, a.ob.passed = nil, nil
	a.navigate(path)
}

// obLeave closes the guide for the page at path, which offers the way
// back to its step (obBackBar).
func (a *app) obLeave(path string) {
	a.closeOnboarding()
	a.ob.awayTo = path
	a.navigate(path)
}

func (a *app) dismissOnboardingBack() { a.ob.awayTo = "" }

// maybeOnboard opens the guide on the first start (openWindow).
func (a *app) maybeOnboard() {
	if !a.prefs.Setup.Done {
		a.openOnboarding()
	}
}

// obOwns reports whether the session id is one of the guide's tries.
func (a *app) obOwns(id string) bool {
	if id == "" {
		return false
	}
	for _, t := range a.ob.tries {
		if t == id {
			return true
		}
	}
	return false
}

// obBusy reports whether one of the guide's tries runs.
func (a *app) obBusy() bool { return a.sess.State.Live() && a.obOwns(a.sess.ID) }

// obTry starts the step's try (Start); text starts Translate with it as
// the selection instead of recording.
func (a *app) obTry(text string) *sessionNotice {
	kind, ok := obStepKind(a.ob.step)
	if !ok {
		return nil
	}
	req := session.StartRequest{Kind: kind, Try: true}
	if text != "" {
		req.Text = &session.TextInput{Selection: text}
	}
	id, n := a.startSessionID(req)
	a.obTryStarted(id, n)
	return n
}

// obTryStarted keeps the step's try (id; "" when it did not start, and n
// says why).
func (a *app) obTryStarted(id string, n *sessionNotice) {
	a.ob.block, a.ob.hint = n, ""
	if id != "" {
		a.ob.tries[a.ob.step] = id
	}
}

// tryIgnored says, on the guide's try step, that the shortcut pressed is
// another feature's.
func (a *app) tryIgnored(kind store.Kind) {
	want, ok := obStepKind(a.ob.step)
	if !a.ob.open || !ok {
		return
	}
	keys := a.featureKeys(string(want))
	if keys == "" {
		a.ob.hint = i18n.T("That was the %s shortcut. This step tests %s.", uitext.KindTitle(kind), uitext.KindTitle(want))
		return
	}
	a.ob.hint = i18n.T("That was the %s shortcut. This step tests %s (%s).", uitext.KindTitle(kind), uitext.KindTitle(want), keys)
}

// obSessionChanged passes a step whose own try ended with a result.
func (a *app) obSessionChanged(s session.Snapshot) {
	if s.State != session.Done || strings.TrimSpace(s.Result) == "" {
		return
	}
	for step, id := range a.ob.tries {
		if id == s.ID {
			a.ob.passed[step] = true
		}
	}
}

// closeOnboarding closes the guide for good (Settings → General opens it
// again), cancelling its try.
func (a *app) closeOnboarding() {
	if a.obBusy() {
		a.cancelSession()
	}
	a.ob.open = false
	if !a.prefs.Setup.Done {
		a.updateSettings(func(s *settings.Settings) { s.Setup.Done = true })
	}
	a.shortcutPrompt() // settings_shortcuts_family.go: it waited for the guide
}

// obGo moves to a step (Back, Continue, Skip).
func (a *app) obGo(step int) {
	if a.obBusy() {
		return
	}
	a.ob.step = min(max(step, 0), obSteps-1)
	a.ob.block, a.ob.hint = nil, ""
	switch a.ob.step {
	case obWelcome:
		a.ob.legacyAvailable = legacyLibrary() != ""
	case obPermissions:
		a.recheckPermissions()
	case obModels:
		a.obSurvey() // onboarding_models.go
	}
}

// obSpeech is the pack's speech model: the catalogue's recommended one,
// MOSS-Transcribe-Diarize (#4067). MTD knows only Chinese and English, so
// a user whose main language it does not know gets Qwen3-ASR 1.7B (8bit),
// which runs well with 8 GB of memory too.
func (a *app) obSpeech() string {
	if spec, ok := omni.Lookup(models.ASRMoss); ok {
		if _, fit := lang.Match(spec.Languages, session.MainLanguage(a.prefs, i18n.Locale())); fit {
			return models.ASRMoss
		}
	}
	return models.ASRAccurate
}

// obPack is the recommended pack: what the guide downloads. The speech
// model's draft comes with it, so that Faster transcription is on from
// the start (settings.Models.FasterOn). The catalogue names the draft
// (Model.Draft); with none, the pack is the speech model and the VAD.
func (a *app) obPack() []string {
	pack := []string{a.obSpeech()}
	if d := a.obDraft(); d != "" {
		pack = append(pack, d)
	}
	return append(pack, models.VADSilero)
}

// obDraft is the pack speech model's catalogue draft, "" when the
// catalogue names none.
func (a *app) obDraft() string {
	m, _ := a.models.Lookup(a.obSpeech())
	return m.Draft
}

// obDownload downloads the pack's speech model and its draft.
func (a *app) obDownload() {
	a.downloadModel(a.obSpeech())
	if d := a.obDraft(); d != "" {
		a.downloadModel(d)
	}
}

// downloadPercent is how much of a download has come, in percent.
func downloadPercent(st models.Status) int {
	if st.Total <= 0 {
		return 0
	}
	return int(100 * float64(st.Done) / float64(st.Total))
}

// obDownloadLeft is what the recommended speech model and its draft still
// have to download, for the buttons that start it.
func (a *app) obDownloadLeft() string {
	var left int64
	for _, id := range []string{a.obSpeech(), a.obDraft()} {
		if id == "" {
			continue
		}
		if st := a.modelStatus[id]; !st.Installed {
			m, _ := a.models.Lookup(id)
			left += m.Size() - st.Done
		}
	}
	return uitext.HumanBytes(left)
}

// obSurvey looks for the pack's models and their alternatives in the
// family's library, off the main thread (when the models step shows).
func (a *app) obSurvey() {
	if a.models == nil {
		return
	}
	var ids []string
	for _, id := range a.obPack() {
		ids = append(ids, id)
		if m, ok := models.Lookup(id); ok {
			ids = append(ids, m.Alternatives...)
		}
	}
	m := a.models
	a.async(func() {
		found := m.Survey(ids...)
		a.post(func() { a.ob.found = found })
	})
}

// obUse claims the models ids for VoxtNew ("Use them", "Use … for now"),
// off the main thread, then downloads what they need that is missing
// (the voice detection of a speech model).
func (a *app) obUse(ids ...string) {
	m := a.models
	a.async(func() {
		err := m.Use(context.Background(), ids...)
		a.post(func() {
			if err != nil {
				a.toast(i18n.T("Could not use the models: %v", err))
			}
			a.modelsChanged()
			for _, id := range ids {
				mod, _ := models.Lookup(id)
				for _, need := range mod.Needs {
					if st := a.modelStatus[need]; !st.Installed && !st.Downloading {
						a.downloadModel(need)
					}
				}
			}
		})
	})
}

func (a *app) clearOnboardingBlock() { a.ob.block = nil }

func (a *app) onboardingNoticeAction(action string) {
	a.clearOnboardingBlock()
	a.noticeAction(action)
}

func (a *app) onboardingTryLanguage() {
	a.showMainWindow()
	a.obLeave(remotePath)
}

func (a *app) onboardingDownloadAndContinue() {
	a.obDownload()
	a.obGo(a.ob.step + 1)
}

func (a *app) copyOnboardingResult(text string) {
	if a.copyText != nil {
		_ = a.copyText(text)
	} else {
		_ = mygo.Clipboard.Write(transfer.TextData(text))
	}
	a.toast(i18n.T("Copied"))
}

func (a *app) onboardingOpenPrivacy(url string) { openExternal(url) }

func (a *app) completeOnboardingLegacyTodo(kind, id string) {
	a.completeLegacyTodo(legacyTodo{kind: local.Kind(kind), id: id})
}
