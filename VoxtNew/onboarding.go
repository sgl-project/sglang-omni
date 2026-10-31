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
// (settings.Setup), so it does not open again by itself.

import (
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
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

// obStepTitles are the steps' short names, for the breadcrumbs.
func obStepTitles() []string {
	return []string{i18n.T("Welcome"), i18n.T("Permissions"), i18n.T("Models"), i18n.T("Transcribe"),
		i18n.T("Translate"), i18n.T("Rewrite"), i18n.T("More")}
}

// openOnboarding opens the guide at its first step.
func (a *app) openOnboarding() {
	a.settingsOpen = false
	a.ob.open, a.ob.step = true, obWelcome
	a.ob.tries, a.ob.passed = map[int]string{}, map[int]bool{}
	a.ob.block, a.ob.hint = nil, ""
	a.ob.found = nil
	a.recheckPermissions()
}

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
		a.ob.hint = i18n.T("That was the %s shortcut: this step tries %s.", uitext.KindTitle(kind), uitext.KindTitle(want))
		return
	}
	a.ob.hint = i18n.T("That was the %s shortcut: this step tries %s (%s).", uitext.KindTitle(kind), uitext.KindTitle(want), keys)
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
	case obPermissions:
		a.recheckPermissions()
	case obModels:
		a.obSurvey() // onboarding_models.go
	}
}

// onboardingRow is Settings → General's way back to the guide.
func (a *app) onboardingRow(c *ui.Context) {
	settingRow(c, "setup-guide", i18n.T("Setup guide"), i18n.T("Permissions, the local models, and a try of each feature, as on the first day."), func() {
		if ui.Button(c.Key("setup-guide-open"), i18n.T("Open the setup guide")).Clicked() {
			a.openOnboarding()
		}
	})
}
