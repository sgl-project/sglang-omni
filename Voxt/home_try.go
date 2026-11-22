package main

// Where a try runs: a shortcut pressed while Voxt is in front starts a try
// on a feature page or the setup guide's matching step. Home starts normal
// sessions, as other apps and pages without a Try area do.

import (
	"os"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// ---- Try places ----

// tryPlace is where a shortcut pressed while Voxt is in front runs as a try.
type tryPlace int

const (
	placeNone       tryPlace = iota // not a try: a session as anywhere else
	placeIgnore                     // the setup guide waits for another feature's shortcut
	placeFeature                    // the feature page's Try area
	placeOnboarding                 // the setup guide's step
)

// tryPlace decides where a shortcut of kind, pressed with target in
// front, runs: a try when target is Voxt and what shows has a Try area
// for it (the setup guide's step or the feature's page), and no text field
// of the window has the focus: then a session types into that field, as in
// another app (Voxt 1.x; FB10, a feature's rules being edited).
func (a *app) tryPlace(kind store.Kind, target proto.App) tryPlace {
	if target.PID == 0 || target.PID != os.Getpid() {
		return placeNone
	}
	if a.ob.open {
		if k, ok := obStepKind(a.ob.step); ok && k == kind {
			return placeOnboarding
		}
		return placeIgnore
	}
	if a.settingsOpen || a.textFocused() {
		return placeNone
	}
	if a.router.Path() == "/"+string(kind) {
		return placeFeature
	}
	return placeNone
}

// tryStarted keeps a try a shortcut started (id, "" when it did not:
// then n says why) for the place that shows it.
func (a *app) tryStarted(place tryPlace, kind store.Kind, id string, n *sessionNotice) {
	switch place {
	case placeFeature:
		a.fui(string(kind)).block = n
	case placeOnboarding:
		a.obTryStarted(id, n) // onboarding.go
	}
}

// markTried keeps the legacy first-result flag consistent when settings
// move between versions, although Home no longer reads it.
func (a *app) markTried() {
	if !a.prefs.Setup.Tried {
		a.updateSettings(func(s *settings.Settings) { s.Setup.Tried = true })
	}
}

// textFocused says whether the user types in a field of the main window
// (MyGo patch 0010).
func (a *app) textFocused() bool { return a.mainUI.TextFocused() }
