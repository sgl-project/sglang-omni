package main

// Where a try runs (home_page.go): a shortcut pressed while VoxtNew is in
// front starts a try on the page it shows, and Home offers its Try area
// until the first result.

import (
	"os"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
)

// ---- Try places ----

// tryPlace is where a shortcut pressed while VoxtNew is in front runs
// as a try.
type tryPlace int

const (
	placeNone       tryPlace = iota // not a try: a session as anywhere else
	placeIgnore                     // the setup guide waits for another feature's shortcut
	placeHome                       // Home's Try area
	placeFeature                    // the feature page's Try area
	placeOnboarding                 // the setup guide's step
)

// tryPlace decides where a shortcut of kind, pressed with target in
// front, runs: a try when target is VoxtNew and what shows has a Try
// area for it (the setup guide's step, Home, the feature's page).
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
	if a.settingsOpen {
		return placeNone
	}
	switch a.router.Path() {
	case "/home":
		return placeHome
	case "/" + string(kind):
		return placeFeature
	}
	return placeNone
}

// tryStarted keeps a try a shortcut started (id, "" when it did not:
// then n says why) for the place that shows it.
func (a *app) tryStarted(place tryPlace, kind store.Kind, id string, n *sessionNotice) {
	switch place {
	case placeHome:
		a.hm.block = n
	case placeFeature:
		a.fui(string(kind)).block = n
	case placeOnboarding:
		a.obTryStarted(id, n) // onboarding.go
	}
}

// homeShowsTry says whether Home offers its Try area: until VoxtNew has
// given a result once (Setup.Tried, or a result in the history, read
// before the flag existed). Once shown, it stays through this visit to
// Home while its try runs or shows a result, or says why it did not
// start, so a finished try does not take its result away.
func (a *app) homeShowsTry(mine bool) bool {
	hm := &a.hm
	if !a.prefs.Setup.Tried && len(hm.recent) == 0 {
		hm.tryOn = true
		return true
	}
	return hm.tryOn && (mine || hm.block != nil)
}

// markTried records that VoxtNew has given a result (Home's Try area
// goes for good, the history cleared or not).
func (a *app) markTried() {
	if !a.prefs.Setup.Tried {
		a.updateSettings(func(s *settings.Settings) { s.Setup.Tried = true })
	}
}
