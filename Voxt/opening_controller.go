package main

// The opening (#4173): a few seconds of Voxt's mark, once per app version
// (before the setup guide on a new install); opening.go draws it.

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

const openingDuration = 3200 * time.Millisecond

// startOpening plays the opening on the first start of each app version,
// and remembers the version it played for.
func (a *app) startOpening() {
	version := a.version
	if a.prefs.Setup.OpeningShown && a.prefs.Setup.OpeningVersion == version {
		return
	}
	a.openingStarted = time.Now()
	a.updateSettings(func(s *settings.Settings) {
		s.Setup.OpeningShown, s.Setup.OpeningVersion = true, version
	})
}

// openingModel is the opening's view state this frame, nil once it has
// played or when the app's or the system's Reduce motion is on.
func (a *app) openingModel(systemReduceMotion bool) *openingViewState {
	if a.openingStarted.IsZero() {
		return nil
	}
	elapsed := time.Since(a.openingStarted)
	if elapsed >= openingDuration || a.prefs.Appearance.ReduceMotion || systemReduceMotion {
		a.openingStarted = time.Time{}
		return nil
	}
	return &openingViewState{Elapsed: elapsed}
}
