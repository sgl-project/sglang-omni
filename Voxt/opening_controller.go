package main

// The first-use opening (#4173): a few seconds of Voxt's mark before the
// setup guide, once per install; opening.go draws it.

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

const openingDuration = 3200 * time.Millisecond

// startOpening plays the opening on the first start of a setup not done
// yet, and remembers it played.
func (a *app) startOpening() {
	if a.prefs.Setup.Done || a.prefs.Setup.OpeningShown {
		return
	}
	a.openingStarted = time.Now()
	a.updateSettings(func(s *settings.Settings) { s.Setup.OpeningShown = true })
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
