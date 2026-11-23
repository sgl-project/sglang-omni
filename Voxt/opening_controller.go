package main

// The opening (#4173): a few seconds of Voxt's mark at each start of the
// app (before the setup guide on a new install); opening.go draws it.

import "time"

const openingDuration = 3200 * time.Millisecond

// startOpening plays the opening when the app starts. A window that waits
// hidden (opened at login into the menu bar) skips it: it would play
// unseen. Reopening the window while the app runs does not come here.
func (a *app) startOpening(hidden bool) {
	if !hidden {
		a.openingStarted = time.Now()
	}
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
