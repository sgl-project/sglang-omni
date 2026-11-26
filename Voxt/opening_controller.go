package main

// The opening (#4173): a few seconds of Voxt's mark at each start of the
// app (before the setup guide on a new install); opening.go draws it.

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// openingTimeline is the opening's length as opening.go draws it. It plays
// in the length Settings → Appearance chose, faster or slower along the
// same timeline.
const openingTimeline = 3200 * time.Millisecond

// openingLength is the length the opening plays in.
func (a *app) openingLength() time.Duration {
	if n := a.prefs.Appearance.OpeningMillis; n > 0 {
		return time.Duration(n) * time.Millisecond
	}
	return settings.DefaultOpeningMillis * time.Millisecond
}

// startOpening plays the opening when the app starts, unless Settings →
// Appearance turned it off. A window that waits hidden (opened at login
// into the menu bar) skips it: it would play unseen. Reopening the window
// while the app runs does not come here.
func (a *app) startOpening(hidden bool) {
	if !hidden && a.prefs.Appearance.Opening {
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
	length := a.openingLength()
	if elapsed >= length || a.prefs.Appearance.ReduceMotion || systemReduceMotion {
		a.openingStarted = time.Time{}
		return nil
	}
	return &openingViewState{Elapsed: time.Duration(float64(elapsed) * float64(openingTimeline) / float64(length))}
}
