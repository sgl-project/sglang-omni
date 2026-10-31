package main

// The app's events between its areas (refactor R10): who hears of a
// session's new snapshot, of its end, and of a change of the history.
// The publishers (session_link.go, history_page.go) run the observers
// listed here, in this order, instead of calling each area; an area that
// cares adds itself in connectEvents. The order is the one the calls had,
// and matters (docs/architecture.md §3):
//
//	onSession     each new snapshot, before the session's own work (its
//	              notice, its end, the shortcuts): the setup guide, the
//	              recording's sounds and the output's mute
//	afterSession  each new snapshot, after it: the menu bar (Start or
//	              Stop), the overlay
//	onSessionEnd  a session that ended, before its card or notice: Home
//	onHistory     the history changed (an entry kept, changed, deleted, a
//	              try ended): Home, the menu bar's recent results, the
//	              History page, the automatic clean-up

import (
	"strings"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
)

// events are the observers of the app's events, main-thread only.
type events struct {
	onSession    []func(was, s session.Snapshot)
	afterSession []func(was, s session.Snapshot)
	onSessionEnd []func(s session.Snapshot)
	onHistory    []func()
}

// connectEvents adds every area's observers, in their order; newApp
// calls it.
func (a *app) connectEvents() {
	a.onSession = append(a.onSession,
		func(_, s session.Snapshot) { a.obSessionChanged(s) }, // onboarding.go: a try of the setup guide passes
		a.recordingAudio, // mute_link.go: the start and stop sounds, the output's mute
	)
	a.afterSession = append(a.afterSession,
		func(was, s session.Snapshot) {
			if s.State.Live() != was.State.Live() {
				a.buildTrayMenu() // tray.go: Start or Stop
			}
		},
		func(_, _ session.Snapshot) { a.overlaySync() }, // overlay.go
	)
	a.onSessionEnd = append(a.onSessionEnd, func(s session.Snapshot) {
		if s.State == session.Done && strings.TrimSpace(s.Result) != "" {
			a.markTried() // home_try.go: Home stops offering its Try area
		}
	})
	a.onHistory = append(a.onHistory,
		a.homeChanged,    // home_page.go: today's numbers, the recent results
		a.loadTrayRecent, // tray.go: the menu bar's recent results
		a.historyReload,  // history_page.go: the page, now or when it shows next
		a.pruneHistory,   // history_settings_ops.go: the automatic clean-up
	)
}
