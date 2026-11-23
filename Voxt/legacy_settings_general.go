package main

import (
	"context"
	"log"
	"path/filepath"
	"strconv"

	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// legacyApplyGeneral reads showInDock from the previous Voxt's preferences
// (Voxt 1.x's General → Show in Dock) into General.HideDock. VoxtNew
// already had the toggle; upgrades lost it because import never copied it.
func legacyApplyGeneral(voxtDir string, s *settings.Settings) bool {
	raw := legacyPreference(voxtDir, "showInDock")
	if raw == "" {
		return false
	}
	show, err := strconv.ParseBool(raw)
	if err != nil {
		return false
	}
	s.General.HideDock = !show
	if s.General.HideDock && !s.General.MenuBar {
		s.General.MenuBar = true
	}
	return true
}

// legacyDockSettled is the document that says the previous Voxt's Show in
// Dock was read, by an import or at the start (legacyDockAtStart): once,
// so the switch here is the user's from then on.
const legacyDockSettled = "legacy-dock-settled"

// legacyDockAtStart reads, once, the previous Voxt's Show in Dock after
// the setup: 2.0.0 and 2.0.1 did not, which left their users in the Dock
// when Voxt 1.x was not (#3788). An import since 2.0.2
// (legacyEnhanceSettled) read it already, and the user may have changed
// the switch since; Dock off here is the user's own. Off the main thread.
func (a *app) legacyDockAtStart() {
	lib := legacyLibrary()
	if lib == "" || !a.prefs.Setup.Done || a.prefs.General.HideDock {
		return
	}
	docs := a.lib.store.Documents()
	a.async(func() {
		ctx := context.Background()
		for _, id := range []string{legacyDockSettled, legacyEnhanceSettled} {
			if _, ok, err := docs.Get(ctx, id); err != nil || ok {
				return
			}
		}
		path := legacyDatabase(lib)
		if path == "" || !legacyReadable(lib) {
			return
		}
		if err := docs.Put(ctx, legacyDockSettled, []byte("true")); err != nil {
			log.Printf("legacy dock: %v", err)
			return
		}
		voxtDir := filepath.Dir(path)
		var s settings.Settings
		if !legacyApplyGeneral(voxtDir, &s) || !s.General.HideDock {
			return
		}
		log.Printf("legacy dock: out of the Dock, as in the previous Voxt")
		a.post(func() {
			a.updateSettings(func(s *settings.Settings) { legacyApplyGeneral(voxtDir, s) })
		})
	})
}
