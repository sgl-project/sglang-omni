package main

import (
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
