package main

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func (a *app) experimentalSettingsModel() settingsExperimentalViewState {
	return buildSettingsExperimentalViewState(i18n.CurrentTranslator(), settingsExperimentalStateInput{
		Features: a.prefs.Features, HelperRecordsKeys: a.helperRecordsKeys(), KeepHistory: a.prefs.Privacy.KeepHistory,
	})
}

// setSessionKeys saves a session slot's shortcut (Settings → Experimental's
// Sessions); the settings' change re-syncs the shortcuts.
func (a *app) setSessionKeys(slot int, keys string) {
	if slot < 0 || slot >= len(hotkey.Sessions) {
		return
	}
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Sessions[slot] = settings.Hotkey{Keys: keys, Trigger: settings.TriggerTap}
	})
}

// setSessionsOn turns sessions on or off. Off goes back to the first
// session and takes the session row off the cards; the other sessions'
// cards wait for on.
func (a *app) setSessionsOn(on bool) {
	a.updateSettings(func(s *settings.Settings) { s.Features.SessionsOn = on })
	if !on {
		a.switchSlot(a.sessions()[0].id)
		a.emptyCard = false
	}
	a.buildTrayMenu() // its session list comes and goes with the switch
	a.overlaySync()
}
