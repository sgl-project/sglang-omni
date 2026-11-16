package main

import (
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type settingsUpdatesViewState struct {
	Present             bool   `json:"present"`
	Known               bool   `json:"known"`
	Enabled             bool   `json:"enabled"`
	CheckOnly           bool   `json:"check_only"`
	BundleUpdated       bool   `json:"bundle_updated"`
	Description         string `json:"description"`
	AutoCheckText       string `json:"auto_check_text"`
	BetaNote            string `json:"beta_note"`
	AutoInstallDisabled bool   `json:"auto_install_disabled"`
	AutoCheck           bool   `json:"auto_check"`
	AutoInstall         bool   `json:"auto_install"`
	Beta                bool   `json:"beta"`
}

type settingsUpdatesStateInput struct {
	Present       bool
	Known         bool
	Prefs         updatePrefs
	BundleUpdated bool
	Beta          bool
	Version       string
}

func buildSettingsUpdatesViewState(translator *i18n.Translator, input settingsUpdatesStateInput, now time.Time) settingsUpdatesViewState {
	state := settingsUpdatesViewState{
		Present: input.Present, Known: input.Known, Enabled: input.Prefs.Enabled, CheckOnly: input.Prefs.CheckOnly,
		BundleUpdated: input.BundleUpdated, AutoInstallDisabled: !input.Prefs.AutoCheck,
		AutoCheck: input.Prefs.AutoCheck, AutoInstall: input.Prefs.AutoInstall, Beta: input.Beta,
		AutoCheckText: translator.T("Once a day. Voxt asks before installing."),
	}
	switch {
	case !input.Known:
	case !input.Prefs.Enabled:
		state.Description = translator.T("This copy cannot update itself because it is a development build or is in a folder Voxt cannot write to. Move Voxt to Applications.")
	case input.Prefs.LastCheck.IsZero():
		state.Description = translator.T("Not checked yet.")
	default:
		state.Description = translator.T("Last checked %s.", uitext.TimeAgoFor(translator, input.Prefs.LastCheck, now))
	}
	if input.Prefs.CheckOnly {
		state.AutoCheckText = translator.T("Once a day. A card says when a new version is out.")
	}
	switch {
	case input.Beta:
		state.BetaNote = translator.T("Betas and releases, whichever is newer.")
	case strings.Contains(input.Version, "-"):
		state.BetaNote = translator.T("Releases only. This beta stays until a newer release comes out.")
	default:
		state.BetaNote = translator.T("Releases only.")
	}
	return state
}
