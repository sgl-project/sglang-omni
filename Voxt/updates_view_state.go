package main

import (
	"context"
	"errors"
	"net"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type updatesViewStateInput struct {
	Available      bool
	Known          bool
	Prefs          updatePrefs
	BundleUpdated  bool
	Beta           bool
	RunningVersion string
	Now            time.Time
}

type updatesViewState struct {
	Available            bool   `json:"available"`
	Known                bool   `json:"known"`
	Enabled              bool   `json:"enabled"`
	CheckOnly            bool   `json:"check_only"`
	BundleUpdated        bool   `json:"bundle_updated"`
	AutoCheck            bool   `json:"auto_check"`
	AutoInstall          bool   `json:"auto_install"`
	Beta                 bool   `json:"beta"`
	CheckDescription     string `json:"check_description"`
	AutomaticDescription string `json:"automatic_description"`
	BetaNote             string `json:"beta_note"`
}

func buildUpdatesViewState(translator *i18n.Translator, input updatesViewStateInput) updatesViewState {
	state := updatesViewState{
		Available: input.Available, Known: input.Known, Enabled: input.Prefs.Enabled,
		CheckOnly: input.Prefs.CheckOnly, BundleUpdated: input.BundleUpdated,
		AutoCheck: input.Prefs.AutoCheck, AutoInstall: input.Prefs.AutoInstall, Beta: input.Beta,
	}
	if !input.Available {
		return state
	}
	switch {
	case !input.Known:
	case !input.Prefs.Enabled:
		state.CheckDescription = translator.T("This copy cannot update itself because it is a development build or is in a folder Voxt cannot write to. Move Voxt to Applications.")
	case input.Prefs.LastCheck.IsZero():
		state.CheckDescription = translator.T("Not checked yet.")
	default:
		state.CheckDescription = translator.T("Last checked %s.", uitext.TimeAgoFor(translator, input.Prefs.LastCheck, input.Now))
	}
	state.AutomaticDescription = translator.T("Once a day. Voxt asks before installing.")
	if input.Prefs.CheckOnly {
		state.AutomaticDescription = translator.T("Once a day. An Update button shows when a new version is out.")
	}
	state.BetaNote = translator.T("Releases only.")
	switch {
	case input.Beta:
		state.BetaNote = translator.T("Betas and releases, whichever is newer.")
	case strings.Contains(input.RunningVersion, "-"):
		state.BetaNote = translator.T("Releases only. This beta stays until a newer release comes out.")
	}
	return state
}

type releaseViewStateInput struct {
	Available      bool
	RunningVersion string
	Done           bool
	Newer          bool
	Found          release
	Err            error
}

type releaseViewState struct {
	Available bool   `json:"available"`
	Done      bool   `json:"done"`
	Newer     bool   `json:"newer"`
	Title     string `json:"title"`
	Subtitle  string `json:"subtitle"`
	ErrorText string `json:"error_text"`
	Notes     string `json:"notes"`
	Download  string `json:"download"`
}

func buildReleaseViewState(translator *i18n.Translator, input releaseViewStateInput) releaseViewState {
	state := releaseViewState{
		Available: input.Available, Done: input.Done, Newer: input.Newer,
		Title: translator.T("Checking for updates…"),
	}
	switch {
	case !input.Done:
	case input.Err != nil:
		state.Title = translator.T("Couldn't check for updates")
		state.ErrorText = releaseErrTextFor(translator, input.Err)
	case input.Newer:
		state.Title = translator.T("Voxt %s is available", input.Found.Version)
		state.Subtitle = translator.T("You have Voxt %s.", input.RunningVersion)
		state.Notes, state.Download = input.Found.Notes, input.Found.Download
	default:
		state.Title = translator.T("You're up to date")
		state.Subtitle = translator.T("Voxt %s is the newest version.", input.RunningVersion)
	}
	return state
}

func releaseErrTextFor(translator *i18n.Translator, err error) string {
	var netError net.Error
	switch {
	case errors.Is(err, errRateLimited):
		return translator.T("GitHub limits how often Voxt can check. Try again in an hour.")
	case errors.As(err, &netError) && netError.Timeout(), errors.Is(err, context.DeadlineExceeded):
		return translator.T("GitHub did not answer in time. Check your connection, or the proxy in Speech models → Network.")
	}
	return translator.T("Voxt could not read the list of releases on GitHub: %v", err)
}

// sidebarUpdateViewState is the update control at the foot of the sidebar
// (a translucent or solid bar when expanded, a circular icon button in the
// collapsed icon rail): nil when up to date, else "Update …" or
// "Restart to update".
type sidebarUpdateViewState struct {
	Ready   bool   `json:"ready,omitempty"`
	Version string `json:"version,omitempty"`
	Label   string `json:"label"`
	Tooltip string `json:"tooltip"`
}

func buildSidebarUpdateViewState(translator *i18n.Translator, bundleUpdated bool, installedVersion, availableVersion string) *sidebarUpdateViewState {
	if bundleUpdated {
		tip := translator.T("Restart Voxt to finish updating.")
		if installedVersion != "" {
			tip = translator.T("Voxt %s is ready. Restart to finish updating.", installedVersion)
		}
		return &sidebarUpdateViewState{
			Ready:   true,
			Version: installedVersion,
			Label:   translator.T("Restart to update"),
			Tooltip: tip,
		}
	}
	if availableVersion == "" {
		return nil
	}
	return &sidebarUpdateViewState{
		Version: availableVersion,
		Label:   translator.T("Update %s", availableVersion),
		Tooltip: translator.T("Voxt %s is available", availableVersion),
	}
}
