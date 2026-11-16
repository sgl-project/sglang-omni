package main

// Settings controller: the settings service seen from the UI (a snapshot
// in a.prefs, kept by settings.Changed), and the language and theme it drives.

import (
	"context"
	"reflect"
	"slices"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// settingsState is the settings dialog's sections' view state, embedded
// in app: each section keeps its own struct in its file.
type settingsState struct {
	storageState // settings_storage_controller.go
	aboutState   // settings_about_controller.go
	settingsForm // shell_settings.go: the controls that edit a setting

	settingsRevision uint64
	generalCache     struct {
		key   generalSettingsCacheKey
		state settingsGeneralViewState
		valid bool
	}
}

// updateSettings changes the settings: at once in the snapshot, and in
// the service (saved in the background, then announced with Changed).
// fn must only set fields, as it runs on both.
func (a *app) updateSettings(fn func(*settings.Settings)) {
	a.settingsRevision++
	a.featureRevision++
	fn(&a.prefs)
	st := a.settings
	a.async(func() {
		if err := st.Update(context.Background(), fn); err != nil {
			a.post(func() { a.toast(i18n.T("Could not save the settings: %v", err)) })
		}
	})
}

// settingsChanged applies saved settings to the app.
func (a *app) settingsChanged(old, new settings.Settings) {
	a.settingsRevision++
	a.featureRevision++
	a.prefs = new
	a.familySettingsChanged(old, new) // family.go: the shared settings, the manifest
	if (old.Dictionary.Learn && !new.Dictionary.Learn) || (old.Dictionary.Enabled && !new.Dictionary.Enabled) {
		a.learner.Stop() // learn_link.go: nothing read from now on, nothing proposed
	}
	if old.General.Language != new.General.Language {
		a.applyLocale()
	}
	if old.General.MenuBar != new.General.MenuBar {
		a.applyTray() // tray.go
	}
	if old.General.HideDock != new.General.HideDock {
		a.applyDock() // tray.go
	}
	if old.Updates.BetaOn(a.version) != new.Updates.BetaOn(a.version) {
		a.applyBeta() // updates.go
	}
	if !reflect.DeepEqual(old.AppEnhancement, new.AppEnhancement) {
		a.syncTrees() // appenhance.go
	}
	if !reflect.DeepEqual(old.Features, new.Features) {
		a.syncHotkeys()   // session_hotkeys.go
		a.buildTrayMenu() // the shortcuts it shows
	}
	if !slices.Equal(old.Audio.Devices, new.Audio.Devices) {
		a.buildTrayMenu() // the microphone it checks
	}
	if old.General.LaunchAtLogin != new.General.LaunchAtLogin && new.General.LaunchAtLogin != a.loginIs && a.hasMenu {
		on := new.General.LaunchAtLogin
		go func() {
			err := mygo.App.SetOpenAtLogin(on)
			a.post(func() {
				if err != nil {
					a.toast(i18n.T("Could not change “Open at login”: %v", err))
					return
				}
				a.loginIs = on
			})
		}()
	}
	if old.Overlay != new.Overlay {
		a.overlayRestyle() // overlay.go
	}
	if old.Features.Transcribe.ASR != new.Features.Transcribe.ASR {
		a.warmSpeech() // models_omni.go: the new model loads before its first session
	}
	if old.Models.KeepLoaded != new.Models.KeepLoaded {
		a.applyKeepLoaded(new.Models) // models_keep.go
	}
	if !reflect.DeepEqual(old.Providers, new.Providers) || old.Network != new.Network {
		a.readUnreadKeys() // remote_link.go: a setup that came over
		a.syncRemote()
	}
	a.historySettingsChanged(old, new) // history_settings_ops.go
	a.syncHistoryAccess()              // history_access.go
	// The problems' cards: the language of their text may have changed
	// (shell_notices.go).
	a.checkModels()
	a.checkHelper()
	a.loadSettingsForm()
}

// loadSettingsForm sets the controls from a.prefs.
func (a *app) loadSettingsForm() {
	s := a.prefs
	f := &a.settingsForm
	general := a.generalSettingsModel()
	f.language, f.mainLanguage = general.Language, general.MainLanguage
	f.menuBar, f.closeToTray, f.openAtLogin = general.MenuBar, general.CloseToTray, general.OpenAtLogin
	f.dock, f.copyResult = general.Dock, general.CopyResult
	appearance := a.appearanceSettingsModel()
	f.theme, f.scale, f.reduceMotion, f.sidebar = appearance.Theme, appearance.Scale, appearance.ReduceMotion, appearance.Sidebar
	f.strictShortcuts = s.Features.StrictShortcuts
	f.keepHistory, f.keepAudio, f.logText = s.Privacy.KeepHistory, s.Privacy.KeepAudio, s.Privacy.LogText
	f.sendWindow, f.sendClipboard = s.Privacy.SendWindow, s.Privacy.SendClipboardImage
	a.histAccessUI.historyAccess, a.histAccessUI.historySelection = s.Privacy.HistoryAccess, s.Privacy.HistorySelection
	f.skipSilence = !s.Audio.VAD.Off
	f.liveResult = s.Audio.LiveResult
	f.cleanSound = !s.Audio.Clean.Off
	f.muteOutput = s.Audio.MuteOn()
	f.uploadMB, f.maxMinutes = float64(s.Audio.UploadMB), float64(s.Audio.MaxSeconds)/60
	a.loadOverlayForm() // settings_overlay_controller.go
}

// syncOpenAtLogin reads whether macOS opens VoxtNew at login (the user
// may have turned it off in System Settings) and makes the setting say
// so; it runs when the window opens.
func (a *app) syncOpenAtLogin() {
	go func() {
		on := mygo.App.OpenAtLogin()
		a.post(func() {
			a.loginIs = on
			if on != a.prefs.General.LaunchAtLogin {
				a.updateSettings(func(s *settings.Settings) { s.General.LaunchAtLogin = on })
			}
		})
	}()
}

// openSettings shows the settings dialog at a section's path ("/audio").
func (a *app) openSettings(sub string) {
	if a.setRouter.Path() != sub || !a.settingsOpen {
		a.settingsShown(sub)
	}
	if a.setRouter.Path() != sub {
		a.setRouter.Replace(sub)
	}
	a.settingsOpen = true
}

// settingsShown runs when the settings dialog goes to a section: sections
// that read something when they show start it here.
func (a *app) settingsShown(sub string) {
	switch sub {
	case "/permissions":
		a.recheckPermissions() // permissions.go
	case "/audio":
		a.loadDevices()         // settings_audio_controller.go
		a.prepareSoundChoices() // settings_soundchoice_controller.go
	case "/storage":
		a.loadStorage() // settings_storage_controller.go
	case "/about":
		a.loadSysInfo() // sysinfo.go
		a.loadUpdates()
	}
}

// shownPath is the path of what shows: the settings dialog's section
// ("/settings/audio") while it is open, else the page's.
func (a *app) shownPath() string {
	if a.settingsOpen {
		return "/settings" + a.setRouter.Path()
	}
	return a.router.Path()
}

func (a *app) settingsDialogModel(now time.Time) settingsDialogViewState {
	state := buildSettingsDialogViewState(i18n.CurrentTranslator(), a.settingsOpen, a.setRouter.Path(), a.shownPath())
	if !state.Open {
		return state
	}
	switch strings.TrimPrefix(state.Path, "/") {
	case "general":
		a.lastSettings = "general"
		section := a.generalSettingsModel()
		state.General = &section
	case "permissions":
		a.lastSettings = "permissions"
		section := a.permissionsModel()
		state.Permissions = &section
	case "audio":
		a.lastSettings = "audio"
		section := a.audioSettingsModel(now)
		state.Audio = &section
	case "overlay":
		a.lastSettings = "overlay"
		section := a.overlaySettingsModel()
		state.Overlay = &section
	case "shortcuts":
		a.lastSettings = "shortcuts"
		section := a.shortcutsSettingsModel()
		state.Shortcuts = &section
	case "appearance":
		a.lastSettings = "appearance"
		section := a.appearanceSettingsModel()
		state.Appearance = &section
	case "privacy":
		a.lastSettings = "privacy"
		section := a.privacySettingsModel(now)
		state.Privacy = &section
	case "storage":
		a.lastSettings = "storage"
		section := a.storageSettingsModel()
		state.Storage = &section
	case "about":
		a.lastSettings = "about"
		section := a.aboutSettingsModel()
		state.About = &section
		updates := a.updatesSettingsModel(now)
		state.Updates = &updates
	}
	return state
}
