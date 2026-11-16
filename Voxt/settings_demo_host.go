package main

import "github.com/egoist/mygo/ui"

type settingsDemoRouter = ui.Router

func (host *meetingDemoHost) resetSettingsDemo(fixture meetingDemoFixture) {
	host.settingsOpen = false
	host.settingsRouter = ui.NewRouter("/general")
	host.settingsForm = settingsForm{}
	host.settingsStorageUI = settingsStorageLocalUI{}
	host.settingsMeetingUI = meetingSettingsUI{}

	if state := fixture.state.Settings; state != nil {
		host.settingsOpen = state.Open
		host.settingsRouter = ui.NewRouter(state.Path)
		if general := state.General; general != nil {
			host.settingsForm.settingsGeneralLocalUI = settingsGeneralLocalUI{
				language: general.Language, mainLanguage: general.MainLanguage, copyResult: general.CopyResult,
				menuBar: general.MenuBar, dock: general.Dock, closeToTray: general.CloseToTray, openAtLogin: general.OpenAtLogin,
			}
			host.settingsForm.loginIs = general.OpenAtLogin
		}
		if appearance := state.Appearance; appearance != nil {
			host.settingsForm.settingsAppearanceLocalUI = settingsAppearanceLocalUI{
				theme: appearance.Theme, scale: appearance.Scale, sidebar: appearance.Sidebar, reduceMotion: appearance.ReduceMotion,
			}
		}
		if privacy := state.Privacy; privacy != nil {
			host.settingsForm.settingsPrivacyLocalUI = settingsPrivacyLocalUI{
				keepHistory: privacy.KeepHistory, keepAudio: privacy.KeepAudio, logText: privacy.LogText,
				sendWindow: privacy.SendWindow, sendClipboard: privacy.SendClipboard, clearHistoryAsk: privacy.ClearHistoryAsk,
			}
			access := privacy.HistoryAccess
			host.historyAccessUI = historyAccessLocalUI{
				historyAccess: access.On, historySelection: access.Selection, ask: access.Ask, listOpen: access.ListOpen,
			}
		}
		if audio := state.Audio; audio != nil {
			host.settingsForm.settingsAudioLocalUI = settingsAudioLocalUI{
				skipSilence: audio.SkipSilence, liveResult: audio.LiveResult, cleanSound: audio.CleanSound,
				muteOutput: audio.MuteOutput, uploadMB: audio.UploadMB, maxMinutes: audio.MaxMinutes,
			}
		}
		if shortcuts := state.Shortcuts; shortcuts != nil {
			host.settingsForm.strictShortcuts = shortcuts.StrictShortcuts
		}
		if overlay := state.Overlay; overlay != nil {
			host.settingsForm.ovStyle = overlayForm{
				position: overlay.Position, opacity: overlay.Opacity, radius: overlay.Radius, margin: overlay.Margin,
				showTime: overlay.ShowTime, liveText: overlay.LiveText, liveLayout: overlay.LiveLayout,
			}
		}
		if storage := state.Storage; storage != nil {
			host.settingsStorageUI = settingsStorageLocalUI{resetAsk: storage.ResetAsk, resetText: storage.ResetText}
		}
	}
	if state := fixture.state.MeetingSettings; state != nil {
		host.pageUI.setOpen = true
		host.settingsMeetingUI = meetingSettingsUI{
			mode: settingsLanguageCode(state.CaptureModeSelected, state.CaptureModeOptions), hide: state.HideFromSharing,
		}
	}
	host.settingsDialogUI = settingsDialogLocalUI{
		Open: &host.settingsOpen, Router: host.settingsRouter, Form: &host.settingsForm,
		HistoryAccess: &host.historyAccessUI, Storage: &host.settingsStorageUI,
	}
}

func (host *meetingDemoHost) setAppLanguage(languageTag string) {
	host.record("setAppLanguage %s", languageTag)
}
func (host *meetingDemoHost) setMainLanguage(languageCode string) {
	host.record("setMainLanguage %s", languageCode)
}
func (host *meetingDemoHost) addOtherLanguage(languageCode string) {
	host.record("addOtherLanguage %s", languageCode)
}
func (host *meetingDemoHost) removeOtherLanguage(languageCode string) {
	host.record("removeOtherLanguage %s", languageCode)
}
func (host *meetingDemoHost) setCopyResult(on bool)  { host.record("setCopyResult %t", on) }
func (host *meetingDemoHost) setMenuBar(on bool)     { host.record("setMenuBar %t", on) }
func (host *meetingDemoHost) setDock(on bool)        { host.record("setDock %t", on) }
func (host *meetingDemoHost) setCloseToTray(on bool) { host.record("setCloseToTray %t", on) }
func (host *meetingDemoHost) setOpenAtLogin(on bool) { host.record("setOpenAtLogin %t", on) }
func (host *meetingDemoHost) openOnboarding()        { host.record("openOnboarding") }
func (host *meetingDemoHost) setTheme(theme string)  { host.record("setTheme %s", theme) }
func (host *meetingDemoHost) setSidebarChoice(choice string) {
	host.record("setSidebarChoice %s", choice)
}
func (host *meetingDemoHost) setReduceMotion(on bool) { host.record("setReduceMotion %t", on) }
func (host *meetingDemoHost) setInterfaceScale(scale float64) {
	host.record("setInterfaceScale %g", scale)
}
func (host *meetingDemoHost) setFeatureShown(featureID string, shown bool) {
	host.record("setFeatureShown %s %t", featureID, shown)
}

func (host *meetingDemoHost) askRequired() { host.record("askRequired") }
func (host *meetingDemoHost) requestPermission(which string) {
	host.record("requestPermission %s", which)
}
func (host *meetingDemoHost) recheckPermissions() { host.record("recheckPermissions") }
func (host *meetingDemoHost) openPrivacy(_ *permissionUIContext, url string) {
	host.record("openPrivacy %s", url)
}

func (host *meetingDemoHost) loadDevices()   { host.record("loadDevices") }
func (host *meetingDemoHost) restartHelper() { host.record("restartHelper") }
func (host *meetingDemoHost) addDevice(deviceID string) {
	host.record("addDevice %s", deviceID)
}
func (host *meetingDemoHost) moveDevice(deviceID string, delta int) {
	host.record("moveDevice %s %d", deviceID, delta)
}
func (host *meetingDemoHost) removeDevice(deviceID string) {
	host.record("removeDevice %s", deviceID)
}
func (host *meetingDemoHost) setSkipSilence(on bool) { host.record("setSkipSilence %t", on) }
func (host *meetingDemoHost) setCleanSound(on bool)  { host.record("setCleanSound %t", on) }
func (host *meetingDemoHost) setLiveResult(on bool)  { host.record("setLiveResult %t", on) }
func (host *meetingDemoHost) setMuteOutput(on bool)  { host.record("setMuteOutput %t", on) }
func (host *meetingDemoHost) setUploadMB(megabytes float64) {
	host.record("setUploadMB %g", megabytes)
}
func (host *meetingDemoHost) setMaxMinutes(minutes float64) {
	host.record("setMaxMinutes %g", minutes)
}
func (host *meetingDemoHost) chooseSoundSource(sourceID string) {
	host.record("chooseSoundSource %s", sourceID)
}
func (host *meetingDemoHost) chooseSoundPair(start, stop string) {
	host.record("chooseSoundPair %s %s", start, stop)
}
func (host *meetingDemoHost) previewSoundPair(start, stop string) {
	host.record("previewSoundPair %s %s", start, stop)
}
func (host *meetingDemoHost) stopPreview() { host.record("stopPreview") }
func (host *meetingDemoHost) chooseSystemSound(eventID, choice string) {
	host.record("chooseSystemSound %s %s", eventID, choice)
}
func (host *meetingDemoHost) tryCueSound(eventID string) { host.record("tryCueSound %s", eventID) }
func (host *meetingDemoHost) setSoundVolume(volume int)  { host.record("setSoundVolume %d", volume) }

func (host *meetingDemoHost) overlayEdited() { host.record("overlayEdited") }
func (host *meetingDemoHost) applyShortcutPreset(preset string) {
	host.record("applyShortcutPreset %s", preset)
}
func (host *meetingDemoHost) openShortcutPermissions() { host.record("openShortcutPermissions") }
func (host *meetingDemoHost) openShortcutFeature(path string) {
	host.record("openShortcutFeature %s", path)
}
func (host *meetingDemoHost) setStrictShortcuts(on bool) { host.record("setStrictShortcuts %t", on) }

func (host *meetingDemoHost) setSendWindow(on bool) { host.record("setSendWindow %t", on) }
func (host *meetingDemoHost) setSendClipboardImage(on bool) {
	host.record("setSendClipboardImage %t", on)
}
func (host *meetingDemoHost) setKeepHistory(on bool) { host.record("setKeepHistory %t", on) }
func (host *meetingDemoHost) setKeepAudio(on bool)   { host.record("setKeepAudio %t", on) }
func (host *meetingDemoHost) showHistorySettings()   { host.record("showHistorySettings") }
func (host *meetingDemoHost) clearHistory()          { host.record("clearHistory") }
func (host *meetingDemoHost) setLogText(on bool)     { host.record("setLogText %t", on) }

func (host *meetingDemoHost) openStorageFolder(directory string) {
	host.record("openStorageFolder %s", directory)
}
func (host *meetingDemoHost) chooseStorageDataFolder()  { host.record("chooseStorageDataFolder") }
func (host *meetingDemoHost) chooseStorageModelFolder() { host.record("chooseStorageModelFolder") }
func (host *meetingDemoHost) removePrevious()           { host.record("removePrevious") }
func (host *meetingDemoHost) dismissMoveFailure()       { host.record("dismissMoveFailure") }
func (host *meetingDemoHost) cancelStorageDataMove()    { host.record("cancelStorageDataMove") }
func (host *meetingDemoHost) cancelStorageModelMove()   { host.record("cancelStorageModelMove") }
func (host *meetingDemoHost) resetLook()                { host.record("resetLook") }
func (host *meetingDemoHost) resetSettings()            { host.record("resetSettings") }
func (host *meetingDemoHost) askErase() {
	host.settingsStorageUI.resetAsk = "everything"
	host.settingsStorageUI.resetText = ""
	host.record("askErase")
}
func (host *meetingDemoHost) eraseAll() {
	host.settingsStorageUI.resetAsk = ""
	host.settingsStorageUI.resetText = ""
	host.record("eraseAll")
}
func (host *meetingDemoHost) setStorageModelBackup(on bool) {
	host.record("setStorageModelBackup %t", on)
}

func (host *meetingDemoHost) runSelfCheck() { host.record("runSelfCheck") }
func (host *meetingDemoHost) saveBundle()   { host.record("saveBundle") }
func (host *meetingDemoHost) copyDiagnostics(_ *aboutUIContext) {
	host.record("copyDiagnostics")
}
func (host *meetingDemoHost) reportBug(_ *aboutUIContext) { host.record("reportBug") }
func (host *meetingDemoHost) navigate(path string)        { host.record("navigate %s", path) }

func (host *meetingDemoHost) setMeetingCaptureMode(captureMode string) {
	host.record("setMeetingCaptureMode %s", captureMode)
}
func (host *meetingDemoHost) setMeetingHideFromSharing(hideFromSharing bool) {
	host.record("setMeetingHideFromSharing %t", hideFromSharing)
}
func (host *meetingDemoHost) setMeetingASR(speechModelID string) {
	host.record("setMeetingASR %s", speechModelID)
}
func (host *meetingDemoHost) setMeetingLanguage(languageCode string) {
	host.record("setMeetingLanguage %s", languageCode)
}
func (host *meetingDemoHost) setMeetingLLM(languageModelID string) {
	host.record("setMeetingLLM %s", languageModelID)
}

func (host *meetingDemoHost) answerShortcutPrompt() { host.record("answerShortcutPrompt") }
func (host *meetingDemoHost) openShortcutPromptLink(_ *shortcutPromptUIContext, link string) {
	host.record("openShortcutPromptLink %s", link)
}
func (host *meetingDemoHost) changeShortcutPrompt(path string) {
	host.record("changeShortcutPrompt %s", path)
}
