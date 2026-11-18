package main

type settingsGeneralActions interface {
	setAppLanguage(languageTag string)
	setMainLanguage(languageCode string)
	addOtherLanguage(languageCode string)
	removeOtherLanguage(languageCode string)
	setCopyResult(on bool)
	setMenuBar(on bool)
	setDock(on bool)
	setCloseToTray(on bool)
	setOpenAtLogin(on bool)
	setFeatureShown(featureID string, shown bool)
	openOnboarding()
}

type settingsAppearanceActions interface {
	setTheme(theme string)
	setPaletteColor(color string)
	resetPaletteColor()
	saveWindowTransparency()
	setInterfaceScale(scale float64)
	setSidebarChoice(choice string)
	setReduceMotion(on bool)
}

type settingsPrivacyActions interface {
	historyAccessActions
	setSendWindow(on bool)
	setSendClipboardImage(on bool)
	setKeepHistory(on bool)
	setKeepAudio(on bool)
	showHistorySettings()
	clearHistory()
	setLogText(on bool)
}

type settingsMeetingActions interface {
	setMeetingCaptureMode(captureMode string)
	setMeetingHideFromSharing(hideFromSharing bool)
	setMeetingASR(speechModelID string)
	setMeetingLanguage(languageCode string)
	setMeetingLLM(languageModelID string)
}

type settingsStorageActions interface {
	openStorageFolder(directory string)
	chooseStorageDataFolder()
	chooseStorageModelFolder()
	showHistorySettings()
	removePrevious()
	dismissMoveFailure()
	cancelStorageDataMove()
	cancelStorageModelMove()
	resetLook()
	resetSettings()
	askErase()
	eraseAll()
	setStorageModelBackup(on bool)
}

type settingsOverlayActions interface {
	overlayEdited()
}

type shortcutPresetActions interface {
	applyShortcutPreset(preset string)
}

type settingsShortcutsActions interface {
	shortcutPresetActions
	openShortcutPermissions()
	openShortcutFeature(path string)
	setStrictShortcuts(on bool)
}

type keyRecorderActions interface {
	beginHotkeyRecord(row *keyRow, save func(keys string)) string
	saveShortcutRecorderKeys(keys string, row *keyRow, save func(keys string)) string
	typeShortcutKeys(row *keyRow) customKeysViewState
	saveTypedShortcutKeys(row *keyRow, save func(keys string)) customKeysViewState
	cancelHotkeyRecord()
	openShortcutPermissions()
	markShortcutRecorderFocused()
}

type shortcutPromptActions interface {
	answerShortcutPrompt()
	openShortcutPromptLink(services shortcutPromptServices, link string)
	changeShortcutPrompt(path string)
}

type settingsAudioActions interface {
	settingsSoundsActions
	loadDevices()
	restartHelper()
	addDevice(deviceID string)
	moveDevice(deviceID string, delta int)
	removeDevice(deviceID string)
	setSkipSilence(on bool)
	setCleanSound(on bool)
	setLiveResult(on bool)
	setUploadMB(megabytes float64)
	setMaxMinutes(minutes float64)
	setMuteOutput(on bool)
}

type settingsSoundsActions interface {
	chooseSoundSource(sourceID string)
	chooseSoundPair(start, stop string)
	previewSoundPair(start, stop string)
	stopPreview()
	chooseSystemSound(eventID, choice string)
	tryCueSound(eventID string)
	setSoundVolume(volume int)
}
