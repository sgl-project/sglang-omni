package main

type settingsDialogActions interface {
	settingsGeneralActions
	permissionsActions
	settingsAudioActions
	settingsOverlayActions
	settingsShortcutsActions
	settingsAppearanceActions
	settingsPrivacyActions
	settingsStorageActions
	settingsAboutActions
	updatesActions
	navigate(path string)
}
