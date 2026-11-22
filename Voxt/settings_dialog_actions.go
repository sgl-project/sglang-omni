package main

type settingsDialogActions interface {
	settingsGeneralActions
	modelsActions
	remoteActions
	onboardingActions
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
