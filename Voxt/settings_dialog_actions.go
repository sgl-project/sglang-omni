package main

type settingsDialogActions interface {
	settingsGeneralActions
	featureActions
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
