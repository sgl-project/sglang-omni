package main

type onboardingActions interface {
	openOnboarding()
	closeOnboarding()
	finishOnboarding(path string)
	dismissOnboardingBack()
	obGo(step int)
	obLeave(path string)
	obTry(text string) *sessionNotice
	stopSession()
	cancelSession()
	copyOnboardingResult(text string)
	clearOnboardingBlock()
	onboardingNoticeAction(action string)
	onboardingTryLanguage()
	onboardingDownloadAndContinue()
	downloadModel(id string)
	obUse(ids ...string)
	askRequired()
	askPermission(which string, then func())
	recheckPermissions()
	onboardingOpenPrivacy(url string)
	importLegacy()
	completeOnboardingLegacyTodo(kind, id string)
	applyShortcutPreset(preset string)
}
