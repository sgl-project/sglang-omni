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
	setInterfaceScale(scale float64)
	setSidebarChoice(choice string)
	setReduceMotion(on bool)
}
