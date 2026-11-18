package main

type settingsAboutActions interface {
	runSelfCheck()
	openWhatsNew()
	saveBundle()
	copyDiagnostics(services aboutServices)
	reportBug(services aboutServices)
}
