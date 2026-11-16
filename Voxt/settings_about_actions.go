package main

type settingsAboutActions interface {
	runSelfCheck()
	saveBundle()
	copyDiagnostics(services aboutServices)
	reportBug(services aboutServices)
}
