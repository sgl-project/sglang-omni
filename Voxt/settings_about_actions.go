package main

type settingsAboutActions interface {
	runSelfCheck()
	saveBundle()
	copyDiagnostics(context *aboutUIContext)
	reportBug(context *aboutUIContext)
}
