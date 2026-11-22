package main

const (
	homeFixSpeechDownload = "download-speech"
	homeFixSpeechModels   = "speech-models"
	homeFixLanguageModels = "language-models"
	homeFixMicrophone     = "microphone"
	homeFixPermissions    = "permissions"
	homeFixTranscribe     = "transcribe"
	homeFixRestartHelper  = "restart-helper"
	homeFixRelaunch       = "relaunch"
)

type homeActions interface {
	historyActions
	openHomeGuide()
	openHomeHistory()
	openHomeRecent(id string)
	copyHomeRecent(id string)
	addHomeShortcut()
	fixHomeReadiness(fix string)
	activityBack()
	activityForward()
}
