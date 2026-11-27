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
	openHomeGuide()
	addHomeShortcut()
	fixHomeReadiness(fix string)
	activityBack()
	activityForward()
	// openShare opens the share card (share_card.go).
	openShare()
	// setInsightsClosed folds Looking back to its title row, or opens it.
	setInsightsClosed(closed bool)
	insightsActions
}
