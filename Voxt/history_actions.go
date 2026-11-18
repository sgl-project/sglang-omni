package main

type historyTextFormat string

const (
	historyTextMarkdown historyTextFormat = "md"
	historyTextJSONL    historyTextFormat = "jsonl"
)

type historyActions interface {
	loadHistory()
	searchHistory(query string)
	showHistoryTab(tab int)
	openHistoryDetail(id string)
	closeHistoryDetail()
	openHistoryFeature(feature string)
	openHistoryPrivacy()
	openHistorySettingsDialog()
	copyHistoryText(text string)
	showHistoryRecording(id string)
	retryHistoryEntry(id string)
	deleteHistoryEntry(id string)
	// transcriptResult shows a result with its Markdown, code and math
	// (#4105, transcript_preview_controller.go) and says whether it did;
	// the view draws the plain text when not.
	transcriptResult(c *uiContext, key, source string, transcript, streaming bool) bool
}

type historySettingsActions interface {
	chooseHistoryRecordingFolder()
	showHistoryRecordingFolder()
	useDefaultHistoryRecordingFolder()
	exportHistoryRecordings(shown bool)
	exportShownHistoryText(shown bool, format historyTextFormat)
	setHistoryCleanup(on bool)
	setHistoryCleanupDays(keepDays int)
	setHistoryCleanupCount(keepCount int)
	savePendingHistoryCleanup()
	deleteHistoryRecordings()
	cancelHistoryWork()
}

type historyAccessActions interface {
	setHistoryAccess(on bool)
	setHistorySelection(on bool)
	copyHistoryAccessCommand()
	openHistoryAccessDocs()
}
