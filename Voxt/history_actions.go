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
