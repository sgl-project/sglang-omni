package main

type meetingExportFormat string

const (
	meetingExportText     meetingExportFormat = "txt"
	meetingExportMarkdown meetingExportFormat = "md"
)

type meetingsActions interface {
	startMeeting()
	meetingFinish()
	meetingImport()
	loadMeetings()
	searchMeetings(query string)
	showMeetingsPage(page int)
	openMeetingDetail(id string)
	closeMeetingDetail()
	deleteMeeting(id string)
	meetingReprocess(id string)
	meetingCorrect(id string)
	showMeetingInFinder(id string)
	regenerateMeetingSummary(id string)
	askMeetingFollowUp(id, question string)
	openMeetingLanguageModels()
	openMeetingSpeechModels()
	openMeetingPrivacySettings()
	copyMeetingTranscript(id string)
	saveMeetingExport(id string, format meetingExportFormat)
	toggleMeetingPlayback(id string)
	seekMeetingPlayback(id string, startMS int64)
}

type meetingOverlayActions interface {
	meetingPauseResume()
	meetingFinish()
	meetingSave()
	meetingDiscard()
	meetingSetModeID(id string)
	toggleMeetingTranslation(id string)
	copyLiveMeetingTranscript()
	toggleMeetingMenu(menu meetingMenu)
	setMeetingOverlayCollapsed(collapsed bool)
	setMeetingOverlayCardSize(size int)
	requestMeetingDiscard()
}
