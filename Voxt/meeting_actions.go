package main

type meetingExportFormat string

const (
	meetingExportText     meetingExportFormat = "txt"
	meetingExportMarkdown meetingExportFormat = "md"
)

type meetingsActions interface {
	startMeeting()
	meetingFinish()
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
	requestMeetingFinish()
	keepMeetingRecording()
}

// meetingOverlayAfterFrame runs the overlay's actions after the frame that
// handled the click. A click is handled while the view builds, and an
// action that resized the window then (expand, collapse, End's bubble, a
// menu) had that frame, drawn for the old size, stretched over the new one:
// the card flashed. After it, the resize draws the new size at once.
type meetingOverlayAfterFrame struct{ a *app }

func (o meetingOverlayAfterFrame) meetingPauseResume() { o.a.post(o.a.meetingPauseResume) }
func (o meetingOverlayAfterFrame) meetingFinish()      { o.a.post(o.a.meetingFinish) }
func (o meetingOverlayAfterFrame) meetingSave()        { o.a.post(o.a.meetingSave) }
func (o meetingOverlayAfterFrame) meetingDiscard()     { o.a.post(o.a.meetingDiscard) }
func (o meetingOverlayAfterFrame) meetingSetModeID(id string) {
	o.a.post(func() { o.a.meetingSetModeID(id) })
}
func (o meetingOverlayAfterFrame) toggleMeetingTranslation(id string) {
	o.a.post(func() { o.a.toggleMeetingTranslation(id) })
}
func (o meetingOverlayAfterFrame) copyLiveMeetingTranscript() {
	o.a.post(o.a.copyLiveMeetingTranscript)
}
func (o meetingOverlayAfterFrame) toggleMeetingMenu(menu meetingMenu) {
	o.a.post(func() { o.a.toggleMeetingMenu(menu) })
}
func (o meetingOverlayAfterFrame) setMeetingOverlayCollapsed(collapsed bool) {
	o.a.post(func() { o.a.setMeetingOverlayCollapsed(collapsed) })
}
func (o meetingOverlayAfterFrame) setMeetingOverlayCardSize(size int) {
	o.a.post(func() { o.a.setMeetingOverlayCardSize(size) })
}
func (o meetingOverlayAfterFrame) requestMeetingDiscard() { o.a.post(o.a.requestMeetingDiscard) }
func (o meetingOverlayAfterFrame) requestMeetingFinish()  { o.a.post(o.a.requestMeetingFinish) }
func (o meetingOverlayAfterFrame) keepMeetingRecording()  { o.a.post(o.a.keepMeetingRecording) }
