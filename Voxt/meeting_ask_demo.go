package main

// The demo's side of the card that offers to record a detected meeting:
// the fixture seeds its compact and menu states, and the controls change
// them as the app's controller would.

type meetingAskDemoUI struct {
	compact, menu bool
}

func (host *meetingDemoHost) startDetectedMeeting() { host.record("startDetectedMeeting") }
func (host *meetingDemoHost) dismissMeetingAsk()    { host.record("dismissMeetingAsk") }

func (host *meetingDemoHost) toggleMeetingAskMenu() {
	host.askUI.menu = !host.askUI.menu
	host.record("toggleMeetingAskMenu")
}

func (host *meetingDemoHost) chooseMeetingAskMenu(id string) {
	host.askUI.menu = false
	host.record("chooseMeetingAskMenu %s", id)
}

// meetingAskHovered opens a compact card; it is no control, so it is not
// recorded.
func (host *meetingDemoHost) meetingAskHovered() { host.askUI.compact = false }
