package main

// The demo's side of End's question (meeting_overlay.go: meetingEndBubble):
// End opens the bubble, Keep recording closes it, and the overlay's layout
// makes room for it as the app's rebuildMeetingOverlay does.

func (host *meetingDemoHost) requestMeetingFinish() {
	host.overlayUI.menu = meetingMenuNone
	host.overlayUI.endAsk = true
	host.record("requestMeetingFinish")
}

func (host *meetingDemoHost) keepMeetingRecording() {
	host.overlayUI.endAsk = false
	host.record("keepMeetingRecording")
}

// overlayMenu is the overlay's menu for its layout, End's bubble included
// (as the app's rebuildMeetingOverlay).
func (host *meetingDemoHost) overlayMenu() meetingMenu {
	if host.overlayUI.endAsk {
		return meetingMenuEnd
	}
	return host.overlayUI.menu
}
