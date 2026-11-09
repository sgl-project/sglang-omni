package main

import "testing"

func TestMeetingOverlayContentProtection(t *testing.T) {
	a, _, _ := newTestApp(t)
	var got []bool
	t.Cleanup(func() { meetingContentProtectionHook = nil })
	meetingContentProtectionHook = func(on bool) { got = append(got, on) }

	a.prefs.Features.Meeting.HideFromSharing = false
	a.applyMeetingOverlayProtection()
	a.prefs.Features.Meeting.HideFromSharing = true
	a.applyMeetingOverlayProtection()
	if len(got) != 2 || got[0] || !got[1] {
		t.Fatalf("content protection calls %v", got)
	}
}
