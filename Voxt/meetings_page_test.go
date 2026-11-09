package main

import "testing"

func TestMeetingsNavHiddenWhenDisabled(t *testing.T) {
	_, _, tt := newTestApp(t)
	meetingDevHook = func() bool { return false }
	t.Cleanup(func() { meetingDevHook = nil })
	frames(tt)
	if has(tt, "Meetings") {
		t.Fatalf("Meetings in sidebar when disabled: %q", firstTexts(tt))
	}
}
