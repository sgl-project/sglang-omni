package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
)

// A meeting filed after its final pass says so where the overlay was,
// and its button opens the meeting (11-25: the overlay went and nothing
// said where the meeting was). A pass that failed says nothing here.
func TestMeetingSavedNotice(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.meetingChanged(meeting.Snapshot{ID: "m1", State: meeting.StateRecording})
	a.meetingChanged(meeting.Snapshot{ID: "m1", State: meeting.StateFinishing, Stage: meeting.StageTranscribing, Progress: 0.4})
	if a.notice != nil {
		t.Fatalf("while it is filed: notice %+v", a.notice)
	}
	a.meetingChanged(meeting.Snapshot{State: meeting.StateIdle})
	n := a.notice
	if n == nil || n.text != "Meeting saved" || n.action != actionMeeting || n.entry != "m1" || !n.ok {
		t.Fatalf("filed: notice %+v", n)
	}
	if label := noticeActionLabelFor(i18n.NewTranslator("en"), actionMeeting); label != "View meeting" {
		t.Fatalf("button %q", label)
	}
	a.noticeAction(actionMeeting)
	if a.router.Path() != "/meetings" || a.mv.detail == nil || a.mv.detail.id != "m1" {
		t.Fatalf("View meeting: at %q, detail %+v", a.router.Path(), a.mv.detail)
	}

	b, _, _ := newTestApp(t)
	b.meetingChanged(meeting.Snapshot{ID: "m1", State: meeting.StateFinishing, Stage: meeting.StageTranscribing})
	b.meetingChanged(meeting.Snapshot{ID: "m1", State: meeting.StateInterrupted})
	if b.notice != nil {
		t.Fatalf("interrupted: notice %+v", b.notice)
	}
}
