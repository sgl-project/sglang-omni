package main

import (
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
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

// The overlay's End asks first, in a small bubble out of the overlay, not
// a card over it: a collapsed overlay stays a pill, its window grows by the
// bubble, and the blur stays the face's. Keep recording closes it; a meeting
// ended elsewhere (the menu bar) takes it away.
func TestMeetingEndAsksFirst(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.prefs.Features.Meeting.Collapsed = true
	a.meetingChanged(meeting.Snapshot{ID: "m1", State: meeting.StateRecording})
	pill := a.meetingOverlayModel()
	a.requestMeetingFinish()
	asked := a.meetingOverlayModel()
	if !a.meetOvUI.endAsk || !a.meetingOverlayCollapsed() || !asked.EndAsk {
		t.Fatalf("asked: endAsk %v, collapsed %v, bubble %v", a.meetOvUI.endAsk, a.meetingOverlayCollapsed(), asked.EndAsk)
	}
	if asked.Layout.Height != pill.Layout.Height+meetingOvEndAskH || asked.Layout.Width < meetingOvEndBubbleW {
		t.Fatalf("the window %+v for the bubble, from %+v", asked.Layout, pill.Layout)
	}
	if _, face := meetingOverlayFace(asked); face != pill.Layout.Height-2*layout.Pad {
		t.Fatalf("the face %d tall with the bubble, %d without", face, pill.Layout.Height-2*layout.Pad)
	}
	if a.meetSnap.State != meeting.StateRecording {
		t.Fatalf("ended before the answer: %s", a.meetSnap.State)
	}
	a.keepMeetingRecording()
	if a.meetOvUI.endAsk || a.meetingOverlayModel().EndAsk || a.meetSnap.State != meeting.StateRecording {
		t.Fatalf("Keep recording: endAsk %v, state %s", a.meetOvUI.endAsk, a.meetSnap.State)
	}
	a.requestMeetingFinish()
	a.meetingChanged(meeting.Snapshot{ID: "m1", State: meeting.StateFinishing, Stage: meeting.StageSaving})
	if a.meetOvUI.endAsk || !a.meetingOverlayCollapsed() {
		t.Fatalf("ended from the menu bar: endAsk %v, collapsed %v", a.meetOvUI.endAsk, a.meetingOverlayCollapsed())
	}
}

// The bubble's two buttons: Keep recording and End.
func TestMeetingEndBubbleControls(t *testing.T) {
	for label, want := range map[string]string{"Keep recording": "keepMeetingRecording", "End": "meetingFinish"} {
		host, err := loadMeetingDemoHost("testdata/meeting")
		if err != nil {
			t.Fatal(err)
		}
		selectMeetingDemoFixture(t, host, "en/state-collapsed")
		host.overlayUI.endAsk = true
		tester := newMeetingDemoTester(t, host)
		frames(tester)
		if !tester.HasText("End this meeting?") {
			t.Fatalf("no bubble: %q", tester.Texts())
		}
		if err := tester.ClickControl(label); err != nil {
			t.Fatalf("%s: %v (controls %q)", label, err, tester.AccessLabels())
		}
		frames(tester)
		if !slices.Contains(host.calls, want) {
			t.Fatalf("%s: calls %q, want %s", label, host.calls, want)
		}
	}
}
