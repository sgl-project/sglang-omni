package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
)

func TestMeetingOverlayModelSizes(t *testing.T) {
	s := meeting.Snapshot{State: meeting.StateRecording}
	collapsed := overlayModel(s, true)
	if collapsed.Layout.Width != meetingOvCollapsedW || collapsed.Layout.Height != meetingOvCollapsedH || !collapsed.Layout.Collapsed {
		t.Fatalf("collapsed %+v", collapsed.Layout)
	}
	if !collapsed.ShowBar || collapsed.ShowExpanded {
		t.Fatalf("collapsed flags bar=%v expanded=%v", collapsed.ShowBar, collapsed.ShowExpanded)
	}
	expanded := overlayModel(s, false)
	if expanded.Layout.Width != meetingOvExpandedW || expanded.Layout.Height != meetingOvExpandedH {
		t.Fatalf("expanded %+v", expanded.Layout)
	}
	if !expanded.ShowExpanded {
		t.Fatal("want expanded body")
	}
	ask := overlayModel(meeting.Snapshot{State: meeting.StateAsking}, false)
	if !ask.ShowAsking || ask.ShowBar {
		t.Fatalf("asking %+v", ask)
	}
}

func TestMeetingProblemMessageKeys(t *testing.T) {
	for _, tc := range []struct {
		key  string
		want string
	}{
		{meeting.ProblemKeySystemSilent, "If you see no text when others speak, allow Voxt in System Settings → Privacy & Security → Screen & System Audio Recording."},
		{meeting.ProblemKeyLiveBacklog, "Live captions are falling behind. The final transcript will catch up after you end the meeting."},
		{meeting.ProblemKeyDiskLow, "Disk space is running low. End the meeting soon or free space on this Mac."},
		{meeting.ProblemKeyHelperOld, "Recording system audio needs a newer Voxt helper. Update Voxt, or record the microphone only."},
		{"meeting.track.degraded.system", "One audio track stopped. Voxt switched to a simpler capture mode."},
	} {
		if got := meetingOverlayProblemText(tc.key); got != platformText(tc.want) {
			t.Errorf("key %q: got %q want %q", tc.key, got, tc.want)
		}
	}
	s := meeting.Snapshot{
		State:   meeting.StateRecording,
		Problem: &meeting.Problem{Key: meeting.ProblemKeyDiskLow},
	}
	if m := overlayModel(s, true); m.ProblemKey == "" {
		t.Fatal("problem key missing on model")
	}
}
