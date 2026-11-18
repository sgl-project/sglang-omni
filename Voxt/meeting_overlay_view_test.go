package main

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

func overlayModel(snapshot meeting.Snapshot, collapsed bool, size int, menu meetingMenu) meetingOverlayModel {
	return meetingOverlayLayoutModel(buildMeetingOverlayViewState(i18n.NewTranslator(i18n.Locale()), snapshot, collapsed, size), menu)
}

func (a *app) meetingOverlayView(c *ui.Context) {
	model := a.meetingOverlayModel()
	mode, faceHeight := meetingOverlayFace(model)
	style := meetingOverlayFaceStyle{radius: faceRadius(a.prefs.Overlay, mode, faceHeight), alpha: layout.FaceAlpha(a.prefs.Overlay), palette: a.prefs.Appearance.Palette}
	animation := meetingOverlayAnimation{motion: a.overlayMotion(), phase: a.overlayPhase()}
	meetingOverlayView(c, a.meetOvView, model, a, &a.meetOvUI, animation, style)
}

func (a *app) meetingOverlayBenchmarkView(c *ui.Context) {
	model := a.meetOvModel
	mode, faceHeight := meetingOverlayFace(model)
	style := meetingOverlayFaceStyle{radius: faceRadius(a.prefs.Overlay, mode, faceHeight), alpha: layout.FaceAlpha(a.prefs.Overlay), palette: a.prefs.Appearance.Palette}
	animation := meetingOverlayAnimation{motion: a.overlayMotion(), phase: a.overlayPhase()}
	meetingOverlayView(c, a.meetOvView, model, a, &a.meetOvUI, animation, style)
}

func meetingOverlayRows(snapshot meeting.Snapshot) []meetingOverlayTranscriptRow {
	return meetingOverlayTranscriptRows(i18n.NewTranslator(i18n.Locale()), snapshot)
}

func meetingOverlayProblemText(key string) string {
	return meetingOverlayStateProblemText(i18n.NewTranslator(i18n.Locale()), key)
}

func formatMeetingElapsed(duration time.Duration) string {
	return formatMeetingOverlayElapsed(duration)
}

func TestMeetingOverlayModelSizes(t *testing.T) {
	s := meeting.Snapshot{State: meeting.StateRecording}
	collapsed := overlayModel(s, true, 0, meetingMenuNone)
	if collapsed.Layout.Width != meetingOvCollapsedW || collapsed.Layout.Height != meetingOvCollapsedH || !collapsed.Layout.Collapsed {
		t.Fatalf("collapsed %+v", collapsed.Layout)
	}
	if !collapsed.ShowBar || collapsed.ShowExpanded {
		t.Fatalf("collapsed flags bar=%v expanded=%v", collapsed.ShowBar, collapsed.ShowExpanded)
	}
	// The capsule is the dictation pill's height (#3986).
	if mode, h := meetingOverlayFace(collapsed); mode != ovRecording || h != layout.PillH {
		t.Fatalf("collapsed face %v %d, want the pill's %d", mode, h, layout.PillH)
	}
	expanded := overlayModel(s, false, 0, meetingMenuNone)
	if expanded.Layout.Width != meetingOvWidths[0]+2*layout.Pad || expanded.Layout.Height >= meetingOvMaxFaceH {
		t.Fatalf("expanded %+v: want %d wide, short before any text", expanded.Layout, meetingOvWidths[0])
	}
	if !expanded.ShowExpanded {
		t.Fatal("want expanded body")
	}
	ask := overlayModel(meeting.Snapshot{State: meeting.StateAsking}, false, 0, meetingMenuNone)
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
		if got := meetingOverlayProblemText(tc.key); got != tc.want {
			t.Errorf("key %q: got %q want %q", tc.key, got, tc.want)
		}
	}
	s := meeting.Snapshot{
		State:   meeting.StateRecording,
		Problem: &meeting.Problem{Key: meeting.ProblemKeyDiskLow},
	}
	if m := overlayModel(s, true, 0, meetingMenuNone); m.ProblemKey == "" {
		t.Fatal("problem key missing on model")
	}
}

// The expanded window grows with its lines up to its cap (the answer
// card's way), and takes the size the corner button steps to.
func TestMeetingOverlayGrowsWithText(t *testing.T) {
	s := meeting.Snapshot{State: meeting.StateRecording}
	h0 := overlayModel(s, false, 0, meetingMenuNone).Layout.Height
	s.Segments = []meeting.Segment{{ID: "1", Source: meeting.SourceMic, Speaker: "me", Text: "One line.", Final: true}}
	h1 := overlayModel(s, false, 0, meetingMenuNone).Layout.Height
	for i := range 40 {
		s.Segments = append(s.Segments, meeting.Segment{ID: fmt.Sprint(i + 2), Source: []meeting.Source{meeting.SourceMic, meeting.SourceSystem}[i%2], Text: "A sentence of the meeting, said in full."})
	}
	full := overlayModel(s, false, 0, meetingMenuNone).Layout
	if !(h0 < h1 && h1 < full.Height) || full.Height != meetingOvMaxFaceH+2*layout.Pad {
		t.Fatalf("heights %d, %d, %d: want growing to %d", h0, h1, full.Height, meetingOvMaxFaceH+2*layout.Pad)
	}
	if w := overlayModel(s, false, 2, meetingMenuNone).Layout.Width; w != meetingOvWidths[2]+2*layout.Pad {
		t.Fatalf("size 2: %d wide", w)
	}
	// The modes' menu fits before any text.
	if h := overlayModel(meeting.Snapshot{State: meeting.StateRecording}, false, 0, meetingMenuMode).Layout.Height; h < meetingOvChrome+menuHeight(meetingMenuMode, buildMeetingOverlayViewState(i18n.NewTranslator(i18n.Locale()), s, false, 0))+2*layout.Pad {
		t.Fatalf("menu height %d", h)
	}
}

// Rows: the speaker's name on the first of their lines only; a sentence
// the recogniser split is one row; the sentence still heard follows.
func TestMeetingOverlayRows(t *testing.T) {
	sec := func(n float64) time.Duration { return time.Duration(n * float64(time.Second)) }
	s := meeting.Snapshot{
		Segments: []meeting.Segment{
			{ID: "a", Source: meeting.SourceSystem, Start: sec(0), End: sec(1), Text: "Sounds"},
			{ID: "b", Source: meeting.SourceSystem, Start: sec(1.4), End: sec(4), Text: "good. I will test it."},
			{ID: "c", Source: meeting.SourceSystem, Start: sec(4.2), End: sec(6), Text: "Then ship."},
			{ID: "d", Source: meeting.SourceMic, Speaker: "me", Start: sec(7), End: sec(8), Text: "好的"},
			{ID: "e", Source: meeting.SourceMic, Speaker: "me", Start: sec(8.5), End: sec(9), Text: "我来跟进。"},
		},
		Live: map[meeting.Source]string{meeting.SourceMic: "还有"},
	}
	var got []string
	for _, r := range meetingOverlayRows(s) {
		who := ""
		if r.First {
			who = r.Who + ": "
		}
		got = append(got, who+r.Text)
	}
	want := []string{"Others: Sounds good. I will test it.", "Then ship.", "You: 好的我来跟进。", "还有"}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("rows %q\nwant %q", got, want)
	}
}

func TestFormatMeetingElapsed(t *testing.T) {
	for d, want := range map[time.Duration]string{7 * time.Second: "0:07", 754 * time.Second: "12:34", time.Hour + 62*time.Second: "1:01:02"} {
		if got := formatMeetingElapsed(d); got != want {
			t.Errorf("%v: %q, want %q", d, got, want)
		}
	}
}
