package main

import (
	"context"
	"fmt"
	"image/png"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/egoist/mygo/plugins/glass"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestSnapshotsMeetings saves the Meetings page (empty, the list, a
// meeting's details with and without a language model, an interrupted
// one, the settings and the delete dialog) at the default and the
// smallest window, Chinese dark and English light.
func TestSnapshotsMeetings(t *testing.T) { runSnapshots(t, "VOXTNEW_SNAPSHOTS", snapshotsMeetings) }

func snapshotsMeetings(t *testing.T, out string) {
	meetingDevHook = func() bool { return true }
	t.Cleanup(func() { meetingDevHook = nil })
	for _, size := range [][2]int{{windowWidth, windowHeight}, {windowMinWidth, windowMinHeight}} {
		for _, run := range []struct {
			lang string
			dark bool
		}{{"zh-Hans", true}, {"en", false}} {
			a, _, tt := newTestApp(t)
			tt.SetSize(size[0], size[1])
			if run.lang != "en" {
				a.updateSettings(func(s *settings.Settings) { s.General.Language = run.lang })
			}
			tt.SetDark(run.dark)
			tt.SetScale(2)
			a.version = "0.1.0"
			a.clearNotices()
			suffix := fmt.Sprintf("%dx%d-light", size[0], size[1])
			if run.dark {
				suffix = fmt.Sprintf("%dx%d-dark", size[0], size[1])
			}
			if run.lang != "en" {
				suffix += "-" + run.lang
			}
			save := func(name string) {
				t.Helper()
				a.toasts = nil
				frames(tt)
				f, err := os.Create(filepath.Join(out, name+"-"+suffix+".png"))
				if err != nil {
					t.Fatal(err)
				}
				defer f.Close()
				if err := png.Encode(f, tt.Image()); err != nil {
					t.Fatal(err)
				}
			}
			a.navigate("/meetings")
			save("meetings-empty")
			seedSnapshotMeetings(t, a)
			a.loadMeetings()
			save("meetings-list")
			a.openMeetingDetail("m-0")
			save("meetings-detail")
			a.openMeetingDetail("m-4")
			save("meetings-detail-interrupted")
			a.local = readyLocal()
			a.openMeetingDetail("m-0")
			save("meetings-detail-llm")
			a.mv.delID = "m-0"
			save("meetings-delete")
			a.mv.delID, a.mv.detail = "", nil
			a.mv.setOpen = true
			save("meetings-settings")
		}
	}
}

// TestOverlayShotsMeeting renders the meeting overlay's states alone,
// English over a light desktop and Chinese over a dark one.
func TestOverlayShotsMeeting(t *testing.T) {
	runSnapshots(t, "VOXTNEW_OVERLAY_SHOTS", snapshotsMeetingOverlay)
}

func snapshotsMeetingOverlay(t *testing.T, out string) {
	now := time.Now()
	segs := []meeting.Segment{
		{Source: meeting.SourceSystem, Speaker: "s1", Text: "Mail and Voice Memos group by day and show two lines per row.", Final: true},
		{Source: meeting.SourceMic, Speaker: "me", Text: "Yes. I'll ship the list first and the detail panel after.", Final: true},
	}
	level := map[meeting.Source]float64{meeting.SourceMic: -18, meeting.SourceSystem: -30}
	live := map[meeting.Source]string{meeting.SourceSystem: "Sounds good, let's review again"}
	rec := meeting.Snapshot{ID: "ov", Mode: meeting.ModeMeeting, State: meeting.StateRecording, Started: now.Add(-754 * time.Second), Elapsed: 754 * time.Second,
		Level: level, Live: live, Segments: segs}
	for _, run := range []struct {
		lang string
		dark bool
	}{{"en", false}, {"zh-Hans", true}} {
		a, _, _ := newTestApp(t)
		if run.lang != "en" {
			a.updateSettings(func(s *settings.Settings) { s.General.Language = run.lang })
		}
		suffix := "light"
		desk, line := ui.RGB(246, 246, 248), ui.RGB(150, 156, 166)
		if run.dark {
			suffix = "dark-" + run.lang
			desk, line = ui.RGB(30, 31, 36), ui.RGB(120, 128, 140)
		}
		paused, problem, asking, finishing := rec, rec, rec, rec
		paused.State = meeting.StatePaused
		problem.Problem = &meeting.Problem{Key: meeting.ProblemKeyDiskLow}
		asking.State = meeting.StateAsking
		finishing.State, finishing.Stage, finishing.Progress = meeting.StateFinishing, meeting.StageTranscribing, 0.4
		for _, st := range []struct {
			name      string
			s         meeting.Snapshot
			collapsed bool
		}{
			{"meeting-overlay-starting", meeting.Snapshot{ID: "ov", Mode: meeting.ModeMeeting, State: meeting.StateStarting, Started: now,
				Level: map[meeting.Source]float64{meeting.SourceMic: -120, meeting.SourceSystem: -120}}, false},
			{"meeting-overlay-recording", rec, false},
			{"meeting-overlay-paused", paused, false},
			{"meeting-overlay-problem", problem, false},
			{"meeting-overlay-pill", rec, true},
			{"meeting-overlay-pill-problem", problem, true},
			{"meeting-overlay-asking", asking, false},
			{"meeting-overlay-finishing", finishing, false},
		} {
			a.meetSnap, a.meetOvCollapsed = st.s, st.collapsed
			model := overlayModel(st.s, st.collapsed)
			mode, faceH := meetingOverlayFace(model)
			radius := faceRadius(a.prefs.Overlay, mode, faceH)
			w, h := model.Layout.Width, model.Layout.Height
			tt := hudTester(func(c *ui.Context) {
				ui.Column(c).Absolute().Fill().Background(desk).Padding(10, 14).Gap(8).Children(func() {
					for j := range 12 {
						ui.Box(c.Key(itoa(j))).Height(6).Width(float32(w) * (0.45 + 0.4*float32((j*37)%10)/10)).Radius(3).Background(line)
					}
				})
				ui.Box(c).Absolute().Left(layout.Pad).Top(layout.Pad).Right(layout.Pad).Bottom(layout.Pad).Radius(radius).Material(glass.Blur{Radius: 8})
				ui.Box(c).Absolute().Fill().Children(func() { a.meetingOverlayView(c) })
			}, w, h)
			tt.SetScale(2)
			frames(tt)
			f, err := os.Create(filepath.Join(out, st.name+"-"+suffix+".png"))
			if err != nil {
				t.Fatal(err)
			}
			if err := png.Encode(f, tt.Image()); err != nil {
				t.Fatal(err)
			}
			f.Close()
		}
	}
}

// seedSnapshotMeetings stores seven meetings over the last month: done,
// in progress and interrupted ones, with and without titles and
// summaries; m-0 has a summary with action items and a transcript of
// three speakers.
func seedSnapshotMeetings(t *testing.T, a *app) {
	t.Helper()
	repo := a.lib.store.Meetings()
	ctx := context.Background()
	now := time.Now()
	day := func(d, h int) time.Time {
		y, m, dd := now.Date()
		return time.Date(y, m, dd-d, h, 0, 0, 0, now.Location())
	}
	for i, m := range []struct {
		at    time.Time
		title string
		min   int64
		sum   string
		st    store.MeetingStatus
	}{
		{day(0, 9), "Design review: meetings page", 48, "", store.MeetingDone},
		{day(0, 11), "", 0, "", store.MeetingRecording},
		{day(1, 15), "Budget review Q3", 72, "Travel budget cut by 10%; hiring plan unchanged until January.", store.MeetingDone},
		{day(1, 10), "Weekly sync", 31, "", store.MeetingDone},
		{day(4, 14), "", 6, "", store.MeetingInterrupted},
		{day(5, 16), "1:1 with Lin", 25, "Lin takes over the export work; next check-in in two weeks.", store.MeetingDone},
		{day(30, 9), "Kickoff: Voxt 2.x", 95, "Scope set for Group 1: meetings, overlay, exports.", store.MeetingDone},
	} {
		var sum string
		if m.sum != "" {
			sum = `{"title":"","body":` + strconv.Quote(m.sum) + `,"todos":[]}`
		}
		segs := []store.MeetingSegment{{Source: store.SegmentMic, Speaker: "me", Text: "We should ship the export first.", EndMS: 6000, Final: true}}
		if i == 0 {
			sum = `{"title":"Ship the new meetings list this week","body":"Rebuild the list on the History layout, then move the summary above the transcript.","todos":["Ship the list redesign","Draft the detail panel layout","Review again on Friday"]}`
			segs = []store.MeetingSegment{
				{Idx: 0, Source: store.SegmentMic, Speaker: "me", Text: "Let's start with the meetings list. It looks too small next to the search bar.", EndMS: 9000, Final: true},
				{Idx: 1, Source: store.SegmentSystem, Speaker: "s1", Text: "Agreed. Mail and Voice Memos group by day and show two lines per row.", StartMS: 9000, EndMS: 21000, Final: true},
				{Idx: 2, Source: store.SegmentSystem, Speaker: "s2", Text: "Can we also put the summary on top of the transcript in the detail panel?", StartMS: 21000, EndMS: 33000, Final: true},
				{Idx: 3, Source: store.SegmentMic, Speaker: "me", Text: "Yes. I'll ship the list first and the detail panel after.", StartMS: 33000, EndMS: 41000, Final: true},
				{Idx: 4, Source: store.SegmentSystem, Speaker: "s1", Text: "Sounds good, let's review again on Friday.", StartMS: 1810000, EndMS: 1815000, Final: true},
			}
		}
		rec, err := repo.CreateMeeting(ctx, store.Meeting{
			ID: fmt.Sprintf("m-%d", i), CreatedAt: m.at, Status: m.st, Mode: store.MeetingModeMeeting,
			Title: m.title, DurationMS: m.min * 60000, Summary: sum,
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := repo.AppendSegments(ctx, rec.Seq, segs); err != nil {
			t.Fatal(err)
		}
	}
}
