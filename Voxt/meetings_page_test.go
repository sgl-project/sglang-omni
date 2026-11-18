package main

import (
	"context"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/egoist/mygo"

	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// Meetings are in release builds (C1-10): the sidebar, ⌘K and the menu bar
// menu offer them, as in development builds.
func TestMeetingsInReleaseBuild(t *testing.T) {
	t.Setenv("MYGO_ENV", "production")
	a, _, tt := newTestApp(t)
	frames(tt)
	if !has(tt, "Meetings") {
		t.Fatalf("no Meetings in the sidebar: %q", firstTexts(tt))
	}
	a.paletteQuery = "meetings"
	if !slices.ContainsFunc(a.paletteItems(), func(cmd command) bool { return cmd.id == "go/meetings" }) {
		t.Fatal("⌘K does not list Go to Meetings")
	}
	if !slices.ContainsFunc(a.trayItems(), func(it *mygo.MenuItem) bool { return it.Label == "Start meeting" }) {
		t.Fatal("the menu bar menu has no Start meeting")
	}
	a.navigate("/meetings")
	frames(tt)
	if !has(tt, "No meetings yet.") {
		t.Fatalf("the Meetings page: %q", firstTexts(tt))
	}
	// 11-15: the empty state repeated the header's settings button.
	settings := 0
	for _, l := range tt.AccessLabels() {
		if l == "Meeting settings" {
			settings++
		}
	}
	if settings != 1 {
		t.Fatalf("%d Meeting settings buttons, want the header's one", settings)
	}
}

func TestMeetingsPageSearchPaginationEmpty(t *testing.T) {
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1400) // Linux's taller lines put the pager near 1200
	a.navigate("/meetings")
	frames(tt)
	if !has(tt, "No meetings yet.") {
		t.Fatalf("empty list: %q", firstTexts(tt))
	}

	repo := a.lib.store.Meetings()
	ctx := context.Background()
	t0 := time.Date(2026, 11, 8, 10, 0, 0, 0, time.UTC)
	for i := range 21 {
		id := fmt.Sprintf("m-%02d", i)
		rec, err := repo.CreateMeeting(ctx, store.Meeting{
			ID: id, CreatedAt: t0.Add(time.Duration(i) * time.Minute),
			Status: store.MeetingDone, Mode: store.MeetingModeMeeting,
			Title: fmt.Sprintf("sync %d", i),
		})
		if err != nil {
			t.Fatal(err)
		}
		if i == 3 {
			if err := repo.AppendSegments(ctx, rec.Seq, []store.MeetingSegment{
				{Source: store.SegmentMic, Text: "budget review", StartMS: 0, EndMS: 1000},
			}); err != nil {
				t.Fatal(err)
			}
		}
	}
	a.loadMeetings()
	frames(tt)
	if !hasExact(tt, "Page 1") || !has(tt, "sync 20") || hasExact(tt, "sync 0") {
		t.Fatalf("page 1: %q", firstTexts(tt))
	}
	if err := tt.ClickControl("Next page"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if !hasExact(tt, "Page 2") || !hasExact(tt, "sync 0") {
		t.Fatalf("page 2: %q", firstTexts(tt))
	}

	a.navigate("/meetings")
	a.mvUI.query = "budget"
	a.mvUI.page = 0
	a.loadMeetings()
	frames(tt)
	if !hasExact(tt, "sync 3") || hasExact(tt, "sync 20") {
		t.Fatalf("search: %q", firstTexts(tt))
	}
	a.mvUI.query = "missing-xyz"
	a.loadMeetings()
	frames(tt)
	if !has(tt, "Nothing matches") {
		t.Fatalf("no matches: %q", firstTexts(tt))
	}
}

// A meeting that starts while the Meetings page is open is listed at once,
// with its running time, instead of the empty state (#3990).
func TestMeetingsPageListsMeetingInProgress(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.navigate("/meetings")
	frames(tt)
	if !has(tt, "No meetings yet.") {
		t.Fatalf("empty list: %q", firstTexts(tt))
	}
	// The manager writes the row, then reports the recording state.
	rec, err := a.lib.store.Meetings().CreateMeeting(context.Background(), store.Meeting{
		ID: "live", CreatedAt: time.Now(), Status: store.MeetingRecording, Mode: store.MeetingModeMeeting,
	})
	if err != nil {
		t.Fatal(err)
	}
	a.meetingChanged(meeting.Snapshot{ID: "live", State: meeting.StateRecording, Elapsed: 3 * time.Minute})
	frames(tt)
	if has(tt, "No meetings yet.") || !hasExact(tt, "In progress") || !hasExact(tt, "3 min") {
		t.Fatalf("meeting in progress: %q", firstTexts(tt))
	}
	// Ending it updates the row: no "In progress", the stored length.
	if err := a.lib.store.Meetings().UpdateMeeting(context.Background(), store.Meeting{
		Seq: rec.Seq, ID: "live", CreatedAt: rec.CreatedAt, Status: store.MeetingDone, Mode: store.MeetingModeMeeting, DurationMS: (5 * time.Minute).Milliseconds(),
	}); err != nil {
		t.Fatal(err)
	}
	a.meetingChanged(meeting.Snapshot{ID: "live", State: meeting.StateDone, Elapsed: 5 * time.Minute})
	frames(tt)
	if hasExact(tt, "In progress") || !hasExact(tt, "5 min") {
		t.Fatalf("ended meeting: %q", firstTexts(tt))
	}
}

// The details of the meeting being recorded follow it (11-15: they kept the
// lines from when they were opened): new lines and translations show as
// they come, before their store writes land.
func TestMeetingDetailFollowsRecording(t *testing.T) {
	a, _, tt := newTestApp(t)
	ctx := context.Background()
	repo := a.lib.store.Meetings()
	rec, err := repo.CreateMeeting(ctx, store.Meeting{ID: "live", CreatedAt: time.Now(), Status: store.MeetingRecording, Mode: store.MeetingModeMeeting})
	if err != nil {
		t.Fatal(err)
	}
	if err := repo.AppendSegments(ctx, rec.Seq, []store.MeetingSegment{{Idx: 0, Source: store.SegmentSystem, Text: "first line", EndMS: 1000, Final: true}}); err != nil {
		t.Fatal(err)
	}
	first := meeting.Segment{ID: "s0", Source: meeting.SourceSystem, End: time.Second, Text: "first line", Final: true}
	snap := meeting.Snapshot{ID: "live", State: meeting.StateRecording, Segments: []meeting.Segment{first}}
	a.meetingChanged(snap)
	a.navigate("/meetings")
	a.openMeetingDetail("live")
	for i := 0; i < 50 && a.mv.detail.loading; i++ {
		frames(tt)
	}
	if !has(tt, "first line") {
		t.Fatalf("details: %q", firstTexts(tt))
	}
	first.Translation = "第一行"
	firstRow := &a.mv.detailView.Segments[0]
	snap.Segments = []meeting.Segment{first, {ID: "s1", StoreIdx: 1, Source: meeting.SourceMic, Start: 2 * time.Second, End: 3 * time.Second, Text: "second line", Final: true}}
	a.meetingChanged(snap)
	frames(tt)
	if &a.mv.detailView.Segments[0] != firstRow {
		t.Fatal("live update rebuilt the transcript")
	}
	if !has(tt, "second line") || !has(tt, "第一行") {
		t.Fatalf("the details did not follow the meeting: %q", firstTexts(tt))
	}
	// It ends: the final pass rewrote the lines and the manager lets the
	// meeting go; then the correction and summary land.
	if err := repo.AppendSegments(ctx, rec.Seq, []store.MeetingSegment{{Idx: 5, Source: store.SegmentMic, StartMS: 4000, EndMS: 5000, Text: "final pass line", Final: true}}); err != nil {
		t.Fatal(err)
	}
	a.meetingChanged(meeting.Snapshot{State: meeting.StateIdle})
	for i := 0; i < 50 && !has(tt, "final pass line"); i++ {
		frames(tt)
	}
	if !has(tt, "final pass line") {
		t.Fatalf("the details missed the final pass: %q", firstTexts(tt))
	}
	rec.Status, rec.Title, rec.Summary = store.MeetingDone, "Launch plan", `{"title":"Launch plan","body":"We agreed the plan.","todos":[]}`
	if err := repo.UpdateMeeting(ctx, rec); err != nil {
		t.Fatal(err)
	}
	a.meetingUpdated("live")
	for i := 0; i < 50 && !has(tt, "We agreed the plan."); i++ {
		frames(tt)
	}
	if !has(tt, "We agreed the plan.") || has(tt, "No summary yet.") {
		t.Fatalf("the details missed the summary: %q", firstTexts(tt))
	}
}
