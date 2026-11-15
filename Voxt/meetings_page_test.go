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
	a.mv.query = "budget"
	a.mv.page = 0
	a.loadMeetings()
	frames(tt)
	if !hasExact(tt, "sync 3") || hasExact(tt, "sync 20") {
		t.Fatalf("search: %q", firstTexts(tt))
	}
	a.mv.query = "missing-xyz"
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
