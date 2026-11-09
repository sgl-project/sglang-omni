package main

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestMeetingsNavHiddenWhenDisabled(t *testing.T) {
	_, _, tt := newTestApp(t)
	meetingDevHook = func() bool { return false }
	t.Cleanup(func() { meetingDevHook = nil })
	frames(tt)
	if has(tt, "Meetings") {
		t.Fatalf("Meetings in sidebar when disabled: %q", firstTexts(tt))
	}
}

func TestMeetingsPageSearchPaginationEmpty(t *testing.T) {
	a, _, tt := newTestApp(t)
	meetingDevHook = func() bool { return true }
	t.Cleanup(func() { meetingDevHook = nil })
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
