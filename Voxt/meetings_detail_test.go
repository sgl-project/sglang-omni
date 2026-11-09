package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestMeetingDetailExportsFiles(t *testing.T) {
	a, _, tt := newTestApp(t)
	meetingDevHook = func() bool { return true }
	t.Cleanup(func() { meetingDevHook = nil })
	repo := a.lib.store.Meetings()
	ctx := context.Background()
	rec, err := repo.CreateMeeting(ctx, store.Meeting{
		ID: "m-exp", CreatedAt: time.Date(2026, 11, 8, 10, 0, 0, 0, time.UTC),
		Status: store.MeetingDone, Mode: store.MeetingModeMeeting, Title: "Budget/Q3",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := repo.AppendSegments(ctx, rec.Seq, []store.MeetingSegment{
		{Source: store.SegmentMic, Text: "budget review", StartMS: 0, EndMS: 1000},
	}); err != nil {
		t.Fatal(err)
	}
	saved := chooseExportFile
	t.Cleanup(func() { chooseExportFile = saved })
	dir := t.TempDir()
	var names []string
	chooseExportFile = func(a *app, _, name string, then func(string)) {
		names = append(names, name)
		then(filepath.Join(dir, name))
	}
	tt.SetSize(windowWidth, 1400)
	a.navigate("/meetings")
	a.openMeetingDetail("m-exp")
	frames(tt)
	for _, label := range []string{"Export .txt…", "Export Markdown…"} {
		if err := tt.ClickControl("Export transcript"); err != nil {
			t.Fatalf("%v (%q)", err, firstTexts(tt))
		}
		frames(tt)
		choose(t, tt, label)
	}
	if strings.Join(names, ",") != "Budget-Q3.txt,Budget-Q3.md" {
		t.Fatalf("default names %q", names)
	}
	for _, name := range names {
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil || !strings.Contains(string(data), "budget review") {
			t.Errorf("%s: %q %v", name, data, err)
		}
	}
	if md, _ := os.ReadFile(filepath.Join(dir, "Budget-Q3.md")); !strings.Contains(string(md), "Budget/Q3") {
		t.Errorf("markdown lacks the title: %q", md)
	}
}

func TestMeetingRowOpensDetailAndDeleteConfirms(t *testing.T) {
	a, _, tt := newTestApp(t)
	repo := a.lib.store.Meetings()
	ctx := context.Background()
	if _, err := repo.CreateMeeting(ctx, store.Meeting{
		ID: "m-del", CreatedAt: time.Date(2026, 11, 8, 10, 0, 0, 0, time.UTC),
		Status: store.MeetingDone, Mode: store.MeetingModeMeeting, Title: "Weekly sync",
	}); err != nil {
		t.Fatal(err)
	}
	tt.SetSize(windowWidth, 900)
	a.navigate("/meetings")
	frames(tt)
	if err := tt.Click("Weekly sync"); err != nil {
		t.Fatalf("%v (%q)", err, firstTexts(tt))
	}
	frames(tt)
	if a.mv.detail == nil || a.mv.detail.id != "m-del" {
		t.Fatal("clicking the row did not open its detail")
	}
	for _, step := range []string{"Delete the meeting…", "Cancel"} {
		if err := tt.ClickControl(step); err != nil {
			t.Fatalf("%s: %v (%q)", step, err, firstTexts(tt))
		}
		frames(tt)
	}
	if _, _, err := repo.Meeting(ctx, "m-del"); err != nil || a.mv.detail == nil {
		t.Fatalf("cancel deleted the meeting or closed its detail: %v", err)
	}
	if err := tt.ClickControl("Delete the meeting…"); err != nil {
		t.Fatal(err)
	}
	frames(tt)
	if _, ok := tt.Find("Delete this meeting?"); !ok {
		t.Fatal("Delete did not ask first")
	}
	if err := tt.ClickControl("Delete meeting"); err != nil {
		t.Fatalf("%v (%q)", err, firstTexts(tt))
	}
	frames(tt)
	if _, _, err := repo.Meeting(ctx, "m-del"); err == nil {
		t.Fatal("confirming did not delete the meeting")
	}
}

// A long transcript builds only the rows in view, not every segment.
func TestMeetingLongTranscriptBuildsRowsInView(t *testing.T) {
	a, _, tt := newTestApp(t)
	repo := a.lib.store.Meetings()
	ctx := context.Background()
	rec, err := repo.CreateMeeting(ctx, store.Meeting{
		ID: "m-long", CreatedAt: time.Date(2026, 11, 8, 10, 0, 0, 0, time.UTC),
		Status: store.MeetingDone, Mode: store.MeetingModeMeeting, Title: "Long",
	})
	if err != nil {
		t.Fatal(err)
	}
	var segs []store.MeetingSegment
	for i := range 300 {
		segs = append(segs, store.MeetingSegment{Idx: i, Source: store.SegmentSystem, Text: fmt.Sprintf("line %d", i), StartMS: int64(i) * 3000, EndMS: int64(i)*3000 + 2500})
	}
	if err := repo.AppendSegments(ctx, rec.Seq, segs); err != nil {
		t.Fatal(err)
	}
	tt.SetSize(windowWidth, 900)
	a.navigate("/meetings")
	a.openMeetingDetail("m-long")
	frames(tt)
	if _, ok := tt.Find("line 0"); !ok {
		t.Fatalf("the first line is not shown (%q)", firstTexts(tt))
	}
	if _, ok := tt.Find("line 299"); ok {
		t.Fatal("the last of 300 lines was built with the first in view")
	}
}

func TestMeetingGroupAndDuration(t *testing.T) {
	now := time.Date(2026, 11, 8, 23, 0, 0, 0, time.Local)
	for at, want := range map[time.Time]string{
		time.Date(2026, 11, 8, 0, 0, 0, 0, time.Local):   "Today",
		time.Date(2026, 11, 7, 23, 59, 0, 0, time.Local): "Yesterday",
		time.Date(2026, 11, 1, 0, 0, 0, 0, time.Local):   "Previous 7 days",
		time.Date(2026, 10, 31, 23, 0, 0, 0, time.Local): "2026-10",
	} {
		if got := meetingGroup(at, now); got != want {
			t.Errorf("meetingGroup(%v) = %q, want %q", at, got, want)
		}
	}
	for ms, want := range map[int64]string{20_000: "Under a minute", 31 * 60_000: "31 min", 95 * 60_000: "1 h 35 min"} {
		if got := meetingDuration(ms); got != want {
			t.Errorf("meetingDuration(%d) = %q, want %q", ms, got, want)
		}
	}
	// An untitled meeting's header says "Untitled meeting", so the line under it keeps the date.
	if got, want := meetingMeta(store.Meeting{CreatedAt: now, DurationMS: 6 * 60_000}), "2026-11-08 23:00 · 6 min · Microphone and system audio"; got != want {
		t.Errorf("meetingMeta = %q, want %q", got, want)
	}
}

// Without a language model the summary says so once and links its page.
func TestMeetingDetailOpensLanguageModels(t *testing.T) {
	a, _, tt := newTestApp(t)
	if _, err := a.lib.store.Meetings().CreateMeeting(context.Background(), store.Meeting{
		ID: "m-llm", CreatedAt: time.Date(2026, 11, 8, 10, 0, 0, 0, time.UTC),
		Status: store.MeetingDone, Mode: store.MeetingModeMeeting,
	}); err != nil {
		t.Fatal(err)
	}
	tt.SetSize(windowWidth, 900)
	a.navigate("/meetings")
	a.openMeetingDetail("m-llm")
	frames(tt)
	if _, ok := tt.Find("Untitled meeting"); !ok {
		t.Fatalf("no title (%q)", firstTexts(tt))
	}
	if _, ok := tt.Find("Ask about this meeting"); ok {
		t.Fatal("Ask shows without a language model")
	}
	if err := tt.ClickControl("Open Language models"); err != nil {
		t.Fatalf("%v (%q)", err, firstTexts(tt))
	}
	frames(tt)
	if a.shownPath() != remotePath {
		t.Fatalf("went to %s", a.shownPath())
	}
}
