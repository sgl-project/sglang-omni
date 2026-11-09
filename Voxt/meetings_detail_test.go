package main

import (
	"context"
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
		if err := tt.ClickControl(label); err != nil {
			t.Fatalf("%s: %v (%q)", label, err, firstTexts(tt))
		}
		frames(tt)
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
