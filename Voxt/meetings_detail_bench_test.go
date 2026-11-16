package main

import (
	"context"
	"fmt"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func BenchmarkW0MeetingDetailUpdates(b *testing.B) {
	a, _, _ := newRenderBenchmarkApp(b)
	repo := a.lib.store.Meetings()
	stored, err := repo.CreateMeeting(context.Background(), store.Meeting{ID: "detail-hour", Status: store.MeetingRecording, Mode: store.MeetingModeMeeting})
	if err != nil {
		b.Fatal(err)
	}
	if err := repo.AppendSegments(context.Background(), stored.Seq, []store.MeetingSegment{{Idx: 0, Source: store.SegmentSystem, Text: "first", Final: true}}); err != nil {
		b.Fatal(err)
	}
	a.navigate("/meetings")
	a.openMeetingDetail("detail-hour")
	tester := ui.NewTester(a.view, windowWidth, windowHeight)
	frames(tester)
	snapshot := meeting.Snapshot{ID: "detail-hour", State: meeting.StateRecording}
	segments := make([]meeting.Segment, meetingUpdatesPerHour)
	for index := range segments {
		segments[index] = meeting.Segment{ID: fmt.Sprintf("segment-%d", index), StoreIdx: index + 1, Source: meeting.SourceSystem, Text: fmt.Sprintf("update %d", index), Final: true}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for _, segment := range segments {
		snapshot.Segments = append(snapshot.Segments, segment)
		if len(snapshot.Segments) > meetingSnapshotSegmentCap {
			snapshot.Segments = snapshot.Segments[len(snapshot.Segments)-meetingSnapshotSegmentCap:]
		}
		a.meetingChanged(snapshot)
		tester.Frame()
	}
	b.StopTimer()
	if len(a.mv.detailView.Segments) != len(segments)+1 {
		b.Fatalf("detail has %d rows, want %d", len(a.mv.detailView.Segments), len(segments)+1)
	}
	if a.mv.detailView.Segments[0].Text != "first" || a.mv.detailView.Segments[len(a.mv.detailView.Segments)-1].Text != "update 899" {
		b.Fatal("detail update lost the first or last row")
	}
}
