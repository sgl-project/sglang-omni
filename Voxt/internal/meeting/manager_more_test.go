package meeting_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store/sqlitestore"
)

func TestDiskTooLowAtStart(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	m := meeting.NewManager(meeting.Config{
		Mic:   newFakeAudio(nil),
		Store: s.Meetings(),
		Dir:   filepath.Join(dir, "meetings"),
		FreeSpace: func(string) (int64, error) {
			return meeting.DiskMinStartBytes() - 1, nil
		},
	})
	if _, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording}); err == nil {
		t.Fatal("expected disk error")
	}
}

func TestAskingDiscardDeletesRow(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	m := meeting.NewManager(meeting.Config{
		Mic:     newFakeAudio([]byte{1}),
		Store:   s.Meetings(),
		Dir:     filepath.Join(dir, "meetings"),
		Privacy: func() settings.Privacy { return settings.Privacy{KeepHistory: false} },
		FreeSpace: func(string) (int64, error) {
			return meeting.DiskMinStartBytes() + 1, nil
		},
	})
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Finish(ctx, id); err != nil {
		t.Fatal(err)
	}
	if m.Current().State != meeting.StateAsking {
		t.Fatalf("state %s", m.Current().State)
	}
	if err := m.Discard(ctx, id); err != nil {
		t.Fatal(err)
	}
	list, err := s.Meetings().ListMeetings(ctx, store.MeetingQuery{})
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 0 {
		t.Fatalf("rows left: %d", len(list))
	}
}

func TestSetModeAddsMicTrack(t *testing.T) {
	m, ctx, clk, _ := testManager(t, []byte{9}, []byte{8}, false)
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeSubtitles})
	if err != nil {
		t.Fatal(err)
	}
	clk.Advance(2 * time.Second)
	elapsed := m.Current().Elapsed
	if err := m.SetMode(ctx, id, meeting.ModeMeeting); err != nil {
		t.Fatal(err)
	}
	clk.Advance(time.Second)
	if m.Current().Elapsed < elapsed {
		t.Fatalf("timeline reset: %v before %v", m.Current().Elapsed, elapsed)
	}
	if err := m.Finish(ctx, id); err != nil {
		t.Fatal(err)
	}
}

func TestRecoverFixesWAV(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	audioDir := filepath.Join(dir, "meetings", "m1")
	if err := os.MkdirAll(audioDir, 0o755); err != nil {
		t.Fatal(err)
	}
	wav := filepath.Join(audioDir, "mic.wav")
	if err := os.WriteFile(wav, make([]byte, 44), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = s.Meetings().CreateMeeting(ctx, store.Meeting{
		ID: "m1", CreatedAt: time.Now(), Status: store.MeetingRecording,
		Mode: store.MeetingModeMeeting, Audio: "m1",
	})
	if err != nil {
		t.Fatal(err)
	}
	m := meeting.NewManager(meeting.Config{
		Store: s.Meetings(),
		Dir:   filepath.Join(dir, "meetings"),
	})
	got, err := m.Recover(ctx)
	if err != nil || len(got) != 1 {
		t.Fatalf("Recover: %v %d", err, len(got))
	}
	if got[0].Status != store.MeetingInterrupted {
		t.Fatalf("status %s", got[0].Status)
	}
	info, _ := os.Stat(wav)
	if info.Size() < 44 {
		t.Fatal("wav header not fixed")
	}
}

func TestFinishKeepHistoryNoAsking(t *testing.T) {
	m, ctx, _, _ := testManager(t, []byte{1}, nil, false)
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Finish(ctx, id); err != nil {
		t.Fatal(err)
	}
	if m.Current().State != meeting.StateIdle {
		t.Fatalf("state %s", m.Current().State)
	}
}

func TestSaveAfterAsking(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	mic := newFakeAudio([]byte{2})
	m := meeting.NewManager(meeting.Config{
		Mic:     mic,
		Store:   s.Meetings(),
		Dir:     filepath.Join(dir, "meetings"),
		Privacy: func() settings.Privacy { return settings.Privacy{KeepHistory: false} },
		FreeSpace: func(string) (int64, error) {
			return meeting.DiskMinStartBytes() + 1, nil
		},
	})
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Finish(ctx, id); err != nil {
		t.Fatal(err)
	}
	if err := m.Save(ctx, id); err != nil {
		t.Fatal(err)
	}
	mt, _, err := s.Meetings().Meeting(ctx, id)
	if err != nil || mt.Status != store.MeetingDone {
		t.Fatalf("saved row: %v %#v", err, mt)
	}
}

func TestResumeUsesNewCaptureJob(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	clk := newFakeClock(time.Unix(1_700_000_100, 0))
	mic := newFakeAudio([]byte{3})
	m := meeting.NewManager(meeting.Config{
		Mic:     mic,
		Store:   s.Meetings(),
		Dir:     filepath.Join(dir, "meetings"),
		Now:     clk.Now,
		Privacy: func() settings.Privacy { return settings.Privacy{KeepHistory: true} },
		FreeSpace: func(string) (int64, error) {
			return meeting.DiskMinStartBytes() + 1, nil
		},
	})
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Pause(ctx, id); err != nil {
		t.Fatal(err)
	}
	clk.Advance(2 * time.Second)
	if err := m.Resume(ctx, id); err != nil {
		t.Fatal(err)
	}
	jobs := mic.jobIDs()
	if len(jobs) < 2 {
		t.Fatalf("want two jobs, got %v", jobs)
	}
	if jobs[0] == jobs[1] {
		t.Fatalf("resume should start a new job: %v", jobs)
	}
	if err := m.Finish(ctx, id); err != nil {
		t.Fatal(err)
	}
}

func TestDiskLowEndsMeeting(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	s, err := sqlitestore.Open(ctx, filepath.Join(dir, "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	var free int64 = meeting.DiskMinStartBytes() + 1
	m := meeting.NewManager(meeting.Config{
		Mic:   newFakeAudio(make([]byte, 32)),
		Store: s.Meetings(),
		Dir:   filepath.Join(dir, "meetings"),
		FreeSpace: func(string) (int64, error) {
			return free, nil
		},
	})
	id, err := m.Start(ctx, meeting.StartRequest{Mode: meeting.ModeRecording})
	if err != nil {
		t.Fatal(err)
	}
	free = meeting.DiskMinRecordBytes() - 1
	deadline := time.Now().Add(6 * time.Second)
	for m.Current().State.Active() && time.Now().Before(deadline) {
		time.Sleep(100 * time.Millisecond)
	}
	if m.Current().State.Active() {
		t.Fatalf("still active after low disk")
	}
	mt, _, err := s.Meetings().Meeting(ctx, id)
	if err != nil || mt.Status != store.MeetingInterrupted {
		t.Fatalf("row status: %v %#v", err, mt)
	}
}
