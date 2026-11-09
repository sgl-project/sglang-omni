package sqlitestore

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo/plugins/sqlite"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func meetingID(n string) store.Meeting {
	t0 := time.Date(2026, 11, 8, 10, 0, 0, 0, time.UTC)
	return store.Meeting{
		ID: n, CreatedAt: t0, Status: store.MeetingRecording, Mode: store.MeetingModeMeeting,
		Title: "standup", Audio: "m1",
	}
}

func seg(src store.SegmentSource, text string) store.MeetingSegment {
	return store.MeetingSegment{Idx: -1, Source: src, Text: text, StartMS: 0, EndMS: 1200}
}

func TestMeetingsCRUDAndFTS(t *testing.T) {
	s := openTest(t, t.TempDir())
	m := s.Meetings()
	ctx := context.Background()
	one, err := m.CreateMeeting(ctx, meetingID("m-alpha"))
	if err != nil {
		t.Fatal(err)
	}
	if one.Seq <= 0 {
		t.Fatalf("seq %d", one.Seq)
	}
	got, segs, err := m.Meeting(ctx, "m-alpha")
	if err != nil || got.ID != "m-alpha" || len(segs) != 0 {
		t.Fatalf("%+v %d %v", got, len(segs), err)
	}
	if err := m.AppendSegments(ctx, one.Seq, []store.MeetingSegment{
		seg(store.SegmentMic, "大家好"), seg(store.SegmentSystem, "Hello team"),
	}); err != nil {
		t.Fatal(err)
	}
	_, segs, err = m.Meeting(ctx, "m-alpha")
	if err != nil || len(segs) != 2 || segs[0].Idx != 0 || segs[1].Text != "Hello team" {
		t.Fatalf("%v", segs)
	}
	two, _ := m.CreateMeeting(ctx, meetingID("m-beta"))
	m.AppendSegments(ctx, two.Seq, []store.MeetingSegment{seg(store.SegmentMic, "only mic here")})

	ids := func(q store.MeetingQuery) string {
		list, err := m.ListMeetings(ctx, q)
		if err != nil {
			t.Fatal(err)
		}
		var out []string
		for _, x := range list {
			out = append(out, x.ID)
		}
		return strings.Join(out, ",")
	}
	for q, want := range map[store.MeetingQuery]string{
		{}:                       "m-beta,m-alpha",
		{Search: "大家好"}:          "m-alpha",
		{Search: "大家"}:           "m-alpha", // LIKE under 3 runes
		{Search: "Hello"}:        "m-alpha",
		{Search: "nothing-here"}: "",
	} {
		if got := ids(q); got != want {
			t.Errorf("%+v: %q want %q", q, got, want)
		}
	}

	one.Status = store.MeetingDone
	one.Title = "done"
	if err := m.UpdateMeeting(ctx, one); err != nil {
		t.Fatal(err)
	}
	if err := m.ReplaceSegments(ctx, one.Seq, store.SegmentMic, []store.MeetingSegment{
		seg(store.SegmentMic, "replacement mic"),
	}); err != nil {
		t.Fatal(err)
	}
	_, segs, _ = m.Meeting(ctx, "m-alpha")
	if len(segs) != 2 || segs[0].Text != "replacement mic" || segs[1].Source != store.SegmentSystem {
		t.Fatalf("replace mic only: %+v", segs)
	}
	if err := m.DeleteMeeting(ctx, "m-alpha"); err != nil {
		t.Fatal(err)
	}
	if _, _, err := m.Meeting(ctx, "m-alpha"); !errors.Is(err, core.ErrNotFound) {
		t.Fatalf("deleted: %v", err)
	}
	rows, _ := s.db.Query(ctx, `SELECT COUNT(*) FROM meeting_segments WHERE meeting_seq = ?`, one.Seq)
	if asInt(rows.Values[0][0]) != 0 {
		t.Error("segments not cascaded")
	}
	if ids(store.MeetingQuery{Search: "replacement"}) != "" {
		t.Error("fts index kept deleted meeting")
	}
}

func TestMeetingsReplaceTransactional(t *testing.T) {
	s := openTest(t, t.TempDir())
	m := s.Meetings()
	ctx := context.Background()
	mt, _ := m.CreateMeeting(ctx, meetingID("m-tx"))
	m.AppendSegments(ctx, mt.Seq, []store.MeetingSegment{seg(store.SegmentMic, "before")})
	bad := []store.MeetingSegment{{Source: "bogus", Text: "x"}}
	if err := m.ReplaceSegments(ctx, mt.Seq, store.SegmentMic, bad); !errors.Is(err, core.ErrInvalid) {
		t.Fatalf("bad source: %v", err)
	}
	_, segs, _ := m.Meeting(ctx, "m-tx")
	if len(segs) != 1 || segs[0].Text != "before" {
		t.Fatalf("partial replace left %v", segs)
	}
}

func TestMeetingsUnfinishedAndExpiredAudio(t *testing.T) {
	s := openTest(t, t.TempDir())
	m := s.Meetings()
	ctx := context.Background()
	t0 := time.Now()
	rec, _ := m.CreateMeeting(ctx, store.Meeting{ID: "r1", CreatedAt: t0, Status: store.MeetingRecording, Mode: store.MeetingModeMeeting})
	ask, _ := m.CreateMeeting(ctx, store.Meeting{ID: "a1", CreatedAt: t0, Status: store.MeetingAsking, Mode: store.MeetingModeMeeting})
	done, _ := m.CreateMeeting(ctx, store.Meeting{ID: "d1", CreatedAt: t0, Status: store.MeetingDone, Mode: store.MeetingModeMeeting})
	exp := done
	exp.Audio, exp.AudioUntil = "folder", t0.Add(-time.Hour)
	m.UpdateMeeting(ctx, exp)

	open, _ := m.Unfinished(ctx)
	if len(open) != 2 {
		t.Fatalf("unfinished %d", len(open))
	}
	expired, _ := m.ExpiredAudio(ctx, t0)
	if len(expired) != 1 || expired[0].ID != "d1" {
		t.Fatalf("expired %+v", expired)
	}
	_ = rec
	_ = ask
}

func TestMeetingsConstraints(t *testing.T) {
	s := openTest(t, t.TempDir())
	m := s.Meetings()
	ctx := context.Background()
	base := meetingID("ok")
	if _, err := m.CreateMeeting(ctx, base); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name string
		run  func() error
	}{
		{"status", func() error {
			bad := base
			bad.ID = "bad-status"
			bad.Status = "bogus"
			_, err := m.CreateMeeting(ctx, bad)
			return err
		}},
		{"mode", func() error {
			bad := base
			bad.ID = "bad-mode"
			bad.Mode = "bogus"
			_, err := m.CreateMeeting(ctx, bad)
			return err
		}},
		{"segment source", func() error {
			mt, _, _ := m.Meeting(ctx, "ok")
			return m.AppendSegments(ctx, mt.Seq, []store.MeetingSegment{{Source: "bogus", Text: "x"}})
		}},
	}
	for _, tc := range cases {
		if err := tc.run(); !errors.Is(err, core.ErrInvalid) {
			t.Errorf("%s: %v", tc.name, err)
		}
	}
	mt, _, _ := m.Meeting(ctx, "ok")
	if _, err := s.db.Execute(ctx, `INSERT INTO meetings (id, created_at, status, mode) VALUES ('db-bad', 1, 'nope', 'meeting')`); err == nil {
		t.Error("db status CHECK: wanted error")
	}
	if _, err := s.db.Execute(ctx, `INSERT INTO meeting_segments (meeting_seq, idx, source, start_ms, end_ms, text) VALUES (?, 9, 'nope', 0, 1, 'x')`, mt.Seq); err == nil {
		t.Error("db segment CHECK: wanted error")
	}
}

func TestMeetingsMigrateFromV6(t *testing.T) {
	dir := t.TempDir()
	s := openTest(t, dir)
	if v, _ := userVersion(ctx, s.db); v != 7 {
		t.Fatalf("version %d", v)
	}
	s.Close()
	s = openTest(t, dir)
	if _, err := s.Meetings().CreateMeeting(ctx, meetingID("after-v6")); err != nil {
		t.Fatal(err)
	}
}

func TestMeetingsChatPersistAndDelete(t *testing.T) {
	s := openTest(t, t.TempDir())
	m := s.Meetings()
	ctx := context.Background()
	mt, err := m.CreateMeeting(ctx, meetingID("m-chat"))
	if err != nil {
		t.Fatal(err)
	}
	mt.Chat = `[{"role":"user","content":"Q1"},{"role":"assistant","content":"A1"}]`
	if err := m.UpdateMeeting(ctx, mt); err != nil {
		t.Fatal(err)
	}
	got, _, err := m.Meeting(ctx, "m-chat")
	if err != nil || got.Chat == "" || !strings.Contains(got.Chat, "Q1") {
		t.Fatalf("%+v %v", got.Chat, err)
	}
	if err := m.DeleteMeeting(ctx, "m-chat"); err != nil {
		t.Fatal(err)
	}
	if _, _, err := m.Meeting(ctx, "m-chat"); !errors.Is(err, core.ErrNotFound) {
		t.Fatalf("delete: %v", err)
	}
}

func TestMeetingsReplaceFailsMidTransaction(t *testing.T) {
	s := openTest(t, t.TempDir())
	m := s.Meetings()
	ctx := context.Background()
	mt, _ := m.CreateMeeting(ctx, meetingID("m-fail"))
	m.AppendSegments(ctx, mt.Seq, []store.MeetingSegment{seg(store.SegmentMic, "keep me")})

	// A second insert in the same transaction with a bad source must roll back the delete.
	repo := s.meetings
	stmts := []sqlite.Statement{
		{SQL: `DELETE FROM meeting_segments WHERE meeting_seq = ? AND source = ?`, Args: []any{mt.Seq, "mic"}},
		{SQL: `INSERT INTO meeting_segments (meeting_seq, idx, source, start_ms, end_ms, text) VALUES (?, 0, 'mic', 0, 1, 'new')`, Args: []any{mt.Seq}},
		{SQL: `INSERT INTO meeting_segments (meeting_seq, idx, source, start_ms, end_ms, text) VALUES (?, 1, 'nope', 0, 1, 'bad')`, Args: []any{mt.Seq}},
	}
	if _, err := repo.db.Transaction(ctx, stmts); err == nil {
		t.Fatal("transaction should fail")
	}
	_, segs, _ := m.Meeting(ctx, "m-fail")
	if len(segs) != 1 || segs[0].Text != "keep me" {
		t.Fatalf("rolled back: %+v", segs)
	}
}
