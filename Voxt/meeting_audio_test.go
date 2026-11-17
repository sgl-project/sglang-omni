package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// A failed meeting's recording, kept for its 7 days, is deleted once
// they pass; one still inside them stays.
func TestSweepMeetingAudio(t *testing.T) {
	a, _, _ := newTestApp(t)
	dir := filepath.Join(a.dataDir(), "meetings")
	repo := a.lib.store.Meetings()
	old := store.Meeting{ID: "old", CreatedAt: time.Now().Add(-8 * 24 * time.Hour), Status: store.MeetingInterrupted,
		Mode: store.MeetingModeMeeting, Audio: "old", AudioUntil: time.Now().Add(-time.Hour)}
	fresh := store.Meeting{ID: "fresh", CreatedAt: time.Now(), Status: store.MeetingInterrupted,
		Mode: store.MeetingModeMeeting, Audio: "fresh", AudioUntil: time.Now().Add(6 * 24 * time.Hour)}
	for _, mt := range []store.Meeting{old, fresh} {
		if _, err := repo.CreateMeeting(context.Background(), mt); err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(filepath.Join(dir, mt.ID), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, mt.ID, "system.wav"), []byte("wav"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	n, err := a.sweepMeetingAudio(dir)
	if err != nil || n != 1 {
		t.Fatalf("swept %d, err %v", n, err)
	}
	got, _, err := repo.Meeting(context.Background(), "old")
	if err != nil || got.Audio != "" || !got.AudioUntil.IsZero() {
		t.Fatalf("expired meeting %+v, %v", got, err)
	}
	if _, err := os.Stat(filepath.Join(dir, "old")); !os.IsNotExist(err) {
		t.Errorf("the old recording's folder stayed: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "fresh", "system.wav")); err != nil {
		t.Errorf("the fresh recording went: %v", err)
	}
	got, _, err = repo.Meeting(context.Background(), "fresh")
	if err != nil || got.Audio != "fresh" {
		t.Errorf("the fresh meeting changed: %+v, %v", got, err)
	}
}
