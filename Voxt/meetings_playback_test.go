package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestMeetingDetailPlaybackButtonNeedsAudio(t *testing.T) {
	dir := t.TempDir()
	meetingsDir := filepath.Join(dir, "meetings")
	if meeting.AudioFolderReady(meetingsDir, "") {
		t.Fatal("empty audio id")
	}
	if meeting.AudioFolderReady(meetingsDir, "gone") {
		t.Fatal("missing folder")
	}
	folder := "m-audio"
	sub := filepath.Join(meetingsDir, folder)
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	if meeting.AudioFolderReady(meetingsDir, folder) {
		t.Fatal("empty wavs")
	}
	if err := os.WriteFile(filepath.Join(sub, "mic.wav"), asr.WAV(make([]byte, 400), asr.SampleRate), 0o644); err != nil {
		t.Fatal(err)
	}
	if !meeting.AudioFolderReady(meetingsDir, folder) {
		t.Fatal("expected audio ready")
	}
}

func TestMeetingAudioProbeWaitsForLoadAndPage(t *testing.T) {
	a, _, _ := newTestApp(t)
	const id = "m-watch-load"
	if _, err := a.lib.store.Meetings().CreateMeeting(context.Background(), store.Meeting{ID: id, Status: store.MeetingDone, Mode: store.MeetingModeMeeting, Audio: "watch-audio"}); err != nil {
		t.Fatal(err)
	}
	var queued []func()
	a.postQueue = func(fn func()) { queued = append(queued, fn) }
	a.openMeetingDetail(id)
	detail := a.mv.detail
	if !detail.loading {
		t.Fatal("detail load did not start")
	}
	a.prepareMeetingsPage()
	if detail.audioProbe != 2 {
		t.Fatalf("probe started during load: generation %d", detail.audioProbe)
	}
	for _, fn := range queued {
		fn()
	}
	a.prepareMeetingsPage()
	if !detail.audioProbing {
		t.Fatal("no watch after load")
	}
	a.pageShown("/home")
	if detail.audioProbing {
		t.Fatal("watch continued off page")
	}
	a.navigate("/meetings")
	a.prepareMeetingsPage()
	if !detail.audioProbing {
		t.Fatal("watch did not restart on return")
	}
}

func TestMeetingAudioProbeStartsOnce(t *testing.T) {
	a, _, _ := newTestApp(t)
	const id = "m-watch"
	if _, err := a.lib.store.Meetings().CreateMeeting(context.Background(), store.Meeting{ID: id, Status: store.MeetingDone, Mode: store.MeetingModeMeeting, Audio: "watch-audio"}); err != nil {
		t.Fatal(err)
	}
	a.openMeetingDetail(id)
	detail := a.mv.detail
	directory := filepath.Join(a.dataDir(), "meetings", "watch-audio")
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(directory, "mic.wav"), asr.WAV(make([]byte, 400), asr.SampleRate), 0o644); err != nil {
		t.Fatal(err)
	}
	a.refreshMeetingAudio()
	if !detail.audioReady {
		t.Fatal("audio is not ready")
	}
	probe := detail.audioProbe
	a.refreshMeetingAudio()
	if detail.audioProbe != probe {
		t.Fatalf("settled frame started probe %d, want %d", detail.audioProbe, probe)
	}
	if err := os.Remove(filepath.Join(directory, "mic.wav")); err != nil {
		t.Fatal(err)
	}
	a.loadMeetingDetail(id)
	if detail.audioReady {
		t.Fatal("removed audio is still ready")
	}
	a.refreshMeetingAudio()
	probe = detail.audioProbe
	var queued []func()
	a.postQueue = func(fn func()) { queued = append(queued, fn) }
	a.refreshMeetingAudio()
	a.closeMeetingDetail()
	for _, fn := range queued {
		fn()
	}
	if a.mv.detail != nil {
		t.Fatal("late audio result reopened the detail")
	}
}
