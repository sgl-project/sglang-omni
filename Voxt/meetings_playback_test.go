package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

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

// TestMeetingPlaybackMissingAudioKeepsState: Play and Seek do not stop or move playback when the WAV folder is gone.
func TestMeetingPlaybackMissingAudioKeepsState(t *testing.T) {
	for _, test := range []struct {
		name       string
		audioReady bool
		playing    bool
		run        func(*app, string)
	}{
		{name: "play", run: func(a *app, id string) { a.toggleMeetingPlayback(id) }},
		{name: "seek", playing: true, run: func(a *app, id string) { a.seekMeetingPlayback(id, 5000) }},
		{name: "stale ready play", audioReady: true, run: func(a *app, id string) { a.toggleMeetingPlayback(id) }},
		{name: "stale ready seek", audioReady: true, playing: true, run: func(a *app, id string) { a.seekMeetingPlayback(id, 5000) }},
	} {
		t.Run(test.name, func(t *testing.T) {
			a, _, _ := newTestApp(t)
			const id = "m-missing-audio"
			a.mv.detail = &meetingDetail{
				id: id, meeting: store.Meeting{Audio: "missing"}, audioReady: test.audioReady,
				playback: meetingDetailPlayback{pos: 2 * time.Second, playing: test.playing, highlight: 1, seq: 7},
			}
			before := a.mv.detail.playback
			stopped := false
			a.mtPlayer.cancel = func() { stopped = true }
			test.run(a, id)
			if got := a.mv.detail.playback; got != before {
				t.Fatalf("playback changed from %+v to %+v", before, got)
			}
			if stopped {
				t.Fatal("missing audio stopped the current player")
			}
		})
	}
}

// TestMeetingPlaybackPauseWithMissingAudio: Pause always stops playback, even when the WAV folder is gone.
func TestMeetingPlaybackPauseWithMissingAudio(t *testing.T) {
	a, _, _ := newTestApp(t)
	const id = "m-pause-missing-audio"
	a.mv.detail = &meetingDetail{
		id: id, meeting: store.Meeting{Audio: "missing"}, audioReady: true,
		playback: meetingDetailPlayback{pos: 2 * time.Second, playing: true, highlight: 1, seq: 7},
	}
	stopped := false
	a.mtPlayer.cancel = func() { stopped = true }
	a.toggleMeetingPlayback(id)
	want := meetingDetailPlayback{pos: 2 * time.Second, playing: false, highlight: -1, seq: 8}
	if got := a.mv.detail.playback; got != want {
		t.Fatalf("playback %+v, want %+v", got, want)
	}
	if !stopped {
		t.Fatal("Pause did not stop the current player")
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

func TestMeetingAudioProbeWaitsForReload(t *testing.T) {
	a, _, _ := newTestApp(t)
	const id = "m-watch-reload"
	if _, err := a.lib.store.Meetings().CreateMeeting(context.Background(), store.Meeting{ID: id, Status: store.MeetingDone, Mode: store.MeetingModeMeeting, Audio: "watch-audio"}); err != nil {
		t.Fatal(err)
	}
	a.openMeetingDetail(id)
	detail := a.mv.detail
	a.refreshMeetingAudio()
	var queued []func()
	a.postQueue = func(fn func()) { queued = append(queued, fn) }
	a.loadMeetingDetail(id)
	if !detail.loadPending {
		t.Fatal("reload is not pending")
	}
	pendingProbe := detail.audioProbe
	a.prepareMeetingsPage()
	if detail.audioProbe != pendingProbe {
		t.Fatalf("reload started a probe: generation %d, want %d", detail.audioProbe, pendingProbe)
	}
	for _, fn := range queued {
		fn()
	}
	a.prepareMeetingsPage()
	if !detail.audioProbing {
		t.Fatal("no watch after reload")
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
