package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
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
