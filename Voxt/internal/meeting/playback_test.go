package meeting

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

func TestSegmentIndexAt(t *testing.T) {
	segs := []PlaybackSegment{
		{Idx: 0, StartMS: 0, EndMS: 1000},
		{Idx: 1, StartMS: 1500, EndMS: 3000},
		{Idx: 2, StartMS: 4000, EndMS: 5000},
	}
	cases := []struct {
		pos  time.Duration
		want int
	}{
		{500 * time.Millisecond, 0},
		{1200 * time.Millisecond, 0}, // gap after first
		{2000 * time.Millisecond, 1},
		{4500 * time.Millisecond, 2},
		{-time.Second, -1},
		{6 * time.Second, 2},
	}
	for _, tc := range cases {
		if got := SegmentIndexAt(segs, tc.pos); got != tc.want {
			t.Fatalf("pos %v got %d want %d", tc.pos, got, tc.want)
		}
	}
}

func TestAudioFolderReady(t *testing.T) {
	dir := t.TempDir()
	if AudioFolderReady(dir, "") {
		t.Fatal("empty folder name")
	}
	folder := "m1"
	if AudioFolderReady(dir, folder) {
		t.Fatal("missing dir")
	}
	sub := filepath.Join(dir, folder)
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatal(err)
	}
	if AudioFolderReady(dir, folder) {
		t.Fatal("no wav yet")
	}
	if err := os.WriteFile(filepath.Join(sub, "mic.wav"), asr.WAV(make([]byte, 200), asr.SampleRate), 0o644); err != nil {
		t.Fatal(err)
	}
	if !AudioFolderReady(dir, folder) {
		t.Fatal("expected ready")
	}
}
