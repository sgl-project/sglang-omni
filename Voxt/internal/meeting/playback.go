package meeting

import (
	"os"
	"path/filepath"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// PlaybackSegment is one sentence for playback highlight and seek.
type PlaybackSegment struct {
	Idx     int
	StartMS int64
	EndMS   int64
}

// SegmentIndexAt returns the idx of the segment playing at pos, or -1.
// When pos sits in a gap between sentences, the latest started segment wins.
func SegmentIndexAt(segs []PlaybackSegment, pos time.Duration) int {
	if len(segs) == 0 {
		return -1
	}
	ms := pos.Milliseconds()
	best := -1
	var bestStart int64 = -1
	for _, s := range segs {
		if s.StartMS <= ms && ms < s.EndMS {
			return s.Idx
		}
		if s.StartMS <= ms && s.StartMS >= bestStart {
			bestStart = s.StartMS
			best = s.Idx
		}
	}
	return best
}

// AudioFolderReady reports whether a meeting's audio folder still has at
// least one non-empty track WAV (Audio is the folder name under meetings/).
func AudioFolderReady(meetingsDir, folder string) bool {
	if folder == "" {
		return false
	}
	dir := filepath.Join(meetingsDir, folder)
	for _, name := range []string{"mic.wav", "system.wav"} {
		st, err := os.Stat(filepath.Join(dir, name))
		if err == nil && st.Size() > 44 {
			return true
		}
	}
	return false
}

// TrackPaths lists existing per-track WAV paths under a meeting audio dir.
func TrackPaths(dir string) []string {
	var out []string
	for _, name := range []string{"mic.wav", "system.wav"} {
		p := filepath.Join(dir, name)
		if st, err := os.Stat(p); err == nil && st.Size() > 44 {
			out = append(out, p)
		}
	}
	return out
}

// FormatPlaybackTime formats a timeline offset like export timestamps.
func FormatPlaybackTime(d time.Duration) string { return formatExportTime(d) }

// MaxTrackDuration is the longest duration among track files in dir.
func MaxTrackDuration(dir string) time.Duration {
	var max time.Duration
	for _, p := range TrackPaths(dir) {
		raw, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		pcm, rate, _, err := asr.ReadWAV(raw)
		if err != nil {
			continue
		}
		d := asr.Duration(pcm, rate)
		if d > max {
			max = d
		}
	}
	return max
}
