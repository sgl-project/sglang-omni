package main

import (
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// TestLimitNote: the pill warns in the last minute before the length
// limit (with the time left), and from 30 s before the upload size limit
// on; the length limit comes first; nothing without limits, before them,
// or once the recording stopped (#3840).
func TestLimitNote(t *testing.T) {
	heard := time.Date(2026, 11, 12, 9, 0, 0, 0, time.UTC)
	rec := func(maxLen, upload time.Duration) session.Snapshot {
		return session.Snapshot{State: session.Recording, HeardAt: heard, MaxLen: maxLen, UploadLen: upload}
	}
	stopped := rec(10*time.Minute, 0)
	stopped.State = session.Recognizing
	for _, c := range []struct {
		name string
		s    session.Snapshot
		at   time.Duration
		want string
	}{
		{"no limits", rec(0, 0), time.Hour, ""},
		{"length, early", rec(10*time.Minute, 0), 8 * time.Minute, ""},
		{"length, last minute", rec(10*time.Minute, 0), 9*time.Minute + 18*time.Second, "The length limit stops the recording in 0:42"},
		{"length, at it", rec(10*time.Minute, 0), 10*time.Minute + time.Second, "The length limit stops the recording in 0:00"},
		{"upload, early", rec(0, 13*time.Minute), 12 * time.Minute, ""},
		{"upload, near", rec(0, 13*time.Minute), 12*time.Minute + 40*time.Second, "Near the upload size limit: Voxt will send the recording in pieces"},
		{"upload, past", rec(0, 13*time.Minute), 40 * time.Minute, "Over the upload size limit: Voxt sends the recording in pieces"},
		{"both near", rec(13*time.Minute, 13*time.Minute), 12*time.Minute + 30*time.Second, "The length limit stops the recording in 0:30"},
		{"stopped", stopped, 9*time.Minute + 30*time.Second, ""},
	} {
		if got := limitNote(c.s, heard.Add(c.at)); got != c.want {
			t.Errorf("%s: %q; want %q", c.name, got, c.want)
		}
	}
}
