package main

// A failed or cancelled meeting whose settings keep no audio keeps the
// recording for seven days (Meeting.AudioUntil), then this deletes it
// (the store's ExpiredAudio). A finished meeting's audio follows the
// settings right away (internal/meeting's applyAudioRetention).

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
)

// expireMeetingAudio deletes those recordings once their time is up:
// now, then every recordingSweep (history_retry.go's).
func (a *app) expireMeetingAudio() {
	dir := filepath.Join(a.dataDir(), "meetings")
	go func() {
		for {
			n, err := a.sweepMeetingAudio(dir)
			if n > 0 || err != nil {
				msg := fmt.Sprintf("deleted %d meetings' recordings kept after a failure", n)
				if err != nil {
					msg += ": " + err.Error()
				}
				a.logs.Append(core.LogEntry{Time: time.Now(), Level: core.SeverityInfo, Source: meetingLogSource, Message: msg})
				a.post(a.loadMeetings)
			}
			time.Sleep(recordingSweep)
		}
	}()
}

// sweepMeetingAudio deletes the recordings of one pass of expired
// meetings and clears their fields.
func (a *app) sweepMeetingAudio(dir string) (int, error) {
	ms := a.lib.store.Meetings()
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	expired, err := ms.ExpiredAudio(ctx, time.Now())
	if err != nil {
		return 0, err
	}
	n := 0
	for _, mt := range expired {
		// The fields clear only once the files are gone: otherwise the
		// recording would stay on disk with nothing ever selecting it again.
		if err := os.RemoveAll(filepath.Join(dir, mt.Audio)); err != nil {
			return n, fmt.Errorf("delete recording %s: %w", mt.Audio, err)
		}
		mt.Audio, mt.AudioUntil = "", time.Time{}
		if err := ms.UpdateMeeting(ctx, mt); err != nil {
			return n, err
		}
		n++
	}
	return n, nil
}
