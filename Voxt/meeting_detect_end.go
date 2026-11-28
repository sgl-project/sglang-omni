package main

// The end of a meeting Voxt records is the user's: nothing ends it by
// itself. A meeting app may keep the microphone after the meeting, or let
// go of it on mute, so neither says the meeting is over. When Voxt's
// capture has heard no sound for meetingQuietAsk, the card asks whether
// it is over; the recording goes on until the user answers.

import (
	"cmp"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
)

// meetingQuietAsk is how long a meeting Voxt records hears no sound
// before the card asks whether it is over.
const meetingQuietAsk = 5 * time.Minute

// meetingEndKey is the card's key while it asks whether the meeting is
// over: no meeting app goes with the question.
const meetingEndKey = "meeting.end"

// meetingRecorded is what the end question knows of the meeting Voxt
// records.
type meetingRecorded struct {
	// asked: the card asked about this quiet stretch (or the user closed
	// it); a sound starts the next one.
	asked bool
	// quietAsk replaces meetingQuietAsk when not zero (tests).
	quietAsk time.Duration
}

// meetingRecords reports whether Voxt's meeting still records (paused
// too). Ended in Voxt, it is not over yet (its final pass runs, or it
// asks whether to keep it), but there is no end left to ask about.
func (a *app) meetingRecords() bool {
	switch a.meetSnap.State {
	case meeting.StateStarting, meeting.StateRecording, meeting.StatePaused:
		return a.meetMgr != nil
	}
	return false
}

// followRecordedMeeting runs when Voxt's meeting changed: once it has
// heard no sound for meetingQuietAsk, the card asks whether the meeting
// is over. A sound, or the meeting ended in Voxt, takes the question
// back.
func (a *app) followRecordedMeeting() {
	r := &a.recorded
	if !a.meetingRecords() || a.meetSnap.Quiet < cmp.Or(r.quietAsk, meetingQuietAsk) {
		r.asked = false
		if a.ask.end {
			a.closeMeetingAsk()
		}
		return
	}
	if r.asked || a.ask.key != "" || !a.detectMeetings() {
		return
	}
	r.asked = true
	a.showMeetingAsk(meetingEndKey, "", true)
}
