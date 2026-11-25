package main

// The other end of a detected meeting: Voxt records it, and its app lets
// go of the microphone. The meeting is probably over, so the card asks
// whether to end the recording. It only asks: an app may let go of the
// microphone while the user is muted, and a recording ended by itself
// would lose the rest of the meeting.

import "time"

// meetingEndAfter is how long the meeting's app must be off the
// microphone before the card asks.
const meetingEndAfter = 20 * time.Second

// meetingRecorded is the meeting app whose meeting Voxt records ("": Voxt
// records none, or one that no app's microphone use goes with).
type meetingRecorded struct {
	key, app string
	// asked: the user answered the card (or closed it) for this meeting.
	asked bool
	// waiting: the wait for meetingEndAfter runs; seq numbers it, so that
	// the app coming back ends it.
	waiting bool
	seq     int
	// endAfter replaces meetingEndAfter when it is not zero (tests).
	endAfter time.Duration
}

// adoptRecordedMeeting runs when Voxt starts to record a meeting that was
// not started from the card (the shortcut, the Meetings page): while the
// app the card last asked about still records, it is that app's meeting,
// and its end is asked about as well.
func (a *app) adoptRecordedMeeting() {
	if a.recorded.key == "" && a.ask.lastKey != "" && a.detect.Recording(a.ask.lastKey) {
		a.recorded.key, a.recorded.app = a.ask.lastKey, a.ask.lastApp
	}
}

// followRecordedMeeting runs when the apps recording changed: once the
// app of the meeting Voxt records has been off the microphone for
// meetingEndAfter, the card asks whether the meeting is over.
func (a *app) followRecordedMeeting() {
	r := &a.recorded
	if r.key == "" || r.asked || !a.meetingActive() || a.detect.Recording(r.key) || a.ask.key != "" {
		r.seq++ // nothing to wait for (any more)
		r.waiting = false
		return
	}
	if r.waiting {
		return // other apps came and went: the wait goes on
	}
	r.waiting = true
	r.seq++
	key, seq := r.key, r.seq
	wait := meetingEndAfter
	if r.endAfter > 0 {
		wait = r.endAfter
	}
	a.later(wait, func() {
		if r.seq != seq {
			return
		}
		r.waiting = false
		if r.key != key || r.asked || !a.meetingActive() || a.detect.Recording(key) || a.ask.key != "" || !a.detectMeetings() {
			return
		}
		a.showMeetingAsk(key, r.app, true)
	})
}
