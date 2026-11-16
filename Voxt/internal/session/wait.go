package session

import "time"

// Wait is what a session waits for before its recording works in full
// (Snapshot.Waiting).
type Wait string

const (
	WaitNone Wait = ""
	// WaitMic: the microphone has given no audio yet (capture.started
	// has not come): what is said meanwhile is not recorded. It takes
	// 150–400 ms; 5.9 s once on 10-20, while three models loaded.
	WaitMic Wait = "mic"
	// WaitModel: the speech model is still starting (its stream is not
	// open, or the services are not resolved): the recording goes on and
	// is sent to it once it is ready, so the live text comes late. Also
	// after Stop, while recognition waits for it.
	WaitModel Wait = "model"
)

// slowMic is a microphone start worth a warning in the log.
const slowMic = time.Second

// waitOf is what s waits for: nothing before Config.WaitNote passed (a
// short wait is not worth a word), the microphone while recording
// without audio, then the speech model while it starts (r.opening).
func (r *run) waitOf(s *Snapshot) Wait {
	if !r.waitDue.Load() {
		return WaitNone
	}
	switch {
	case s.State == Recording && s.HeardAt.IsZero():
		return WaitMic
	case (s.State == Recording || s.State == Recognizing) && r.opening.Load():
		return WaitModel
	}
	return WaitNone
}

// noteWait updates Snapshot.Waiting.
func (r *run) noteWait() {
	r.m.update(r, func(s *Snapshot) {
		if w := r.waitOf(s); w != s.Waiting {
			s.Waiting = w
		}
	})
}

// setOpening records whether the speech model is still starting, and
// what the snapshot waits for.
func (r *run) setOpening(v bool) {
	if r.opening.Swap(v) != v {
		r.noteWait()
	}
}

// waitTimer fires once Config.WaitNote has passed since the session
// started (nil channel when it has).
func (r *run) waitTimer(started time.Time) <-chan time.Time {
	left := r.m.cfg.WaitNote - r.now().Sub(started)
	if left <= 0 {
		r.waitDue.Store(true)
		return nil
	}
	return time.After(left)
}
