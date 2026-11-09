package meeting

import (
	"errors"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// Mode is how the user wants this meeting captured.
type Mode string

const (
	ModeMeeting   Mode = "meeting"
	ModeSubtitles Mode = "subtitles"
	ModeRecording Mode = "recording"
)

// Source is a recording track within a meeting.
type Source string

const (
	SourceMic    Source = "mic"
	SourceSystem Source = "system"
)

// State is where a meeting is in its lifecycle.
type State string

const (
	StateIdle        State = "idle"
	StateStarting    State = "starting"
	StateRecording   State = "recording"
	StatePaused      State = "paused"
	StateAsking      State = "asking"
	StateFinishing   State = "finishing"
	StateDone        State = "done"
	StateDiscarded   State = "discarded"
	StateFailed      State = "failed"
	StateInterrupted State = "interrupted"
)

// Stage is a substage while finishing (transcription and summary, C1-2+).
type Stage string

const (
	StageSaving Stage = "saving"
)

// Utterance is one PCM chunk for file-mode meeting ASR (C1-2).
type Utterance struct {
	Start, End time.Duration
	PCM        []byte
}

// Segment is one sentence on the timeline (filled in C1-2).
type Segment struct {
	ID          string
	Source      Source
	Speaker     string
	Start, End  time.Duration
	Text        string
	Translation string
	Final       bool
}

// Problem is a user-visible warning during a meeting.
type Problem struct {
	Key string
}

// Snapshot is the live view of the active meeting for UI and tests.
type Snapshot struct {
	ID        string
	Mode      Mode
	State     State
	Stage     Stage
	Progress  float64
	Started   time.Time
	Elapsed   time.Duration
	Level     map[Source]float64
	Live      map[Source]string
	Segments  []Segment
	Translate string
	Problem   *Problem
}

// StartRequest names a new meeting.
type StartRequest struct {
	Mode Mode
	Lang string
}

// Config wires a Manager to capture, storage, and settings.
type Config struct {
	Mic, System session.Audio
	Store       store.Meetings
	Dir         string // parent of per-meeting folders (<data>/meetings)
	Settings    func() settings.Meeting
	Privacy     func() settings.Privacy
	Now         func() time.Time
	FreeSpace   func(dir string) (int64, error)
}

// ErrBusy means a meeting is already live.
var ErrBusy = errors.New("meeting: one is running already")

// ErrNoMeeting names a meeting id that is not loaded.
var ErrNoMeeting = errors.New("meeting: not found")

// StartError means a track failed to start; the meeting does not begin.
type StartError struct {
	Source   Source
	Err      error
	Guidance string // i18n key for the overlay
}

func (e *StartError) Error() string {
	if e.Err != nil {
		return e.Err.Error()
	}
	return "meeting: could not start " + string(e.Source)
}

func (e *StartError) Unwrap() error { return e.Err }
