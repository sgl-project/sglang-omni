package meeting

import (
	"context"
	"errors"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/vad"
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
	StageSaving       Stage = "saving"
	StageTranscribing Stage = "transcribing"
	StageSummary      Stage = "summary"
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
	StoreIdx    int // sqlite meeting_segments.idx (-1 before persist)
}

// Problem is a user-visible warning during a meeting.
type Problem struct {
	Key string
}

// Snapshot is the live view of the active meeting for UI and tests.
type Snapshot struct {
	ID       string
	Mode     Mode
	State    State
	Stage    Stage
	Progress float64
	Started  time.Time
	Elapsed  time.Duration
	// Levels are each track's last session.Bars input levels as bar
	// heights (session.Level), oldest first: its waveform.
	Levels    map[Source][]float64
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
	// SystemCaptureOK is false when the helper lacks capture.system (C1-5).
	SystemCaptureOK func() bool
	// ModelHold keeps local speech models loaded for the meeting (C1-2).
	ModelHold local.Holder
	// ASR resolves live recognition per track; nil keeps WAV-only capture (tests).
	ASR func(ctx context.Context, src Source) (TrackASR, error)
	// VAD finds speech regions for the final pass; nil uses energy analysis.
	VAD vad.Model
	// Context is what the language model corrects a transcript with;
	// nil for none.
	Context func(ctx context.Context) TranscriptContext
	// LLM opens the language model for summary and live translation.
	LLM func(ctx context.Context, id string) (SummaryChat, error)
	// MainLanguage names the user's primary language for summary prompts;
	// "" writes them in the transcript's language.
	MainLanguage func() string
	// Warn records a failure no caller waits for (the correction after a
	// meeting); nil drops it.
	Warn func(msg string)
	// Updated reports that a meeting's stored transcript or summary
	// changed after it ended (the correction and summary that follow it);
	// nil reports nothing.
	Updated func(id string)
}

// ErrBusy means a meeting is already live.
var ErrBusy = errors.New("meeting: one is running already")

// ErrNoMeeting names a meeting id that is not loaded.
var ErrNoMeeting = errors.New("meeting: not found")

// StartError means a track failed to start; the meeting does not begin.
// GuidanceModel is StartError.Guidance when the track's speech model did
// not load, as opposed to its capture.
const GuidanceModel = "meeting.model.unavailable"

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
