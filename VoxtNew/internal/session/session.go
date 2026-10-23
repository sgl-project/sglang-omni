// Package session is the one owner of a voice session: a state machine
// that takes a shortcut's press to a delivered result.
//
//	idle ─Start─▶ recording ─Stop─▶ recognizing ─▶ composing ─▶ delivering ─▶ done
//	                  │                  │             │             │
//	                  └──────────────────┴── Cancel ───┴─────────────┴──▶ cancelled
//	                  └──────────── an error with no way on ────────────▶ failed
//
// Translate starts in reading (the selection): with selected text it goes
// straight to composing, else it records. Rewrite reads the selection
// while it records: the spoken instruction rewrites it, or, with nothing
// selected, says what to write. A session started with its text (a retry
// from the answer card, another target language) starts in composing.
//
// Every session has an id; its callbacks (audio frames, the speech
// model's partials, timers) carry it, and anything that arrives for a
// session that is not the current one, or after it ended, is dropped. A
// session ends once: done, cancelled or failed, with one history entry
// (when the settings keep history).
//
// While it records, the PCM goes to a streaming speech model as it comes
// (partial text for the overlay), or is kept for a file transcriber,
// which gets it on Stop; either way the recording is kept in memory as
// the WAV a history entry may keep. After recognition one language-model
// request (internal/compose) cleans the text up, unless the feature has
// it off; if it fails or times out the raw text goes on. Then
// internal/deliver inserts it, or leaves it for the answer card.
//
// Everything platform is behind an interface: the microphone and the
// text I/O (the helper; helperio.go adapts the supervisor), the speech and
// language models (Resolver), the history and the dictionary (store),
// the clock. Tests drive the whole machine on Linux with fakes.
//
// Imports: appctx, asr, compose, core, deliver, dict, llm, settings, store,
// helper/proto, lang.
package session

import (
	"context"
	"errors"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/appctx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/compose"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/deliver"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/vad"
)

// Kind is the feature of a session.
type Kind = store.Kind

// State is where a session is.
type State string

const (
	Idle State = "idle" // no session yet
	// Reading: Translate reads the selection before it records.
	Reading     State = "reading"
	Recording   State = "recording"
	Recognizing State = "recognizing"
	Composing   State = "composing"
	Delivering  State = "delivering"
	Done        State = "done"
	Cancelled   State = "cancelled"
	Failed      State = "failed"
)

// Live reports whether a session in state s has not ended.
func (s State) Live() bool {
	switch s {
	case Reading, Recording, Recognizing, Composing, Delivering:
		return true
	}
	return false
}

// Ended reports whether s is an end.
func (s State) Ended() bool { return s == Done || s == Cancelled || s == Failed }

// Problem codes.
const (
	// ProblemNoASR: no speech model is ready (Local models).
	ProblemNoASR = "noASR"
	// ProblemMicrophone: macOS refused the microphone.
	ProblemMicrophone = "microphone"
	// ProblemDevice: no input gave audio.
	ProblemDevice = "device"
	// ProblemHelper: the helper is not running (or died).
	ProblemHelper = "helper"
	// ProblemASR: the speech model failed (Err is an *asr.Error). Degraded:
	// its Final failed, and the session went on with the text it had (the
	// live preview's, or the pieces it recognised while it recorded).
	ProblemASR = "asr"
	// ProblemLLM: the language model failed or timed out. Transcribe: the
	// raw text went on (Degraded); Translate and Rewrite: the answer card
	// shows the error, to try again.
	ProblemLLM = "llm"
	// ProblemNoSpeech: nothing was recognised.
	ProblemNoSpeech = "noSpeech"
	// ProblemTooShort: the recording was too short to send (the session
	// is cancelled quietly).
	ProblemTooShort = "tooShort"
	// ProblemNoLLM: Translate and Rewrite need a language model, and none
	// is ready (Local models).
	ProblemNoLLM = "noLLM"
	// ProblemEmpty: the language model answered nothing (Translate,
	// Rewrite).
	ProblemEmpty = "empty"
	// ProblemUnsupported: the feature does not work (an unknown kind).
	ProblemUnsupported = "unsupported"
	// ProblemInternal: anything else.
	ProblemInternal = "internal"
)

// Problem is what went wrong in a session.
type Problem struct {
	Code string
	Err  error
	// Degraded: the session went on without the failed part.
	Degraded bool
}

// Model names a model a session used.
type Model struct {
	ID, Name, Model string
}

// Timings are a session's stage times. Sent is how much of the
// recording (Audio) the speech model got: less without the silence the
// local voice activity detection left out (0 when it got none, or
// before recognition).
type Timings struct {
	Audio, Sent, ASR, LLM, Deliver, Total time.Duration
}

// Bars is the number of level bars a snapshot keeps (the overlay's
// waveform).
const Bars = 16

// Snapshot is a session as the UI sees it: a copy, published at each
// change (Config.OnChange) with an increasing Rev.
type Snapshot struct {
	Rev   uint64
	ID    string
	Kind  Kind
	State State
	// Try: started from a page's Try area, the result stays there.
	Try bool
	// HoldBinding is the binding whose long press started the session
	// (its release stops it); "" for none.
	HoldBinding string
	// Target is the app in front when the shortcut was pressed.
	Target proto.App
	// StartedAt: the shortcut; HeardAt: the first audio (zero until then:
	// the user's first words are not heard yet); StoppedAt: Stop.
	StartedAt, HeardAt, StoppedAt time.Time
	// Levels are the last Bars input levels, 0–1, oldest first.
	Levels []float64
	// Waiting is what the session still waits for, once it waited
	// longer than Config.WaitNote (wait.go): the microphone's first audio,
	// or the speech model, still starting. The pill says so.
	Waiting Wait
	// Device is the input that records, FellBack whether it was not the
	// first choice.
	Device   string
	FellBack bool
	// Partial is a streaming speech model's text so far; Raw the final
	// recognised text (Rewrite's spoken instruction); Result what is
	// delivered.
	Partial, Raw, Result string
	// Selection is the selected text read when the shortcut fired
	// (Translate, Rewrite; "" for none), SelectionVia how ("ax", "copy").
	Selection, SelectionVia string
	// TargetLanguage is Translate's (a code: "en", "zh-Hans").
	TargetLanguage string
	// Draft is the answer so far while the language model writes
	// (Translate, Rewrite); Title the answer's title (Rewrite writing
	// from scratch).
	Draft, Title string
	// Retried: Translate asked again, strictly (the first answer looked
	// untranslated).
	Retried bool
	// Inserted: the result was pasted; Card: it shows in the answer card
	// (Delivery says why it was not inserted). While the model writes,
	// Card says the answer streams into the card.
	Inserted bool
	Card     bool
	Delivery deliver.Reason
	// InsertedInto is the app the result was pasted into: Target, or the
	// app the user went to and focused a text field in. Target stays the
	// app the scene was chosen for (the history's app).
	InsertedInto proto.App
	// Scene is the app enhancement group the session falls in (scene.go);
	// nil until it is decided.
	Scene *appctx.Match
	// Profile is the composition's ("transcribe:cleanup+structure").
	Profile  string
	ASR, LLM Model
	Timings  Timings
	// DSP is the sound processing the speech model's copy went through
	// ("highpass+level"; "" for none), GainDB the gain it gave the speech
	// (a stream: at its end), to 0.1 dB.
	DSP     string
	GainDB  float64
	Problem *Problem
	// Recording: the session's history entry keeps its recording (the
	// settings', or for a while because recognition failed: Retry).
	Recording bool
}

// Elapsed is how long the recording has run (from the first audio), at
// now.
func (s Snapshot) Elapsed(now time.Time) time.Duration {
	if s.HeardAt.IsZero() {
		return 0
	}
	end := now
	if !s.StoppedAt.IsZero() {
		end = s.StoppedAt
	}
	if end.Before(s.HeardAt) {
		return 0
	}
	return end.Sub(s.HeardAt)
}

// Audio is the microphone: the helper's capture.start and capture.stop.
type Audio interface {
	// Record records as job id until Stop, the input's end or ctx's end
	// (which discards it), handing its events to fn in order, before it
	// returns.
	Record(ctx context.Context, id string, p proto.CaptureParams, fn func(AudioEvent)) (proto.CaptureResult, error)
	// Stop ends the recording id; Record returns before it does.
	Stop(ctx context.Context, id string) error
}

// AudioEvent is one notification of a recording: exactly one field set.
type AudioEvent struct {
	Started   *proto.CaptureStartedParams
	Level     *proto.LevelParams
	PCM       *proto.PCMParams
	Restarted *proto.RestartedParams
}

// LLM is a language model (an llm.Client over a local model).
type LLM interface {
	Chat(ctx context.Context, req llm.Request) (llm.Response, error)
}

// Services are the speech and language models of one session.
type Services struct {
	// ASR is nil when no speech model is ready (the session fails with
	// ProblemNoASR when it needs one).
	ASR     asr.Provider
	ASRInfo Model
	// LLM is nil when the feature has no language model ready.
	LLM     LLM
	LLMInfo Model
	// VAD is the voice activity model (Silero) that finds the speech in
	// the recording; nil: the energy detector of internal/vad does, as it
	// does when the model fails (vad_model.go).
	VAD vad.Model
}

// Resolver opens the models a session of kind uses with settings s; it
// may block (it may start a runtime). A model that is not ready is nil
// in Services (an error wrapping ErrNoASR works too).
type Resolver func(ctx context.Context, kind Kind, s settings.Settings) (Services, error)

// ErrNoASR: no speech model is ready.
var ErrNoASR = errors.New("session: no speech model is ready")

// ErrBusy: a session is live already.
var ErrBusy = errors.New("session: a session is running")

// Selector reads the selected text of the app in front (the helper's
// textio.selection).
type Selector interface {
	Selection(ctx context.Context, p proto.SelectionParams) (proto.SelectionResult, error)
}

// Config is what a Manager needs. Audio, Text and Resolve are required.
type Config struct {
	Audio   Audio
	Text    deliver.TextIO
	Resolve Resolver
	// Select reads the selection (nil: Translate and Rewrite never see
	// one).
	Select Selector
	// History keeps finished sessions (nil: none); Glossary lists the
	// dictionary's enabled terms (nil: none).
	History  store.History
	Glossary func(ctx context.Context) ([]store.Term, error)
	// Settings returns the current settings (memory only); Locale the UI
	// language, for a main language left to follow it.
	Settings func() settings.Settings
	Locale   func() string
	// Scene is app enhancement (nil: none).
	Scene *SceneConfig
	// Self is VoxtNew's pid and Activate brings an app back to the front
	// (deliver.Request).
	Self     int
	Activate func(pid int) bool
	// OnChange gets every snapshot, on the session's goroutines; it must
	// not block (post it to the UI). Snapshots may arrive out of order:
	// keep the highest Rev.
	OnChange func(Snapshot)
	// Log writes a line about a session: never its text unless the
	// settings allow it (Privacy.LogText).
	Log func(level, msg string)
	// Now is the clock (nil: time.Now); NewID makes session ids.
	Now   func() time.Time
	NewID func() string

	// Limits (0: the defaults).
	MaxRecording time.Duration // DefaultMaxRecording
	MinRecording time.Duration // DefaultMinRecording
	// Recognition after Stop gets ASRTimeout plus ASRPerSecond per second
	// of the recording (asrLimit): a local model that stopped answering is
	// the runtime's to notice (internal/omni's stall watch), so the limit
	// only ends one that runs away, never a long recording recognised
	// slowly (behavior.md 5.5a).
	ASRTimeout   time.Duration // DefaultASRTimeout
	ASRPerSecond time.Duration // DefaultASRPerSecond
	// AudioRetention is how long the history keeps a recording whose
	// recognition failed (or found no text) when the settings keep no
	// recordings: for Retry (DefaultAudioRetention).
	AudioRetention time.Duration
	// A language model's request ends when it makes no progress
	// (behavior.md 5.5, llmDeadlines): no first token within
	// LLMFirstToken (plus llmPrefillPerChar per character of its input),
	// no next one within LLMStall; and in all after LLMPerChar per
	// character, at least LLMTimeout for a piece of Transcribe's clean-up
	// and WriteTimeout for Translate's and Rewrite's answer.
	LLMFirstToken time.Duration // DefaultLLMFirstToken
	LLMStall      time.Duration // DefaultLLMStall
	LLMPerChar    time.Duration // DefaultLLMPerChar
	LLMTimeout    time.Duration // DefaultLLMTimeout
	WriteTimeout  time.Duration // DefaultWriteTimeout
	ReadTimeout   time.Duration // reading the selection; DefaultReadTimeout
	DeliverTime   time.Duration // DefaultDeliverTimeout
	// WaitNote is how long a recording waits for the microphone or the
	// speech model before Snapshot.Waiting says so (DefaultWaitNote).
	WaitNote time.Duration
	// StopGrace is how long Stop waits for the recording to end before
	// discarding the capture job (a capture.stop that raced its start);
	// DefaultStopGrace.
	StopGrace time.Duration
}

// Defaults of Config's limits.
const (
	DefaultMaxRecording   = 10 * time.Minute
	DefaultMinRecording   = 300 * time.Millisecond
	DefaultASRTimeout     = 60 * time.Second
	DefaultASRPerSecond   = time.Second
	DefaultAudioRetention = 7 * 24 * time.Hour
	DefaultLLMFirstToken  = 15 * time.Second
	DefaultLLMStall       = 5 * time.Second
	DefaultLLMPerChar     = 100 * time.Millisecond
	DefaultLLMTimeout     = 20 * time.Second
	DefaultWriteTimeout   = 60 * time.Second
	DefaultReadTimeout    = 1500 * time.Millisecond
	DefaultDeliverTimeout = 5 * time.Second
	DefaultStopGrace      = 3 * time.Second
	DefaultWaitNote       = 400 * time.Millisecond
)

// StartRequest starts a session.
type StartRequest struct {
	Kind Kind
	// Target is the app in front (hotkey.event's front, else the last
	// app.activated).
	Target proto.App
	// HoldBinding: a long press of this binding started it.
	HoldBinding string
	// Try: from a page's Try area; nothing is inserted and nothing kept
	// (and no selection is read).
	Try bool
	// At is when the shortcut fired (zero: now).
	At time.Time
	// Text starts Translate or Rewrite with its input, without recording
	// or reading the selection: a retry from the answer card, or the
	// same text into another language.
	Text *TextInput
	// TargetLanguage overrides Translate's target ("": the settings').
	TargetLanguage string
	// Card keeps the answer in the answer card, never inserted: the
	// card's text into another language (Voxt's language menu replaces
	// the card's answer).
	Card bool
	// Previous are a Rewrite conversation's earlier turns: the session is
	// a follow-up to the answer card.
	Previous []compose.Turn
}

// TextInput is a session's input given at Start.
type TextInput struct {
	// Raw is what was said (Translate's speech, Rewrite's instruction).
	Raw string
	// Selection is the selected text ("" for none).
	Selection string
}

func (s Snapshot) clone() Snapshot {
	if s.Levels != nil {
		s.Levels = append([]float64(nil), s.Levels...)
	}
	if s.Problem != nil {
		p := *s.Problem
		s.Problem = &p
	}
	if s.Scene != nil {
		m := cloneMatch(*s.Scene)
		s.Scene = &m
	}
	return s
}

// Level is a dBFS level as a bar height, 0–1 (−60 dBFS and below is 0).
func Level(db float64) float64 {
	v := (db + 60) / 60
	return min(max(v, 0), 1)
}

// ProblemOf classifies an error of a session.
func ProblemOf(err error) *Problem {
	if err == nil {
		return nil
	}
	p := &Problem{Code: ProblemInternal, Err: err}
	var pe *proto.Error
	var ae *asr.Error
	switch {
	case errors.Is(err, ErrNoASR):
		p.Code = ProblemNoASR
	case errors.As(err, &pe):
		switch pe.Code {
		case proto.CodePermissionDenied:
			p.Code = ProblemMicrophone
		case proto.CodeDeviceUnavailable:
			p.Code = ProblemDevice
		case proto.CodeBusy:
			p.Code = ProblemDevice
		}
	case errors.As(err, &ae):
		p.Code = ProblemASR
	case isHelperErr(err):
		p.Code = ProblemHelper
	}
	return p
}
