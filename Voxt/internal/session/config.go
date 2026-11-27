package session

import (
	"context"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/deliver"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/snippet"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// Config is what a Manager needs. Audio, Text and Resolve are required.
type Config struct {
	Audio   Audio
	Text    deliver.TextIO
	Resolve Resolver
	// ResolveASR and ResolveLLM are the independent readiness lanes for
	// ordinary voice Transcribe; Resolve keeps the other feature paths compatible.
	ResolveASR ASRResolver
	ResolveLLM LLMResolver
	// Select reads the selection (nil: Translate and Rewrite never see
	// one).
	Select Selector
	// Window takes a JPEG of app pid's front window (the helper's
	// screen.window; nil without a picture) for the language model, when
	// the settings send one (Privacy.SendWindow). Nil: never.
	Window func(ctx context.Context, pid int) ([]byte, error)
	// ClipboardImage reads the picture on the clipboard as a JPEG (the
	// helper's clipboard.image; nil without one) for Rewrite's language
	// model, when the settings send it (Privacy.SendClipboardImage).
	// Nil: never.
	ClipboardImage func(ctx context.Context) ([]byte, error)
	// BoxInput returns the Rewrite input box's typed text and pictures
	// (JPEGs, encoded once when they were pasted) as they are when the
	// answer is asked for (StartRequest.Box). It waits for pending image
	// conversions and returns their error, if one needs confirmation. It
	// is called on the session's goroutine. Nil: the box sends nothing.
	BoxInput func(context.Context) (string, [][]byte, error)
	// BoxAutoSuppressed reports whether the user removed the automatic
	// selection or clipboard/window image from the box's preview. Nil: send both.
	BoxAutoSuppressed func() (selection, image bool)
	// BoxReview waits while the user reviews a newly captured automatic
	// window image, before its request. Nil: send without another review.
	BoxReview func(context.Context) error
	// History keeps finished sessions (nil: none); Glossary lists the
	// dictionary's enabled terms (nil: none).
	History  store.History
	Glossary func(ctx context.Context) ([]store.Term, error)
	// Snippets lists the user's snippets (nil: none): a Transcribe of only
	// a trigger is its snippet; said triggers are replaced (snippets.go).
	Snippets func(ctx context.Context) ([]snippet.Snippet, error)
	// Notes keeps the results of note sessions (nil: they show in the
	// card, with a problem).
	Notes store.Notes
	// Settings returns the current settings (memory only); Locale the UI
	// language, for a main language left to follow it.
	Settings func() settings.Settings
	Locale   func() string
	// Languages lists the languages the user reads, most preferred first,
	// as the system orders them (nil: none), for the Chinese script of
	// dictation (SpeechScript).
	Languages func() []string
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
	// AfterFunc schedules a callback; the returned function stops it.
	AfterFunc func(time.Duration, func()) func() bool
	// CleanupLatency is opt-in; non-positive thresholds preserve the current wait.
	CleanupLatency CleanupLatencyPolicy

	// Limits (0: the defaults).
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
	// A language model's request ends when its answer, once it streams,
	// stops for LLMStall, or after LLMBase plus LLMPerChar (and
	// llmPrefillPerChar) per character of its input, at least
	// LLMTimeout for a piece of Transcribe's clean-up and WriteTimeout
	// for Translate's and Rewrite's answer (behavior.md 5.5,
	// llmDeadlines).
	LLMBase      time.Duration // DefaultLLMBase
	LLMStall     time.Duration // DefaultLLMStall
	LLMPerChar   time.Duration // DefaultLLMPerChar
	LLMTimeout   time.Duration // DefaultLLMTimeout
	WriteTimeout time.Duration // DefaultWriteTimeout
	ReadTimeout  time.Duration // reading the selection; DefaultReadTimeout
	// RewriteReadTimeout is Rewrite's: it reads the selection while the
	// user speaks, so a long one (a page of a web editor read through
	// Accessibility's text markers) may take longer without delaying
	// anything (DefaultRewriteReadTimeout). Translate reads it before it
	// knows whether to record.
	RewriteReadTimeout time.Duration
	DeliverTime        time.Duration // DefaultDeliverTimeout
	// WaitNote is how long a recording waits for the microphone or the
	// speech model before Snapshot.Waiting says so (DefaultWaitNote).
	WaitNote time.Duration
	// StopGrace is how long Stop waits for the recording to end before
	// discarding the capture job (a capture.stop that raced its start);
	// DefaultStopGrace.
	StopGrace time.Duration
	// EndWordPause is how long the live text must end with the end word
	// (settings.Features.EndWord) before the recording stops;
	// DefaultEndWordPause.
	EndWordPause time.Duration
}
