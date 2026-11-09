// Package store is VoxtNew's persistence contract: a repository for each
// kind of record. The app uses internal/store/sqlitestore (SQLite through
// the MyGo plugin github.com/egoist/mygo/plugins/sqlite; schema in its
// migrations/*.sql).
//
// Rules:
//   - Repositories are goroutine-safe and may block on I/O: never call
//     them from a MyGo build pass.
//   - Values in, values out: a repository copies what it is given and
//     returns copies.
//   - Not found is an error wrapping core.ErrNotFound.
//   - Blobs (a history entry's audio) are files under the store's
//     directory, referenced by name; the store deletes them with the record.
//
// There is no job queue: a session lives in memory (internal/session)
// and writes one history entry when it ends.
//
// Imports: core.
package store

import (
	"context"
	"time"
)

// Store is the whole database.
type Store interface {
	// Documents holds small JSON documents by key: "settings:app" (the
	// settings), "appctx:rules" (app enhancement), "notifications".
	Documents() Documents
	History() History
	Dictionary() Dictionary
	Notes() Notes
	Meetings() Meetings
	Close() error
}

// Documents is a key → JSON store for small documents.
type Documents interface {
	// Get returns the document and whether it exists.
	Get(ctx context.Context, key string) ([]byte, bool, error)
	Put(ctx context.Context, key string, doc []byte) error
	Delete(ctx context.Context, key string) error
}

// Kind is the feature that made a history entry.
type Kind string

const (
	KindTranscribe Kind = "transcribe"
	KindTranslate  Kind = "translate"
	KindRewrite    Kind = "rewrite"
)

// Kinds lists the kinds, in the History page's order.
var Kinds = []Kind{KindTranscribe, KindTranslate, KindRewrite}

// Status is how a session ended.
type Status string

const (
	StatusDone      Status = "done"
	StatusFailed    Status = "failed"
	StatusCancelled Status = "cancelled"
)

// Entry is one finished session: what was heard or selected, what the
// model made of it, where it went and how long each stage took.
type Entry struct {
	ID        string
	Kind      Kind
	CreatedAt time.Time
	Status    Status
	Error     string // why it failed, in English; "" when done

	// Context: the app in front when the shortcut was pressed, the host of
	// its tab (a browser), and the app enhancement group that matched.
	AppBundle string
	AppName   string
	URLHost   string
	RuleID    string
	RuleName  string // as it was named then
	// RuleSource is how the group matched: "url", "browserApp" or "app"
	// (appctx.Source); "" for none.
	RuleSource string

	// RawText is the recognised speech; SourceText the selected text
	// (translate, rewrite); PromptText the spoken instruction (rewrite);
	// Result what was delivered; TargetLang the translation's target.
	RawText    string
	SourceText string
	PromptText string
	Result     string
	TargetLang string
	// Delivery: "insert" (pasted where the cursor was), "copy" or "card"
	// (shown in the answer card).
	Delivery string
	// InsertedBundle and InsertedName are the app the result was pasted
	// into when that was not AppBundle's (the user moved to another app
	// before it was inserted); "" otherwise. AppBundle stays the app the
	// app enhancement group was chosen for. Kept with Delivery, without a
	// column of their own (sqlitestore).
	InsertedBundle, InsertedName string

	ASRProvider, ASRModel string
	LLMProvider, LLMModel string

	// Stage times, in milliseconds: the recording, recognition after the
	// stop, the language model, and the whole session. SentMS is how much
	// of the recording the speech model got, without the silence left
	// out (0: unknown, or none was sent).
	AudioMS, ASRMS, LLMMS, TotalMS int64
	SentMS                         int64

	// DSP is the sound processing the speech model's copy of the
	// recording went through (internal/dsp): "highpass", "level" or
	// "highpass+level"; "" for none (off, older entries, nothing sent).
	// GainDB is the gain the level stage gave the speech (a stream: at
	// its end). The recording kept (Audio) is the original either way.
	DSP    string
	GainDB float64

	// Audio is the recording kept with the entry, a file name in
	// History.AudioDir(); "" when none was kept.
	Audio string
	// AudioUntil: the recording is kept only until then (zero: as long as
	// the entry): recognition failed or found no text while the settings
	// keep no recordings, so it stays for a retry (session.Retry,
	// behavior.md 5.23a). ExpireAudio deletes it after.
	AudioUntil time.Time
}

// HistoryQuery selects history entries: the zero value is every entry,
// newest first.
type HistoryQuery struct {
	Kind   Kind   // "" = every kind
	Search string // full text over the texts; under 3 characters a substring match
	Limit  int    // 0 = no limit
	Offset int
	// WithAudio selects only the entries with a recording kept.
	WithAudio bool
}

// Retention is what the history keeps (its automatic clean-up): the
// entries since Before (zero: of any age) and the newest PerKind of each
// kind (0: any number). The zero value keeps everything.
type Retention struct {
	Before  time.Time
	PerKind int
}

// Zero reports whether r keeps everything.
func (r Retention) Zero() bool { return r.Before.IsZero() && r.PerKind <= 0 }

// History stores finished sessions.
type History interface {
	// Put inserts or replaces an entry (by ID); its audio file, if any,
	// must already be in AudioDir.
	Put(ctx context.Context, e Entry) error
	Get(ctx context.Context, id string) (Entry, error)
	List(ctx context.Context, q HistoryQuery) ([]Entry, error)
	// Delete removes the entry and its audio file.
	Delete(ctx context.Context, id string) error
	// DeleteAll removes every entry of kind ("" = all) and their audio,
	// and reports how many went.
	DeleteAll(ctx context.Context, kind Kind) (int, error)
	// Prune removes, in one transaction, the entries r does not keep, then
	// their audio, and reports how many went; CountPrune counts them
	// without removing anything.
	Prune(ctx context.Context, r Retention) (int, error)
	CountPrune(ctx context.Context, r Retention) (int, error)
	// DeleteAudio removes every entry's recording and keeps the entries
	// (their Audio becomes ""), and reports how many entries had one.
	DeleteAudio(ctx context.Context) (int, error)
	// AudioDir is the directory new recordings go to: the one SetAudioDir
	// set, else DefaultAudioDir. Memory only.
	AudioDir() string
	// DefaultAudioDir is the store's own audio directory. Pure.
	DefaultAudioDir() string
	// SetAudioDir makes dir the directory new recordings go to; "" is
	// DefaultAudioDir.
	SetAudioDir(dir string)
	// AudioPath is the file of a recording (an Entry.Audio): in AudioDir,
	// else in DefaultAudioDir when it is there (a recording kept while
	// the other folder was missing), else where it would be in AudioDir.
	// It reads the disk.
	AudioPath(name string) string
	// ExpireAudio deletes the recordings kept until now or before
	// (Entry.AudioUntil) and clears them from their entries; it reports
	// how many went.
	ExpireAudio(ctx context.Context, now time.Time) (int, error)
}

// Term is an entry of the user's dictionary: a name or a word of jargon
// the speech model should expect and the language model should keep as
// written.
type Term struct {
	ID   string
	Term string // as it must be written: "VoxtNew", "Kubernetes"
	// Aliases are what speech recognition tends to make of it ("aura
	// link"), for the glossary's "write X for Y".
	Aliases   []string
	Note      string // the user's note; not sent anywhere
	Enabled   bool
	CreatedAt time.Time
	UpdatedAt time.Time
}

// Dictionary stores the user's terms.
type Dictionary interface {
	// Put inserts or replaces a term (by ID); another term with the same
	// spelling (ignoring case) is core.ErrExists.
	Put(ctx context.Context, t Term) error
	Get(ctx context.Context, id string) (Term, error)
	// List returns the terms matching search ("" = all), A to Z.
	List(ctx context.Context, search string) ([]Term, error)
	Delete(ctx context.Context, id string) error
}

// Note is what the user said after the note shortcut (Voxt 1.x's notes):
// kept apart from the history, so its clean-up and Clear never take it.
type Note struct {
	ID        string
	Text      string
	CreatedAt time.Time
}

// Notes stores the user's notes.
type Notes interface {
	// Put inserts or replaces a note (by ID); empty text is core.ErrInvalid.
	Put(ctx context.Context, n Note) error
	// List returns every note, newest first.
	List(ctx context.Context) ([]Note, error)
	Delete(ctx context.Context, id string) error
}

// MeetingStatus is where a meeting is in its lifecycle.
type MeetingStatus string

const (
	MeetingRecording   MeetingStatus = "recording"
	MeetingInterrupted MeetingStatus = "interrupted"
	MeetingAsking      MeetingStatus = "asking"
	MeetingFinishing   MeetingStatus = "finishing"
	MeetingDone        MeetingStatus = "done"
	MeetingFailed      MeetingStatus = "failed"
)

// MeetingMode is what the user chose when the meeting started.
type MeetingMode string

const (
	MeetingModeMeeting   MeetingMode = "meeting"
	MeetingModeSubtitles MeetingMode = "subtitles"
	MeetingModeRecording MeetingMode = "recording"
)

// SegmentSource is which audio track produced a segment.
type SegmentSource string

const (
	SegmentMic    SegmentSource = "mic"
	SegmentSystem SegmentSource = "system"
)

// Meeting is one recorded session (rows in meetings; C1-1b-store).
type Meeting struct {
	Seq        int64 // the database rowid meetings_fts may index later
	ID         string
	CreatedAt  time.Time
	EndedAt    time.Time
	Status     MeetingStatus
	Mode       MeetingMode
	Title      string
	DurationMS int64
	ASRModel   string
	DiarModel  string
	LLMModel   string
	Speakers   string // JSON object, e.g. {"s1":"Alice"}
	Summary    string // JSON {title, body, todos, generated_at}
	Chat       string // JSON array of Q&A (C1-7)
	Audio      string // folder name under <data>/meetings/; "" when deleted
	AudioUntil time.Time
}

// MeetingSegment is one sentence on a track.
type MeetingSegment struct {
	MeetingSeq  int64
	Idx         int // meeting_segments.idx; <0 lets AppendSegments assign the next index
	Source      SegmentSource
	Speaker     string
	StartMS     int64
	EndMS       int64
	Text        string
	Translation string
	Final       bool
}

// MeetingQuery lists meetings; zero lists all, newest first.
type MeetingQuery struct {
	Search string // full text on segment text and translation
	Limit  int
	Offset int
}

// Meetings stores meeting metadata and segments (no call site in C1-1b-store).
type Meetings interface {
	CreateMeeting(ctx context.Context, m Meeting) (Meeting, error)
	UpdateMeeting(ctx context.Context, m Meeting) error
	AppendSegments(ctx context.Context, seq int64, segs []MeetingSegment) error
	UpdateSegmentTranslation(ctx context.Context, seq int64, idx int, translation string) error
	ReplaceSegments(ctx context.Context, seq int64, src SegmentSource, segs []MeetingSegment) error
	Meeting(ctx context.Context, id string) (Meeting, []MeetingSegment, error)
	ListMeetings(ctx context.Context, q MeetingQuery) ([]Meeting, error)
	DeleteMeeting(ctx context.Context, id string) error
	Unfinished(ctx context.Context) ([]Meeting, error)
	ExpiredAudio(ctx context.Context, now time.Time) ([]Meeting, error)
}
