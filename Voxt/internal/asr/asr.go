// Package asr is the shape of VoxtNew's speech recognition, which the
// session uses: a speech model (a Provider) is a file transcriber that
// takes a finished recording, a streamer that recognises while the user
// speaks, or both. The speech models are local (internal/local: the Omni
// native MLX servers of sglang_omni_mlx/native, over loopback HTTP and
// WebSockets, in internal/omni); this package has their types, errors and
// audio helpers only.
//
// Audio is what the helper records: 16-bit little-endian PCM, mono, at
// 16 kHz (SampleRate), sent in frames of about 100 ms. File transcribers
// get it as a WAV file (WAV builds one).
//
// Everything here is goroutine-safe and blocks: call it off the main
// thread. Errors are *Error values with a Kind (config, server, network,
// protocol, audio, cancelled…) that the UI maps to a message and a way
// forward.
//
// Imports: core.
package asr

import (
	"context"
	"time"
)

// SampleRate is the rate of the PCM VoxtNew records and sends, in Hz.
const SampleRate = 16000

// Caps is what a speech model can do.
type Caps struct {
	// Stream: it implements Streamer (partial text while recording);
	// File: FileTranscriber (the whole recording after it stops).
	Stream bool
	File   bool
	// Prompt: it takes a free-text prompt (context, spelling, style),
	// FileRequest.Prompt / StreamRequest.Prompt. Hints: it takes a list
	// of words to favour (hot words), the dictionary's terms.
	Prompt bool
	Hints  bool
	// MaxBytes bounds a recording's size (0: no bound known); MaxDuration
	// a recording (0: none known). Longer recordings fail with KindAudio
	// before anything is sent.
	MaxBytes    int64
	MaxDuration time.Duration
}

// Provider is what every speech model has.
type Provider interface {
	// Caps says what it can do.
	Caps() Caps
}

// FileRequest is one recording to transcribe.
type FileRequest struct {
	// Audio is the recording: a WAV file (Format "wav") or raw PCM16LE
	// mono at SampleRate Hz (Format "pcm"; the speech model wraps it).
	Audio      []byte
	Format     string
	SampleRate int // of "pcm"; 0: SampleRate
	// Lang is a language hint (a lang code, "zh", "en"); "" detects.
	Lang string
	// Hints are words to favour; Prompt is free text about the speech.
	// Speech models without Caps.Hints fold the hints into the prompt.
	Hints  []string
	Prompt string
}

// Result is a finished transcription.
type Result struct {
	Text     string
	Lang     string        // as the speech model reported it, "" when it did not
	Duration time.Duration // of the audio, when the speech model reported it
	Model    string
	Latency  time.Duration // from the end of the audio to the final text
}

// FileTranscriber transcribes a whole recording and returns its text.
type FileTranscriber interface {
	Provider
	Transcribe(ctx context.Context, req FileRequest) (Result, error)
}

// StreamRequest opens a stream.
type StreamRequest struct {
	SampleRate int    // of the PCM Send gets; 0: SampleRate
	Lang       string // a language hint, "" detects
	Hints      []string
	Prompt     string
}

// Streamer opens recognition streams.
type Streamer interface {
	Provider
	// Open connects and starts a session; it returns once the speech
	// model accepted it (or failed: not installed, not started, the
	// connection…), so that a failure shows before the user has said
	// much.
	Open(ctx context.Context, req StreamRequest) (Stream, error)
}

// Stream is one recognition session.
//
// Send and Finish are called from one goroutine (the session's); Close
// from any. Events delivers EventPartial while audio comes in, then
// exactly one EventFinal or EventError, then closes. ctx of Open bounds the whole
// stream: cancelling it ends the stream with a KindCancelled Error.
type Stream interface {
	// Send sends PCM16LE mono audio at the request's sample rate, any
	// length (the stream frames it as its speech model wants). It may
	// block for as long as the connection does.
	Send(pcm []byte) error
	// Finish says there is no more audio; EventFinal follows.
	Finish() error
	// Close abandons the stream (no EventFinal), closes the connection and
	// then Events. Safe to call more than once and after the end.
	Close()
	Events() <-chan Event
}

// Salvager is a Stream that can give the text it has when its final
// result failed or took too long (its live preview's, or the parts of
// the recording it recognised already): what the session inserts
// rather than nothing. "" for none.
type Salvager interface {
	Salvage() string
}

// EventKind is what an Event says.
type EventKind uint8

const (
	// EventPartial: the text so far (the whole hypothesis, not a delta);
	// it may change until the final one.
	EventPartial EventKind = iota + 1
	// EventFinal: the whole final text. The last event.
	EventFinal
	// EventError: the stream failed (Err is an *Error). The last event.
	EventError
)

func (k EventKind) String() string {
	switch k {
	case EventPartial:
		return "partial"
	case EventFinal:
		return "final"
	case EventError:
		return "error"
	}
	return "unknown"
}

// Event is one event of a Stream.
type Event struct {
	Kind EventKind
	Text string // EventPartial and EventFinal
	// Result is the final event's details (Text repeats there).
	Result Result
	Err    error // EventError
}
