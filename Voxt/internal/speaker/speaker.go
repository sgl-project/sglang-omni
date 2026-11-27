// Package speaker defines audio-only, incremental speaker identification.
// Its model runs independently of speech recognition and screen observations.
package speaker

import (
	"context"
	"time"
)

// Frame contains simultaneous speaker probabilities for one audio interval.
// Slots retain their identity until the stream is closed.
type Frame struct {
	Start, End time.Duration
	Scores     [8]float32
}

// Stream consumes PCM16 mono at 16 kHz. Push can return finalized frames
// from earlier audio because the model needs a short right context.
type Stream interface {
	Push(context.Context, []byte) ([]Frame, error)
	Close() error
}

// Model opens a meeting-owned stream. Opening may load a model, so callers
// must keep it off the capture, recognition and UI goroutines.
type Model interface {
	Open(context.Context) (Stream, error)
}
