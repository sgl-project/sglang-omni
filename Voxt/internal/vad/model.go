package vad

import (
	"context"
	"time"
)

// Model is a voice activity model (Silero VAD, on the Omni runtime:
// internal/omni), which tells speech by a neural network's probability
// rather than by energy: noise, music or a fan do not count as speech.
// The session uses it where it has one, and the energy detector of this
// package where it has none or the model fails. Implementations are
// goroutine-safe and may block (a loopback server).
type Model interface {
	// Speech finds the speech in a whole recording (PCM16 mono at rate
	// Hz): its segments in samples, in order, padded a little.
	Speech(ctx context.Context, pcm []byte, rate int) ([]Segment, error)
	// Open starts a detector for one stream (PCM16 mono at 16 kHz); ctx
	// bounds the stream's life, not only its opening (the model bounds
	// how long it waits for its own start).
	Open(ctx context.Context) (ModelStream, error)
}

// ModelStream is a model's detector over one stream. It is not safe for
// concurrent use.
type ModelStream interface {
	// Feed takes the next PCM (any length) and returns the speech
	// probability of each chunk of Chunk samples it completed, in order.
	Feed(pcm []byte) ([]float64, error)
	// Chunk is the samples one probability covers; Threshold the
	// probability that is speech.
	Chunk() int
	Threshold() float64
	// Close ends the stream.
	Close()
}

// FromSegments is the Analysis of a recording of samples samples whose
// speech a model found in segs (clipped to the recording); FloorDB and
// PeakDB are energy's, from a (the energy analysis of the same
// recording, for the log).
func FromSegments(segs []Segment, samples int, energy Analysis) Analysis {
	a := Analysis{FloorDB: energy.FloorDB, PeakDB: energy.PeakDB, Samples: samples}
	for _, s := range segs {
		s.Start, s.End = max(s.Start, 0), min(s.End, samples)
		if s.End > s.Start {
			a.Segments = append(a.Segments, s)
		}
	}
	a.Speech = len(a.Segments) > 0
	return a
}

// modelOnset finds where speech starts in a model's probabilities: the
// first run of chunks at or over the threshold that lasts OnsetMS.
type modelOnset struct {
	s       ModelStream
	chunk   int // samples
	onsetN  int // chunks
	pos     int // chunks seen
	run     int
	first   int // the onset in samples, -1 before any
	err     error
	stopped bool // the onset was found: no more feeding
}

func newModelOnset(s ModelStream, rate int, p Params) *modelOnset {
	chunk := max(s.Chunk(), 1)
	chunkMS := float64(chunk) * 1000 / float64(rate)
	n := max(int(float64(p.OnsetMS)/chunkMS+0.999), 1)
	return &modelOnset{s: s, chunk: chunk, onsetN: n, first: -1}
}

// feed gives the model pcm; false once it failed (the gate falls back to
// energy).
func (m *modelOnset) feed(pcm []byte) bool {
	if m.err != nil {
		return false
	}
	if m.stopped {
		return true
	}
	probs, err := m.s.Feed(pcm)
	if err != nil {
		m.err = err
		return false
	}
	thr := m.s.Threshold()
	for _, p := range probs {
		if p >= thr {
			m.run++
			if m.run >= m.onsetN && m.first < 0 {
				m.first = (m.pos - m.run + 1) * m.chunk
				m.stopped = true
			}
		} else {
			m.run = 0
		}
		m.pos++
	}
	return true
}

// ModelTimeout bounds one exchange with a model's stream.
const ModelTimeout = 2 * time.Second
