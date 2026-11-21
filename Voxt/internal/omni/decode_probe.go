package omni

import (
	"context"
	"sync"
	"sync/atomic"
)

// decodeEvent is one Final request's timing, without its text or audio:
// when it was submitted to the server, when it finished, and how many of
// the session's requests were still in flight ahead of it (the queue).
type decodeEvent struct {
	Kind        string `json:"kind"`
	SubmittedMS int64  `json:"submitted_ms"`
	FinishedMS  int64  `json:"finished_ms"`
	Inflight    int    `json:"inflight"`
}

// decodeProbe collects a live session's request timings. Measurement
// only: it changes no request, response or timing of the session.
type decodeProbe struct {
	inflight atomic.Int64
	mu       sync.Mutex
	events   []decodeEvent
}

type decodeProbeKey struct{}

// withDecodeProbe attaches p to ctx so every Final request made with it
// is timed.
func withDecodeProbe(ctx context.Context, p *decodeProbe) context.Context {
	return context.WithValue(ctx, decodeProbeKey{}, p)
}

// decodeProbeOf returns the probe attached to ctx, if any.
func decodeProbeOf(ctx context.Context) *decodeProbe {
	p, _ := ctx.Value(decodeProbeKey{}).(*decodeProbe)
	return p
}

// record appends one measured request, in order.
func (p *decodeProbe) record(ev decodeEvent) {
	p.mu.Lock()
	p.events = append(p.events, ev)
	p.mu.Unlock()
}

// Events are the request timings so far, in order.
func (p *decodeProbe) Events() []decodeEvent {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]decodeEvent(nil), p.events...)
}

// decodeKind names a request in its decode event: the batch preview's
// provisional requests, a long session's pieces, everything else final.
func (r finalRequest) decodeKind() string {
	if r.preview {
		return "preview"
	}
	if r.piece {
		return "piece"
	}
	return "final"
}
