package llm

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Waits a progress timeout names (CodeTimeout's "wait").
const (
	WaitFirstToken = "first_token" // no delta within Request.FirstTokenTimeout
	WaitStall      = "stall"       // no delta for Request.StallTimeout after the one before
)

// errNoProgress is the cause of an attempt progress ended.
var errNoProgress = errors.New("llm: no progress")

// progress ends an attempt that makes no progress
// (Request.FirstTokenTimeout, StallTimeout): a timer, armed at the
// attempt's start and again at each delta, cancels the attempt's context.
type progress struct {
	firstToken, stall time.Duration

	mu         sync.Mutex
	cancel     context.CancelCauseFunc
	timer      *time.Timer
	generation int // the armed timer's; an older one that fires does nothing
	start      time.Time
	deltas     int
	expired    string // the wait that ended the attempt ("" while none did)
	waited     time.Duration
}

// begin starts watching an attempt: the returned context ends when it
// makes no progress.
func (p *progress) begin(ctx context.Context) context.Context {
	actx, cancel := context.WithCancelCause(ctx)
	p.mu.Lock()
	defer p.mu.Unlock()
	p.cancel, p.start, p.deltas, p.expired, p.waited = cancel, time.Now(), 0, "", 0
	p.arm(WaitFirstToken, p.firstToken)
	return actx
}

// arm sets the timer to end the attempt after d without a delta (p.mu
// held; d 0: no bound).
func (p *progress) arm(wait string, d time.Duration) {
	if p.timer != nil {
		p.timer.Stop()
		p.timer = nil
	}
	p.generation++
	if d <= 0 {
		return
	}
	generation := p.generation
	p.timer = time.AfterFunc(d, func() {
		p.mu.Lock()
		defer p.mu.Unlock()
		if generation != p.generation || p.cancel == nil {
			return
		}
		p.expired, p.waited = wait, d
		p.cancel(errNoProgress)
	})
}

// tick counts a delta.
func (p *progress) tick() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.cancel == nil || p.expired != "" {
		return
	}
	p.deltas++
	p.arm(WaitStall, p.stall)
}

// end stops watching the attempt and returns the wait that ended it, ""
// for none.
func (p *progress) end() (wait string, waited time.Duration, deltas int, elapsed time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.arm("", 0)
	p.cancel(nil)
	p.cancel = nil
	return p.expired, p.waited, p.deltas, time.Since(p.start)
}

// progressErr is the timeout of an attempt that made no progress: a
// deadline, as the attempt's own timeout is (cause is what the backend
// returned on its cancel).
func progressErr(provider, wait string, waited time.Duration, deltas int, elapsed time.Duration, cause error) *Error {
	e := newErr(CodeTimeout, "provider", provider, "seconds", fmt.Sprintf("%.0f", waited.Seconds()), "wait", wait,
		"deltas", fmt.Sprint(deltas), "elapsed", fmt.Sprintf("%.1f", elapsed.Seconds()))
	e.Err = fmt.Errorf("%w (%v)", context.DeadlineExceeded, cause)
	return e
}
