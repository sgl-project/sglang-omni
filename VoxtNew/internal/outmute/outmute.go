// Package outmute mutes the system's audio output while a session records
// (Settings → Audio → Mute system audio while recording, on by default),
// so that what the speakers play (a video, music, a call) does not reach
// the microphone. Only the mute of the default output device changes,
// through the helper's output.* methods (docs/helper-protocol.md); nothing
// of the system's sound is read, and no permission is needed.
//
// The helper does, this package decides. A Controller takes a recording's
// output (Engage, once the start cue has played) and gives it back (Release,
// the moment the recording stops, is cancelled or fails), and follows the
// output meanwhile:
//
//   - a device the user muted is never claimed, so it is never unmuted;
//   - a device that cannot be muted leaves the recording as it is, and the
//     overlay says so (Config.OnUnmutable);
//   - another default output: the old one is given back, the new one taken;
//   - the user unmuting it: it is theirs for the rest of the recording (not
//     muted again, not touched at the end, and no other device taken);
//   - the Mac going to sleep: the helper gives its claims back itself
//     (output.released), and the recording keeps the output it has.
//
// Crash safety lives in the helper's claims: it writes each to a journal
// (Config.Journal) before it mutes, undoes them when the app goes away
// (stdin closing, helper.shutdown, a signal) and, at each new helper, the
// Controller asks it to undo the claims of helpers that are gone
// (HelperReady → output.recover): a crash of the app, of the helper or of
// both never leaves the output muted past the next start.
//
// Calls run in order on a goroutine of the Controller's (its methods
// never block), so that a release cannot overtake the mute it undoes;
// session ids make late calls harmless: a session released once is never
// engaged again.
//
// Imports: helper/proto.
package outmute

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

// Output is the helper's output.* methods (HelperOutput).
type Output interface {
	Watch(ctx context.Context, on bool) (proto.OutputState, error)
	Mute(ctx context.Context, p proto.OutputMuteParams) (proto.OutputMuteResult, error)
	Release(ctx context.Context, p proto.OutputReleaseParams) (proto.OutputReleaseResult, error)
	Recover(ctx context.Context, p proto.OutputRecoverParams) (proto.OutputRecoverResult, error)
}

// Config is what a Controller needs; Output is required.
type Config struct {
	Output Output
	// Journal is the directory the helper keeps its claims in
	// (~/Library/Application Support/VoxtNew/output-mute; "" keeps none).
	Journal string
	// Log writes a line (no device names: they may name a person).
	Log func(level, msg string)
	// OnUnmutable is told that session id's output could not be muted
	// as it started: the recording goes on, and the overlay says so. It
	// runs on the Controller's goroutine and must not block.
	OnUnmutable func(id string)
	// Timeout bounds each helper call (DefaultTimeout).
	Timeout time.Duration
}

// DefaultTimeout bounds a helper call: Core Audio answers in
// milliseconds.
const DefaultTimeout = 2 * time.Second

// Status is the Controller's state (tests, diagnostics).
type Status struct {
	// Supported: the helper has the output methods.
	Supported bool
	// Session is the recording whose output is followed ("" for none);
	// Device the output it watches, Claim the one it muted (nil: none);
	// Uncertain: the mute's answer did not come (the release still goes,
	// and the helper undoes it only if it did mute).
	Session   string
	Device    *proto.OutputDevice
	Claim     *proto.OutputDevice
	Uncertain bool
	// Yielded: the user unmuted during this recording.
	Yielded bool
}

// Controller decides when the output is muted. Goroutine-safe; its
// methods queue the work and return at once.
type Controller struct {
	cfg Config

	mu     sync.Mutex
	queue  []func()
	busy   bool
	idle   *sync.Cond
	closed bool

	// The worker's (one goroutine at a time runs the queue).
	st       Status
	watching bool
	told     bool     // the log said this helper cannot mute
	ended    []string // the sessions released lately, newest last
}

// New returns a Controller; it does nothing until HelperReady.
func New(cfg Config) *Controller {
	if cfg.Log == nil {
		cfg.Log = func(string, string) {}
	}
	if cfg.OnUnmutable == nil {
		cfg.OnUnmutable = func(string) {}
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = DefaultTimeout
	}
	c := &Controller{cfg: cfg}
	c.idle = sync.NewCond(&c.mu)
	return c
}

// do queues fn for the worker.
func (c *Controller) do(fn func()) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.queue = append(c.queue, fn)
	if !c.busy {
		c.busy = true
		go c.drain()
	}
}

func (c *Controller) drain() {
	for {
		c.mu.Lock()
		if len(c.queue) == 0 {
			c.busy = false
			c.idle.Broadcast()
			c.mu.Unlock()
			return
		}
		fn := c.queue[0]
		c.queue = c.queue[1:]
		c.mu.Unlock()
		fn()
	}
}

// Wait returns once the queue is empty (tests).
func (c *Controller) Wait() {
	c.mu.Lock()
	for c.busy {
		c.idle.Wait()
	}
	c.mu.Unlock()
}

// Status is the state once what was queued before ran.
func (c *Controller) Status() Status {
	ch := make(chan Status, 1)
	c.do(func() {
		s := c.st
		s.Device, s.Claim = clone(s.Device), clone(s.Claim)
		ch <- s
	})
	return <-ch
}

func clone(d *proto.OutputDevice) *proto.OutputDevice {
	if d == nil {
		return nil
	}
	v := *d
	return &v
}

func (c *Controller) logf(level, format string, args ...any) {
	c.cfg.Log(level, "output: "+fmt.Sprintf(format, args...))
}

func (c *Controller) ctx() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), c.cfg.Timeout)
}

// HelperReady: a helper (a new one) is ready; supported says whether it
// has the output methods (hello's methods). Claims of the helpers before
// it are undone first (output.recover): they are in the journal.
func (c *Controller) HelperReady(supported bool) {
	c.do(func() {
		// What the last helper muted is its journal's now.
		c.st.Supported, c.st.Claim, c.st.Uncertain, c.st.Device, c.watching, c.told = supported, nil, false, nil, false, false
		if !supported {
			c.logf("info", "the helper cannot mute the output (an older one): recordings leave it as it is")
			c.told = true
			return
		}
		if c.cfg.Journal == "" {
			return // no journal (tests): nothing to recover
		}
		ctx, cancel := c.ctx()
		defer cancel()
		res, err := c.cfg.Output.Recover(ctx, proto.OutputRecoverParams{Journal: c.cfg.Journal})
		switch {
		case err != nil:
			c.logf("warning", "recovering the mutes of an earlier helper: %v", err)
		case len(res.Restored) > 0:
			c.logf("info", "unmuted %d output device(s) an earlier helper had muted (it stopped while recording)", len(res.Restored))
		}
	})
}

// HelperGone: the helper stopped or crashed. Its claims are in its
// journal, for the next one.
func (c *Controller) HelperGone() {
	c.do(func() {
		if c.st.Claim != nil {
			c.logf("warning", "the helper stopped while output device %d was muted: the next helper unmutes it", c.st.Claim.ID)
		}
		c.st.Supported, c.st.Claim, c.st.Uncertain, c.st.Device, c.watching = false, nil, false, nil, false
	})
}

// Engage mutes the output for session id, which records (after its start
// cue). A session released already, or engaged already, is left alone.
func (c *Controller) Engage(id string) {
	c.do(func() { c.engage(id) })
}

// Release gives session id's output back (its recording stopped, was
// cancelled or failed), then runs then (may be nil) on the Controller's
// goroutine: the stop cue plays after it, to be heard.
func (c *Controller) Release(id string, then func()) {
	c.do(func() {
		if id != "" && id == c.st.Session {
			c.disengage()
		}
		c.markEnded(id)
		if then != nil {
			then()
		}
	})
}

// Changed is an output.changed notification.
func (c *Controller) Changed(s proto.OutputState) {
	c.do(func() { c.changed(s) })
}

// Released is an output.released notification: the helper gave its
// claims back itself (the Mac sleeps). The session keeps the output as it
// is from then on.
func (c *Controller) Released(p proto.OutputReleasedParams) {
	c.do(func() {
		if cl := c.st.Claim; cl != nil {
			for _, d := range p.Devices {
				if d.UID == cl.UID {
					c.st.Claim, c.st.Uncertain = nil, false
				}
			}
		}
		if id := c.st.Session; id != "" {
			c.logf("info", "the helper unmuted the output (%s): it stays as it is for the rest of session %s", p.Reason, id)
			c.disengage()
			c.markEnded(id)
		}
	})
}

// Close gives the output back (quitting) and waits for it, at most
// timeout; nothing is engaged after it.
func (c *Controller) Close(timeout time.Duration) {
	done := make(chan struct{})
	c.mu.Lock()
	c.closed = true
	c.mu.Unlock()
	c.do(func() {
		if id := c.st.Session; id != "" {
			c.disengage()
			c.markEnded(id)
		}
		close(done)
	})
	select {
	case <-done:
	case <-time.After(timeout):
		c.logf("warning", "the output was not given back within %v of quitting: the helper unmutes it as it exits", timeout)
	}
}

func (c *Controller) isClosed() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.closed
}

// markEnded remembers a released session, so that a late Engage of it
// does nothing.
func (c *Controller) markEnded(id string) {
	if id == "" {
		return
	}
	c.ended = append(c.ended, id)
	if len(c.ended) > 16 {
		c.ended = c.ended[1:]
	}
}

func (c *Controller) isEnded(id string) bool {
	for _, e := range c.ended {
		if e == id {
			return true
		}
	}
	return false
}

func (c *Controller) engage(id string) {
	if id == "" || id == c.st.Session || c.isEnded(id) || c.isClosed() {
		return
	}
	if old := c.st.Session; old != "" {
		c.disengage()
		c.markEnded(old)
	}
	if !c.st.Supported {
		if !c.told {
			c.logf("info", "no helper that can mute the output: session %s leaves it as it is", id)
			c.told = true
		}
		return
	}
	c.st.Session, c.st.Yielded = id, false
	ctx, cancel := c.ctx()
	st, err := c.cfg.Output.Watch(ctx, true)
	cancel()
	if err != nil {
		c.logf("warning", "reading the output: %v: session %s leaves it as it is", err, id)
		return
	}
	c.watching = true
	c.adopt(st, true)
}

// adopt takes the output of state s, when it can; first: as the
// recording starts (a device that cannot be muted is said in the
// overlay).
func (c *Controller) adopt(s proto.OutputState, first bool) {
	c.st.Device = clone(s.Device)
	d := s.Device
	switch {
	case d == nil:
		c.logf("info", "no output device: nothing to mute")
	case s.IsMuted():
		c.logf("info", "output device %d is muted already (the user's): left as it is", d.ID)
	case s.Muted == nil || !s.Settable:
		c.logf("info", "output device %d has no mute VoxtNew can set: the recording goes on", d.ID)
		c.unmutable(first)
	default:
		c.take(*d, first)
	}
}

func (c *Controller) unmutable(first bool) {
	if first {
		c.cfg.OnUnmutable(c.st.Session)
	}
}

// take mutes d and claims it.
func (c *Controller) take(d proto.OutputDevice, first bool) {
	ctx, cancel := c.ctx()
	res, err := c.cfg.Output.Mute(ctx, proto.OutputMuteParams{Device: d, Journal: c.cfg.Journal})
	cancel()
	var pe *proto.Error
	switch {
	case err == nil && res.Took:
		c.st.Claim = &d
		c.logf("info", "muted output device %d for session %s", d.ID, c.st.Session)
	case err == nil:
		c.logf("info", "output device %d was muted meanwhile (the user's): left as it is", d.ID)
	case errors.As(err, &pe) && pe.Code == proto.CodeUnsupported:
		c.logf("info", "output device %d cannot be muted (%s): the recording goes on", d.ID, pe.Message)
		c.unmutable(first)
	case errors.As(err, &pe) && pe.Code == proto.CodeDeviceUnavailable:
		c.logf("info", "output device %d went away before it was muted", d.ID)
	case errors.As(err, &pe) && pe.Code == proto.CodeInternal:
		c.logf("warning", "output device %d was not muted: %s", d.ID, pe.Message)
		c.unmutable(first)
	default:
		// No answer: the helper may have muted it. The release still
		// goes, and the helper undoes only a mute it made.
		c.st.Claim, c.st.Uncertain = &d, true
		c.logf("warning", "muting output device %d: %v (it is released at the end in case it was muted)", d.ID, err)
	}
}

// giveBack releases the claim; unmute unless the user unmuted it.
func (c *Controller) giveBack(unmute bool) {
	cl := c.st.Claim
	if cl == nil {
		return
	}
	c.st.Claim, c.st.Uncertain = nil, false
	ctx, cancel := c.ctx()
	res, err := c.cfg.Output.Release(ctx, proto.OutputReleaseParams{Device: *cl, Unmute: unmute})
	cancel()
	switch {
	case err != nil:
		c.logf("warning", "unmuting output device %d: %v (the helper unmutes it when it exits; check the output if it stays muted)", cl.ID, err)
	case res.Unmuted:
		c.logf("info", "unmuted output device %d", cl.ID)
	case unmute:
		c.logf("info", "output device %d was not muted any more, or is gone: nothing to unmute", cl.ID)
	}
}

// disengage gives the session's output back and stops watching.
func (c *Controller) disengage() {
	c.giveBack(true)
	if c.watching {
		ctx, cancel := c.ctx()
		if _, err := c.cfg.Output.Watch(ctx, false); err != nil {
			c.logf("warning", "no longer watching the output: %v", err)
		}
		cancel()
	}
	c.st.Session, c.st.Device, c.st.Yielded, c.watching = "", nil, false, false
}

func (c *Controller) changed(s proto.OutputState) {
	if c.st.Session == "" || !c.watching {
		return // late: the recording ended
	}
	// The notification is a hint: it may be older than what this queue
	// did since (a mute, a release). The output as it is now decides.
	ctx, cancel := c.ctx()
	if now, err := c.cfg.Output.Watch(ctx, true); err == nil {
		s = now
	}
	cancel()
	if !sameDevice(s.Device, c.st.Device) {
		c.giveBack(true)
		if c.st.Yielded {
			c.st.Device = clone(s.Device)
			c.logf("info", "the output changed: left as it is (the user unmuted during session %s)", c.st.Session)
			return
		}
		c.logf("info", "the output changed during session %s", c.st.Session)
		c.adopt(s, false)
		return
	}
	cl := c.st.Claim
	if cl == nil || s.Muted == nil {
		return
	}
	switch {
	case *s.Muted && c.st.Uncertain:
		c.st.Uncertain = false // it was muted: the claim is the helper's
	case !*s.Muted:
		// The user unmuted it: theirs until the recording ends, and not
		// touched at its end.
		c.logf("info", "output device %d was unmuted during session %s (by the user): left as it is", cl.ID, c.st.Session)
		c.giveBack(false)
		c.st.Yielded = true
	}
}

// sameDevice: the same device by its UID (nil for none).
func sameDevice(a, b *proto.OutputDevice) bool {
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return a.UID == b.UID
}
