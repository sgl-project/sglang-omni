package llmrt

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/wire"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/resident"
)

// Runtime supervises voxtnew-llm. Nothing runs until the first call;
// goroutine-safe.
type Runtime struct {
	cfg    Config
	events *resident.Queue // Config.OnModel's

	mu       sync.Mutex // guards the fields below
	status   Status
	proc     *proc         // the running process; nil when none
	starting chan struct{} // closed when the current start ends; nil when none
	idle     *time.Timer
	inFlight int // calls between ensure and their end
	stopped  bool
	// mismatch: the process speaks another protocol; it is not started
	// again (rebuild it).
	mismatch string
	// held is the model the process holds (resident.go), heldBytes the
	// memory it reported then; work counts the calls using a model.
	held      string
	heldBytes int64
	work      int
	tracker   resident.Tracker
}

// New returns a Runtime for cfg.
func New(cfg Config) *Runtime {
	cfg.defaults()
	return &Runtime{cfg: cfg, events: newEvents(cfg), status: Status{State: Stopped, Since: time.Now()}}
}

// Status returns a snapshot (memory only).
func (r *Runtime) Status() Status {
	r.mu.Lock()
	defer r.mu.Unlock()
	st := r.status
	if st.Model != nil {
		m := *st.Model
		st.Model = &m
	}
	return st
}

// log hands a line to Config.Log.
func (r *Runtime) log(level core.Severity, msg string) {
	if r.cfg.Log != nil {
		r.cfg.Log(core.LogEntry{Time: time.Now(), Level: level, Source: LogSource, Message: msg})
	}
}

// Call sends one request, starting the process first when none runs, and
// returns its raw result. Errors: *wire.Error for an error response;
// ErrUnavailable when the process cannot start; ErrExited when it died;
// ctx's error when ctx ended (a chat is cancelled first).
func (r *Runtime) Call(ctx context.Context, method string, params any, notify func(wire.Message)) (json.RawMessage, error) {
	p, err := r.ensure(ctx)
	if err != nil {
		return nil, err
	}
	defer r.release()
	return p.call(ctx, method, params, notify)
}

// ensure returns a ready process, starting one (or waiting for the one
// starting), and counts the caller in flight (release ends it).
func (r *Runtime) ensure(ctx context.Context) (*proc, error) {
	for {
		r.mu.Lock()
		switch {
		case r.stopped:
			r.mu.Unlock()
			return nil, ErrStopped
		case r.mismatch != "":
			r.mu.Unlock()
			return nil, fmt.Errorf("%w: %s", ErrUnavailable, r.mismatch)
		case r.proc != nil && r.status.State == Ready:
			r.inFlight++
			if r.idle != nil {
				r.idle.Stop()
			}
			p := r.proc
			r.mu.Unlock()
			return p, nil
		case r.starting != nil:
			wait := r.starting
			r.mu.Unlock()
			select {
			case <-wait:
				continue
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		// Start one; others wait for it.
		done := make(chan struct{})
		r.starting = done
		r.mu.Unlock()
		err := r.start()
		r.mu.Lock()
		r.starting = nil
		close(done)
		r.mu.Unlock()
		if err != nil {
			return nil, err
		}
	}
}

// release ends a call; the last one arms the idle timer.
func (r *Runtime) release() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.inFlight--
	if r.inFlight > 0 || r.proc == nil {
		return
	}
	r.armIdle(r.proc)
}

// armIdle stops p after the idle timeout, from now (r.mu held).
func (r *Runtime) armIdle(p *proc) {
	if r.idle != nil {
		r.idle.Stop()
		r.idle = nil
	}
	d := r.cfg.IdleTimeout
	if d < 0 {
		return
	}
	r.idle = time.AfterFunc(d, func() { r.idleStop(p, d) })
}

// IdleTimeout is how long the process stays without a call (< 0: until
// Stop).
func (r *Runtime) IdleTimeout() time.Duration {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.cfg.IdleTimeout
}

// SetIdleTimeout changes how long the process stays without a call: d <
// 0 until Stop, 0 DefaultIdleTimeout. An unused process counts it from
// now; one in use, from the end of its last call.
func (r *Runtime) SetIdleTimeout(d time.Duration) {
	if d == 0 {
		d = DefaultIdleTimeout
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.cfg.IdleTimeout == d {
		return
	}
	r.cfg.IdleTimeout = d
	if r.proc != nil && r.inFlight == 0 && r.status.State == Ready {
		r.armIdle(r.proc)
	}
}

// idleStop stops p, unused for d, when nothing used it since the timer
// was armed.
func (r *Runtime) idleStop(p *proc, d time.Duration) {
	r.mu.Lock()
	if r.proc != p || r.inFlight > 0 || p.busy() {
		r.mu.Unlock()
		return
	}
	r.mu.Unlock()
	r.log(core.SeverityInfo, fmt.Sprintf("voxtnew-llm unused for %s: stopping it", d))
	r.shutdown(p, Stopped, "")
}

// start spawns a process and handshakes; on success it is r.proc, Ready.
func (r *Runtime) start() error {
	if r.cfg.Bin == "" {
		r.fail(BinName + " was not found: build it (scripts/build-llm.sh), or set VOXTNEW_LLM")
		return fmt.Errorf("%w: %s not found", ErrUnavailable, BinName)
	}
	p, err := r.startProc()
	if err != nil {
		r.fail(err.Error())
		return fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	r.mu.Lock()
	r.proc = p
	r.status.State, r.status.PID, r.status.Since, r.status.Err = Starting, p.cmd.Process.Pid, time.Now(), ""
	r.status.Hello, r.status.Model = nil, nil
	r.status.Starts++
	r.mu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), r.cfg.HelloTimeout)
	defer cancel()
	raw, err := p.call(ctx, wire.MethodHello, wire.HelloParams{Protocol: wire.Version, App: r.cfg.AppVersion}, nil)
	var hello wire.HelloResult
	if err == nil {
		err = json.Unmarshal(raw, &hello)
	}
	var we *wire.Error
	switch {
	case err == nil && hello.Protocol != wire.Version:
		err = fmt.Errorf("it speaks protocol %d; want %d", hello.Protocol, wire.Version)
		fallthrough
	case errors.As(err, &we) && we.Code == wire.CodeProtocolMismatch:
		r.mu.Lock()
		r.mismatch = "version mismatch: rebuild " + BinName + " (" + err.Error() + ")"
		r.mu.Unlock()
	}
	if err != nil {
		msg := "llm.hello failed: " + err.Error()
		if errors.Is(err, context.DeadlineExceeded) {
			msg = fmt.Sprintf("voxtnew-llm did not answer llm.hello within %s", r.cfg.HelloTimeout)
		}
		r.shutdown(p, Failed, msg)
		return fmt.Errorf("%w: %s", ErrUnavailable, msg)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.proc != p || r.stopped {
		return ErrStopped
	}
	r.status.State, r.status.Since, r.status.Hello = Ready, time.Now(), &hello
	return nil
}

// fail marks the runtime failed without a process.
func (r *Runtime) fail(msg string) {
	r.mu.Lock()
	r.status.State, r.status.Err, r.status.PID, r.status.Since = Failed, msg, 0, time.Now()
	r.mu.Unlock()
	r.log(core.SeverityError, msg)
}

// exited is the end of process p (proc.end).
func (r *Runtime) exited(p *proc, msg string) {
	r.mu.Lock()
	if r.proc != p {
		r.mu.Unlock()
		return
	}
	r.gone(p)
	if r.idle != nil {
		r.idle.Stop()
	}
	r.status.PID, r.status.Hello, r.status.Model, r.status.Since = 0, nil, nil, time.Now()
	if p.intentional.Load() {
		r.status.State = Stopped
		r.mu.Unlock()
		return
	}
	if tail := p.tail.String(); tail != "" {
		msg += "\n" + tail
	}
	r.status.State, r.status.Err = Failed, msg
	r.mu.Unlock()
	r.log(core.SeverityError, msg)
}

// shutdown asks p to exit (llm.shutdown), kills it after a grace period,
// and leaves the runtime in state with msg.
func (r *Runtime) shutdown(p *proc, state State, msg string) {
	p.intentional.Store(true)
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	p.call(ctx, wire.MethodShutdown, nil, nil)
	cancel()
	p.stdin.Close()
	select {
	case <-p.exited:
	case <-time.After(3 * time.Second):
		p.kill("did not exit after llm.shutdown")
		<-p.exited
	}
	r.mu.Lock()
	r.gone(p)
	r.status.State, r.status.Err, r.status.PID, r.status.Since = state, msg, 0, time.Now()
	r.status.Hello, r.status.Model = nil, nil
	r.mu.Unlock()
	if msg != "" {
		r.log(core.SeverityError, msg)
	}
}

// Stop stops the process (the app quits); later calls fail with
// ErrStopped.
func (r *Runtime) Stop() {
	r.mu.Lock()
	r.stopped = true
	if r.idle != nil {
		r.idle.Stop()
	}
	p := r.proc
	r.mu.Unlock()
	if p != nil {
		r.shutdown(p, Stopped, "")
	}
}

// Unload stops the process if one runs, freeing its model's memory (a
// model about to be deleted). The next call starts a new one.
func (r *Runtime) Unload() {
	r.mu.Lock()
	p := r.proc
	r.mu.Unlock()
	if p != nil {
		r.shutdown(p, Stopped, "")
	}
}

// Load loads m ahead of its first chat (a warm-up).
func (r *Runtime) Load(ctx context.Context, m wire.Model) (wire.LoadResult, error) {
	raw, err := r.modelCall(ctx, m, wire.MethodLoad, wire.LoadParams{Model: m, CacheLimitMB: r.cfg.CacheLimitMB}, nil)
	var res wire.LoadResult
	if err == nil {
		err = json.Unmarshal(raw, &res)
	}
	if err == nil {
		r.noteModel(&res.Model, res.Memory)
	}
	return res, err
}

// Memory asks a running process for its model and memory; without one it
// returns the zero value (it starts nothing).
func (r *Runtime) Memory(ctx context.Context) (wire.StatusResult, error) {
	r.mu.Lock()
	running := r.proc != nil && r.status.State == Ready
	r.mu.Unlock()
	if !running {
		return wire.StatusResult{}, nil
	}
	raw, err := r.Call(ctx, wire.MethodStatus, nil, nil)
	var res wire.StatusResult
	if err == nil {
		err = json.Unmarshal(raw, &res)
	}
	if err == nil {
		r.noteModel(res.Model, res.Memory)
	}
	return res, err
}

// noteModel keeps what the process said is loaded, and its memory.
func (r *Runtime) noteModel(m *wire.Model, mem wire.Memory) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if m != nil {
		c := *m
		r.status.Model = &c
	} else {
		r.status.Model = nil
	}
	r.status.Memory = mem
}
