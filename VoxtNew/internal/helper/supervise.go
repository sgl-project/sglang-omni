package helper

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"os/exec"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

// LogSource is the source of the helper's lines in the logs.
const LogSource = "helper"

// New returns a Supervisor for cfg. Nothing runs until Start.
func New(cfg Config) Supervisor {
	if cfg.HelloTimeout == 0 {
		cfg.HelloTimeout = 10 * time.Second
	}
	if cfg.MaxRestarts == 0 {
		cfg.MaxRestarts = 3
	}
	if cfg.RestartWindow == 0 {
		cfg.RestartWindow = 10 * time.Minute
	}
	if cfg.CancelGrace == 0 {
		cfg.CancelGrace = 3 * time.Second
	}
	if cfg.Logf == nil {
		cfg.Logf = log.Printf
	}
	if cfg.SlowCall == 0 {
		cfg.SlowCall = 500 * time.Millisecond
	}
	return &supervisor{cfg: cfg, status: Status{State: Stopped, Since: time.Now()}}
}

type supervisor struct {
	cfg    Config
	ctrlID atomic.Int64

	mu       sync.Mutex // guards the fields below
	status   Status
	proc     *proc         // the running process; nil when none
	ready    chan struct{} // closed when the current start succeeded or failed
	restarts []time.Time   // automatic restarts within the window
	timer    *time.Timer   // the pending automatic restart
	stopped  bool
}

// proc is one helper process.
type proc struct {
	s      *supervisor
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	tail   *lineTail
	exited chan struct{}

	writeMu sync.Mutex
	mu      sync.Mutex
	pending map[string]*call
	closed  bool

	helloDone   atomic.Bool
	intentional atomic.Bool // stopped on purpose: no failure, no restart
	// Set before the process is killed for a reason of the supervisor.
	failReason FailReason
	failErr    string
	noRestart  bool
}

type call struct {
	method   string
	notify   func(proto.Message)
	done     chan struct{}
	resp     proto.Message
	err      error
	answered time.Time // when the reader read the response line
}

func (c *call) result() (json.RawMessage, error) {
	if c.err != nil {
		return nil, c.err
	}
	if c.resp.Error != nil {
		return nil, c.resp.Error
	}
	return c.resp.Result, nil
}

func (s *supervisor) Start(ctx context.Context) error {
	s.mu.Lock()
	s.stopped = false
	idle := s.proc == nil && s.timer == nil
	s.mu.Unlock()
	if idle {
		s.spawn(false)
	}
	return nil
}

func (s *supervisor) publish(st Status) { s.cfg.Bus.Publish(StatusChanged{Status: st}) }

// statusLocked is a copy of the status to publish. Caller holds s.mu.
func (s *supervisor) statusLocked() Status {
	st := s.status
	if st.Hello != nil {
		h := *st.Hello
		h.Methods = slices.Clone(h.Methods)
		st.Hello = &h
	}
	if s.proc != nil {
		st.Running = s.proc.running()
	}
	return st
}

func (s *supervisor) Status() Status {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.statusLocked()
}

func (s *supervisor) Call(ctx context.Context, id, method string, params any, notify func(proto.Message)) (json.RawMessage, error) {
	tr := Trace{ID: id, Method: method, Start: time.Now()}
	p, err := s.readyProc(ctx)
	tr.Ready = time.Since(tr.Start)
	var raw json.RawMessage
	if err == nil {
		raw, err = p.call(ctx, id, method, params, notify, &tr)
	}
	tr.Returned = time.Since(tr.Start)
	if err != nil {
		tr.Err = err.Error()
	}
	s.traced(tr)
	return raw, err
}

// traced hands a finished call's trace to Config.OnTrace, and logs a
// control request slower than Config.SlowCall with its phases.
func (s *supervisor) traced(tr Trace) {
	if s.cfg.OnTrace != nil {
		s.cfg.OnTrace(tr)
	}
	if tr.Returned >= s.cfg.SlowCall && !proto.IsJob(tr.Method) {
		s.emit(core.LogEntry{Time: time.Now(), Level: core.SeverityWarning, Source: LogSource, Message: "slow call: " + tr.String()})
	}
}

func (s *supervisor) Restart() error {
	s.mu.Lock()
	s.stopped = false
	if s.timer != nil {
		s.timer.Stop()
		s.timer = nil
	}
	s.restarts = nil
	s.status.Restarts = 0
	p := s.proc
	s.proc = nil
	// Callers wait for the new process instead of failing.
	s.status.State, s.status.Since = Starting, time.Now()
	if s.ready == nil || isClosed(s.ready) {
		s.ready = make(chan struct{})
	}
	s.mu.Unlock()
	if p != nil {
		p.intentional.Store(true)
		p.kill()
		<-p.exited
	}
	s.spawn(false)
	return nil
}

func (s *supervisor) Stop() {
	s.mu.Lock()
	s.stopped = true
	if s.timer != nil {
		s.timer.Stop()
		s.timer = nil
	}
	p := s.proc
	if p != nil {
		p.intentional.Store(true)
	}
	s.proc = nil
	s.status.State, s.status.PID, s.status.Hello = Stopped, 0, nil
	s.status.RestartAt, s.status.Since = time.Time{}, time.Now()
	if s.ready != nil && !isClosed(s.ready) {
		close(s.ready)
	}
	st := s.statusLocked()
	s.mu.Unlock()
	if p != nil {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		p.call(ctx, "", proto.MethodShutdown, nil, nil, &Trace{Start: time.Now()})
		cancel()
		p.stdin.Close()
		select {
		case <-p.exited:
		case <-time.After(3 * time.Second):
			p.kill()
			<-p.exited
		}
	}
	s.publish(st)
}
