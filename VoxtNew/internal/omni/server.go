package omni

import (
	"context"
	"errors"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/resident"
)

// server is one model's server over its life: started on first use,
// stopped when idle, retired (by a switch of speech model or Close), and
// started again after it died. Its process changes; the server stays.
type server struct {
	rt   *Runtime
	spec Spec
	dir  string

	mu      sync.Mutex
	proc    *process // the ready process, nil when none runs
	spawned *process // the process starting, until it is ready or failed
	start   *startOp // the start in progress, shared by its waiters
	leases  int      // uses holding proc
	idle    *time.Timer
	retired bool
	stopped chan struct{} // closed once retired and no process is left
	// loaded is the process that said it is ready, until it exited
	// (Config.OnModel's view: tracker).
	loaded  *process
	tracker resident.Tracker
}

type startOp struct {
	done chan struct{}
	err  error
}

func newServer(rt *Runtime, spec Spec, dir string) *server {
	return &server{rt: rt, spec: spec, dir: dir, stopped: make(chan struct{})}
}

// errRetired: the server was retired while a use waited for it.
var errRetired = errors.New("omni: the server was replaced")

// lease is one use of a server's process; release it once.
type lease struct {
	s    *server
	p    *process
	once sync.Once
}

func (l *lease) endpoint() endpoint { return l.p.ep }

// release ends the use: the last one arms the idle timer, or stops a
// retired server.
func (l *lease) release() {
	l.once.Do(func() { l.s.unlease(l.p) })
}

// broken says the process hung (a stalled Final): it is killed, and the
// next use starts another.
func (l *lease) broken(why string) {
	l.s.rt.logf("warning", "%s stalled (%s): killing it; %s", l.s.spec.ID, why, memoryNote())
	l.p.kill()
}

// acquire leases the running process, starting one when none runs.
func (s *server) acquire(ctx context.Context) (*lease, error) {
	for {
		s.mu.Lock()
		if s.retired {
			s.mu.Unlock()
			return nil, &asr.Error{Kind: asr.KindServer, Provider: Provider, Detail: "the speech model was switched", Err: errRetired}
		}
		if p := s.proc; p != nil && p.alive() {
			s.leases++
			if s.idle != nil {
				s.idle.Stop()
				s.idle = nil
			}
			s.report()
			s.mu.Unlock()
			return &lease{s: s, p: p}, nil
		}
		s.proc = nil
		op := s.start
		if op == nil {
			op = &startOp{done: make(chan struct{})}
			s.start = op
			go s.launch(op)
		}
		s.mu.Unlock()
		select {
		case <-op.done:
			if op.err != nil {
				return nil, op.err
			}
		case <-ctx.Done():
			return nil, asr.NetError(ctx, Provider, ctx.Err())
		}
	}
}

// launch starts a process for op's waiters. It is not bound to their
// contexts: a start nobody waits for any more still ends in a server,
// which the idle timer stops.
func (s *server) launch(op *startOp) {
	s.waitDrained()
	if fn := s.rt.cfg.BeforeLoad; fn != nil {
		ctx, cancel := context.WithTimeout(context.Background(), beforeLoadTimeout)
		fn(ctx, s.spec.ID)
		cancel()
	}
	started := time.Now()
	p, ready, err := spawn(s.rt.cfg, s.spec, s.dir, s.rt.cacheLimit(s.spec))
	if err == nil {
		s.mu.Lock()
		s.spawned = p
		retired := s.retired
		s.mu.Unlock()
		if retired {
			p.kill()
		}
		err = p.waitReady(ready, s.rt.cfg.StartTimeout)
	}
	s.mu.Lock()
	s.start, s.spawned = nil, nil
	switch {
	case err != nil:
		op.err = &asr.Error{Kind: asr.KindServer, Provider: Provider, Detail: "the local speech model did not start: " + err.Error(), Err: err}
	case s.retired:
		op.err = &asr.Error{Kind: asr.KindServer, Provider: Provider, Detail: "the speech model was switched", Err: errRetired}
		go p.stop()
	default:
		s.proc, s.loaded = p, p
		if s.leases == 0 {
			s.armIdle(p)
		}
		s.report()
	}
	s.mu.Unlock()
	if err != nil {
		s.rt.logf("error", "%s did not start: %v", s.spec.ID, err)
		if s.isRetired() {
			s.markStopped()
		}
	} else {
		s.rt.logf("info", "%s ready on %s:%d after %v (pid %d)", s.spec.ID, p.ep.host, p.ep.port, time.Since(started).Round(time.Millisecond), p.ep.pid)
		go s.watch(p)
	}
	close(op.done)
}

// watch notices a process that exits: it is dropped, so the next use
// starts another.
func (s *server) watch(p *process) {
	<-p.exited
	s.mu.Lock()
	if s.loaded == p {
		s.loaded = nil
		s.report()
	}
	unexpected := s.proc == p && !p.stopping()
	if s.proc == p {
		s.proc = nil
	}
	retired := s.retired && s.proc == nil && s.spawned == nil
	s.mu.Unlock()
	if unexpected {
		s.rt.logf("warning", "%s exited unexpectedly (%v): it starts again at the next use; %s", s.spec.ID, p.exitErr, p.tail.String())
		s.rt.forget(s)
	}
	if retired {
		s.markStopped()
	}
}

func (s *server) unlease(p *process) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.leases = max(s.leases-1, 0)
	s.report()
	if s.leases > 0 || s.proc != p {
		return
	}
	if s.retired {
		s.proc = nil
		go s.stopAndMark(p)
		return
	}
	s.armIdle(p)
}

// armIdle stops p after the idle timeout, from now (s.mu held).
func (s *server) armIdle(p *process) {
	if s.idle != nil {
		s.idle.Stop()
		s.idle = nil
	}
	d := s.rt.IdleTimeout()
	if d < 0 {
		return
	}
	s.idle = time.AfterFunc(d, func() {
		s.mu.Lock()
		if s.proc != p || s.leases > 0 {
			s.mu.Unlock()
			return
		}
		s.proc, s.idle = nil, nil
		s.mu.Unlock()
		s.rt.logf("info", "%s unloaded after %v unused", s.spec.ID, d)
		p.stop()
	})
}

// rearm arms the idle timer again with the runtime's idle timeout when
// the server is unused (SetIdleTimeout).
func (s *server) rearm() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.retired || s.proc == nil || s.leases > 0 {
		return
	}
	s.armIdle(s.proc)
}

// retire stops the server once its uses end, for why; no new use gets
// it.
func (s *server) retire(why string) {
	s.mu.Lock()
	s.retired = true
	if s.idle != nil {
		s.idle.Stop()
		s.idle = nil
	}
	p := s.proc
	idle := p != nil && s.leases == 0
	if idle {
		s.proc = nil
	}
	nothing := p == nil && s.spawned == nil && s.start == nil
	s.mu.Unlock()
	switch {
	case idle:
		s.rt.logf("info", "%s stopped: %s", s.spec.ID, why)
		go s.stopAndMark(p)
	case nothing:
		s.markStopped()
	}
}

// shutdown stops the server at once, uses or not, and waits (Close).
func (s *server) shutdown() {
	s.mu.Lock()
	s.retired = true
	if s.idle != nil {
		s.idle.Stop()
		s.idle = nil
	}
	p, sp := s.proc, s.spawned
	s.proc = nil
	s.mu.Unlock()
	if sp != nil {
		sp.kill()
		<-sp.exited
	}
	if p != nil {
		p.stop()
	}
	s.markStopped()
}

// report tells Config.OnModel what changed (s.mu held): the process that
// said it is ready holds the model until it exits, busy while leased.
func (s *server) report() {
	pid := 0
	if s.loaded != nil {
		pid = s.loaded.pidOf()
	}
	s.tracker.Set(s.rt.events, s.spec.ID, pid, s.leases > 0, 0)
}

func (s *server) stopAndMark(p *process) {
	p.stop()
	s.markStopped()
}

func (s *server) markStopped() {
	s.mu.Lock()
	defer s.mu.Unlock()
	select {
	case <-s.stopped:
	default:
		close(s.stopped)
	}
}

func (s *server) isRetired() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.retired
}

func (s *server) running() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.proc != nil && s.proc.alive()
}

// pid is the ready process's id, 0 when none runs.
func (s *server) pid() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.proc == nil || !s.proc.alive() {
		return 0
	}
	return s.proc.pidOf()
}

// waitDrained waits, at most a start's timeout, for the speech models'
// servers being retired to stop, so that two models' weights are never
// in memory at once.
func (s *server) waitDrained() {
	if s.spec.ASR == nil {
		return
	}
	s.rt.mu.Lock()
	var wait []chan struct{}
	for _, o := range s.rt.draining {
		wait = append(wait, o.stopped)
	}
	s.rt.mu.Unlock()
	deadline := time.After(s.rt.cfg.StartTimeout)
	for _, w := range wait {
		select {
		case <-w:
		case <-deadline:
			return
		}
	}
}
