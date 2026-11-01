package helper

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

// spawn starts a process. auto marks an automatic restart.
func (s *supervisor) spawn(auto bool) {
	s.mu.Lock()
	if s.stopped {
		s.mu.Unlock()
		return
	}
	s.timer = nil
	s.status.RestartAt = time.Time{}
	if auto {
		s.status.Restarts++
	}
	if s.ready == nil || isClosed(s.ready) {
		s.ready = make(chan struct{})
	}
	if s.cfg.Bin == "" {
		s.failLocked(FailMissing, BinName+" was not found: build it (helper/README.md), or set VOXTNEW_HELPER", "")
		st := s.statusLocked()
		s.mu.Unlock()
		s.publish(st)
		return
	}
	s.mu.Unlock()

	// After an update, the helper on disk is the new version's.
	if s.stale() {
		s.mu.Lock()
		if s.stopped {
			s.mu.Unlock()
			return
		}
		s.failLocked(FailUpdated, errUpdated, "")
		st := s.statusLocked()
		s.mu.Unlock()
		s.publish(st)
		return
	}

	p, err := s.startProcess()
	s.mu.Lock()
	if err != nil {
		s.failLocked(FailStart, err.Error(), "")
		st := s.statusLocked()
		s.mu.Unlock()
		s.publish(st)
		return
	}
	if s.stopped {
		s.mu.Unlock()
		p.intentional.Store(true)
		p.kill()
		return
	}
	s.proc = p
	s.status.State, s.status.PID, s.status.Since = Starting, p.cmd.Process.Pid, time.Now()
	s.status.Err, s.status.Reason, s.status.StderrTail, s.status.Hello = "", "", "", nil
	st := s.statusLocked()
	s.mu.Unlock()
	s.publish(st)
	go s.handshake(p)
}

// failLocked marks the helper failed and wakes callers waiting for it.
// Caller holds s.mu.
func (s *supervisor) failLocked(reason FailReason, msg, tail string) {
	s.status.State, s.status.Reason, s.status.Err, s.status.StderrTail = Failed, reason, msg, tail
	s.status.PID, s.status.Hello, s.status.Since = 0, nil, time.Now()
	if s.ready != nil && !isClosed(s.ready) {
		close(s.ready)
	}
}

func (s *supervisor) startProcess() (*proc, error) {
	cmd := exec.Command(s.cfg.Bin)
	cmd.Dir = filepath.Dir(s.cfg.Bin)
	cmd.Env = append(os.Environ(), s.cfg.Env...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("starting %s: %w", s.cfg.Bin, err)
	}
	p := &proc{s: s, cmd: cmd, stdin: stdin, tail: newLineTail(40), exited: make(chan struct{}), pending: map[string]*call{}}
	var logFile *os.File
	if s.cfg.LogDir != "" {
		logFile = openLog(filepath.Join(s.cfg.LogDir, "helper.log"))
	}
	stderrDone := make(chan struct{})
	go func() {
		defer close(stderrDone)
		sc := bufio.NewScanner(stderr)
		sc.Buffer(make([]byte, 64*1024), 1024*1024)
		for sc.Scan() {
			line := sc.Text()
			p.tail.add(line)
			if logFile != nil {
				fmt.Fprintln(logFile, line)
			}
			if strings.TrimSpace(line) != "" {
				s.emit(core.LogEntry{Time: time.Now(), Level: stderrLevel(line), Source: LogSource, Message: line})
			}
		}
	}()
	go func() {
		p.read(stdout)
		<-stderrDone
		err := cmd.Wait()
		if logFile != nil {
			logFile.Close()
		}
		s.exited(p, err)
	}()
	return p, nil
}

// openLog opens the stderr log for appending, keeping one previous file
// once it passes 10 MB.
func openLog(path string) *os.File {
	if st, err := os.Stat(path); err == nil && st.Size() > 10<<20 {
		os.Rename(path, path+".1")
	}
	os.MkdirAll(filepath.Dir(path), 0o755)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return nil
	}
	return f
}

func (s *supervisor) handshake(p *proc) {
	ctx, cancel := context.WithTimeout(context.Background(), s.cfg.HelloTimeout)
	defer cancel()
	raw, err := p.call(ctx, "", proto.MethodHello, proto.HelloParams{Protocol: proto.Version, App: s.cfg.AppVersion, LogLevel: "info"}, nil, &Trace{Start: time.Now()})
	var hello proto.HelloResult
	if err == nil {
		err = json.Unmarshal(raw, &hello)
	}
	var perr *proto.Error
	switch {
	case p.isClosed():
		return // exited: exited reports it
	case errors.Is(err, ErrExited):
		// The hello's write found the pipe closed: the helper is exiting
		// (one that dies at start), which exited reports as a crash, not
		// as a failed handshake. One that closed its input and still runs
		// is killed when the hello's time is up.
		select {
		case <-p.exited:
			return
		case <-ctx.Done():
		}
		p.killFor(FailHandshake, "helper.hello failed: "+err.Error(), false)
		return
	case errors.Is(err, context.DeadlineExceeded):
		p.killFor(FailHandshake, fmt.Sprintf("the helper did not answer helper.hello within %s", s.cfg.HelloTimeout), false)
		return
	case errors.As(err, &perr) && perr.Code == proto.CodeProtocolMismatch:
		s.protocolMismatch(p, perr.Message)
		return
	case err != nil:
		p.killFor(FailHandshake, "helper.hello failed: "+err.Error(), false)
		return
	case hello.Protocol != proto.Version:
		s.protocolMismatch(p, fmt.Sprintf("it speaks protocol %d; want %d", hello.Protocol, proto.Version))
		return
	}
	p.helloDone.Store(true)
	s.mu.Lock()
	if s.proc != p {
		s.mu.Unlock()
		return
	}
	s.status.State, s.status.Since, s.status.Hello = Ready, time.Now(), &hello
	if hello.PID != 0 {
		s.status.PID = hello.PID
	}
	if !isClosed(s.ready) {
		close(s.ready)
	}
	st := s.statusLocked()
	s.mu.Unlock()
	s.publish(st)
}

// errUpdated is Status.Err of FailUpdated.
const errUpdated = "VoxtNew was updated: restart it to finish the update (the helper on disk is the new version's)"

// stale asks Config.Stale.
func (s *supervisor) stale() bool { return s.cfg.Stale != nil && s.cfg.Stale() }

// protocolMismatch stops a helper that speaks another protocol, which is
// not restarted: after an update the app is to restart (FailUpdated),
// otherwise the helper is to be rebuilt (FailProtocol).
func (s *supervisor) protocolMismatch(p *proc, detail string) {
	if s.stale() {
		p.killFor(FailUpdated, errUpdated, true)
		return
	}
	p.killFor(FailProtocol, "helper version mismatch: rebuild "+BinName+" ("+detail+")", true)
}

// exited handles the end of a process: its calls fail, and an unexpected
// exit marks the helper failed and plans a restart.
func (s *supervisor) exited(p *proc, waitErr error) {
	p.mu.Lock()
	p.closed = true
	calls := p.pending
	p.pending = map[string]*call{}
	reason, msg, noRestart := p.failReason, p.failErr, p.noRestart
	p.mu.Unlock()
	close(p.exited)
	for _, c := range calls {
		c.err = ErrExited
		close(c.done)
	}

	s.mu.Lock()
	if s.proc != p {
		s.mu.Unlock()
		return
	}
	s.proc = nil
	if p.intentional.Load() {
		s.mu.Unlock()
		return
	}
	if reason == "" {
		reason, msg = FailCrashed, "the helper exited"
		if waitErr != nil {
			msg += ": " + waitErr.Error()
		}
	}
	s.failLocked(reason, msg, p.tail.String())
	s.cfg.Logf("helper: %s", msg)
	if !noRestart && !s.stopped {
		now := time.Now()
		s.restarts = slices.DeleteFunc(s.restarts, func(t time.Time) bool { return now.Sub(t) > s.cfg.RestartWindow })
		if len(s.restarts) < s.cfg.MaxRestarts {
			delay := min(time.Second<<len(s.restarts), 30*time.Second)
			s.restarts = append(s.restarts, now)
			s.status.RestartAt = now.Add(delay)
			s.timer = time.AfterFunc(delay, func() { s.spawn(true) })
			s.ready = make(chan struct{}) // calls wait for the restart
		}
	}
	st := s.statusLocked()
	s.mu.Unlock()
	s.publish(st)
}

// readyProc waits for the helper to be ready and returns its process. A
// helper that is starting, or failed with a restart planned, is waited for
// (within ctx); a stopped one, or a failed one that will not restart, is an
// error at once.
func (s *supervisor) readyProc(ctx context.Context) (*proc, error) {
	for {
		s.mu.Lock()
		st, p, ready := s.status, s.proc, s.ready
		s.mu.Unlock()
		switch st.State {
		case Ready:
			if p != nil {
				return p, nil
			}
		case Starting, Failed:
			if st.State == Failed && st.RestartAt.IsZero() {
				break
			}
			select {
			case <-ready:
				continue
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		msg := string(st.State)
		if st.Err != "" {
			msg = st.Err
		}
		return nil, fmt.Errorf("the helper is not ready (%s): %w", msg, core.ErrHelper)
	}
}
