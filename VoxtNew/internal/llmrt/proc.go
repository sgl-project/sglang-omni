package llmrt

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/wire"
)

// proc is one voxtnew-llm process.
type proc struct {
	rt     *Runtime
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	tail   *lineTail
	exited chan struct{}
	ids    atomic.Int64

	writeMu sync.Mutex
	mu      sync.Mutex
	pending map[string]*call
	closed  bool
	// intentional: stopped on purpose (idle, Stop), not a failure.
	intentional atomic.Bool
	killReason  string
}

type call struct {
	method string
	notify func(wire.Message)
	done   chan struct{}
	resp   wire.Message
	err    error
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

// startProc spawns the process and its readers; onExit runs once it has
// ended and its calls have failed.
func (r *Runtime) startProc() (*proc, error) {
	cmd := exec.Command(r.cfg.Bin)
	cmd.Dir = filepath.Dir(r.cfg.Bin)
	cmd.Env = append(os.Environ(), r.cfg.Env...)
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
		return nil, fmt.Errorf("starting %s: %w", r.cfg.Bin, err)
	}
	p := &proc{rt: r, cmd: cmd, stdin: stdin, tail: newLineTail(40), exited: make(chan struct{}), pending: map[string]*call{}}
	var logFile *os.File
	if r.cfg.LogDir != "" {
		logFile = openLog(filepath.Join(r.cfg.LogDir, "llm.log"))
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
				r.log(stderrLevel(line), line)
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
		p.end(err)
	}()
	return p, nil
}

// read routes stdout: responses to their calls, a job's notifications to
// its call, log lines to the log.
func (p *proc) read(stdout io.Reader) {
	br := bufio.NewReaderSize(stdout, 64*1024)
	for {
		line, err := readLine(br)
		if len(line) > 0 {
			p.handle(line)
		}
		if err != nil {
			if err != io.EOF {
				p.tail.add("stdout: " + err.Error())
			}
			io.Copy(io.Discard, br)
			return
		}
	}
}

// readLine reads one line of at most wire.MaxLine bytes; a longer one is
// skipped.
func readLine(r *bufio.Reader) ([]byte, error) {
	var buf []byte
	for {
		chunk, isPrefix, err := r.ReadLine()
		if len(buf)+len(chunk) <= wire.MaxLine {
			buf = append(buf, chunk...)
		} else {
			buf = buf[:0:0]
		}
		if err != nil || !isPrefix {
			return buf, err
		}
	}
}

func (p *proc) handle(line []byte) {
	m, err := wire.Decode(line)
	if err != nil {
		p.tail.add("stdout: " + string(line))
		return
	}
	switch m.Kind() {
	case wire.KindResponse:
		p.mu.Lock()
		c := p.pending[m.ID]
		delete(p.pending, m.ID)
		p.mu.Unlock()
		if c != nil {
			c.resp = m
			close(c.done)
		}
	case wire.KindNotification:
		var ref struct {
			Job string `json:"job"`
		}
		m.DecodeParams(&ref)
		if m.Method == wire.MethodLog {
			var lp wire.LogParams
			if m.DecodeParams(&lp) == nil {
				p.rt.log(severity(lp.Level), lp.Message)
			}
			return
		}
		if ref.Job == "" {
			return
		}
		p.mu.Lock()
		c := p.pending[ref.Job]
		p.mu.Unlock()
		if c != nil && c.notify != nil {
			c.notify(m)
		}
	}
}

// end fails the process's calls and tells the runtime.
func (p *proc) end(waitErr error) {
	p.mu.Lock()
	p.closed = true
	calls := p.pending
	p.pending = map[string]*call{}
	reason := p.killReason
	p.mu.Unlock()
	msg := "voxtnew-llm exited"
	switch {
	case reason != "":
		msg += ": " + reason
	case waitErr != nil:
		msg += ": " + waitErr.Error()
	}
	close(p.exited)
	// The runtime first, so that a failed call finds it without the
	// process (its status, the model unloaded).
	p.rt.exited(p, msg)
	for _, c := range calls {
		c.err = ErrExited
		close(c.done)
	}
}

func (p *proc) kill(reason string) {
	p.mu.Lock()
	if p.killReason == "" {
		p.killReason = reason
	}
	p.mu.Unlock()
	if p.cmd.Process != nil {
		p.cmd.Process.Kill()
	}
}

func (p *proc) write(m wire.Message) error {
	line, err := wire.Encode(m)
	if err != nil {
		return err
	}
	if len(line) > wire.MaxLine {
		return fmt.Errorf("request of %d bytes is over the protocol's limit: %w", len(line), core.ErrInvalid)
	}
	p.writeMu.Lock()
	defer p.writeMu.Unlock()
	if _, err := p.stdin.Write(line); err != nil {
		return fmt.Errorf("writing to voxtnew-llm: %v: %w", err, ErrExited)
	}
	return nil
}

// call sends a request and waits for its response. notify receives a
// job's notifications on the reader goroutine; it must not block.
// Cancelling ctx sends job.cancel for a job and waits for its answer
// (killing the process after Config.CancelGrace); a control request is
// abandoned.
func (p *proc) call(ctx context.Context, method string, params any, notify func(wire.Message)) (json.RawMessage, error) {
	id := fmt.Sprintf("c%d", p.ids.Add(1))
	req, err := wire.NewRequest(id, method, params)
	if err != nil {
		return nil, fmt.Errorf("%s params: %w", method, err)
	}
	c := &call{method: method, notify: notify, done: make(chan struct{})}
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil, ErrExited
	}
	p.pending[id] = c
	p.mu.Unlock()
	if err := p.write(req); err != nil {
		p.drop(id)
		return nil, err
	}
	select {
	case <-c.done:
		return c.result()
	case <-ctx.Done():
	}
	if !wire.IsJob(method) {
		p.drop(id)
		return nil, ctx.Err()
	}
	go func() {
		cctx, cancel := context.WithTimeout(context.Background(), p.rt.cfg.CancelGrace)
		defer cancel()
		p.call(cctx, wire.MethodJobCancel, wire.CancelParams{Job: id}, nil)
	}()
	grace := time.NewTimer(p.rt.cfg.CancelGrace)
	defer grace.Stop()
	select {
	case <-c.done:
	case <-grace.C:
		p.kill(fmt.Sprintf("chat %s did not stop within %s of being cancelled", id, p.rt.cfg.CancelGrace))
		<-c.done
	}
	if _, err := c.result(); err != nil {
		return nil, fmt.Errorf("%w (%v)", ctx.Err(), err)
	}
	return nil, ctx.Err()
}

func (p *proc) drop(id string) {
	p.mu.Lock()
	delete(p.pending, id)
	p.mu.Unlock()
}

func (p *proc) busy() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.pending) > 0
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

// stderrLevel guesses the level of a free-form stderr line.
func stderrLevel(line string) core.Severity {
	l := strings.ToLower(line)
	switch {
	case strings.Contains(l, "fatal") || strings.Contains(l, "error"):
		return core.SeverityError
	case strings.Contains(l, "warning"):
		return core.SeverityWarning
	}
	return core.SeverityInfo
}

func severity(level string) core.Severity {
	switch s := core.Severity(level); s {
	case core.SeverityDebug, core.SeverityInfo, core.SeverityWarning, core.SeverityError:
		return s
	}
	return core.SeverityInfo
}

// lineTail keeps the last lines written to it.
type lineTail struct {
	mu    sync.Mutex
	max   int
	lines []string
}

func newLineTail(max int) *lineTail { return &lineTail{max: max} }

func (t *lineTail) add(line string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.lines = append(t.lines, line)
	if len(t.lines) > t.max {
		t.lines = t.lines[len(t.lines)-t.max:]
	}
}

func (t *lineTail) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return strings.Join(t.lines, "\n")
}
