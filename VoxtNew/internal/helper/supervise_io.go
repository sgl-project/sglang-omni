package helper

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

func isClosed(ch chan struct{}) bool {
	select {
	case <-ch:
		return true
	default:
		return false
	}
}

// read routes the helper's stdout: responses to their calls, a job's
// notifications to its call's notify, others to OnNotify, log lines to
// the log.
func (p *proc) read(stdout io.Reader) {
	r := bufio.NewReaderSize(stdout, 64*1024)
	for {
		line, err := readLine(r)
		if len(line) > 0 {
			p.handleLine(line, time.Now())
		}
		if err != nil {
			if err != io.EOF {
				p.tail.add("stdout: " + err.Error())
			}
			io.Copy(io.Discard, r) // drain, so the process is not blocked writing
			return
		}
	}
}

// readLine reads one line of at most proto.MaxLine bytes; a longer one is
// skipped.
func readLine(r *bufio.Reader) ([]byte, error) {
	var buf []byte
	for {
		chunk, isPrefix, err := r.ReadLine()
		if len(buf)+len(chunk) <= proto.MaxLine {
			buf = append(buf, chunk...)
		} else {
			buf = buf[:0:0]
		}
		if err != nil || !isPrefix {
			return buf, err
		}
	}
}

// handleLine routes one line, read at at.
func (p *proc) handleLine(line []byte, at time.Time) {
	m, err := proto.Decode(line)
	if err != nil {
		p.tail.add("stdout: " + string(line))
		return
	}
	switch m.Kind() {
	case proto.KindResponse:
		p.mu.Lock()
		c := p.pending[m.ID]
		delete(p.pending, m.ID)
		p.mu.Unlock()
		if c != nil {
			c.resp, c.answered = m, at
			close(c.done)
		}
	case proto.KindNotification:
		var ref struct {
			Job string `json:"job"`
		}
		m.DecodeParams(&ref)
		if m.Method == proto.MethodLog {
			p.log(m)
		}
		var c *call
		if ref.Job != "" {
			p.mu.Lock()
			c = p.pending[ref.Job]
			p.mu.Unlock()
		}
		switch {
		case c != nil:
			if c.notify != nil {
				c.notify(m)
			}
		case ref.Job != "":
			// A job that answered already (or was never ours): dropped.
		case m.Method != proto.MethodLog && p.s.cfg.OnNotify != nil:
			p.s.cfg.OnNotify(m)
		}
	}
}

func (p *proc) log(m proto.Message) {
	var lp proto.LogParams
	if m.DecodeParams(&lp) != nil {
		return
	}
	level := core.Severity(lp.Level)
	switch level {
	case core.SeverityDebug, core.SeverityInfo, core.SeverityWarning, core.SeverityError:
	default:
		level = core.SeverityInfo
	}
	p.s.emit(core.LogEntry{Time: time.Now(), Level: level, Source: LogSource, Message: lp.Message})
}

// emit hands a log line to Config.Log, or the bus.
func (s *supervisor) emit(e core.LogEntry) {
	if s.cfg.Log != nil {
		s.cfg.Log(e)
		return
	}
	s.cfg.Bus.Publish(core.LogAppended{Entry: e})
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

func (p *proc) isClosed() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.closed
}

func (p *proc) running() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	var ids []string
	for id, c := range p.pending {
		if proto.IsJob(c.method) {
			ids = append(ids, id)
		}
	}
	sort.Strings(ids)
	return ids
}

// killFor kills the process for a reason of the supervisor's; exited
// reports it (and restarts it unless noRestart).
func (p *proc) killFor(reason FailReason, msg string, noRestart bool) {
	p.mu.Lock()
	if p.failReason == "" {
		p.failReason, p.failErr, p.noRestart = reason, msg, noRestart
	}
	p.mu.Unlock()
	p.kill()
}

func (p *proc) kill() {
	if p.cmd.Process != nil {
		p.cmd.Process.Kill()
	}
}

func (p *proc) write(m proto.Message) error {
	line, err := proto.Encode(m)
	if err != nil {
		return err
	}
	if len(line) > proto.MaxLine {
		return fmt.Errorf("request of %d bytes is over the protocol's limit", len(line))
	}
	p.writeMu.Lock()
	defer p.writeMu.Unlock()
	if _, err := p.stdin.Write(line); err != nil {
		return fmt.Errorf("writing to the helper: %v: %w", err, ErrExited)
	}
	return nil
}

// call sends a request and waits for its response (see Supervisor.Call),
// filling in tr's phases after Ready.
func (p *proc) call(ctx context.Context, id, method string, params any, notify func(proto.Message), tr *Trace) (json.RawMessage, error) {
	if id == "" {
		id = fmt.Sprintf("c%d", p.s.ctrlID.Add(1))
	}
	tr.ID = id
	req, err := proto.NewRequest(id, method, params)
	if err != nil {
		return nil, fmt.Errorf("%s params: %w", method, err)
	}
	c := &call{method: method, notify: notify, done: make(chan struct{})}
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil, ErrExited
	}
	if _, dup := p.pending[id]; dup {
		p.mu.Unlock()
		return nil, fmt.Errorf("request id %s is already in flight: %w", id, core.ErrInvalid)
	}
	p.pending[id] = c
	p.mu.Unlock()
	if err := p.write(req); err != nil {
		p.drop(id)
		return nil, err
	}
	tr.Written = time.Since(tr.Start)

	select {
	case <-c.done:
		tr.answered(c)
		return c.result()
	case <-ctx.Done():
	}
	if !proto.IsJob(method) {
		p.drop(id)
		return nil, ctx.Err()
	}
	// Ask the helper to stop the job; it answers the job with "cancelled".
	go func() {
		cctx, cancel := context.WithTimeout(context.Background(), p.s.cfg.CancelGrace)
		defer cancel()
		p.call(cctx, "", proto.MethodJobCancel, proto.CancelParams{Job: id}, nil, &Trace{Start: time.Now()})
	}()
	grace := time.NewTimer(p.s.cfg.CancelGrace)
	defer grace.Stop()
	select {
	case <-c.done:
	case <-grace.C:
		p.s.cfg.Logf("helper: job %s did not stop within %s of being cancelled: restarting the helper", id, p.s.cfg.CancelGrace)
		p.killFor(FailStuck, fmt.Sprintf("job %s did not stop within %s of being cancelled", id, p.s.cfg.CancelGrace), false)
		<-c.done
	}
	tr.answered(c)
	return c.result()
}

func (p *proc) drop(id string) {
	p.mu.Lock()
	delete(p.pending, id)
	p.mu.Unlock()
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
