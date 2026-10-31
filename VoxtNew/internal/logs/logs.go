// Package logs is the log behind the logs panel: core.LogService as a
// ring of the newest lines in memory, also written to rotated files
// (<dir>/voxtnew.log, .1, .2) by a goroutine of its own, so Append
// never blocks on the disk.
//
// Sources: "app" (the Go log package, through Writer) and "helper" (the
// Swift helper's stderr and log notifications, from the supervisor).
//
// Imports: core.
package logs

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/hehehai/actnowkit/redact"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
)

// Options configure a Ring.
type Options struct {
	// Size is how many lines the ring keeps; 0 = 2000.
	Size int
	// Dir is where the files go (mygo.PathLogs); "" = memory only.
	Dir string
	// MaxFileBytes rotates voxtnew.log at this size; 0 = 5 MB. Keep
	// is how many rotated files stay; 0 = 2.
	MaxFileBytes int64
	Keep         int
	// Bus receives LogAppended for every line (may be nil).
	Bus *core.Bus
}

// Ring implements core.LogService.
type Ring struct {
	opts Options

	mu    sync.Mutex
	lines []core.LogEntry // ring buffer
	next  int             // index of the slot to write
	full  bool
	seq   uint64 // the last line's Seq

	queue   chan core.LogEntry // to the file writer
	dropped int                // lines the file writer missed (queue full)
	done    chan struct{}
	close   sync.Once
}

var _ core.LogService = (*Ring)(nil)

// FileName is the name of the current log file in Options.Dir.
const FileName = "voxtnew.log"

// New starts a Ring (and its file writer when Dir is set).
func New(o Options) *Ring {
	if o.Size <= 0 {
		o.Size = 2000
	}
	if o.MaxFileBytes <= 0 {
		o.MaxFileBytes = 5 << 20
	}
	if o.Keep <= 0 {
		o.Keep = 2
	}
	r := &Ring{opts: o, lines: make([]core.LogEntry, o.Size), done: make(chan struct{})}
	if o.Dir != "" {
		r.queue = make(chan core.LogEntry, 1024)
		go r.writeFiles(r.queue)
	} else {
		close(r.done)
	}
	return r
}

// Append adds a line; Time defaults to now, Level to info, and Seq is
// the next number (the line's identity in the view). Never blocks:
// when the file writer falls behind, lines miss the file (not the ring)
// and a note of how many is written later.
func (r *Ring) Append(e core.LogEntry) {
	if e.Time.IsZero() {
		e.Time = time.Now()
	}
	if e.Level == "" {
		e.Level = core.SeverityInfo
	}
	// Keys, tokens and proxy passwords never reach the panel, the files
	// or a diagnostic bundle (family spec §13, audit S5/S10).
	e.Message = redact.Text(strings.TrimRight(e.Message, "\r\n"))
	r.mu.Lock()
	r.seq++
	e.Seq = r.seq
	r.lines[r.next] = e
	r.next = (r.next + 1) % len(r.lines)
	if r.next == 0 {
		r.full = true
	}
	if r.queue != nil {
		select {
		case r.queue <- e:
		default:
			r.dropped++
		}
	}
	r.mu.Unlock()
	r.opts.Bus.Publish(core.LogAppended{Entry: e})
}

// Tail returns up to n newest lines of source ("" = all; a source ending
// in ":" matches its prefix, "helper:" every helper source), oldest first. It
// reads memory only: views may call it.
func (r *Ring) Tail(n int, source string) []core.LogEntry {
	r.mu.Lock()
	defer r.mu.Unlock()
	count := r.next
	if r.full {
		count = len(r.lines)
	}
	var out []core.LogEntry
	for i := 1; i <= count && len(out) < n; i++ {
		e := r.lines[(r.next-i+len(r.lines))%len(r.lines)]
		if Matches(e.Source, source) {
			out = append(out, e)
		}
	}
	// Newest first so far: reverse.
	for i, j := 0, len(out)-1; i < j; i, j = i+1, j-1 {
		out[i], out[j] = out[j], out[i]
	}
	return out
}

// Matches reports whether a line's source is selected by filter ("" = all,
// "helper:" = every source with that prefix).
func Matches(source, filter string) bool {
	return filter == "" || source == filter || strings.HasSuffix(filter, ":") && strings.HasPrefix(source, filter)
}

// Close stops the file writer after it wrote what is queued.
func (r *Ring) Close() {
	r.close.Do(func() {
		r.mu.Lock()
		if r.queue != nil {
			close(r.queue)
			r.queue = nil
		}
		r.mu.Unlock()
	})
	<-r.done
}

// Path is the current log file, "" when memory only.
func (r *Ring) Path() string {
	if r.opts.Dir == "" {
		return ""
	}
	return filepath.Join(r.opts.Dir, FileName)
}

// Format renders a line as the files and "Copy" show it.
func Format(e core.LogEntry) string {
	return fmt.Sprintf("%s %-7s %-12s %s", e.Time.Format("2006-01-02 15:04:05.000"), strings.ToUpper(string(e.Level)), e.Source, e.Message)
}

func (r *Ring) writeFiles(queue <-chan core.LogEntry) {
	defer close(r.done)
	if err := os.MkdirAll(r.opts.Dir, 0o755); err != nil {
		for range queue {
		}
		return
	}
	path := r.Path()
	var f *os.File
	var size int64
	open := func() {
		var err error
		f, err = os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			f = nil
			return
		}
		if st, err := f.Stat(); err == nil {
			size = st.Size()
		}
	}
	open()
	var buf bytes.Buffer
	for e := range queue {
		buf.Reset()
		r.mu.Lock()
		dropped := r.dropped
		r.dropped = 0
		r.mu.Unlock()
		if dropped > 0 {
			buf.WriteString(Format(core.LogEntry{Time: e.Time, Level: core.SeverityWarning, Source: "logs", Message: fmt.Sprintf("%d lines were not written to this file", dropped)}) + "\n")
		}
		buf.WriteString(Format(e) + "\n")
		if f == nil {
			continue
		}
		if size+int64(buf.Len()) > r.opts.MaxFileBytes && size > 0 {
			f.Close()
			rotate(path, r.opts.Keep)
			open()
			if f == nil {
				continue
			}
		}
		n, _ := f.Write(buf.Bytes())
		size += int64(n)
	}
	if f != nil {
		f.Close()
	}
}

// rotate shifts path.(k-1) → path.k … path → path.1, dropping path.keep.
func rotate(path string, keep int) {
	os.Remove(fmt.Sprintf("%s.%d", path, keep))
	for i := keep - 1; i >= 1; i-- {
		os.Rename(fmt.Sprintf("%s.%d", path, i), fmt.Sprintf("%s.%d", path, i+1))
	}
	os.Rename(path, path+".1")
}

// Writer returns an io.Writer that appends each line written to it, as a
// line of source: log.SetOutput(io.MultiWriter(os.Stderr, ring.Writer("app"))).
// Lines starting with "error"/"warning" (any case, after the log prefix)
// get those levels.
func (r *Ring) Writer(source string) io.Writer { return &lineWriter{r: r, source: source} }

type lineWriter struct {
	r      *Ring
	source string
	mu     sync.Mutex
	buf    []byte
}

func (w *lineWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	w.buf = append(w.buf, p...)
	var lines []string
	for {
		i := bytes.IndexByte(w.buf, '\n')
		if i < 0 {
			break
		}
		lines = append(lines, string(w.buf[:i]))
		w.buf = w.buf[i+1:]
	}
	w.mu.Unlock()
	for _, l := range lines {
		if strings.TrimSpace(l) == "" {
			continue
		}
		w.r.Append(core.LogEntry{Level: levelOf(l), Source: w.source, Message: l})
	}
	return len(p), nil
}

func levelOf(line string) core.Severity {
	l := strings.ToLower(line)
	switch {
	case strings.Contains(l, "panic") || strings.Contains(l, "error") || strings.Contains(l, "fatal"):
		return core.SeverityError
	case strings.Contains(l, "warn"):
		return core.SeverityWarning
	}
	return core.SeverityInfo
}
