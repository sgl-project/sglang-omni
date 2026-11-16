package historyapi

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"os"
	"slices"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// SocketName is the socket's name in the default data folder.
const SocketName = "history.sock"

// WriteTimeout is how long one write may take before the client is
// dropped: a stuck client holds up the queue for at most this.
const WriteTimeout = 30 * time.Second

// maxRequest bounds the one request line; a real one is well under 1 KiB.
const maxRequest = 64 << 10

// maxName bounds the name a client gives itself, for the read log's rows.
const maxName = 64

// Ops of a request.
const (
	OpExport = "export"
	OpStatus = "status"
)

// Request is a client's one line. Since and Until are Unix milliseconds,
// 0 for no bound. The socket's wire format is internal to Voxt.
type Request struct {
	Op    string `json:"op"`
	Since int64  `json:"since,omitempty"`
	Until int64  `json:"until,omitempty"`
	Kind  string `json:"kind,omitempty"`
	As    string `json:"as,omitempty"`
}

// Status answers `history status`.
type Status struct {
	Schema      int       `json:"schema"`
	KeepHistory bool      `json:"keep_history"`
	KeepDays    int       `json:"keep_days"` // 0: cleaned up by no age
	Entries     int       `json:"entries"`
	Oldest      time.Time `json:"oldest,omitzero"`
}

// Policy is what the settings say right now.
type Policy struct {
	Selection   bool
	KeepHistory bool
	KeepDays    int
}

// Read is the log's record of one request: never any history text.
type Read struct {
	At      time.Time `json:"at"`
	Op      string    `json:"op"`
	Count   int       `json:"count"`
	Program string    `json:"program,omitempty"` // from the process tree
	As      string    `json:"as,omitempty"`      // the name it gave itself
}

// doneLine ends a successful reply; without it the reply was cut short.
type doneLine struct {
	Done  bool `json:"done"`
	Count int  `json:"count"`
}

type errorLine struct {
	Error string `json:"error"`
}

type statusLine struct {
	Status Status `json:"status"`
}

// Server answers one connection at a time, in the app's process. Audit
// runs on its goroutine, one call at a time.
type Server struct {
	History store.History
	Policy  func() Policy
	Audit   func(Read)
	// Peer names the client's program; nil names none.
	Peer func(net.Conn) string
	// Timeout replaces WriteTimeout (tests).
	Timeout time.Duration
}

// Listen makes the socket at path, 0600, replacing one a crash left
// behind. Binding then changing the mode leaves a moment with the umask's
// mode, inside a home other accounts cannot enter.
func Listen(path string) (net.Listener, error) {
	if fi, err := os.Lstat(path); err == nil && fi.Mode()&os.ModeSocket != 0 {
		if err := os.Remove(path); err != nil {
			return nil, err
		}
	}
	ln, err := net.Listen("unix", path)
	if err != nil {
		return nil, err
	}
	if err := os.Chmod(path, 0o600); err != nil {
		ln.Close()
		return nil, err
	}
	return ln, nil
}

// Serve answers connections until ctx ends; ln closes then, and with a
// Unix listener the socket file goes with it.
func (s *Server) Serve(ctx context.Context, ln net.Listener) error {
	stop := context.AfterFunc(ctx, func() { ln.Close() })
	defer stop()
	for {
		conn, err := ln.Accept()
		if err != nil {
			if ctx.Err() != nil || errors.Is(err, net.ErrClosed) {
				return nil
			}
			return err
		}
		s.handle(ctx, conn)
	}
}

func (s *Server) timeout() time.Duration {
	if s.Timeout > 0 {
		return s.Timeout
	}
	return WriteTimeout
}

func (s *Server) handle(ctx context.Context, conn net.Conn) {
	defer conn.Close()
	stop := context.AfterFunc(ctx, func() { conn.Close() })
	defer stop()
	program := ""
	if s.Peer != nil {
		program = s.Peer(conn)
	}
	conn.SetReadDeadline(time.Now().Add(s.timeout()))
	var req Request
	if err := json.NewDecoder(io.LimitReader(conn, maxRequest)).Decode(&req); err != nil {
		return
	}
	w := newLines(conn, s.timeout())
	if req.Op != OpExport && req.Op != OpStatus || req.Kind != "" && !slices.Contains(store.Kinds, store.Kind(req.Kind)) {
		w.line(errorLine{"bad request"})
		return
	}
	n := 0
	var err error
	if req.Op == OpStatus {
		err = s.status(ctx, w)
	} else {
		n, err = s.export(ctx, w, req)
	}
	if s.Audit != nil {
		s.Audit(Read{At: time.Now(), Op: req.Op, Count: n, Program: program, As: cleanName(req.As)})
	}
	if err != nil {
		if ctx.Err() == nil && !isWriteError(err) {
			w.line(errorLine{err.Error()})
		}
		return
	}
	w.line(doneLine{Done: true, Count: n})
}

func (s *Server) export(ctx context.Context, w *lines, req Request) (int, error) {
	q := store.HistoryQuery{Kind: store.Kind(req.Kind)}
	if req.Since != 0 {
		q.After = time.UnixMilli(req.Since)
	}
	if req.Until != 0 {
		q.Before = time.UnixMilli(req.Until)
	}
	n := 0
	err := Walk(ctx, s.History, q, func() bool { return s.Policy().Selection }, func(r Row) error {
		if err := w.line(r); err != nil {
			return writeError{err}
		}
		n++
		return nil
	})
	return n, err
}

func (s *Server) status(ctx context.Context, w *lines) error {
	p := s.Policy()
	st := Status{Schema: 1, KeepHistory: p.KeepHistory, KeepDays: p.KeepDays}
	q := store.HistoryQuery{Finished: true}
	var err error
	if st.Entries, err = s.History.Count(ctx, q); err != nil {
		return err
	}
	q.Oldest, q.Limit = true, 1
	first, err := s.History.List(ctx, q)
	if err != nil {
		return err
	}
	if len(first) > 0 {
		st.Oldest = first[0].CreatedAt.Local()
	}
	if err := w.line(statusLine{st}); err != nil {
		return writeError{err}
	}
	return nil
}

// writeError marks a failure to send, as against one to read the store.
type writeError struct{ error }

func isWriteError(err error) bool {
	var w writeError
	return errors.As(err, &w)
}

// cleanName keeps a client-given name to a short single line for the
// read log and a notice.
func cleanName(s string) string {
	s = strings.Map(func(r rune) rune {
		if r < 0x20 || r == 0x7f {
			return -1
		}
		return r
	}, strings.TrimSpace(s))
	if r := []rune(s); len(r) > maxName {
		s = string(r[:maxName])
	}
	return s
}

// lines writes JSON lines, each under its own deadline.
type lines struct {
	conn    net.Conn
	bw      *bufio.Writer
	enc     *json.Encoder
	timeout time.Duration
}

func newLines(conn net.Conn, timeout time.Duration) *lines {
	bw := bufio.NewWriter(conn)
	enc := json.NewEncoder(bw)
	enc.SetEscapeHTML(false)
	return &lines{conn: conn, bw: bw, enc: enc, timeout: timeout}
}

func (l *lines) line(v any) error {
	l.conn.SetWriteDeadline(time.Now().Add(l.timeout))
	if err := l.enc.Encode(v); err != nil {
		return err
	}
	return l.bw.Flush()
}
