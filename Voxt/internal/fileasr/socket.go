package fileasr

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// SocketName is the socket's name in the default data folder.
const SocketName = "transcribe.sock"

// OpTranscribe is the one op of a request.
const OpTranscribe = "transcribe"

// lineTimeout bounds reading the request line and writing each reply
// line; clientIdle how long the command waits for the next line (one
// utterance's recognition).
const (
	lineTimeout = 30 * time.Second
	clientIdle  = 5 * time.Minute
	maxRequest  = 4 << 10
)

// ErrUnavailable: no socket to reach (Voxt is not running).
var ErrUnavailable = errors.New("fileasr: unavailable")

// Request is the command's one line; Bytes of PCM follow it.
type Request struct {
	Op    string `json:"op"`
	Bytes int64  `json:"bytes"`
	Lang  string `json:"lang,omitempty"` // "" for the app's language hint
}

type reply struct {
	Segment *Segment `json:"segment,omitempty"`
	Done    bool     `json:"done,omitempty"`
	Count   int      `json:"count,omitempty"`
	Error   string   `json:"error,omitempty"`
}

// Server answers each connection on a goroutine of its own, in the
// app's process.
type Server struct {
	// Open gives the recognizer of one request.
	Open func(ctx context.Context, lang string) (Recognize, error)
	// Done reports a request's end: its segments, its audio and how it
	// failed (nil when it did not). Nil reports nothing.
	Done func(segments int, audio time.Duration, err error)
}

// Serve answers connections until ctx ends; ln closes then.
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
		go s.handle(ctx, conn)
	}
}

func (s *Server) handle(ctx context.Context, conn net.Conn) {
	defer conn.Close()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	stop := context.AfterFunc(ctx, func() { conn.Close() })
	defer stop()
	conn.SetReadDeadline(time.Now().Add(lineTimeout))
	br := bufio.NewReaderSize(conn, maxRequest)
	line, err := br.ReadSlice('\n')
	if err != nil {
		return
	}
	enc := json.NewEncoder(conn)
	enc.SetEscapeHTML(false)
	send := func(r reply) error {
		conn.SetWriteDeadline(time.Now().Add(lineTimeout))
		return enc.Encode(r)
	}
	var req Request
	if json.Unmarshal(line, &req) != nil || req.Op != OpTranscribe || req.Bytes < 0 || req.Bytes%2 != 0 {
		send(reply{Error: "bad request"})
		return
	}
	conn.SetReadDeadline(time.Time{}) // the upload waits while an utterance is recognised
	n, err := s.transcribe(ctx, br, req, send)
	if s.Done != nil {
		s.Done(n, time.Duration(req.Bytes/2)*time.Second/asr.SampleRate, err)
	}
	if err != nil {
		send(reply{Error: err.Error()})
		return
	}
	send(reply{Done: true, Count: n})
}

func (s *Server) transcribe(ctx context.Context, r io.Reader, req Request, send func(reply) error) (int, error) {
	rec, err := s.Open(ctx, req.Lang)
	if err != nil {
		return 0, err
	}
	body := &counter{r: io.LimitReader(r, req.Bytes)}
	n := 0
	err = Run(ctx, body, rec, func(seg Segment) error {
		n++
		return send(reply{Segment: &seg})
	})
	if err == nil && body.n != req.Bytes {
		err = errors.New("the audio ended early")
	}
	return n, err
}

type counter struct {
	r io.Reader
	n int64
}

func (c *counter) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

// Transcribe sends pcm to the app at path and gives each segment to
// each as it comes. It writes while it reads, so that neither side waits
// on the other's full buffer.
func Transcribe(path, lang string, pcm []byte, each func(Segment) error) error {
	conn, err := net.Dial("unix", path)
	if err != nil {
		return fmt.Errorf("%w: %v", ErrUnavailable, err)
	}
	defer conn.Close() // also ends the writer
	head, err := json.Marshal(Request{Op: OpTranscribe, Bytes: int64(len(pcm)), Lang: lang})
	if err != nil {
		return err
	}
	go func() {
		if _, err := conn.Write(append(head, '\n')); err == nil {
			conn.Write(pcm)
		}
	}()
	br := bufio.NewReader(conn)
	for {
		conn.SetReadDeadline(time.Now().Add(clientIdle))
		line, err := br.ReadBytes('\n')
		if err != nil {
			return fmt.Errorf("the connection to Voxt broke: %v", err)
		}
		var r reply
		if err := json.Unmarshal(line, &r); err != nil {
			return fmt.Errorf("unexpected answer from Voxt: %v", err)
		}
		switch {
		case r.Error != "":
			return errors.New(r.Error)
		case r.Done:
			return nil
		case r.Segment != nil:
			if err := each(*r.Segment); err != nil {
				return err
			}
		}
	}
}
