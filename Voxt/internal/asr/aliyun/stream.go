package aliyun

import (
	"context"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// protocol is one of the two wire protocols: it builds the client's
// messages and reads the server's, keeping the text. Its methods are
// pure; only handle and current keep state, and only the stream's reader
// calls them (current also after the reader is done).
type protocol interface {
	start() [][]byte                                    // text messages that open the task or session
	audio(frame []byte) (websocket.MessageType, []byte) // one frame of PCM
	finish() [][]byte                                   // text messages that end the audio
	frameSize() int
	// handle reads one text message from the service; finishing: the
	// client has sent finish(). An error is an *asr.Error that ends the
	// stream.
	handle(data []byte, finishing bool) (step, error)
	current() string  // the whole text so far
	language() string // as the service reported it, ""
}

// step is what a server message meant.
type step uint8

const (
	stepIgnore  step = iota // nothing for the caller
	stepStarted             // the task or session is accepted: audio may flow
	stepText                // the text may have changed
	stepDone                // the final text is in
)

// stream is an open task or session. The reader goroutine owns the
// protocol's state and is the only sender on events.
type stream struct {
	ctx    context.Context
	conn   *websocket.Conn
	p      protocol
	model  string
	rate   int
	key    string // blanked out of errors (asr.HideKey)
	events chan asr.Event

	startc     chan error    // the reader's verdict on the start (buffered)
	done       chan struct{} // closed by Close
	readerDone chan struct{}
	closeOnce  sync.Once

	rechunk    asr.Rechunk // Send's goroutine only
	sent       atomic.Int64
	finishing  atomic.Bool
	finishedAt atomic.Int64 // unix nanoseconds of Finish
	timedOut   atomic.Bool
	timerMu    sync.Mutex
	timer      *time.Timer
}

func open(ctx context.Context, c *Client, h http.Header, p protocol, req asr.StreamRequest) (*stream, error) {
	conn, _, err := asr.Dial(ctx, ID, c.endpoint, asr.WSOptions{Client: c.http, Header: h})
	if err != nil {
		return nil, err
	}
	s := &stream{
		ctx: ctx, conn: conn, p: p, model: c.model, rate: req.SampleRate, key: c.key,
		events:     make(chan asr.Event, 32),
		startc:     make(chan error, 1),
		done:       make(chan struct{}),
		readerDone: make(chan struct{}),
	}
	s.rechunk.Size = p.frameSize()
	go s.read()
	for _, m := range p.start() {
		if err := conn.Write(ctx, websocket.MessageText, m); err != nil {
			e := asr.WSError(ctx, ID, err)
			s.Close()
			return nil, e
		}
	}
	timer := time.NewTimer(asr.ConnectTimeout)
	defer timer.Stop()
	select {
	case err := <-s.startc:
		if err != nil {
			s.Close()
			return nil, err
		}
		return s, nil
	case <-timer.C:
		s.Close()
		return nil, &asr.Error{Kind: asr.KindNetwork, Provider: ID, Detail: "the service did not start the session in time", Retryable: true}
	case <-ctx.Done():
		s.Close()
		return nil, asr.NetError(ctx, ID, ctx.Err())
	}
}

// read reads the service's messages until the end: it answers the start
// on startc, then sends partials and the one final or error on events.
func (s *stream) read() {
	defer close(s.readerDone)
	defer close(s.events)
	started := false
	fail := func(e error) {
		e = asr.HideKey(e, s.key)
		if !started {
			s.startc <- e
			return
		}
		s.emit(asr.Event{Kind: asr.EventError, Err: e})
	}
	last := ""
	for {
		typ, data, err := s.conn.Read(s.ctx)
		if err != nil {
			fail(s.readError(err))
			return
		}
		if typ != websocket.MessageText {
			continue
		}
		st, herr := s.p.handle(data, s.finishing.Load())
		if herr != nil {
			fail(herr)
			s.conn.CloseNow()
			return
		}
		switch st {
		case stepStarted:
			if !started {
				started = true
				s.startc <- nil
			}
		case stepText:
			if t := s.p.current(); started && t != last {
				last = t
				if !s.emit(asr.Event{Kind: asr.EventPartial, Text: t}) {
					return
				}
			}
		case stepDone:
			if !started {
				s.startc <- &asr.Error{Kind: asr.KindProtocol, Provider: ID, Detail: "the service ended the session before starting it"}
				return
			}
			s.stopTimer()
			text := s.p.current()
			r := asr.Result{
				Text:     text,
				Lang:     s.p.language(),
				Duration: time.Duration(s.sent.Load()/2) * time.Second / time.Duration(s.rate),
				Model:    s.model,
			}
			if at := s.finishedAt.Load(); at != 0 {
				r.Latency = time.Since(time.Unix(0, at))
			}
			s.emit(asr.Event{Kind: asr.EventFinal, Text: text, Result: r})
			// Not Close: its handshake would hold the session's Close, and
			// the text's insertion, until the service answers it.
			s.conn.CloseNow()
			return
		}
	}
}

// emit sends ev unless the stream was closed; it reports whether it did.
func (s *stream) emit(ev asr.Event) bool {
	select {
	case <-s.done:
		return false
	default:
	}
	select {
	case s.events <- ev:
		return true
	case <-s.done:
		return false
	}
}

func (s *stream) readError(err error) *asr.Error {
	if s.timedOut.Load() && s.ctx.Err() == nil {
		return &asr.Error{Kind: asr.KindNetwork, Provider: ID, Detail: "no final result in time", Retryable: true, Err: err}
	}
	return asr.WSError(s.ctx, ID, err)
}

// Send frames pcm as the protocol wants and sends the full frames.
func (s *stream) Send(pcm []byte) error {
	if s.finishing.Load() {
		return asr.Errorf(ID, asr.KindProtocol, "send after finish")
	}
	if err := s.closed(); err != nil {
		return err
	}
	for _, f := range s.rechunk.Add(pcm) {
		if err := s.write(f); err != nil {
			return err
		}
	}
	return nil
}

func (s *stream) write(frame []byte) error {
	typ, msg := s.p.audio(frame)
	if err := s.conn.Write(s.ctx, typ, msg); err != nil {
		return asr.WSError(s.ctx, ID, err)
	}
	s.sent.Add(int64(len(frame)))
	return nil
}

// Finish sends what is left of the audio and the protocol's end, then
// bounds the wait for the final text by asr.FinalTimeout.
func (s *stream) Finish() error {
	if !s.finishing.CompareAndSwap(false, true) {
		return nil
	}
	if err := s.closed(); err != nil {
		return err
	}
	if rest := s.rechunk.Rest(); len(rest) > 0 {
		if err := s.write(rest); err != nil {
			return err
		}
	}
	s.finishedAt.Store(time.Now().UnixNano())
	for _, m := range s.p.finish() {
		if err := s.conn.Write(s.ctx, websocket.MessageText, m); err != nil {
			return asr.WSError(s.ctx, ID, err)
		}
	}
	s.timerMu.Lock()
	s.timer = time.AfterFunc(asr.FinalTimeout, func() {
		s.timedOut.Store(true)
		s.conn.CloseNow()
	})
	s.timerMu.Unlock()
	return nil
}

func (s *stream) stopTimer() {
	s.timerMu.Lock()
	if s.timer != nil {
		s.timer.Stop()
	}
	s.timerMu.Unlock()
}

func (s *stream) closed() error {
	select {
	case <-s.done:
		return asr.Errorf(ID, asr.KindCancelled, "the stream is closed")
	default:
		return nil
	}
}

// Close abandons the stream: no final event, the connection closed, and
// Events closed when it returns.
func (s *stream) Close() {
	s.closeOnce.Do(func() {
		close(s.done)
		s.stopTimer()
		s.conn.CloseNow()
	})
	<-s.readerDone
}

func (s *stream) Events() <-chan asr.Event { return s.events }
