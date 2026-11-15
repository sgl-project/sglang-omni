package doubao

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// stream is one recognition session.
//
// Sequences: the full client request is 1, audio packets 2, 3, …; the
// last packet (Finish) carries the negated next number and the last flag.
// The service answers with full server responses whose result.text is the
// whole text so far; its last answer has the last flag (and a negative
// sequence).
type stream struct {
	c      *Client
	conn   *websocket.Conn
	parent context.Context // Open's: cancelling it is KindCancelled
	ctx    context.Context // parent, also cancelled by Close
	cancel context.CancelFunc
	logID  string

	events     chan asr.Event
	done       chan struct{} // closed by Close
	readerDone chan struct{}
	closeOnce  sync.Once
	closed     atomic.Bool

	// Send and Finish (one goroutine).
	chunk    asr.Rechunk
	seq      int32
	finished bool

	sent       atomic.Int64 // PCM bytes sent
	finishedAt atomic.Int64 // UnixNano of Finish, 0 before
	timedOut   atomic.Bool  // no final within finalTimeout of Finish
	timerMu    sync.Mutex
	timer      *time.Timer
	ended      atomic.Pointer[endState]
}

// endState is how the stream ended: an error, or nil Err after the final.
type endState struct{ Err error }

func newStream(parent context.Context, c *Client, conn *websocket.Conn, logID string) *stream {
	ctx, cancel := context.WithCancel(parent)
	return &stream{
		c: c, conn: conn, parent: parent, ctx: ctx, cancel: cancel, logID: logID,
		events:     make(chan asr.Event, 32),
		done:       make(chan struct{}),
		readerDone: make(chan struct{}),
		chunk:      asr.Rechunk{Size: int(packet.Seconds()*asr.SampleRate) * 2},
		seq:        2,
	}
}

// start sends the full client request and reads the service's first
// answer: an acknowledgement, or an error frame.
func (s *stream) start(params []byte) error {
	if err := s.write(Frame{Type: FullClientRequest, Flags: FlagSeq, Serialization: SerJSON, Compression: CompGzip, Seq: 1, Payload: params}); err != nil {
		return err
	}
	actx, cancel := context.WithTimeout(s.ctx, s.c.ackTimeout)
	defer cancel()
	typ, data, err := s.conn.Read(actx)
	if err != nil {
		if s.parent.Err() == nil && actx.Err() != nil {
			return &asr.Error{Kind: asr.KindNetwork, Provider: ID, Detail: "no answer to the request", Retryable: true, Err: err}
		}
		return asr.WSError(s.parent, ID, err)
	}
	f, err := s.decode(typ, data)
	if err != nil {
		return err
	}
	if f.Type == FullServerResponse {
		if _, err := parseResponse(f.Payload); err != nil {
			return withLog(err, s.logID)
		}
	}
	return nil
}

// decode decodes a server message: binary frames only; an error frame
// becomes its *asr.Error.
func (s *stream) decode(typ websocket.MessageType, data []byte) (Frame, error) {
	if typ != websocket.MessageBinary {
		return Frame{}, withLog(protoErr("unexpected text message: %s", asr.Clip(string(data), 200)), s.logID)
	}
	f, err := Decode(data)
	if err != nil {
		return f, withLog(err, s.logID)
	}
	switch f.Type {
	case ServerError:
		return f, withLog(codeError(f.Code, f.Payload), s.logID)
	case FullServerResponse, ServerAck:
		return f, nil
	}
	return f, withLog(protoErr("unexpected %s from the service", f.Type), s.logID)
}

// withLog adds the service's log id (X-Tt-Logid, what its support asks
// for) to an error's detail.
func withLog(err error, logID string) error {
	var e *asr.Error
	if logID != "" && errors.As(err, &e) {
		e.Detail += " (logid " + logID + ")"
	}
	return err
}

// read delivers the service's answers as events until the final one, an
// error, or Close.
func (s *stream) read() {
	defer close(s.readerDone)
	defer close(s.events)
	defer s.stopTimer()
	var text, partial string
	var durMS int64
	for {
		typ, data, err := s.conn.Read(s.ctx)
		if err != nil {
			s.fail(s.connErr(err))
			return
		}
		f, err := s.decode(typ, data)
		if err != nil {
			s.fail(err)
			return
		}
		if f.Type != FullServerResponse {
			continue
		}
		p, err := parseResponse(f.Payload)
		if err != nil {
			s.fail(withLog(err, s.logID))
			return
		}
		// An empty answer (an acknowledgement, silence) keeps the text: the
		// last answer may come without one.
		if p.Text != "" {
			text = p.Text
		}
		if p.DurationMS > 0 {
			durMS = p.DurationMS
		}
		if f.Last() {
			s.final(text, durMS)
			return
		}
		if text != partial {
			partial = text
			select {
			case s.events <- asr.Event{Kind: asr.EventPartial, Text: text}:
			default: // a later partial (or the final) carries the whole text again
			}
		}
	}
}

func (s *stream) final(text string, durMS int64) {
	r := asr.Result{Text: text, Model: s.c.model, Duration: time.Duration(durMS) * time.Millisecond}
	if durMS == 0 {
		r.Duration = time.Duration(s.sent.Load()/2) * time.Second / asr.SampleRate
	}
	if t := s.finishedAt.Load(); t != 0 {
		r.Latency = time.Since(time.Unix(0, t))
	}
	s.ended.CompareAndSwap(nil, &endState{})
	s.emit(asr.Event{Kind: asr.EventFinal, Text: text, Result: r})
	s.conn.CloseNow()
}

// fail ends the stream with err, unless Close abandoned it.
func (s *stream) fail(err error) {
	if err == nil || s.closed.Load() {
		return
	}
	err = asr.HideKey(err, s.c.token)
	s.ended.CompareAndSwap(nil, &endState{Err: err})
	s.emit(asr.Event{Kind: asr.EventError, Err: err})
	s.conn.CloseNow()
}

// emit delivers a last event, or drops it when Close comes first.
func (s *stream) emit(ev asr.Event) {
	select {
	case s.events <- ev:
	case <-s.done:
	}
}

// connErr classifies a failed read or write: nil after Close, the final
// timeout, the caller's cancellation, else what the connection says.
func (s *stream) connErr(err error) error {
	if s.closed.Load() {
		return nil
	}
	if s.timedOut.Load() {
		return &asr.Error{Kind: asr.KindNetwork, Provider: ID, Detail: "no final result in time", Retryable: true}
	}
	return withLog(asr.WSError(s.parent, ID, err), s.logID)
}

func (s *stream) write(f Frame) error {
	b, err := Encode(f)
	if err != nil {
		return asr.Errorf(ID, asr.KindProtocol, "cannot encode a frame: %v", err)
	}
	if err := s.conn.Write(s.ctx, websocket.MessageBinary, b); err != nil {
		if e := s.ended.Load(); e != nil && e.Err != nil {
			return e.Err // the reader saw why
		}
		if err := s.connErr(err); err != nil {
			return err
		}
		return asr.Errorf(ID, asr.KindCancelled, "the stream was closed")
	}
	return nil
}

func (s *stream) usable() error {
	if s.closed.Load() {
		return asr.Errorf(ID, asr.KindCancelled, "the stream was closed")
	}
	if e := s.ended.Load(); e != nil {
		if e.Err != nil {
			return e.Err
		}
		return asr.Errorf(ID, asr.KindProtocol, "the stream has ended")
	}
	if s.finished {
		return asr.Errorf(ID, asr.KindProtocol, "the stream is finished")
	}
	return nil
}

// Send sends audio in packets of 200 ms.
func (s *stream) Send(pcm []byte) error {
	if err := s.usable(); err != nil {
		return err
	}
	for _, fr := range s.chunk.Add(pcm) {
		if err := s.write(Frame{Type: AudioOnlyRequest, Flags: FlagSeq, Compression: CompGzip, Seq: s.seq, Payload: fr}); err != nil {
			return err
		}
		s.seq++
		s.sent.Add(int64(len(fr)))
	}
	return nil
}

// Finish sends what is left as the last packet (negative sequence, last
// flag; an empty one when nothing is left) and starts the wait for the
// final result.
func (s *stream) Finish() error {
	if err := s.usable(); err != nil {
		return err
	}
	s.finished = true
	rest := s.chunk.Rest()
	f := Frame{Type: AudioOnlyRequest, Flags: FlagSeq | FlagLast, Seq: -s.seq, Payload: rest}
	if len(rest) > 0 {
		f.Compression = CompGzip
	}
	s.finishedAt.Store(time.Now().UnixNano())
	s.timerMu.Lock()
	s.timer = time.AfterFunc(s.c.finalTimeout, func() {
		s.timedOut.Store(true)
		s.conn.CloseNow()
	})
	s.timerMu.Unlock()
	if err := s.write(f); err != nil {
		return err
	}
	s.sent.Add(int64(len(rest)))
	return nil
}

func (s *stream) stopTimer() {
	s.timerMu.Lock()
	if s.timer != nil {
		s.timer.Stop()
	}
	s.timerMu.Unlock()
}

// Close abandons the stream and waits for Events to close.
func (s *stream) Close() {
	s.abort()
	<-s.readerDone
}

// abort closes the connection without waiting (Open's failures, before
// the reader runs).
func (s *stream) abort() {
	s.closeOnce.Do(func() {
		s.closed.Store(true)
		close(s.done)
		s.cancel()
		s.conn.CloseNow()
	})
}

func (s *stream) Events() <-chan asr.Event { return s.events }
