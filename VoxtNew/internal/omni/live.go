package omni

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/url"
	"slices"
	"strconv"
	"sync"
	"time"

	"github.com/coder/websocket"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
)

// maxQueued is the preview's backlog: audio not yet written to the socket
// beyond this (30 s, Voxt's) means the server cannot keep up, and the
// preview stops. The Final does not depend on it.
const maxQueued = 30 * sampleRate * 2

// eventBuffer is a stream's channel's room; partials never take the last
// place, so the final always fits.
const eventBuffer = 64

// live is one live session (asr.Stream): the realtime socket shows the
// partial text; the Final at Finish is the result. A preview that fails
// stops quietly (the log says why): the Final still runs on the audio.
type live struct {
	a      *ASR
	l      *lease
	req    asr.StreamRequest
	ctx    context.Context // Open's: it bounds the whole stream
	cancel context.CancelFunc
	conn   *websocket.Conn
	out    chan []byte // messages for the socket's writer

	// language and bias are the Final's: the model's language hint ("":
	// it detects) and the context bias.
	language string
	bias     bias

	mu        sync.Mutex
	audio     []byte  // everything sent, for the Final
	queued    int     // bytes of audio in out
	previewed int     // bytes of audio written to the socket
	preview   bool    // the socket still takes audio
	pc        *pieces // a long recording's pieces (pieces.go), nil when the model has none
	finished  bool
	ended     bool // the last event was sent (or Close): events is closed
	events    chan asr.Event
	asm       assembler
	once      sync.Once // release
}

// openLive connects the realtime socket and starts the session; it
// returns once the server accepted it.
func openLive(ctx context.Context, a *ASR, l *lease, req asr.StreamRequest) (*live, error) {
	ep := l.endpoint()
	u := url.URL{Scheme: "ws", Host: ep.host + ":" + strconv.Itoa(ep.port), Path: "/v1/realtime",
		RawQuery: url.Values{"intent": {"transcription"}, "model": {ep.model}}.Encode()}
	dctx, dcancel := context.WithTimeout(ctx, asr.ConnectTimeout)
	defer dcancel()
	conn, _, err := websocket.Dial(dctx, u.String(), &websocket.DialOptions{HTTPClient: loopback})
	if err != nil {
		return nil, asr.NetError(dctx, Provider, err)
	}
	conn.SetReadLimit(1 << 20)
	session := map[string]any{"input_audio_format": "pcm16", "turn_detection": nil}
	if hint := a.spec.languageHint(req.Lang); hint != "" {
		session["language"] = hint
	}
	if err := writeJSON(dctx, conn, map[string]any{"type": "session.update", "session": session}); err != nil {
		conn.CloseNow()
		return nil, asr.NetError(dctx, Provider, err)
	}
	for {
		ev, err := readEvent(dctx, conn)
		if err != nil {
			conn.CloseNow()
			return nil, asr.NetError(dctx, Provider, err)
		}
		if ev.Type == "transcription_session.updated" {
			break
		}
		if ev.Type == "error" {
			conn.CloseNow()
			return nil, &asr.Error{Kind: asr.KindProtocol, Provider: Provider, Code: ev.code(), Detail: "the live session was refused"}
		}
	}
	sctx, cancel := context.WithCancel(ctx)
	s := &live{a: a, l: l, req: req, ctx: sctx, cancel: cancel, conn: conn,
		language: a.spec.languageHint(req.Lang), bias: a.spec.ASR.contextBias(req.Hints, req.Prompt),
		out: make(chan []byte, 1024), preview: true, events: make(chan asr.Event, eventBuffer)}
	go s.write()
	go s.read()
	if a.spec.ASR.PiecesAfter > 0 {
		s.pc = &pieces{wake: make(chan struct{}, 1), exited: make(chan struct{})}
		go s.recognisePieces(s.language, s.bias)
	}
	go func() {
		<-sctx.Done()
		s.mu.Lock()
		finished := s.finished
		s.mu.Unlock()
		if !finished {
			// Open's context ended the stream before Finish.
			s.end(asr.Event{Kind: asr.EventError, Err: asr.NetError(ctx, Provider, ctx.Err())})
		}
		conn.CloseNow()
	}()
	return s, nil
}

// realtimeEvent is a message of the realtime socket.
type realtimeEvent struct {
	Type    string `json:"type"`
	Index   int    `json:"event_index"`
	Segment int    `json:"segment_id"`
	Text    string `json:"text"`
	Final   bool   `json:"is_final"`
	Error   *struct {
		Type string `json:"type"`
		Code string `json:"code"`
	} `json:"error"`
}

func (e realtimeEvent) code() string {
	if e.Error != nil && e.Error.Code != "" {
		return e.Error.Code
	}
	return "error"
}

func writeJSON(ctx context.Context, c *websocket.Conn, v any) error {
	b, _ := json.Marshal(v)
	return c.Write(ctx, websocket.MessageText, b)
}

func readEvent(ctx context.Context, c *websocket.Conn) (realtimeEvent, error) {
	for {
		typ, b, err := c.Read(ctx)
		if err != nil {
			return realtimeEvent{}, err
		}
		var ev realtimeEvent
		if typ == websocket.MessageText && json.Unmarshal(b, &ev) == nil && ev.Type != "" {
			return ev, nil
		}
	}
}

// Send keeps pcm for the Final and queues it for the preview; it never
// waits for the server.
func (s *live) Send(pcm []byte) error {
	if len(pcm) == 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.finished || s.ended {
		return &asr.Error{Kind: asr.KindProtocol, Provider: Provider, Detail: "the stream has ended"}
	}
	s.audio = append(s.audio, pcm...)
	if s.pc != nil {
		select {
		case s.pc.wake <- struct{}{}:
		default:
		}
	}
	if !s.preview {
		return nil
	}
	if s.queued+len(pcm) > maxQueued {
		s.stopPreview("the preview fell behind the microphone")
		return nil
	}
	msg, _ := json.Marshal(map[string]string{"type": "input_audio_buffer.append", "audio": base64.StdEncoding.EncodeToString(pcm)})
	select {
	case s.out <- msg:
		s.queued += len(pcm)
	default:
		s.stopPreview("the preview's queue is full")
	}
	return nil
}

// stopPreview stops sending audio to the socket (s.mu held).
func (s *live) stopPreview(why string) {
	if s.preview {
		s.preview = false
		s.a.rt.logf("warning", "%s: live preview stopped: %s", s.a.spec.ID, why)
	}
}

// write sends the queued messages in order.
func (s *live) write() {
	for {
		select {
		case msg := <-s.out:
			n := audioBytes(msg)
			if err := s.conn.Write(s.ctx, websocket.MessageText, msg); err != nil {
				s.mu.Lock()
				if !s.finished {
					s.stopPreview("writing: " + err.Error())
				}
				s.mu.Unlock()
				return
			}
			s.mu.Lock()
			s.queued -= n
			s.previewed += n
			s.mu.Unlock()
		case <-s.ctx.Done():
			return
		}
	}
}

// audioBytes is the PCM an append message carries.
func audioBytes(msg []byte) int {
	var m struct {
		Audio string `json:"audio"`
	}
	if json.Unmarshal(msg, &m) != nil {
		return 0
	}
	return base64.StdEncoding.DecodedLen(len(m.Audio))
}

// read turns the socket's segments into partials.
func (s *live) read() {
	for {
		ev, err := readEvent(s.ctx, s.conn)
		if err != nil {
			s.mu.Lock()
			if !s.finished && s.ctx.Err() == nil {
				s.stopPreview("the socket closed: " + err.Error())
			}
			s.mu.Unlock()
			return
		}
		switch ev.Type {
		case "transcription.segment":
			s.mu.Lock()
			text, ok := s.asm.apply(ev.Index, ev.Segment, ev.Text, ev.Final)
			if ok && !s.finished {
				s.emit(asr.Event{Kind: asr.EventPartial, Text: text})
			}
			s.mu.Unlock()
		case "error":
			s.mu.Lock()
			s.stopPreview("the server said " + ev.code())
			s.mu.Unlock()
			s.conn.CloseNow()
			return
		}
	}
}

// Finish ends the audio: the preview's socket closes and the Final runs
// on everything sent (on what the pieces left, after the piece on its
// way); its result is the last event.
func (s *live) Finish() error {
	s.mu.Lock()
	if s.finished || s.ended {
		s.mu.Unlock()
		return nil
	}
	s.finished, s.preview = true, false
	pcm := s.audio
	s.mu.Unlock()
	go func() {
		s.conn.CloseNow()
		start := time.Now()
		language := s.language
		var texts []string
		done, busy := 0, time.Duration(0)
		if pc := s.pc; pc != nil {
			select {
			case pc.wake <- struct{}{}:
			default:
			}
			<-pc.exited
			s.mu.Lock()
			done, texts, busy = pc.done, slices.Clone(pc.texts), pc.busy
			if pc.language != "" {
				language = pc.language
			}
			s.mu.Unlock()
		}
		text := ""
		var err error
		if tail := pcm[done:]; len(tail) > 0 {
			text, language, err = s.a.final(s.ctx, s.l, tail, language, s.bias)
		}
		took := time.Since(start)
		s.logFinal(pcm, done, len(texts), busy, took, err)
		if err != nil {
			if s.ctx.Err() != nil {
				err = asr.NetError(s.ctx, Provider, s.ctx.Err())
			}
			s.end(asr.Event{Kind: asr.EventError, Err: err})
			return
		}
		text = joinParts(append(texts, text))
		res := asr.Result{Text: text, Lang: language, Duration: asr.Duration(pcm, sampleRate), Model: s.a.spec.ID, Latency: took}
		s.end(asr.Event{Kind: asr.EventFinal, Text: text, Result: res})
	}()
	return nil
}

// slowFinal: a Final after Stop this long gets the Mac's memory in its
// log line (a model swapped out recognises several times slower).
const slowFinal = 5 * time.Second

// logFinal logs the Final after Stop: the audio, the pieces recognised
// while it recorded, the rest, and the Mac's memory when it was slow or
// failed.
func (s *live) logFinal(pcm []byte, done, pieces int, busy, took time.Duration, err error) {
	d := func(b []byte) time.Duration { return asr.Duration(b, sampleRate).Round(100 * time.Millisecond) }
	line := fmt.Sprintf("%s: Final of %v of audio", s.a.spec.ID, d(pcm))
	if pieces > 0 {
		line += fmt.Sprintf(": %d pieces while it recorded (%v of audio in %v), the rest (%v)", pieces, d(pcm[:done]), busy.Round(100*time.Millisecond), d(pcm[done:]))
	}
	level := "info"
	switch {
	case err != nil && s.ctx.Err() != nil:
		level, line = "warning", line+fmt.Sprintf(" stopped after %v", took.Round(time.Millisecond))
	case err != nil:
		level, line = "warning", line+fmt.Sprintf(" failed after %v: %v", took.Round(time.Millisecond), err)
	default:
		line += fmt.Sprintf(" in %v", took.Round(time.Millisecond))
	}
	if err != nil || took > slowFinal {
		line += "; " + memoryNote()
	}
	s.a.rt.logf(level, "%s", line)
}

// Salvage is the best text the stream has when its Final failed or took
// too long (asr.Salvager): the pieces' when they cover at least what the
// preview heard, else the preview's; "" for none.
func (s *live) Salvage() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	preview := s.asm.text()
	if s.pc != nil && len(s.pc.texts) > 0 && (preview == "" || s.pc.done >= s.previewed) {
		return joinParts(s.pc.texts)
	}
	return preview
}

// Close abandons the stream.
func (s *live) Close() {
	s.mu.Lock()
	s.finished = true
	s.mu.Unlock()
	s.cancel()
	s.end(asr.Event{})
}

func (s *live) Events() <-chan asr.Event { return s.events }

// emit sends a partial if there is room for it and the final (s.mu held).
func (s *live) emit(ev asr.Event) {
	if s.ended || len(s.events) >= cap(s.events)-1 {
		return
	}
	s.events <- ev
}

// end sends the last event (none for a zero Kind), closes the events and
// releases the server; only the first call counts.
func (s *live) end(ev asr.Event) {
	s.mu.Lock()
	if !s.ended {
		if ev.Kind != 0 {
			s.events <- ev
		}
		s.ended = true
		close(s.events)
	}
	s.mu.Unlock()
	s.once.Do(func() {
		s.cancel()
		s.l.release()
	})
}

// assembler rebuilds the live text from the socket's segments (Voxt's
// OmniLiveTranscriptAssembler): a segment's latest hypothesis replaces
// the one before, a final one is kept, and a replayed or reordered event
// is ignored.
type assembler struct {
	lastIndex   int
	finals      map[int]string
	provSegment int
	provText    string
	hasProv     bool
}

// apply takes one segment event and returns the text to show, false when
// nothing changed.
func (a *assembler) apply(index, segment int, text string, final bool) (string, bool) {
	if index > 0 {
		if index <= a.lastIndex {
			return "", false
		}
		a.lastIndex = index
	}
	if a.finals == nil {
		a.finals = map[int]string{}
	}
	if final {
		a.finals[segment] = text
		if a.hasProv && a.provSegment == segment {
			a.hasProv, a.provText = false, ""
		}
	} else if _, done := a.finals[segment]; !done {
		a.provSegment, a.provText, a.hasProv = segment, text, true
	} else {
		return "", false
	}
	return a.text(), true
}

// text is the confirmed segments in order, then the provisional one.
func (a *assembler) text() string {
	ids := make([]int, 0, len(a.finals))
	for id := range a.finals {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	parts := make([]string, 0, len(ids)+1)
	for _, id := range ids {
		parts = append(parts, a.finals[id])
	}
	if a.hasProv {
		parts = append(parts, a.provText)
	}
	return joinParts(parts)
}
