// Package gemini is Google Gemini's Live API input transcription as an
// asr.Streamer, as Voxt 1.x used it: a WebSocket to BidiGenerateContent,
// a setup that asks for text answers and the input's transcription (the
// dictionary's terms as its custom vocabulary), the recording as realtime
// PCM chunks, then audioStreamEnd. The service sends the transcription in
// segments (inputTranscription) and the one in progress
// (interimInputTranscription); the text is their join.
//
// The key goes in the x-goog-api-key header, never in the URL.
package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"

	"github.com/coder/websocket"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// ID is the service's preset id.
const ID = "gemini"

// DefaultEndpoint and DefaultModel are Voxt 1.x's.
const (
	DefaultEndpoint = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
	DefaultModel    = "gemini-3.5-transcribe-live"
)

// Models are the model menu's (any id may be typed).
var Models = []string{DefaultModel}

// endGrace ends the wait for the last segments after audioStreamEnd: the
// Live API documents no end of the session, and a turn completes at every
// pause (Voxt 1.x waits 2.5 s too).
const endGrace = 2500 * time.Millisecond

// maxVocabulary bounds the custom vocabulary (the API's limit).
const maxVocabulary = 1000

// Client opens Gemini Live transcription streams.
type Client struct {
	endpoint, model, key string
	http                 *http.Client
	// ackTimeout bounds the wait for setupComplete, endGrace the wait
	// after audioStreamEnd; tests shorten them.
	ackTimeout, endGrace time.Duration
}

var _ asr.Streamer = (*Client)(nil)

// New checks cfg: a key, and an endpoint that is wss:// (ws:// only on
// loopback).
func New(cfg asr.Config) (*Client, error) {
	key := strings.TrimSpace(cfg.Key)
	if key == "" {
		return nil, asr.Errorf(ID, asr.KindConfig, "no API key")
	}
	ep := strings.TrimSpace(cfg.Endpoint)
	if ep == "" {
		ep = DefaultEndpoint
	}
	u, err := asr.CheckEndpoint(ID, ep, "wss")
	if err != nil {
		return nil, err
	}
	model := strings.TrimPrefix(strings.TrimSpace(cfg.Model), "models/")
	if model == "" {
		model = DefaultModel
	}
	return &Client{endpoint: u.String(), model: model, key: key, http: asr.WSClient(cfg.HTTPClient),
		ackTimeout: asr.ConnectTimeout, endGrace: endGrace}, nil
}

// Caps: streaming, with the hints as custom vocabulary; no prompt.
func (c *Client) Caps() asr.Caps { return asr.Caps{Stream: true, Hints: true} }

// setup is the session's first message. Its languageCodes stay empty:
// Gemini then detects the language, which keeps mixed speech (Chinese
// with English words) working.
func setup(model string, hints []string) []byte {
	tr := map[string]any{"languageCodes": []string{}, "mode": "SMART"}
	if v := vocabulary(hints); len(v) > 0 {
		tr["customVocabulary"] = v
	}
	b, _ := json.Marshal(map[string]any{"setup": map[string]any{
		"model":                   "models/" + model,
		"generationConfig":        map[string]any{"responseModalities": []string{"TEXT"}},
		"inputAudioTranscription": tr,
	}})
	return b
}

// vocabulary is the hints without blanks and repeats, at most
// maxVocabulary of them.
func vocabulary(hints []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, h := range hints {
		if h = strings.TrimSpace(h); h != "" && !seen[h] && len(out) < maxVocabulary {
			seen[h] = true
			out = append(out, h)
		}
	}
	return out
}

// Open connects, sends the setup and waits for setupComplete, so that a
// refused key or model fails here.
func (c *Client) Open(ctx context.Context, req asr.StreamRequest) (asr.Stream, error) {
	if req.SampleRate != 0 && req.SampleRate != asr.SampleRate {
		return nil, asr.Errorf(ID, asr.KindAudio, "takes %d Hz audio only, not %d Hz", asr.SampleRate, req.SampleRate)
	}
	h := http.Header{}
	h.Set("x-goog-api-key", c.key)
	conn, _, err := asr.Dial(ctx, ID, c.endpoint, asr.WSOptions{Client: c.http, Header: h})
	if err != nil {
		return nil, err
	}
	sctx, cancel := context.WithCancel(ctx)
	s := &stream{c: c, conn: conn, parent: ctx, ctx: sctx, cancel: cancel,
		events: make(chan asr.Event, 32), done: make(chan struct{}), readerDone: make(chan struct{})}
	if err := s.start(setup(c.model, req.Hints)); err != nil {
		s.abort()
		return nil, err
	}
	go s.read()
	return s, nil
}

// message is what the service sends, in proto JSON: camelCase, or
// snake_case (Voxt 1.x saw both).
type message map[string]any

func (m message) get(camel, snake string) (message, bool) {
	v, ok := m[camel]
	if !ok {
		v, ok = m[snake]
	}
	o, _ := v.(map[string]any)
	return o, ok
}

func (m message) text(camel, snake string) string {
	o, _ := m.get(camel, snake)
	s, _ := o["text"].(string)
	return strings.TrimSpace(s)
}

func (m message) flag(camel, snake string) bool {
	v, ok := m[camel].(bool)
	if !ok {
		v, _ = m[snake].(bool)
	}
	return v
}

// decode reads a frame; an error message ({"error":{"code":400,
// "message":…,"status":…}}) becomes its *asr.Error.
func decode(data []byte) (message, error) {
	var m message
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, &asr.Error{Kind: asr.KindProtocol, Provider: ID, Detail: "not JSON: " + asr.Clip(string(data), 200), Err: err}
	}
	e, ok := m.get("error", "error")
	if !ok {
		return m, nil
	}
	code, _ := e["code"].(float64)
	msg, _ := e["message"].(string)
	status, _ := e["status"].(string)
	if msg == "" {
		msg = status
	}
	if code < 400 {
		code = 500
	}
	return nil, asr.HTTPError(ID, int(code), status, msg)
}

// wsError is asr.WSError, with a refused model (Gemini closes with a
// policy violation that names it) a config error rather than the key's.
func wsError(ctx context.Context, err error) *asr.Error {
	e := asr.WSError(ctx, ID, err)
	if d := strings.ToLower(e.Detail); e.Kind == asr.KindAuth && (strings.Contains(d, "not found") || strings.Contains(d, "not supported")) {
		e.Kind = asr.KindConfig
	}
	return e
}

// stream is one transcription session. Send and Finish run on one
// goroutine, read on its own.
type stream struct {
	c      *Client
	conn   *websocket.Conn
	parent context.Context // Open's: cancelling it is KindCancelled
	ctx    context.Context // parent, also cancelled by Close
	cancel context.CancelFunc

	events     chan asr.Event
	done       chan struct{} // closed by Close
	readerDone chan struct{}
	closeOnce  sync.Once
	closed     atomic.Bool
	ended      atomic.Bool
	finished   bool         // Finish was called (the sender's)
	finishedAt atomic.Int64 // UnixNano of Finish, 0 before
	graceOver  atomic.Bool  // endGrace passed after Finish
	sent       atomic.Int64 // PCM bytes sent
}

// start sends the setup and waits for setupComplete.
func (s *stream) start(setup []byte) error {
	if err := s.write(setup); err != nil {
		return err
	}
	actx, cancel := context.WithTimeout(s.ctx, s.c.ackTimeout)
	defer cancel()
	for {
		_, data, err := s.conn.Read(actx)
		if err != nil {
			if s.parent.Err() == nil && actx.Err() != nil {
				return &asr.Error{Kind: asr.KindNetwork, Provider: ID, Detail: "no answer to the setup", Retryable: true, Err: err}
			}
			return wsError(s.parent, err)
		}
		m, err := decode(data)
		if err != nil {
			return err
		}
		if _, ok := m.get("setupComplete", "setup_complete"); ok {
			return nil
		}
	}
}

// read delivers the transcription until the session ends after
// audioStreamEnd: a completed turn or goAway, the grace, or the service
// closing the connection (which it does once the audio ended).
func (s *stream) read() {
	defer close(s.readerDone)
	defer close(s.events)
	var segments []string
	var interim, partial string
	for {
		_, data, err := s.conn.Read(s.ctx)
		var m message
		if err == nil {
			m, err = decode(data)
		}
		if err != nil {
			st := websocket.CloseStatus(err)
			switch {
			case s.closed.Load():
			case s.finishedAt.Load() != 0 && (s.graceOver.Load() || st == websocket.StatusNormalClosure || st == websocket.StatusGoingAway || join(segments, interim) != ""):
				s.final(join(segments, interim))
			case asr.KindOf(err) != "" && st == -1:
				s.fail(err) // the service's error message
			default:
				s.fail(wsError(s.parent, err))
			}
			return
		}
		sc, _ := m.get("serverContent", "server_content")
		if t := sc.text("inputTranscription", "input_transcription"); t != "" {
			if len(segments) == 0 || segments[len(segments)-1] != t {
				segments = append(segments, t)
			}
			interim = ""
		} else if t := sc.text("interimInputTranscription", "interim_input_transcription"); t != "" {
			interim = t
		}
		_, away := m.get("goAway", "go_away")
		if s.finishedAt.Load() != 0 && (away || sc.flag("turnComplete", "turn_complete")) {
			s.final(join(segments, interim))
			return
		}
		if text := join(segments, interim); text != partial {
			partial = text
			select {
			case s.events <- asr.Event{Kind: asr.EventPartial, Text: text}:
			default: // a later partial (or the final) has the whole text
			}
		}
	}
}

// join is the segments, then the interim one: a space between two only
// where neither side is CJK (Voxt 1.x's GeminiLiveTranscriptJoining).
func join(segments []string, interim string) string {
	var b strings.Builder
	var last rune
	for _, p := range append(segments, interim) {
		if p = strings.TrimSpace(p); p == "" {
			continue
		}
		first := []rune(p)[0]
		if b.Len() > 0 && !cjk(last) && !cjk(first) {
			b.WriteByte(' ')
		}
		b.WriteString(p)
		r := []rune(p)
		last = r[len(r)-1]
	}
	return b.String()
}

func cjk(r rune) bool {
	return unicode.In(r, unicode.Han, unicode.Hiragana, unicode.Katakana, unicode.Hangul) ||
		r >= 0x3000 && r <= 0x303F || r >= 0xFF00 && r <= 0xFFEF
}

func (s *stream) final(text string) {
	s.ended.Store(true)
	r := asr.Result{Text: text, Model: s.c.model, Duration: time.Duration(s.sent.Load()/2) * time.Second / asr.SampleRate}
	if t := s.finishedAt.Load(); t != 0 {
		r.Latency = time.Since(time.Unix(0, t))
	}
	s.emit(asr.Event{Kind: asr.EventFinal, Text: text, Result: r})
	s.conn.CloseNow()
}

func (s *stream) fail(err error) {
	s.ended.Store(true)
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

func (s *stream) write(msg []byte) error {
	if err := s.conn.Write(s.ctx, websocket.MessageText, msg); err != nil {
		if s.closed.Load() {
			return asr.Errorf(ID, asr.KindCancelled, "the stream was closed")
		}
		return wsError(s.parent, err)
	}
	return nil
}

func (s *stream) usable() error {
	switch {
	case s.closed.Load():
		return asr.Errorf(ID, asr.KindCancelled, "the stream was closed")
	case s.ended.Load():
		return asr.Errorf(ID, asr.KindProtocol, "the stream has ended")
	case s.finished:
		return asr.Errorf(ID, asr.KindProtocol, "the stream is finished")
	}
	return nil
}

// Send sends pcm as one realtime chunk.
func (s *stream) Send(pcm []byte) error {
	if err := s.usable(); err != nil || len(pcm) == 0 {
		return err
	}
	msg, _ := json.Marshal(map[string]any{"realtimeInput": map[string]any{"audio": map[string]string{
		"data": base64.StdEncoding.EncodeToString(pcm), "mimeType": "audio/pcm;rate=16000",
	}}})
	if err := s.write(msg); err != nil {
		return err
	}
	s.sent.Add(int64(len(pcm)))
	return nil
}

// Finish sends audioStreamEnd; the final follows within endGrace.
func (s *stream) Finish() error {
	if err := s.usable(); err != nil {
		return err
	}
	s.finished = true
	s.finishedAt.Store(time.Now().UnixNano())
	if err := s.write([]byte(`{"realtimeInput":{"audioStreamEnd":true}}`)); err != nil {
		return err
	}
	time.AfterFunc(s.c.endGrace, func() {
		s.graceOver.Store(true)
		s.conn.CloseNow() // the reader then delivers the text it has
	})
	return nil
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
