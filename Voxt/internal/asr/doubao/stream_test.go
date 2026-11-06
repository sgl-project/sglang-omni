package doubao

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/asrtest"
)

// ---- The scripted server ----

// script is a recorded exchange the fake server plays back
// (testdata/server/*.json): which frames it sends after which of the
// client's frames.
type script struct {
	// Source says where the frames come from (the docs, Voxt).
	Source string        `json:"source"`
	Frames []scriptFrame `json:"frames"`
}

type scriptFrame struct {
	// After: send once this many client frames have arrived (1: after
	// the full client request); -1: after the client's last frame.
	After int `json:"after"`
	// Kind: "response" (a full server response, gzip JSON, the default),
	// "error" (an error frame: Code and Payload, uncompressed JSON),
	// "raw" (Hex as is), "close" (end the connection normally).
	Kind    string          `json:"kind"`
	Seq     int32           `json:"seq"`
	Last    bool            `json:"last"`
	Code    uint32          `json:"code"`
	Payload json.RawMessage `json:"payload"`
	Hex     string          `json:"hex"`
}

func loadScript(t testing.TB, name string) script {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "server", name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var s script
	if err := json.Unmarshal(b, &s); err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	if s.Source == "" {
		t.Fatalf("%s: no source", name)
	}
	return s
}

// encode packs a scripted frame with the package's encoder.
func (sf scriptFrame) encode(t testing.TB) []byte {
	t.Helper()
	switch sf.Kind {
	case "raw":
		b, err := hex.DecodeString(sf.Hex)
		if err != nil {
			t.Fatal(err)
		}
		return b
	case "error":
		b, err := Encode(Frame{Type: ServerError, Serialization: SerJSON, Code: sf.Code, Payload: sf.Payload})
		if err != nil {
			t.Fatal(err)
		}
		return b
	case "", "response":
		f := Frame{Type: FullServerResponse, Flags: FlagSeq, Serialization: SerJSON, Compression: CompGzip, Seq: sf.Seq, Payload: sf.Payload}
		if sf.Last {
			f.Flags |= FlagLast
		}
		b, err := Encode(f)
		if err != nil {
			t.Fatal(err)
		}
		return b
	}
	t.Fatalf("unknown scripted frame kind %q", sf.Kind)
	return nil
}

// server is a fake service: it records the handshake and the client's
// frames (as text for a golden file) and plays a script back.
type server struct {
	URL   string
	mu    sync.Mutex
	log   strings.Builder
	audio bytes.Buffer // the client's PCM, decoded
	done  chan struct{}
}

func newServer(t *testing.T, path string, sc script) *server {
	s := &server{done: make(chan struct{})}
	var once sync.Once
	s.URL = asrtest.WSServer(t, path, func(c *asrtest.Conn) {
		defer once.Do(func() { close(s.done) })
		s.record(describeHandshake(c))
		frames := slices.Clone(sc.Frames)
		send := func(after int) bool {
			for len(frames) > 0 && frames[0].After == after {
				sf := frames[0]
				frames = frames[1:]
				if sf.Kind == "close" {
					s.record("(server closes)\n")
					return false
				}
				c.WriteBinary(sf.encode(t))
				if sf.Kind == "error" || sf.Kind == "raw" {
					// The service ends the connection after an error;
					// stopping here also keeps the log deterministic.
					s.record("(server sent " + sf.Kind + " and closes)\n")
					return false
				}
			}
			return true
		}
		for n := 1; ; n++ {
			typ, data, ok := c.TryRead()
			if !ok {
				s.record("(client closed)\n")
				return
			}
			if typ != websocket.MessageBinary {
				t.Errorf("client sent a text message: %q", data)
				return
			}
			f, err := Decode(data)
			if err != nil {
				t.Errorf("client frame %d: %v", n, err)
				return
			}
			s.record(s.describe(t, f))
			if !send(n) {
				return
			}
			if f.Type == AudioOnlyRequest && f.Last() {
				send(-1)
				return
			}
		}
	})
	return s
}

func (s *server) record(line string) {
	s.mu.Lock()
	s.log.WriteString(line)
	s.mu.Unlock()
}

// wait returns what the server saw once its connection ended.
func (s *server) wait(t testing.TB) string {
	t.Helper()
	select {
	case <-s.done:
	case <-time.After(5 * time.Second):
		t.Fatal("the server's connection did not end")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.log.String()
}

var uuidRE = regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`)

func describeHandshake(c *asrtest.Conn) string {
	var b strings.Builder
	fmt.Fprintf(&b, "GET %s\n", c.Req.URL.Path)
	var names []string
	for k := range c.Req.Header {
		if strings.HasPrefix(k, "X-Api-") {
			names = append(names, k)
		}
	}
	slices.Sort(names)
	for _, k := range names {
		v := c.Req.Header.Get(k)
		if k == "X-Api-Connect-Id" || k == "X-Api-Request-Id" {
			if !uuidRE.MatchString(v) {
				c.T.Errorf("%s %q is not a UUID", k, v)
			}
			if v != c.Req.Header.Get("X-Api-Connect-Id") {
				c.T.Errorf("%s differs from X-Api-Connect-Id", k)
			}
			v = "<uuid>"
		}
		fmt.Fprintf(&b, "%s: %s\n", k, v)
	}
	return b.String()
}

func flagNames(f uint8) string {
	var n []string
	for _, x := range []struct {
		bit  uint8
		name string
	}{{FlagSeq, "seq"}, {FlagLast, "last"}, {FlagEvent, "event"}} {
		if f&x.bit != 0 {
			n = append(n, x.name)
		}
	}
	if len(n) == 0 {
		return "none"
	}
	return strings.Join(n, "|")
}

func (s *server) describe(t testing.TB, f Frame) string {
	ser := map[uint8]string{SerNone: "none", SerJSON: "json"}[f.Serialization]
	comp := map[uint8]string{CompNone: "none", CompGzip: "gzip"}[f.Compression]
	line := fmt.Sprintf("> %s flags=%s ser=%s comp=%s", f.Type, flagNames(f.Flags), ser, comp)
	if f.Flags&FlagSeq != 0 {
		line += fmt.Sprintf(" seq=%d", f.Seq)
	}
	if f.Type == AudioOnlyRequest {
		s.mu.Lock()
		s.audio.Write(f.Payload)
		s.mu.Unlock()
		return line + fmt.Sprintf(" bytes=%d\n", len(f.Payload))
	}
	return line + "\n" + prettyJSON(t, f.Payload) + "\n"
}

// ---- Driving a stream ----

const testApp, testToken = "test-app", "test-token"

func testClient(t testing.TB, url string, mutate ...func(*asr.Config)) *Client {
	t.Helper()
	cfg := asr.Config{Endpoint: url, AppID: testApp, Key: testToken}
	for _, m := range mutate {
		m(&cfg)
	}
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	c.finalTimeout = 2 * time.Second
	return c
}

// pcm is d of a recognisable pattern (not silence), so that the server
// can check the audio arrives intact.
func pcm(d time.Duration) []byte {
	b := make([]byte, int(d.Seconds()*asr.SampleRate)*2)
	for i := range b {
		b[i] = byte(i * 7)
	}
	return b
}

// drive opens a stream, sends audio in 100 ms pieces as the helper does,
// finishes, and returns every event as text.
func drive(t *testing.T, c *Client, req asr.StreamRequest, audio []byte) (string, []asr.Event) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	st, err := c.Open(ctx, req)
	if err != nil {
		return "open: " + describeErr(err) + "\n", nil
	}
	defer st.Close()
	go func() {
		for _, fr := range asr.Frames(audio, 100*time.Millisecond, 0) {
			if st.Send(fr) != nil {
				return
			}
		}
		st.Finish()
	}()
	return collect(t, st)
}

func collect(t *testing.T, st asr.Stream) (string, []asr.Event) {
	t.Helper()
	var b strings.Builder
	var evs []asr.Event
	timeout := time.After(5 * time.Second)
	for {
		select {
		case ev, ok := <-st.Events():
			if !ok {
				return b.String(), evs
			}
			evs = append(evs, ev)
			switch ev.Kind {
			case asr.EventPartial:
				fmt.Fprintf(&b, "< partial: %s\n", ev.Text)
			case asr.EventFinal:
				fmt.Fprintf(&b, "< final: %s (model %s, duration %s)\n", ev.Text, ev.Result.Model, ev.Result.Duration)
				if ev.Text != ev.Result.Text {
					t.Errorf("final text %q, result %q", ev.Text, ev.Result.Text)
				}
			case asr.EventError:
				fmt.Fprintf(&b, "< error: %s\n", describeErr(ev.Err))
			}
		case <-timeout:
			t.Fatal("events did not close")
		}
	}
}

func describeErr(err error) string {
	var e *asr.Error
	if !errors.As(err, &e) {
		return "not an *asr.Error: " + err.Error()
	}
	return fmt.Sprintf("kind=%s status=%d code=%q retryable=%v detail=%q", e.Kind, e.Status, e.Code, e.Retryable, e.Detail)
}

// checkEvents checks the Stream contract: partials, then exactly one
// final or error, last.
func checkEvents(t *testing.T, evs []asr.Event) {
	t.Helper()
	for i, ev := range evs {
		last := i == len(evs)-1
		if (ev.Kind == asr.EventFinal || ev.Kind == asr.EventError) != last {
			t.Errorf("event %d (%s) of %d", i, ev.Kind, len(evs))
		}
	}
}

// ---- Protocol tests (golden: what the client sent, and what it made of
// the answers) ----

const asyncPath = "/api/v3/sauc/bigmodel_async"

func TestStreamNormal(t *testing.T) {
	srv := newServer(t, asyncPath, loadScript(t, "normal"))
	audio := pcm(1100 * time.Millisecond)
	events, evs := drive(t, testClient(t, srv.URL), asr.StreamRequest{}, audio)
	got := srv.wait(t)
	checkEvents(t, evs)
	asrtest.Golden(t, "normal", got+"--- events\n"+events)
	if !bytes.Equal(srv.audio.Bytes(), audio) {
		t.Errorf("the server got %d bytes of audio, not the %d sent", srv.audio.Len(), len(audio))
	}
	if f := evs[len(evs)-1]; f.Kind != asr.EventFinal || f.Result.Latency <= 0 || f.Result.Latency > 2*time.Second {
		t.Errorf("final %+v", f)
	}
}

func TestStreamHints(t *testing.T) {
	srv := newServer(t, asyncPath, loadScript(t, "short"))
	c := testClient(t, srv.URL, func(cfg *asr.Config) { cfg.Model = ModelV1 })
	req := asr.StreamRequest{Lang: "zh-TW", Hints: []string{"Voxt", " voxt ", "火山引擎"}, Prompt: "Meeting about Voxt."}
	events, evs := drive(t, c, req, pcm(400*time.Millisecond))
	got := srv.wait(t)
	checkEvents(t, evs)
	asrtest.Golden(t, "hints", got+"--- events\n"+events)
}

// The new console's API key: X-Api-Key alone, no App ID or Access Token.
func TestStreamAPIKey(t *testing.T) {
	srv := newServer(t, asyncPath, loadScript(t, "short"))
	c := testClient(t, srv.URL, func(cfg *asr.Config) { cfg.AppID = "" })
	events, evs := drive(t, c, asr.StreamRequest{}, pcm(400*time.Millisecond))
	got := srv.wait(t)
	checkEvents(t, evs)
	asrtest.Golden(t, "apikey", got+"--- events\n"+events)
}

func TestStreamErrorFrames(t *testing.T) {
	for _, name := range []string{"error-open", "error-quota", "error-busy", "bad-gzip", "early-close"} {
		t.Run(name, func(t *testing.T) {
			srv := newServer(t, asyncPath, loadScript(t, name))
			events, evs := drive(t, testClient(t, srv.URL), asr.StreamRequest{}, pcm(time.Second))
			got := srv.wait(t)
			checkEvents(t, evs)
			asrtest.Golden(t, name, got+"--- events\n"+events)
		})
	}
}

// TestStreamErrorHidesToken: an error frame mid-stream that quotes the
// access token ends the stream with the token blanked out.
func TestStreamErrorHidesToken(t *testing.T) {
	srv := newServer(t, asyncPath, script{Frames: []scriptFrame{
		{After: 1, Seq: 1, Payload: json.RawMessage(`{"audio_info":{"duration":0},"result":{"text":""}}`)},
		{After: 3, Kind: "error", Code: 45000001, Payload: json.RawMessage(`{"error":"invalid access token ` + testToken + `"}`)},
	}})
	_, evs := drive(t, testClient(t, srv.URL), asr.StreamRequest{}, pcm(time.Second))
	if len(evs) == 0 || evs[len(evs)-1].Kind != asr.EventError {
		t.Fatalf("events %+v", evs)
	}
	if err := evs[len(evs)-1].Err; strings.Contains(err.Error(), testToken) || !strings.Contains(err.Error(), "[key]") {
		t.Fatalf("the stream's error: %v", err)
	}
}

func TestHandshakeRefused(t *testing.T) {
	cases := []struct {
		status int
		body   string
		kind   asr.Kind
	}{
		{401, `{"error":"load grant: requested grant not found"}`, asr.KindAuth},
		{403, `{"error":"[resource_id=volc.seedasr.sauc.duration] requested resource not granted"}`, asr.KindQuota},
		{500, `oops`, asr.KindServer},
	}
	for _, c := range cases {
		url := asrtest.RefuseServer(t, c.status, c.body)
		_, err := testClient(t, url).Open(context.Background(), asr.StreamRequest{})
		var e *asr.Error
		if !errors.As(err, &e) || e.Kind != c.kind || e.Status != c.status || e.Provider != ID {
			t.Errorf("%d: %v, want %s", c.status, err, c.kind)
		}
		if strings.Contains(err.Error(), testToken) {
			t.Errorf("the error shows the token: %v", err)
		}
	}
}

func TestCancel(t *testing.T) {
	srv := newServer(t, asyncPath, loadScript(t, "short"))
	ctx, cancel := context.WithCancel(context.Background())
	st, err := testClient(t, srv.URL).Open(ctx, asr.StreamRequest{})
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	if err := st.Send(pcm(400 * time.Millisecond)); err != nil {
		t.Fatal(err)
	}
	cancel()
	events, evs := collect(t, st)
	checkEvents(t, evs)
	if len(evs) == 0 || asr.KindOf(evs[len(evs)-1].Err) != asr.KindCancelled {
		t.Fatalf("events after cancel:\n%s", events)
	}
	if err := st.Send(pcm(200 * time.Millisecond)); asr.KindOf(err) != asr.KindCancelled {
		t.Errorf("Send after cancel: %v", err)
	}
	srv.wait(t)
}

func TestCancelOpen(t *testing.T) {
	// The server never answers the request.
	url := asrtest.WSServer(t, asyncPath, func(c *asrtest.Conn) { c.TryRead(); c.TryRead() })
	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(100*time.Millisecond, cancel)
	_, err := testClient(t, url).Open(ctx, asr.StreamRequest{})
	if asr.KindOf(err) != asr.KindCancelled {
		t.Errorf("Open: %v, want cancelled", err)
	}
	// And without a cancel, Open gives up after the ack timeout.
	c := testClient(t, url)
	c.ackTimeout = 100 * time.Millisecond
	_, err = c.Open(context.Background(), asr.StreamRequest{})
	if asr.KindOf(err) != asr.KindNetwork {
		t.Errorf("Open: %v, want network", err)
	}
}

func TestFinalTimeout(t *testing.T) {
	// Acknowledges, then never sends the final result.
	url := asrtest.WSServer(t, asyncPath, func(c *asrtest.Conn) {
		c.Read()
		c.WriteBinary(scriptFrame{Seq: 1, Payload: json.RawMessage(`{"result":{"text":""}}`)}.encode(t))
		for {
			if _, _, ok := c.TryRead(); !ok {
				return
			}
		}
	})
	c := testClient(t, url)
	c.finalTimeout = 150 * time.Millisecond
	events, evs := drive(t, c, asr.StreamRequest{}, pcm(300*time.Millisecond))
	checkEvents(t, evs)
	if len(evs) != 1 || asr.KindOf(evs[0].Err) != asr.KindNetwork || !strings.Contains(events, "no final result in time") {
		t.Fatalf("events:\n%s", events)
	}
}

func TestClose(t *testing.T) {
	srv := newServer(t, asyncPath, loadScript(t, "short"))
	st, err := testClient(t, srv.URL).Open(context.Background(), asr.StreamRequest{})
	if err != nil {
		t.Fatal(err)
	}
	st.Send(pcm(200 * time.Millisecond))
	st.Close()
	st.Close()
	for ev := range st.Events() {
		if ev.Kind != asr.EventPartial {
			t.Errorf("event after Close: %s", ev.Kind)
		}
	}
	if err := st.Send(pcm(200 * time.Millisecond)); asr.KindOf(err) != asr.KindCancelled {
		t.Errorf("Send after Close: %v", err)
	}
	if err := st.Finish(); err == nil {
		t.Error("Finish after Close succeeded")
	}
	srv.wait(t)
}

func TestStreamAll(t *testing.T) {
	srv := newServer(t, asyncPath, loadScript(t, "normal"))
	var partials []string
	r, err := asr.StreamAll(context.Background(), testClient(t, srv.URL), asr.StreamRequest{}, pcm(1100*time.Millisecond),
		asr.Feed{OnPartial: func(s string) { partials = append(partials, s) }})
	if err != nil {
		t.Fatal(err)
	}
	if r.Text != "这是字节跳动，今日头条母公司。" || r.Model != DefaultModel || r.Duration != 1100*time.Millisecond || len(partials) == 0 {
		t.Errorf("%+v, partials %q", r, partials)
	}
}

func TestOpenRejectsRate(t *testing.T) {
	c := testClient(t, "ws://127.0.0.1:1/x")
	if _, err := c.Open(context.Background(), asr.StreamRequest{SampleRate: 48000}); asr.KindOf(err) != asr.KindAudio {
		t.Errorf("48 kHz: %v", err)
	}
}

func TestNew(t *testing.T) {
	for _, cfg := range []asr.Config{
		{},
		{AppID: "a"},
		{AppID: "a", Key: "  "},
		{AppID: "a", Key: "k", Endpoint: "ws://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"},
		{AppID: "a", Key: "k", Endpoint: "https://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"},
		{AppID: "a", Key: "k", Endpoint: "wss://user:pw@openspeech.bytedance.com/"},
	} {
		if _, err := New(cfg); asr.KindOf(err) != asr.KindConfig {
			t.Errorf("%+v: %v, want a config error", cfg, err)
		}
	}
	c, err := New(asr.Config{AppID: "a", Key: "k"})
	if err != nil || c.Endpoint() != DefaultEndpoint || c.Model() != DefaultModel || c.mode != modeAsync {
		t.Fatalf("defaults: %v %v", c, err)
	}
	if c, err := New(asr.Config{Key: "api-key"}); err != nil || c.authHeader().Get("X-Api-Key") != "api-key" || c.authHeader().Get("X-Api-App-Key") != "" {
		t.Errorf("API key only: %v %v", c, err)
	}
	c, _ = New(asr.Config{AppID: "a", Key: "k", Model: ModelV1})
	if c.Endpoint() != DuplexEndpoint || c.mode != modeDuplex {
		t.Errorf("v1: %s", c.Endpoint())
	}
	c, _ = New(asr.Config{AppID: "a", Key: "k", Endpoint: NostreamEndpoint})
	if c.mode != modeNostream {
		t.Errorf("nostream: %d", c.mode)
	}
	if c, err := New(asr.Config{AppID: "a", Key: "k", Endpoint: "ws://localhost:8080/asr"}); err != nil || c.mode != modeAsync {
		t.Errorf("loopback ws: %v", err)
	}
	if got := c.Caps(); got != (asr.Caps{Stream: true, Hints: true, Prompt: true}) {
		t.Errorf("caps %+v", got)
	}
	c, _ = New(asr.Config{AppID: "app-1234", Key: "secret-token"})
	if s := fmt.Sprint(c); strings.Contains(s, "secret-token") || strings.Contains(s, "app-1234") {
		t.Errorf("String shows credentials: %s", s)
	}
}
