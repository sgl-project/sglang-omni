package asr_test

import (
	"bytes"
	"context"
	"errors"
	"net/http"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/asrtest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
)

func TestCheckEndpoint(t *testing.T) {
	ok := map[string][]string{
		"wss://openspeech.bytedance.com/api/v3/sauc/bigmodel": {"wss"},
		"WSS://dashscope.aliyuncs.com/api-ws/v1/inference":    {"wss"},
		"ws://127.0.0.1:9000/x":                               {"wss"},
		"ws://[::1]:9000/x":                                   {"wss"},
		"http://localhost:8080/v1":                            {"https"},
		"https://api.openai.com/v1?x=1":                       {"https"},
	}
	for raw, schemes := range ok {
		if _, err := asr.CheckEndpoint("p", raw, schemes...); err != nil {
			t.Errorf("%s: %v", raw, err)
		}
	}
	bad := map[string][]string{
		"":                                    {"wss"},
		"ws://example.com/x":                  {"wss"},
		"https://example.com/x":               {"wss"},
		"http://192.168.1.2/v1":               {"https"},
		"https://user:pass@api.openai.com/v1": {"https"},
		"/v1/audio":                           {"https"},
		"localhost:8080":                      {"https"},
	}
	for raw, schemes := range bad {
		_, err := asr.CheckEndpoint("p", raw, schemes...)
		if asr.KindOf(err) != asr.KindConfig || !errors.Is(err, core.ErrInvalid) {
			t.Errorf("%q: %v", raw, err)
		}
		if strings.Contains(err.Error(), "pass@") {
			t.Errorf("leaks the password: %v", err)
		}
	}
	if got := asr.SanitizeEndpoint("wss://u:p@h.test/x?key=secret#f"); got != "wss://h.test/x" {
		t.Errorf("sanitized %q", got)
	}
}

func TestDialRefusedAndClosed(t *testing.T) {
	u := asrtest.RefuseServer(t, 401, `{"error":{"code":"InvalidApiKey","message":"Invalid API-key provided."}}`)
	_, _, err := asr.Dial(context.Background(), "p", u, asr.WSOptions{})
	var e *asr.Error
	if !errors.As(err, &e) || e.Kind != asr.KindAuth || e.Code != "InvalidApiKey" || e.Status != 401 {
		t.Fatalf("refused: %#v", err)
	}

	u = asrtest.WSServer(t, "/ws", func(c *asrtest.Conn) {
		c.Read()
		c.Close(websocket.StatusPolicyViolation, "quota exceeded for this model")
	})
	conn, _, err := asr.Dial(context.Background(), "p", u, asr.WSOptions{})
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	conn.Write(ctx, websocket.MessageText, []byte("hi"))
	_, _, err = conn.Read(ctx)
	if k := asr.WSError(ctx, "p", err).Kind; k != asr.KindQuota {
		t.Errorf("policy close: %s (%v)", k, err)
	}
}

// TestWSClient: a WebSocket's handshake goes through the proxy of the
// app's client, over HTTP/1.1.
func TestWSClient(t *testing.T) {
	proxy, _ := url.Parse("http://proxy.test:3128")
	base := &http.Client{Transport: &http.Transport{Proxy: http.ProxyURL(proxy)}}
	tr := asr.WSClient(base).Transport.(*http.Transport)
	req, _ := http.NewRequest("GET", "https://openspeech.bytedance.com/", nil)
	if got, _ := tr.Proxy(req); got == nil || got.Host != "proxy.test:3128" {
		t.Errorf("proxy %v", got)
	}
	if tr.TLSNextProto == nil || tr.ForceAttemptHTTP2 {
		t.Error("HTTP/2 is not off")
	}
	if tr := asr.WSClient(nil).Transport.(*http.Transport); tr.Proxy != nil {
		t.Error("a nil client has a proxy")
	}
}

func TestWAVAndFrames(t *testing.T) {
	pcm := make([]byte, 3200*3+100)
	for i := range pcm {
		pcm[i] = byte(i)
	}
	got, rate, ch, err := asr.ReadWAV(asr.WAV(pcm, 0))
	if err != nil || rate != 16000 || ch != 1 || !bytes.Equal(got, pcm) {
		t.Fatalf("round trip: %v %d %d %d", err, rate, ch, len(got))
	}
	if _, _, _, err := asr.ReadWAV([]byte("RIFF0000WAVE")); err == nil {
		t.Error("no data chunk accepted")
	}
	if d := asr.Duration(asr.Silence(2*time.Second), 0); d != 2*time.Second {
		t.Errorf("duration %v", d)
	}
	fr := asr.Frames(pcm, 100*time.Millisecond, 0)
	if len(fr) != 4 || len(fr[0]) != 3200 || len(fr[3]) != 100 {
		t.Errorf("frames %d", len(fr))
	}
	r := asr.Rechunk{Size: 6400}
	var n int
	for _, f := range fr {
		for _, full := range r.Add(f) {
			n++
			if len(full) != 6400 {
				t.Errorf("frame of %d", len(full))
			}
		}
	}
	if rest := r.Rest(); n != 1 || len(rest) != 3200+100 {
		t.Errorf("%d full, rest %d", n, len(rest))
	}
}

func TestHTTPErrorKinds(t *testing.T) {
	for _, tc := range []struct {
		status int
		msg    string
		kind   asr.Kind
		retry  bool
	}{
		{401, "invalid key", asr.KindAuth, false},
		{403, "Arrearage: the account is overdue", asr.KindQuota, false},
		{403, "forbidden", asr.KindAuth, false},
		{429, "rate limit", asr.KindQuota, true},
		{429, "insufficient_quota", asr.KindQuota, false},
		{413, "too big", asr.KindAudio, false},
		{400, "audio duration too long", asr.KindAudio, false},
		{400, "model foo does not exist", asr.KindConfig, false},
		{400, "bad field", asr.KindProtocol, false},
		{404, "", asr.KindConfig, false},
		{502, "", asr.KindServer, true},
	} {
		e := asr.HTTPError("p", tc.status, "", tc.msg)
		if e.Kind != tc.kind || e.Retryable != tc.retry {
			t.Errorf("%d %q: %s retry %v", tc.status, tc.msg, e.Kind, e.Retryable)
		}
	}
	e := asr.HTTPError("p", 401, "", "x")
	if !errors.Is(e, &asr.Error{Kind: asr.KindAuth}) || errors.Is(e, &asr.Error{Kind: asr.KindQuota}) {
		t.Error("Is by kind")
	}
	if c, m := asr.ErrorBody([]byte(`{"header":{"event":"task-failed","error_code":"InvalidApiKey","error_message":"Invalid API-key provided."}}`)); c != "InvalidApiKey" || m != "Invalid API-key provided." {
		t.Errorf("dashscope body %q %q", c, m)
	}
	if c, m := asr.ErrorBody([]byte(`{"code":45000001,"message":"bad"}`)); c != "45000001" || m != "bad" {
		t.Errorf("code body %q %q", c, m)
	}
}

// fakeStreamer echoes how many bytes it got as partials.
type fakeStreamer struct{ fail error }

func (fakeStreamer) Caps() asr.Caps { return asr.Caps{Stream: true} }

func (f fakeStreamer) Open(ctx context.Context, req asr.StreamRequest) (asr.Stream, error) {
	if f.fail != nil {
		return nil, f.fail
	}
	s := &fakeStream{ev: make(chan asr.Event, 100)}
	return s, nil
}

type fakeStream struct {
	ev chan asr.Event
	n  int
}

func (s *fakeStream) Send(p []byte) error {
	s.n += len(p)
	s.ev <- asr.Event{Kind: asr.EventPartial, Text: strings.Repeat("a", s.n/3200)}
	return nil
}
func (s *fakeStream) Finish() error {
	s.ev <- asr.Event{Kind: asr.EventFinal, Text: "done", Result: asr.Result{Model: "m"}}
	close(s.ev)
	return nil
}
func (s *fakeStream) Close()                   {}
func (s *fakeStream) Events() <-chan asr.Event { return s.ev }

func TestStreamAll(t *testing.T) {
	var partials []string
	res, err := asr.StreamAll(context.Background(), fakeStreamer{}, asr.StreamRequest{}, asr.Silence(500*time.Millisecond),
		asr.Feed{OnPartial: func(s string) { partials = append(partials, s) }})
	if err != nil || res.Text != "done" || res.Model != "m" || len(partials) != 5 || partials[4] != "aaaaa" {
		t.Fatalf("%v %+v %q", err, res, partials)
	}
	start := time.Now()
	if _, err := asr.StreamAll(context.Background(), fakeStreamer{}, asr.StreamRequest{}, asr.Silence(300*time.Millisecond), asr.Feed{Pace: 2}); err != nil {
		t.Fatal(err)
	}
	if d := time.Since(start); d < 100*time.Millisecond {
		t.Errorf("paced feed took %v", d)
	}
	want := &asr.Error{Kind: asr.KindAuth}
	if _, err := asr.Recognize(context.Background(), fakeStreamer{fail: want}, nil, "", nil, "", asr.Feed{}); err != want {
		t.Errorf("open error %v", err)
	}
}

func TestNoSpeech(t *testing.T) {
	for _, tc := range []struct {
		code, detail string
		want         bool
	}{
		{"", "no speech found", true},
		{"request_params_invalid", "No speech found", true},
		{"NO_VALID_AUDIO_ERROR", "", true},
		{"", "no valid audio", true},
		{"", "Speech not detected in the audio", true},
		{"", "invalid model", false},
		{"", "audio too long", false},
		{"InternalError", "", false},
	} {
		if got := asr.NoSpeech(tc.code, tc.detail); got != tc.want {
			t.Errorf("NoSpeech(%q, %q) = %v", tc.code, tc.detail, got)
		}
	}
}
