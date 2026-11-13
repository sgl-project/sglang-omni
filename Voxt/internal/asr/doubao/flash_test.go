package doubao

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/asrtest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/dict"
)

// flashServer answers every request with status, headers and body, and
// counts the requests; dump prints the last one.
type flashServer struct {
	*httptest.Server
	calls atomic.Int32
	last  atomic.Pointer[string]
}

func newFlashServer(t *testing.T, status int, headers map[string]string, body string, pcm []byte) *flashServer {
	t.Helper()
	s := &flashServer{}
	s.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.calls.Add(1)
		raw, _ := io.ReadAll(r.Body)
		dump := dumpFlashRequest(t, r, raw, pcm)
		s.last.Store(&dump)
		for k, v := range headers {
			w.Header().Set(k, v)
		}
		w.WriteHeader(status)
		io.WriteString(w, body)
	}))
	t.Cleanup(s.Close)
	return s
}

// dumpFlashRequest prints a request for a golden file: the method, the
// path, the headers (the request id as <uuid>) and the JSON body, whose
// audio must be pcm as WAV.
func dumpFlashRequest(t *testing.T, r *http.Request, raw, pcm []byte) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s %s\n", r.Method, r.URL.Path)
	var names []string
	for k := range r.Header {
		if k != "Accept-Encoding" && k != "Content-Length" {
			names = append(names, k)
		}
	}
	sort.Strings(names)
	for _, k := range names {
		v := r.Header.Get(k)
		if k == "X-Api-Request-Id" && len(v) == 36 {
			v = "<uuid>"
		}
		fmt.Fprintf(&b, "%s: %s\n", k, v)
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		t.Errorf("body %q: %v", raw, err)
		return b.String()
	}
	audio := body["audio"].(map[string]any)
	data, _ := base64.StdEncoding.DecodeString(audio["data"].(string))
	if pcm != nil && !bytes.Equal(data, asr.WAV(pcm, asr.SampleRate)) {
		t.Errorf("audio.data is not the recording as WAV (%d bytes)", len(data))
	}
	audio["data"] = fmt.Sprintf("<%d bytes of WAV>", len(data))
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	enc.Encode(body)
	return b.String()
}

// flashOK is the docs' example answer.
var flashOK = map[string]string{"X-Api-Status-Code": "20000000", "X-Api-Message": "OK", "X-Tt-Logid": "20250718T1"}

const flashOKBody = `{"audio_info":{"duration":2499},"result":{"additions":{"duration":"2499"},"text":"关闭透传。","utterances":[{"end_time":1530,"start_time":450,"text":"关闭透传。"}]}}`

func testFlash(t *testing.T, endpoint, appID string) *Flash {
	t.Helper()
	f, err := NewFlash(asr.Config{Endpoint: endpoint, AppID: appID, Key: "token-123"})
	if err != nil {
		t.Fatal(err)
	}
	f.Sleep = func(time.Duration) {}
	return f
}

// TestFlashRequest: one POST of the recording as base64 WAV, with the
// old console's App ID and Access Token or the new console's API key, the
// dictionary's hot words, the prompt as dialog context and the language;
// the answer's text and duration.
func TestFlashRequest(t *testing.T) {
	pcm := bytes.Repeat([]byte{1, 0}, asr.SampleRate/10)
	var dumps []string
	for _, appID := range []string{"app-1", ""} {
		srv := newFlashServer(t, 200, flashOK, flashOKBody, pcm)
		res, err := testFlash(t, srv.URL+"/api/v3/auc/bigmodel/recognize/flash", appID).Transcribe(context.Background(),
			asr.FileRequest{Audio: pcm, Format: "pcm", SampleRate: asr.SampleRate, Lang: "ja", Hints: dict.Hints(dictionaryTerms()), Prompt: " Dictation in Chinese and English. "})
		if err != nil || res.Text != "关闭透传。" || res.Duration != 2499*time.Millisecond || res.Model != ModelFlash || srv.calls.Load() != 1 {
			t.Fatalf("result %+v, %v, %d calls", res, err, srv.calls.Load())
		}
		dumps = append(dumps, *srv.last.Load())
	}
	asrtest.Golden(t, "flash-request", strings.Join(dumps, "---\n"))
}

// TestFlashAnswers: the status code header decides, with its message;
// silence is no text; busy and internal errors are tried again once.
func TestFlashAnswers(t *testing.T) {
	for _, c := range []struct {
		name    string
		status  int
		headers map[string]string
		body    string
		kind    asr.Kind
		calls   int32
	}{
		{"silence", 200, map[string]string{"X-Api-Status-Code": "20000003", "X-Api-Message": "silent audio"}, `{}`, "", 1},
		{"bad parameters", 200, map[string]string{"X-Api-Status-Code": "45000001", "X-Api-Message": "invalid request param", "X-Tt-Logid": "L1"}, ``, asr.KindProtocol, 1},
		{"bad audio", 200, map[string]string{"X-Api-Status-Code": "45000151"}, `{"message":"audio format error"}`, asr.KindAudio, 1},
		{"busy", 200, map[string]string{"X-Api-Status-Code": "55000031", "X-Api-Message": "server busy"}, ``, asr.KindServer, 2},
		{"resource not opened", 403, nil, `{"message":"requested resource not granted"}`, asr.KindQuota, 1},
		{"bad token", 401, nil, `{"message":"grant not found"}`, asr.KindAuth, 1},
		{"unreadable", 200, flashOK, `not json`, asr.KindProtocol, 1},
	} {
		t.Run(c.name, func(t *testing.T) {
			srv := newFlashServer(t, c.status, c.headers, c.body, nil)
			res, err := testFlash(t, srv.URL, "app-1").Transcribe(context.Background(), asr.FileRequest{Audio: []byte{0, 0}, Format: "pcm"})
			var e *asr.Error
			switch {
			case c.kind == "" && (err != nil || res.Text != ""):
				t.Errorf("result %+v, %v", res, err)
			case c.kind != "" && (!errors.As(err, &e) || e.Kind != c.kind):
				t.Errorf("error %v, want kind %s", err, c.kind)
			case c.headers["X-Tt-Logid"] != "" && !strings.Contains(err.Error(), "logid "+c.headers["X-Tt-Logid"]):
				t.Errorf("no log id in %v", err)
			}
			if n := srv.calls.Load(); n != c.calls {
				t.Errorf("%d calls, want %d", n, c.calls)
			}
		})
	}
}

// TestFlashErrorHidesToken: the token never shows in an error, also when
// the service quotes it.
func TestFlashErrorHidesToken(t *testing.T) {
	srv := newFlashServer(t, 200, map[string]string{"X-Api-Status-Code": "45000001", "X-Api-Message": "bad key token-123"}, ``, nil)
	_, err := testFlash(t, srv.URL, "").Transcribe(context.Background(), asr.FileRequest{Audio: []byte{0, 0}, Format: "pcm"})
	if err == nil || strings.Contains(err.Error(), "token-123") {
		t.Fatalf("error %v", err)
	}
}

// TestFlashTooLarge: a recording over the docs' limit is not sent.
func TestFlashTooLarge(t *testing.T) {
	srv := newFlashServer(t, 200, flashOK, flashOKBody, nil)
	_, err := testFlash(t, srv.URL, "app-1").Transcribe(context.Background(), asr.FileRequest{Audio: make([]byte, FlashMaxBytes+1), Format: "wav"})
	var e *asr.Error
	if !errors.As(err, &e) || e.Kind != asr.KindAudio || srv.calls.Load() != 0 {
		t.Fatalf("error %v, %d calls", err, srv.calls.Load())
	}
}

// TestFlashLongPCMInPieces: a PCM recording over the limit as WAV goes
// up in two requests and the text comes back joined (#3840).
func TestFlashLongPCMInPieces(t *testing.T) {
	srv := newFlashServer(t, 200, flashOK, flashOKBody, nil)
	res, err := testFlash(t, srv.URL, "app-1").Transcribe(context.Background(), asr.FileRequest{Audio: make([]byte, FlashMaxBytes), Format: "pcm"})
	if err != nil || res.Text != "关闭透传。关闭透传。" || srv.calls.Load() != 2 {
		t.Fatalf("%+v %v, %d calls", res, err, srv.calls.Load())
	}
}

// TestNewFlash: a key is required, the address must be https:// (http://
// only on this Mac), and the default is the service's.
func TestNewFlash(t *testing.T) {
	for _, cfg := range []asr.Config{
		{AppID: "app-1"},
		{},
		{Key: "k", Endpoint: "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"},
		{Key: "k", Endpoint: "http://relay.example/flash"},
	} {
		if _, err := NewFlash(cfg); err == nil {
			t.Errorf("%+v accepted", cfg)
		}
	}
	f, err := NewFlash(asr.Config{Key: "token-123"})
	if err != nil || f.Endpoint() != FlashEndpoint || f.Caps().Stream || !f.Caps().File || strings.Contains(f.String(), "token-123") {
		t.Fatalf("%v %v %+v", f, err, f.Caps())
	}
}
