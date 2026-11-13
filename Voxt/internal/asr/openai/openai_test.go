package openai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/asrtest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
)

// recorder is a fake transcription server: it prints each request (its
// headers and multipart parts) and answers with a testdata file.
type recorder struct {
	log     strings.Builder
	status  []int
	answers []string
	calls   int
}

func (r *recorder) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	fmt.Fprintf(&r.log, "%s %s\n", req.Method, req.URL.Path)
	for _, h := range []string{"Authorization", "Accept", "User-Agent"} {
		fmt.Fprintf(&r.log, "%s: %s\n", h, req.Header.Get(h))
	}
	mt, params, err := mime.ParseMediaType(req.Header.Get("Content-Type"))
	fmt.Fprintf(&r.log, "Content-Type: %s (boundary set: %v, %v)\n", mt, params["boundary"] != "", err)
	mr := multipart.NewReader(req.Body, params["boundary"])
	for {
		p, err := mr.NextPart()
		if err != nil {
			break
		}
		data, _ := io.ReadAll(p)
		if p.FileName() != "" {
			pcm, rate, ch, err := asr.ReadWAV(data)
			fmt.Fprintf(&r.log, "part %s: file %q %s, %d bytes; wav %d Hz %d ch %d PCM bytes %v\n",
				p.FormName(), p.FileName(), p.Header.Get("Content-Type"), len(data), rate, ch, len(pcm), err)
			continue
		}
		fmt.Fprintf(&r.log, "part %s: %q\n", p.FormName(), data)
	}
	r.log.WriteString("--\n")
	i := min(r.calls, len(r.answers)-1)
	r.calls++
	body, err := os.ReadFile(filepath.Join("testdata", r.answers[i]))
	if err != nil {
		body = []byte(r.answers[i])
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(r.status[i])
	w.Write(body)
}

func serve(t *testing.T, r *recorder) string {
	srv := httptest.NewServer(r)
	t.Cleanup(srv.Close)
	return srv.URL + "/v1"
}

func newClient(t *testing.T, endpoint string, cfg asr.Config) *Client {
	t.Helper()
	cfg.Endpoint, cfg.Key = endpoint, "sk-test"
	c, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	c.Sleep = func(time.Duration) {}
	return c
}

// TestGoldenRequest checks what the client uploads: a WAV built from the
// PCM, the model, json, the language and the prompt with the hints.
func TestGoldenRequest(t *testing.T) {
	r := &recorder{status: []int{200}, answers: []string{"response-json.json"}}
	c := newClient(t, serve(t, r), asr.Config{})
	res, err := c.Transcribe(context.Background(), asr.FileRequest{
		Audio: asr.Silence(time.Second), Format: "pcm", Lang: "zh-Hans", Hints: []string{"Voxt", "DeepSeek", " "}, Prompt: "Voxt, a voice input app.",
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "今天我们用 Voxt 测试一下 speech recognition。" || res.Model != DefaultModel {
		t.Errorf("result %+v", res)
	}
	asrtest.Golden(t, "request-pcm-hints", r.log.String())
}

func TestGoldenRequestPlain(t *testing.T) {
	r := &recorder{status: []int{200}, answers: []string{"response-verbose.json"}}
	c := newClient(t, serve(t, r)+"/", asr.Config{Model: "whisper-large-v3-turbo"})
	res, err := c.Transcribe(context.Background(), asr.FileRequest{Audio: asr.WAV(asr.Silence(500*time.Millisecond), 0), Format: "wav"})
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "今天我们用 Voxt 测试一下 speech recognition。" || res.Lang != "chinese" || res.Duration != 5200*time.Millisecond {
		t.Errorf("result %+v", res)
	}
	asrtest.Golden(t, "request-wav-plain", r.log.String())
}

// TestGoldenRequestBasic: Zhipu's GLM-ASR gets the file, the model and
// the prompt with the hints, no response_format or language.
func TestGoldenRequestBasic(t *testing.T) {
	r := &recorder{status: []int{200}, answers: []string{"response-json.json"}}
	c := newClient(t, serve(t, r), asr.Config{Model: "glm-asr-2512"})
	c.Basic = true
	if _, err := c.Transcribe(context.Background(), asr.FileRequest{
		Audio: asr.Silence(time.Second), Format: "pcm", Lang: "zh-Hans", Hints: []string{"Voxt", "智谱"},
	}); err != nil {
		t.Fatal(err)
	}
	asrtest.Golden(t, "request-basic", r.log.String())
}

func TestErrors(t *testing.T) {
	for _, tc := range []struct {
		status   int
		answer   string
		kind     asr.Kind
		calls    int
		sentinel error
	}{
		{401, "error-401.json", asr.KindAuth, 1, nil},
		{429, "error-429-quota.json", asr.KindQuota, 1, core.ErrBusy},
		{400, "error-400-model.json", asr.KindConfig, 1, core.ErrInvalid},
		{413, `{"error":{"message":"Maximum content size limit exceeded"}}`, asr.KindAudio, 1, core.ErrInvalid},
		{503, `{"error":{"message":"overloaded"}}`, asr.KindServer, 2, nil},
		{200, `{"error":{"message":"internal"}}`, asr.KindServer, 2, nil},
		{200, `{"task":"transcribe"}`, asr.KindProtocol, 1, nil},
	} {
		t.Run(fmt.Sprint(tc.status, tc.answer), func(t *testing.T) {
			r := &recorder{status: []int{tc.status}, answers: []string{tc.answer}}
			c := newClient(t, serve(t, r), asr.Config{})
			_, err := c.Transcribe(context.Background(), asr.FileRequest{Audio: asr.Silence(100 * time.Millisecond), Format: "pcm"})
			if asr.KindOf(err) != tc.kind {
				t.Fatalf("kind %q, want %q: %v", asr.KindOf(err), tc.kind, err)
			}
			if r.calls != tc.calls {
				t.Errorf("%d calls, want %d", r.calls, tc.calls)
			}
			if tc.sentinel != nil && !errors.Is(err, tc.sentinel) {
				t.Errorf("%v is not %v", err, tc.sentinel)
			}
			if strings.Contains(err.Error(), "Bearer") {
				t.Errorf("error leaks the header: %v", err)
			}
		})
	}
}

func TestRetryThenSuccess(t *testing.T) {
	r := &recorder{status: []int{502, 200}, answers: []string{`bad gateway`, "response-json.json"}}
	c := newClient(t, serve(t, r), asr.Config{})
	res, err := c.Transcribe(context.Background(), asr.FileRequest{Audio: asr.Silence(100 * time.Millisecond), Format: "pcm"})
	if err != nil || r.calls != 2 || res.Text == "" {
		t.Fatalf("%v after %d calls: %+v", err, r.calls, res)
	}
}

func TestCancelAndNetwork(t *testing.T) {
	block := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { <-block }))
	t.Cleanup(func() { close(block); srv.Close() })
	c := newClient(t, srv.URL+"/v1", asr.Config{})
	ctx, cancel := context.WithCancel(context.Background())
	go func() { time.Sleep(50 * time.Millisecond); cancel() }()
	_, err := c.Transcribe(ctx, asr.FileRequest{Audio: asr.Silence(100 * time.Millisecond), Format: "pcm"})
	if asr.KindOf(err) != asr.KindCancelled || !errors.Is(err, core.ErrCancelled) {
		t.Fatalf("cancel: %v", err)
	}

	dead := httptest.NewServer(http.NotFoundHandler())
	url := dead.URL
	dead.Close()
	c = newClient(t, url+"/v1", asr.Config{})
	_, err = c.Transcribe(context.Background(), asr.FileRequest{Audio: asr.Silence(100 * time.Millisecond), Format: "pcm"})
	if asr.KindOf(err) != asr.KindNetwork {
		t.Fatalf("refused: %v", err)
	}
}

func TestConfig(t *testing.T) {
	if _, err := New(asr.Config{}); asr.KindOf(err) != asr.KindConfig {
		t.Errorf("no key: %v", err)
	}
	for _, ep := range []string{"http://api.example.com/v1", "ftp://x/y", "https://user:pw@api.example.com/v1", "not a url"} {
		if _, err := New(asr.Config{Key: "k", Endpoint: ep}); asr.KindOf(err) != asr.KindConfig {
			t.Errorf("%s: %v", ep, err)
		}
	}
	for in, want := range map[string]string{
		"":                                     DefaultEndpoint,
		"https://api.groq.com/openai/v1":       "https://api.groq.com/openai/v1/audio/transcriptions",
		"https://api.groq.com/openai/v1/":      "https://api.groq.com/openai/v1/audio/transcriptions",
		"https://x.test/v1/audio/translations": "https://x.test/v1/audio/translations",
		"http://127.0.0.1:8080/v1?k=1":         "http://127.0.0.1:8080/v1/audio/transcriptions?k=1",
	} {
		if got := Endpoint(in); got != want {
			t.Errorf("Endpoint(%q) = %q, want %q", in, got, want)
		}
	}
	c, err := New(asr.Config{Key: "k", Endpoint: "http://localhost:9000/v1"})
	if err != nil || c.Caps().File != true || c.Caps().Stream || c.Model() != DefaultModel {
		t.Errorf("loopback: %v %+v", err, c)
	}
	big := &Client{model: DefaultModel}
	if _, err := big.Transcribe(context.Background(), asr.FileRequest{Audio: make([]byte, MaxBytes+1), Format: "wav"}); asr.KindOf(err) != asr.KindAudio {
		t.Errorf("too large: %v", err)
	}
}

func TestPrompt(t *testing.T) {
	if got := Prompt("", nil); got != "" {
		t.Errorf("%q", got)
	}
	if got := Prompt(" Names: ", []string{"A", "", "B"}); got != "Names:\nA, B" {
		t.Errorf("%q", got)
	}
	long := Prompt(strings.Repeat("词", 400), nil)
	if len(long) > maxPrompt || !strings.HasPrefix(long, "词") || strings.ToValidUTF8(long, "?") != long {
		t.Errorf("clipped badly: %d bytes", len(long))
	}
}

// TestKeyless: a server on this Mac may take no key (no Authorization
// header goes); any other address needs one.
func TestKeyless(t *testing.T) {
	auth := "unset"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")
		io.WriteString(w, `{"text":"ok"}`)
	}))
	defer srv.Close()
	c, err := New(asr.Config{Endpoint: srv.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}
	if res, err := c.Transcribe(context.Background(), asr.FileRequest{Audio: asr.Silence(time.Second), Format: "pcm"}); err != nil || res.Text != "ok" || auth != "" {
		t.Errorf("keyless: %+v %v, Authorization %q", res, err, auth)
	}
	if _, err := New(asr.Config{Endpoint: "https://api.openai.com/v1"}); asr.KindOf(err) != asr.KindConfig {
		t.Errorf("a remote server without a key: %v", err)
	}
}

// TestNoSpeech: StepFun answers silence with a 400 "no speech found";
// that is an empty text, not an error, and is not retried.
func TestNoSpeech(t *testing.T) {
	r := &recorder{status: []int{400}, answers: []string{"error-400-no-speech-stepfun.json"}}
	c := newClient(t, serve(t, r), asr.Config{Model: "step-asr"})
	res, err := c.Transcribe(context.Background(), asr.FileRequest{Audio: asr.Silence(time.Second), Format: "pcm"})
	if err != nil {
		t.Fatalf("silence is an error: %v", err)
	}
	if res.Text != "" || res.Model != "step-asr" || r.calls != 1 {
		t.Errorf("result %+v after %d calls", res, r.calls)
	}
}

// TestLongRecordingInPieces: a 15-minute recording (over 25 MB as WAV)
// goes up in two uploads under the limit, cut in its pause, and the
// text comes back joined (#3840).
func TestLongRecordingInPieces(t *testing.T) {
	pcm := make([]byte, 15*60*asr.BytesPerSecond)
	for i := range pcm {
		pcm[i] = 0x40 // loud enough, everywhere but the pause
	}
	pause := 12 * 60 * asr.BytesPerSecond
	clear(pcm[pause : pause+asr.BytesPerSecond])
	r := &recorder{status: []int{200, 200}, answers: []string{`{"text":"First half."}`, `{"text":"后半段。"}`}}
	c := newClient(t, serve(t, r), asr.Config{})
	res, err := c.Transcribe(context.Background(), asr.FileRequest{Audio: pcm, Format: "pcm", SampleRate: asr.SampleRate})
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "First half.后半段。" || res.Model != DefaultModel {
		t.Errorf("result %+v", res)
	}
	var sizes []int
	for _, m := range regexp.MustCompile(`, (\d+) bytes; wav 16000 Hz 1 ch (\d+) PCM`).FindAllStringSubmatch(r.log.String(), -1) {
		n, _ := strconv.Atoi(m[1])
		if n > MaxBytes {
			t.Errorf("upload of %d bytes, over %d", n, MaxBytes)
		}
		p, _ := strconv.Atoi(m[2])
		sizes = append(sizes, p)
	}
	if len(sizes) != 2 || sizes[0]+sizes[1] != len(pcm) || sizes[0] <= pause || sizes[0] >= pause+asr.BytesPerSecond {
		t.Errorf("pieces %v of %d bytes; want two, cut in the pause at %d", sizes, len(pcm), pause)
	}
}
