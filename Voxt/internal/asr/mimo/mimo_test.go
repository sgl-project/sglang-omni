package mimo

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
)

// TestTranscribe: the request Voxt 1.x sent (the recording as a WAV data
// URI, the language as asr_options), the answer's text, and errors.
func TestTranscribe(t *testing.T) {
	var got map[string]any
	status, answer := 200, `{"choices":[{"message":{"role":"assistant","content":" 你好，Voxt。 "}}]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer mimo-key" {
			t.Errorf("%s %s, auth %q", r.Method, r.URL.Path, r.Header.Get("Authorization"))
		}
		b, _ := io.ReadAll(r.Body)
		json.Unmarshal(b, &got)
		w.WriteHeader(status)
		io.WriteString(w, answer)
	}))
	defer srv.Close()
	c, err := New(asr.Config{Key: "mimo-key", Endpoint: srv.URL + "/v1"})
	if err != nil {
		t.Fatal(err)
	}
	pcm := asr.Silence(500 * time.Millisecond)
	res, err := c.Transcribe(context.Background(), asr.FileRequest{Audio: pcm, Format: "pcm", Lang: "zh-Hans", Hints: []string{"ignored"}})
	if err != nil || res.Text != "你好，Voxt。" || res.Model != DefaultModel {
		t.Fatalf("%+v, %v", res, err)
	}
	msg := got["messages"].([]any)[0].(map[string]any)
	part := msg["content"].([]any)[0].(map[string]any)
	uri := part["input_audio"].(map[string]any)["data"].(string)
	wav, _ := base64.StdEncoding.DecodeString(strings.TrimPrefix(uri, "data:audio/wav;base64,"))
	if back, rate, _, err := asr.ReadWAV(wav); err != nil || len(back) != len(pcm) || rate != asr.SampleRate || msg["role"] != "user" || part["type"] != "input_audio" {
		t.Errorf("audio part %v (%v)", part["type"], err)
	}
	if got["model"] != DefaultModel || got["asr_options"].(map[string]any)["language"] != "zh" {
		t.Errorf("request %v %v", got["model"], got["asr_options"])
	}
	if strings.Contains(string(must(json.Marshal(got))), "ignored") {
		t.Error("hints were sent")
	}

	status, answer = 401, `{"error":{"code":"401","message":"Invalid API Key","type":"invalid_request_error"}}`
	if _, err := c.Transcribe(context.Background(), asr.FileRequest{Audio: pcm, Format: "pcm"}); asr.KindOf(err) != asr.KindAuth || strings.Contains(err.Error(), "mimo-key") {
		t.Errorf("401: %v", err)
	}
	if language("ja") != "auto" || language("en-US") != "en" {
		t.Error("language")
	}
}

func TestNew(t *testing.T) {
	if _, err := New(asr.Config{}); asr.KindOf(err) != asr.KindConfig {
		t.Errorf("no key: %v", err)
	}
	if _, err := New(asr.Config{Key: "k", Endpoint: "http://api.xiaomimimo.com/v1"}); asr.KindOf(err) != asr.KindConfig {
		t.Errorf("plain http: %v", err)
	}
	if c, _ := New(asr.Config{Key: "k"}); c.endpoint != DefaultEndpoint {
		t.Errorf("endpoint %s", c.endpoint)
	}
}

func must(b []byte, _ error) []byte { return b }
