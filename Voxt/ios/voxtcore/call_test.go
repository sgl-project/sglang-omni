package voxtcore

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type replyOf[T any] struct {
	Result T      `json:"result"`
	Error  *Error `json:"error"`
}

func call[T any](t *testing.T, c *Calls, id, method string, params any) replyOf[T] {
	t.Helper()
	data, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	var r replyOf[T]
	if err := json.Unmarshal(c.Call(id, method, data), &r); err != nil {
		t.Fatal(err)
	}
	return r
}

func TestCallPresetsLeaveOutSignIn(t *testing.T) {
	r := call[[]Preset](t, &Calls{}, "", MethodPresets, nil)
	if r.Error != nil || len(r.Result) == 0 {
		t.Fatalf("reply = %+v", r)
	}
	ids := map[string]bool{}
	for _, p := range r.Result {
		ids[p.ID] = true
	}
	if ids["codex"] || !ids["openai"] || !ids["custom"] || !ids["ollama"] {
		t.Fatalf("presets %v: want the key and local services, not the ChatGPT sign-in", ids)
	}
}

func TestCallLanguages(t *testing.T) {
	r := call[[]Language](t, &Calls{}, "", MethodLanguages, nil)
	byCode := map[string]Language{}
	for _, l := range r.Result {
		byCode[l.Code] = l
	}
	if r.Error != nil || len(r.Result) < 50 || byCode["zh"].Chinese != "中文" || byCode["en"].Name != "English" || byCode["ja"].Native == "" {
		t.Fatalf("%d languages, zh %+v en %+v ja %+v", len(r.Result), byCode["zh"], byCode["en"], byCode["ja"])
	}
}

func TestCallProcessAndErrors(t *testing.T) {
	_, p := serveChat(t, "Hello.")
	c := &Calls{}
	ok := call[ProcessResult](t, c, "a", MethodProcess, ProcessRequest{Task: "rewrite", Text: "hello", Instruction: "fix", MainLanguage: "en", Provider: p})
	if ok.Error != nil || ok.Result.Text != "Hello." {
		t.Fatalf("reply = %+v", ok)
	}
	for _, tc := range []struct {
		method      string
		params      any
		code, field string
	}{
		{"nope", nil, CodeMethod, ""},
		{MethodProcess, "not an object", CodeParams, ""},
		{MethodProcess, ProcessRequest{Task: "transcribe", Provider: p}, CodeNoText, ""},
		{MethodProcess, ProcessRequest{Task: "translate", Text: "x", Provider: Provider{Preset: "nope"}}, CodeConfig, FieldPreset},
		{MethodTestProvider, Provider{Preset: "codex"}, CodeConfig, FieldSignIn},
		{MethodTestProvider, Provider{Preset: "openai"}, CodeConfig, FieldKey},
		{MethodTestProvider, Provider{Preset: "custom", Model: "m"}, CodeConfig, FieldAddress},
		{MethodTestProvider, Provider{Preset: "omlx"}, CodeConfig, FieldModel},
		{MethodTestProvider, Provider{Preset: "custom", BaseURL: "ftp://x/v1", Model: "m"}, CodeConfig, ""}, // Resolve's own error
	} {
		r := call[json.RawMessage](t, c, "", tc.method, tc.params)
		if r.Error == nil || r.Error.Code != tc.code || r.Error.Field != tc.field {
			t.Errorf("%s %v: error %+v, want code %s field %q", tc.method, tc.params, r.Error, tc.code, tc.field)
		}
	}
	if len(c.running) != 0 {
		t.Fatalf("%d calls still registered", len(c.running))
	}
}

func TestCallLanguageModelFailure(t *testing.T) {
	f := &fakeChat{t: t, answers: []fakeAnswer{{status: http.StatusUnauthorized}}}
	p := startChat(t, f)
	r := call[json.RawMessage](t, &Calls{}, "", MethodProcess, ProcessRequest{Task: "rewrite", Text: "x", Instruction: "y", Provider: p})
	if r.Error == nil || r.Error.Code != CodeConfig {
		t.Fatalf("error %+v, want %s: a refused key is the settings to fix", r.Error, CodeConfig)
	}
	f = &fakeChat{t: t, answers: []fakeAnswer{{status: http.StatusBadRequest}}}
	p = startChat(t, f)
	r = call[json.RawMessage](t, &Calls{}, "", MethodProcess, ProcessRequest{Task: "rewrite", Text: "x", Instruction: "y", Provider: p})
	if r.Error == nil || r.Error.Code != CodeLanguageModel {
		t.Fatalf("error %+v, want %s", r.Error, CodeLanguageModel)
	}
}

func TestCancelEndsARunningCall(t *testing.T) {
	f := &fakeChat{t: t, block: true, arrived: make(chan struct{}, 1)}
	p := startChat(t, f)
	c := &Calls{}
	done := make(chan replyOf[json.RawMessage], 1)
	go func() {
		done <- call[json.RawMessage](t, c, "run-1", MethodProcess, ProcessRequest{Task: "rewrite", Text: "x", Instruction: "y", Provider: p})
	}()
	<-f.arrived
	if !c.Cancel("run-1") {
		t.Fatal("Cancel found no call")
	}
	if r := <-done; r.Error == nil || r.Error.Code != CodeCancelled {
		t.Fatalf("error %+v, want %s", r.Error, CodeCancelled)
	}
	if c.Cancel("run-1") {
		t.Fatal("a finished call is still cancellable")
	}
}

func TestCallTestProvider(t *testing.T) {
	f, p := serveChat(t, "ok")
	r := call[TestResult](t, &Calls{}, "", MethodTestProvider, p)
	if r.Error != nil || len(f.calls()) != 1 {
		t.Fatalf("reply %+v after %d requests", r, len(f.calls()))
	}
}

func TestCallTranscribeSendsTheRecording(t *testing.T) {
	wav := []byte("RIFF....WAVEfmt fake")
	var got struct {
		file, model, prompt string
		auth                string
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/audio/transcriptions" {
			http.NotFound(w, r)
			return
		}
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			t.Errorf("form: %v", err)
		}
		f, _, err := r.FormFile("file")
		if err == nil {
			data, _ := io.ReadAll(f) // a small test upload
			got.file = string(data)
		}
		got.model, got.prompt, got.auth = r.FormValue("model"), r.FormValue("prompt"), r.Header.Get("Authorization")
		if got.auth != "Bearer sk-test" {
			http.Error(w, `{"error":{"message":"bad key"}}`, http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"text":"hello world","language":"en"}`) // a response writer in a test
	}))
	t.Cleanup(srv.Close)
	path := filepath.Join(t.TempDir(), "take.wav")
	if err := os.WriteFile(path, wav, 0o600); err != nil {
		t.Fatal(err)
	}
	r := call[SpeechResult](t, &Calls{}, "", MethodTranscribe, SpeechRequest{
		Path: path, Endpoint: srv.URL + "/v1", Model: "whisper-1", Key: "sk-test", Terms: []Term{{Term: "Voxt"}},
	})
	if r.Error != nil || r.Result.Text != "hello world" {
		t.Fatalf("reply = %+v", r)
	}
	if got.file != string(wav) || got.model != "whisper-1" || got.auth != "Bearer sk-test" || !strings.Contains(got.prompt, "Voxt") {
		t.Fatalf("request: %+v", got)
	}
	missing := call[SpeechResult](t, &Calls{}, "", MethodTranscribe, SpeechRequest{Path: filepath.Join(t.TempDir(), "none.wav"), Endpoint: srv.URL + "/v1", Key: "k"})
	if missing.Error == nil || missing.Error.Code != CodeSpeech {
		t.Fatalf("missing file: %+v", missing.Error)
	}
	refused := call[SpeechResult](t, &Calls{}, "", MethodTranscribe, SpeechRequest{Path: path, Endpoint: srv.URL + "/v1", Key: "wrong"})
	if refused.Error == nil || refused.Error.Code != CodeConfig {
		t.Fatalf("refused key: %+v, want %s (the settings to check)", refused.Error, CodeConfig)
	}
}

func TestALanguageModelFailureWorthRetryingIsRetried(t *testing.T) {
	f := &fakeChat{t: t, answers: []fakeAnswer{{status: http.StatusServiceUnavailable}, {text: "Hello."}}}
	p := startChat(t, f)
	r := call[ProcessResult](t, &Calls{}, "", MethodProcess, ProcessRequest{Task: "rewrite", Text: "hello", Instruction: "fix", MainLanguage: "en", Provider: p})
	if r.Error != nil || r.Result.Text != "Hello." || len(f.calls()) != 2 {
		t.Fatalf("reply %+v after %d calls, want the second attempt's answer", r, len(f.calls()))
	}
}
