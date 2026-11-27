package aliyun

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/asrtest"
)

func TestProtocolFor(t *testing.T) {
	for model, want := range map[string]Protocol{
		"qwen3-asr-flash-realtime":            Realtime,
		"qwen3-asr-flash-realtime-2026-02-10": Realtime,
		" QWEN3-ASR-FLASH-REALTIME ":          Realtime,
		"fun-asr-realtime":                    Inference,
		"fun-asr-realtime-2025-09-15":         Inference,
		"paraformer-realtime-v2":              Inference,
		"qwen-audio-3.0-asr-flash-streaming":  Inference,
		"something-new":                       Inference,
		"":                                    Inference,
	} {
		if got := ProtocolFor(model); got != want {
			t.Errorf("ProtocolFor(%q) = %v, want %v", model, got, want)
		}
	}
	for _, m := range Models {
		if modelInfo(m).proto != ProtocolFor(m) || modelInfo(m).rate8k {
			t.Errorf("Models lists %q badly", m)
		}
	}
	if ProtocolFor(DefaultModel) != Inference {
		t.Error("the default model")
	}
}

func TestEndpoint(t *testing.T) {
	const fun, qwen = "fun-asr-realtime", "qwen3-asr-flash-realtime"
	for _, c := range []struct{ raw, model, want string }{
		{"", fun, "wss://dashscope.aliyuncs.com/api-ws/v1/inference"},
		{"", qwen, "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-asr-flash-realtime"},
		// The other protocol's path is switched (Voxt's docs give the realtime URL for all).
		{"wss://dashscope.aliyuncs.com/api-ws/v1/realtime", fun, "wss://dashscope.aliyuncs.com/api-ws/v1/inference"},
		{"wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-asr-flash-realtime", fun, "wss://dashscope.aliyuncs.com/api-ws/v1/inference"},
		{"wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference/", qwen, "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime?model=qwen3-asr-flash-realtime"},
		{"wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=custom", qwen, "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=custom"},
		// Hosts alone, console URLs, the compatible-mode URL of the LLM page.
		{"wss://dashscope-intl.aliyuncs.com", fun, "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference"},
		{"https://ws1.cn-beijing.maas.aliyuncs.com/", qwen, "wss://ws1.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime?model=qwen3-asr-flash-realtime"},
		{"https://dashscope.aliyuncs.com/compatible-mode/v1", fun, "wss://dashscope.aliyuncs.com/api-ws/v1/inference"},
		{"https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription", qwen, "wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=qwen3-asr-flash-realtime"},
		// A relay keeps its path.
		{"wss://relay.example.com/aliyun/ws", fun, "wss://relay.example.com/aliyun/ws"},
		{"wss://relay.example.com/aliyun/ws", qwen, "wss://relay.example.com/aliyun/ws?model=qwen3-asr-flash-realtime"},
		{"ws://127.0.0.1:9000/api-ws/v1/inference", qwen, "ws://127.0.0.1:9000/api-ws/v1/realtime?model=qwen3-asr-flash-realtime"},
	} {
		if got := Endpoint(c.raw, c.model); got != c.want {
			t.Errorf("Endpoint(%q, %s) = %q, want %q", c.raw, c.model, got, c.want)
		}
	}
}

func TestNew(t *testing.T) {
	if _, err := New(asr.Config{}); asr.KindOf(err) != asr.KindConfig {
		t.Errorf("no key: %v", err)
	}
	if _, err := New(asr.Config{Key: "k", Endpoint: "ws://dashscope.aliyuncs.com/api-ws/v1/inference"}); asr.KindOf(err) != asr.KindConfig {
		t.Errorf("plain ws to a remote host: %v", err)
	}
	if _, err := New(asr.Config{Key: "k", Endpoint: "wss://user:pw@dashscope.aliyuncs.com"}); asr.KindOf(err) != asr.KindConfig {
		t.Errorf("credentials in the URL: %v", err)
	}
	if _, err := New(asr.Config{Key: "k", Model: "fun-asr-flash-8k-realtime"}); asr.KindOf(err) != asr.KindConfig {
		t.Errorf("8 kHz model: %v", err)
	}
	c, err := New(asr.Config{Key: " k ", Endpoint: "http://localhost:1234"})
	if err != nil {
		t.Fatal(err)
	}
	if c.Model() != DefaultModel || c.endpoint != "ws://localhost:1234/api-ws/v1/inference" || c.key != "k" {
		t.Errorf("client %+v", c)
	}
}

func TestCaps(t *testing.T) {
	for model, want := range map[string]asr.Caps{
		"fun-asr-realtime":                   {Stream: true, Hints: true, Prompt: true},
		"fun-asr-realtime-2025-11-07":        {Stream: true, Hints: true, Prompt: true},
		"fun-asr-realtime-2026-02-28":        {Stream: true},
		"paraformer-realtime-v2":             {Stream: true},
		"qwen-audio-3.0-asr-flash-streaming": {Stream: true, Hints: true, Prompt: true},
		"qwen3-asr-flash-realtime":           {Stream: true},
	} {
		c, err := New(asr.Config{Key: "k", Model: model})
		if err != nil {
			t.Fatal(err)
		}
		if got := c.Caps(); got != want {
			t.Errorf("%s: %+v, want %+v", model, got, want)
		}
	}
}

func TestLang(t *testing.T) {
	fun, qwen, para := modelInfo("fun-asr-realtime"), modelInfo("qwen3-asr-flash-realtime"), modelInfo("paraformer-realtime-v2")
	for _, c := range []struct {
		m        model
		in, want string
	}{
		{fun, "zh-Hans", "zh"}, {fun, "EN_us", "en"}, {fun, "yue", ""}, {fun, "fil", "tl"}, {fun, "", ""},
		{qwen, "yue", "yue"}, {qwen, "tl", "fil"}, {qwen, "nl", ""},
		{para, "ja", "ja"}, {para, "vi", ""},
		{modelInfo("fun-asr-realtime-2026-02-28"), "ko", ""},
	} {
		if got := c.m.lang(c.in); got != c.want {
			t.Errorf("lang(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestRunTaskParams(t *testing.T) {
	long := strings.Repeat("词", 500)
	p := &inferenceProto{model: "fun-asr-realtime", info: modelInfo("fun-asr-realtime"), taskID: "t1",
		req: asr.StreamRequest{SampleRate: 16000, Lang: "en", Hints: []string{" a ", "", "b", "a"}, Prompt: long}}
	var v struct {
		Payload struct {
			Parameters map[string]any `json:"parameters"`
			Input      struct {
				Context []infMessage `json:"context"`
			} `json:"input"`
		} `json:"payload"`
	}
	if err := json.Unmarshal(p.runTask(), &v); err != nil {
		t.Fatal(err)
	}
	text := v.Payload.Input.Context[0].Content[0].Text
	if !strings.HasPrefix(text, "a\nb\n词") || len([]rune(text)) != maxContext {
		t.Errorf("context %q (%d runes)", text[:20], len([]rune(text)))
	}
	if _, ok := v.Payload.Parameters["vocabulary"]; ok {
		t.Error("fun-asr-realtime got inline vocabulary")
	}

	hints := make([]string, maxVocabulary+10)
	for i := range hints {
		hints[i] = strings.Repeat("x", i+1)
	}
	if got := vocabulary(hints); len(got) != maxVocabulary || got["x"] != vocabularyWeight {
		t.Errorf("vocabulary: %d terms", len(got))
	}
	if contextText(nil, "  ", true) != "" {
		t.Error("empty context")
	}
}

func TestTranscript(t *testing.T) {
	var tr transcript
	tr.set("1", "Hello", false)
	tr.set("1", "Hello there.", true)
	tr.set("", "How", false)
	if got := tr.String(); got != "Hello there. How" {
		t.Errorf("%q", got)
	}
	tr.set("", "How are you?", true)
	tr.set("", "好", false)
	if got := tr.String(); got != "Hello there. How are you?好" {
		t.Errorf("%q", got)
	}
	for _, c := range []struct {
		parts []string
		want  string
	}{
		{[]string{"你好。", "世界。"}, "你好。世界。"},
		{[]string{"Hi.", "你好"}, "Hi.你好"},
		{[]string{"用 Go", "写代码"}, "用 Go写代码"},
		{[]string{"One.", "Two."}, "One. Two."},
		{[]string{"こんにちは。", "OK"}, "こんにちは。OK"},
	} {
		if got := asr.JoinText(c.parts); got != c.want {
			t.Errorf("JoinText(%q) = %q, want %q", c.parts, got, c.want)
		}
	}
}

func TestServiceError(t *testing.T) {
	for _, c := range []struct {
		code, msg string
		kind      asr.Kind
		retry     bool
	}{
		{"InvalidApiKey", "Invalid API-key provided.", asr.KindAuth, false},
		{"invalid_api_key", "", asr.KindAuth, false},
		{"AccessDenied", "Access denied.", asr.KindAuth, false},
		{"Arrearage", "Access denied, please make sure your account is in good standing.", asr.KindQuota, false},
		{"Throttling.RateQuota", "Requests rate limit exceeded", asr.KindQuota, true},
		{"Throttling.AllocationQuota", "Allocated quota exceeded", asr.KindQuota, true},
		{"AllocationQuota.FreeTierOnly", "The free tier of the model has been exhausted.", asr.KindQuota, false},
		{"Model.AccessDenied", "Model access denied.", asr.KindQuota, false},
		{"ModelNotFound", "Model not found (fun-asr-x)", asr.KindConfig, false},
		{"InternalError", "An internal error has occured", asr.KindServer, true},
		{"CLIENT_ERROR", "request timeout after 23 seconds.", asr.KindNetwork, true},
		{"InvalidParameter", "sample_rate is invalid", asr.KindAudio, false},
		{"InvalidParameter", "Invalid task, task_id is required", asr.KindProtocol, false},
		{"", "", asr.KindProtocol, false},
	} {
		e := serviceError(c.code, c.msg)
		if e.Kind != c.kind || e.Retryable != c.retry || e.Provider != ID || e.Code != c.code || e.Detail == "" {
			t.Errorf("%s %q: %+v", c.code, c.msg, e)
		}
	}
	if !noSpeech("NO_VALID_AUDIO_ERROR", "") || noSpeech("InternalError", "") {
		t.Error("noSpeech")
	}
}

// Parsing alone: events in the wrong phase or malformed are ignored, and
// a "no valid audio" failure is an empty final.
func TestHandle(t *testing.T) {
	p := &inferenceProto{}
	for _, m := range []string{`not json`, `{}`, `{"header":{"event":"result-generated"},"payload":{}}`} {
		if st, err := p.handle([]byte(m), false); st != stepIgnore || err != nil {
			t.Errorf("%s: %v %v", m, st, err)
		}
	}
	if st, err := p.handle([]byte(`{"header":{"event":"task-failed","error_code":"NO_VALID_AUDIO_ERROR","error_message":"no valid audio"}}`), true); st != stepDone || err != nil {
		t.Errorf("no valid audio: %v %v", st, err)
	}
	r := &realtimeProto{}
	if st, err := r.handle([]byte(`{"type":"error","error":{"message":"Task failed"}}`), true); err == nil || st != stepIgnore {
		t.Error("a generic error without text must fail")
	}
	st, err := r.handle([]byte(`{"type":"conversation.item.input_audio_transcription.failed","item_id":"i","error":{"code":"Oops","message":"boom"}}`), false)
	if asr.KindOf(err) != asr.KindServer || st != stepIgnore {
		t.Errorf("item failed: %v", err)
	}
}

func TestOpenRejectsRealtimeRate(t *testing.T) {
	c, _ := New(asr.Config{Key: "k", Model: "qwen3-asr-flash-realtime", Endpoint: "ws://127.0.0.1:1"})
	st, err := c.Open(t.Context(), asr.StreamRequest{SampleRate: 44100})
	if st != nil || asr.KindOf(err) != asr.KindAudio {
		t.Fatalf("%v %v", st, err)
	}
}

// TestCloseAfterFinal: once the final text is in, Close returns at once,
// although the service answers the close frame only later: the session
// closes the stream before it inserts the text.
func TestCloseAfterFinal(t *testing.T) {
	url := asrtest.WSServer(t, inferencePath, func(c *asrtest.Conn) {
		c.Read() // run-task
		c.WriteText(`{"header":{"event":"task-started"},"payload":{}}`)
		for {
			if _, data := c.Read(); len(data) > 0 && data[0] == '{' {
				break // finish-task
			}
		}
		c.WriteText(`{"header":{"event":"task-finished"},"payload":{}}`)
		time.Sleep(2 * time.Second) // reads nothing: the close frame waits
	})
	c, _ := New(asr.Config{Key: testKey, Endpoint: url})
	st, err := c.Open(t.Context(), asr.StreamRequest{})
	if err != nil {
		t.Fatal(err)
	}
	st.Send(make([]byte, 3200))
	st.Finish()
	for range st.Events() {
	}
	start := time.Now()
	st.Close()
	if d := time.Since(start); d > time.Second {
		t.Errorf("Close took %v", d)
	}
}
