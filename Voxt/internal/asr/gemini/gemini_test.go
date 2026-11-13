package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/asrtest"
)

const testKey = "AIza-test-0000"

func testClient(t *testing.T, url string) *Client {
	t.Helper()
	c, err := New(asr.Config{Key: testKey, Endpoint: url})
	if err != nil {
		t.Fatal(err)
	}
	c.endGrace = 300 * time.Millisecond
	return c
}

// serverSetup checks the handshake and the setup, then acknowledges it
// (in a binary frame, as Gemini does).
func serverSetup(c *asrtest.Conn) map[string]any {
	c.T.Helper()
	if c.Req.Header.Get("x-goog-api-key") != testKey || strings.Contains(c.Req.URL.RawQuery, testKey) {
		c.T.Errorf("key in header %q, query %q", c.Req.Header.Get("x-goog-api-key"), c.Req.URL.RawQuery)
	}
	_, data := c.Read()
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		c.T.Fatalf("setup %s: %v", data, err)
	}
	c.WriteBinary([]byte(`{"setupComplete":{}}`))
	return m
}

// readAudio reads realtime audio up to audioStreamEnd and returns the
// PCM it carried.
func readAudio(c *asrtest.Conn) []byte {
	c.T.Helper()
	var pcm []byte
	for {
		_, data := c.Read()
		var m struct {
			RealtimeInput struct {
				Audio *struct {
					Data, MimeType string
				}
				AudioStreamEnd bool
			}
		}
		if err := json.Unmarshal(data, &m); err != nil {
			c.T.Fatalf("%s: %v", data, err)
		}
		if m.RealtimeInput.AudioStreamEnd {
			return pcm
		}
		if a := m.RealtimeInput.Audio; a == nil || a.MimeType != "audio/pcm;rate=16000" {
			c.T.Fatalf("not audio: %s", data)
		}
		b, _ := base64.StdEncoding.DecodeString(m.RealtimeInput.Audio.Data)
		pcm = append(pcm, b...)
		if len(pcm) == 3200 {
			c.WriteText(`{"serverContent":{"interimInputTranscription":{"text":"你好"}}}`)
			c.WriteText(`{"serverContent":{"inputTranscription":{"text":"你好，世界。"}}}`)
			c.WriteText(`{"server_content":{"turn_complete":true}}`) // a pause, not the end
			c.WriteText(`{"serverContent":{"interimInputTranscription":{"text":"Hello"}}}`)
		}
	}
}

// TestStream: the setup (model, vocabulary without repeats, language
// detected), the audio as base64 chunks, partials as the segments come,
// and the final once a turn completes after audioStreamEnd.
func TestStream(t *testing.T) {
	audio := make([]byte, 9600)
	for i := range audio {
		audio[i] = byte(i % 251)
	}
	got := make(chan []byte, 1)
	var setupMsg map[string]any
	url := asrtest.WSServer(t, "/ws", func(c *asrtest.Conn) {
		setupMsg = serverSetup(c)
		got <- readAudio(c)
		c.WriteText(`{"serverContent":{"inputTranscription":{"text":"Hello world."}}}`)
		c.WriteText(`{"serverContent":{"turnComplete":true}}`)
		c.TryRead()
	})
	var partials []string
	res, err := asr.StreamAll(context.Background(), testClient(t, url), asr.StreamRequest{Hints: []string{"Voxt", " Voxt ", "", "豆包"}},
		audio, asr.Feed{Pace: 1, OnPartial: func(s string) { partials = append(partials, s) }})
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "你好，世界。Hello world." || res.Model != DefaultModel || res.Duration != 300*time.Millisecond {
		t.Errorf("result %+v", res)
	}
	if strings.Join(partials, "|") != "你好|你好，世界。|你好，世界。Hello|你好，世界。Hello world." {
		t.Errorf("partials %q", partials)
	}
	if pcm := <-got; string(pcm) != string(audio) {
		t.Errorf("the server got %d bytes, not the %d sent", len(pcm), len(audio))
	}
	b, _ := json.Marshal(setupMsg)
	want := `{"setup":{"generationConfig":{"responseModalities":["TEXT"]},"inputAudioTranscription":{"customVocabulary":["Voxt","豆包"],"languageCodes":[],"mode":"SMART"},"model":"models/gemini-3.5-transcribe-live"}}`
	if string(b) != want {
		t.Errorf("setup\n%s\nwant\n%s", b, want)
	}
}

// TestEnd: without a completed turn the final comes after the grace, with
// the interim text; a service that closes after silence gives "".
func TestEnd(t *testing.T) {
	url := asrtest.WSServer(t, "/ws", func(c *asrtest.Conn) {
		serverSetup(c)
		readAudio(c)
		c.WriteText(`{"serverContent":{"interimInputTranscription":{"text":"Hello world"}}}`)
		c.TryRead() // holds until the client goes
	})
	start := time.Now()
	res, err := asr.StreamAll(context.Background(), testClient(t, url), asr.StreamRequest{}, make([]byte, 6400), asr.Feed{Pace: 1})
	if err != nil || res.Text != "你好，世界。Hello world" || time.Since(start) < 500*time.Millisecond {
		t.Errorf("after the grace: %+v, %v", res, err)
	}
	url = asrtest.WSServer(t, "/ws", func(c *asrtest.Conn) {
		serverSetup(c)
		readAudio(c)
	})
	if res, err := asr.StreamAll(context.Background(), testClient(t, url), asr.StreamRequest{}, make([]byte, 1600), asr.Feed{}); err != nil || res.Text != "" {
		t.Errorf("silence: %+v, %v", res, err)
	}
}

// TestErrors: a refused key, an error message, a refused model and a
// connection that breaks while recording.
func TestErrors(t *testing.T) {
	open := func(url string) error {
		_, err := testClient(t, url).Open(context.Background(), asr.StreamRequest{})
		return err
	}
	if err := open(asrtest.RefuseServer(t, 403, `{"error":{"code":403,"message":"API key not valid.","status":"PERMISSION_DENIED"}}`)); asr.KindOf(err) != asr.KindAuth || strings.Contains(err.Error(), testKey) {
		t.Errorf("refused: %v", err)
	}
	url := asrtest.WSServer(t, "/ws", func(c *asrtest.Conn) {
		c.Read()
		c.WriteText(`{"error":{"code":429,"message":"Resource has been exhausted","status":"RESOURCE_EXHAUSTED"}}`)
	})
	if err := open(url); asr.KindOf(err) != asr.KindQuota {
		t.Errorf("quota: %v", err)
	}
	url = asrtest.WSServer(t, "/ws", func(c *asrtest.Conn) {
		c.Read()
		c.Close(websocket.StatusPolicyViolation, "models/gemini-x is not found for API version v1beta, or is not supported for bidiGenerateContent.")
	})
	if err := open(url); asr.KindOf(err) != asr.KindConfig {
		t.Errorf("model: %v", err)
	}
	url = asrtest.WSServer(t, "/ws", func(c *asrtest.Conn) {
		serverSetup(c)
		c.Read()
		c.Close(websocket.StatusInternalError, "internal error")
	})
	if _, err := asr.StreamAll(context.Background(), testClient(t, url), asr.StreamRequest{}, make([]byte, 32000), asr.Feed{Pace: 4}); asr.KindOf(err) != asr.KindServer {
		t.Errorf("broken: %v", err)
	}
	url = asrtest.WSServer(t, "/ws", func(c *asrtest.Conn) {
		serverSetup(c)
		c.Read()
		c.Close(websocket.StatusInternalError, "internal error for "+testKey)
	})
	if _, err := asr.StreamAll(context.Background(), testClient(t, url), asr.StreamRequest{}, make([]byte, 32000), asr.Feed{Pace: 4}); err == nil || strings.Contains(err.Error(), testKey) || !strings.Contains(err.Error(), "[key]") {
		t.Errorf("a stream's error quotes the key: %v", err)
	}
}

func TestJoin(t *testing.T) {
	for _, tc := range []struct {
		segments []string
		interim  string
		want     string
	}{
		{[]string{"Hello", "world."}, "", "Hello world."},
		{[]string{"你好，", "世界。"}, "", "你好，世界。"},
		{[]string{"我用", "Voxt"}, "写代码", "我用Voxt写代码"},
		{[]string{" ", "Hi"}, " there ", "Hi there"},
	} {
		if got := join(tc.segments, tc.interim); got != tc.want {
			t.Errorf("join(%q, %q) = %q, want %q", tc.segments, tc.interim, got, tc.want)
		}
	}
}
