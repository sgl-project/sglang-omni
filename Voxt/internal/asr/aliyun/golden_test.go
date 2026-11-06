package aliyun

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/asrtest"
)

const testKey = "sk-test-0000"

// scenario is one protocol golden test: the client streams audio to a
// fake service that plays testdata/<name>.jsonl; what the client sent and
// the events it delivered are compared with testdata/<name>.golden.
type scenario struct {
	name  string
	model string
	// path is the endpoint path the user typed (Endpoint fits it to the
	// model's protocol).
	path string
	req  asr.StreamRequest
	// audio bytes, sent in chunks of chunk bytes (not a multiple of the
	// 3200-byte frame, so the stream has to rechunk).
	audio, chunk int
	// cancelAfter: cancel the context once this many partials came,
	// instead of finishing.
	cancelAfter int
	// The outcome: openErr is the Kind Open fails with; else final (the
	// text of EventFinal) or errKind (of EventError).
	openErr asr.Kind
	final   string
	errKind asr.Kind
}

func (sc scenario) run(t *testing.T) {
	t.Helper()
	rec := newRecorder()
	url := asrtest.WSServer(t, sc.path, rec.handler(t, loadScript(t, sc.name+".jsonl")))
	c, err := New(asr.Config{Key: testKey, Model: sc.model, Endpoint: url})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	var log strings.Builder
	defer func() {
		select {
		case <-rec.done:
		case <-time.After(3 * time.Second):
			t.Error("the fake service did not finish")
		}
		rec.check(t)
		asrtest.Golden(t, sc.name, "## client → service\n"+rec.text()+"\n## events\n"+log.String())
	}()

	st, err := c.Open(ctx, sc.req)
	if err != nil {
		if st != nil {
			t.Error("Open returned a stream with its error")
		}
		fmt.Fprintf(&log, "open: %s\n", describeErr(err))
		if k := asr.KindOf(err); k != sc.openErr {
			t.Errorf("Open: kind %q, want %q (%v)", k, sc.openErr, err)
		}
		return
	}
	defer st.Close()
	if sc.openErr != "" {
		t.Fatalf("Open succeeded, want %q", sc.openErr)
	}
	pcm := make([]byte, sc.audio)
	for i := range pcm {
		pcm[i] = byte(i)
	}
	for len(pcm) > 0 {
		n := min(sc.chunk, len(pcm))
		if err := st.Send(pcm[:n]); err != nil {
			t.Logf("send: %v", err) // timing-dependent; the events tell the cause
			break
		}
		pcm = pcm[n:]
	}
	if sc.cancelAfter == 0 {
		if err := st.Finish(); err != nil {
			t.Logf("finish: %v", err) // the reader reports the cause
		}
	}
	partials, terminal := 0, 0
	for ev := range st.Events() {
		switch ev.Kind {
		case asr.EventPartial:
			partials++
			fmt.Fprintf(&log, "partial: %s\n", ev.Text)
			if partials == sc.cancelAfter {
				cancel()
			}
		case asr.EventFinal:
			terminal++
			r := ev.Result
			fmt.Fprintf(&log, "%s\n  model=%s lang=%s duration=%s\n", strings.TrimSpace("final: "+ev.Text), r.Model, r.Lang, r.Duration) // no trailing space: the repository's hooks strip it
			if ev.Text != sc.final || r.Text != ev.Text {
				t.Errorf("final %q (result %q), want %q", ev.Text, r.Text, sc.final)
			}
			if sc.errKind != "" {
				t.Errorf("final, want an error %q", sc.errKind)
			}
		case asr.EventError:
			terminal++
			fmt.Fprintf(&log, "error: %s\n", describeErr(ev.Err))
			if k := asr.KindOf(ev.Err); k != sc.errKind {
				t.Errorf("error kind %q, want %q (%v)", k, sc.errKind, ev.Err)
			}
		}
	}
	if terminal != 1 {
		t.Errorf("%d final/error events, want 1", terminal)
	}
}

func describeErr(err error) string {
	var e *asr.Error
	if !errors.As(err, &e) {
		return "not an *asr.Error: " + err.Error()
	}
	s := fmt.Sprintf("kind=%s", e.Kind)
	if e.Status != 0 {
		s += fmt.Sprintf(" status=%d", e.Status)
	}
	if e.Code != "" {
		s += " code=" + e.Code
	}
	if e.Retryable {
		s += " retryable"
	}
	if e.Detail != "" {
		s += " detail=" + e.Detail
	}
	if strings.Contains(e.Error(), testKey) {
		s += " (LEAKS THE KEY)"
	}
	return s
}

const (
	second = asr.BytesPerSecond
	// audioLen is 1.0625 s: ten full frames and a 2000-byte rest.
	audioLen = second + 2000
)

func TestInferenceGolden(t *testing.T) {
	for _, sc := range []scenario{{
		// Several sentences with partials, a heartbeat, an empty sentence
		// start and an unknown event; language, hot words and prompt in
		// run-task (fun-asr-realtime takes them as input.context).
		name: "inference-multi", model: "fun-asr-realtime", path: inferencePath,
		req:   asr.StreamRequest{Lang: "zh-Hans", Hints: []string{"Voxt", "百炼", "Voxt"}, Prompt: "A note about Go."},
		audio: audioLen, chunk: 2000,
		final: "你好，世界。我在用 Voxt。Hello world.",
	}, {
		// qwen-audio streaming: hot words inline (vocabulary), the prompt
		// as context; the user typed the realtime URL (Voxt's docs say
		// so), which becomes the inference URL.
		name: "inference-vocabulary", model: "qwen-audio-3.0-asr-flash-streaming", path: realtimePath,
		req:   asr.StreamRequest{Lang: "en", Hints: []string{"Kubernetes", "MyGo"}, Prompt: "Engineering stand-up."},
		audio: 6400, chunk: 3200,
		final: "Kubernetes and MyGo.",
	}, {
		// Paraformer takes neither hints nor a prompt: none in run-task.
		name: "inference-plain", model: "paraformer-realtime-v2", path: inferencePath,
		req:   asr.StreamRequest{Lang: "ko", Hints: []string{"ignored"}, Prompt: "ignored"},
		audio: 3200, chunk: 3200,
		final: "",
	}, {
		name: "inference-auth", model: "fun-asr-realtime", path: inferencePath,
		openErr: asr.KindAuth,
	}, {
		name: "inference-quota", model: "fun-asr-realtime", path: inferencePath,
		audio: audioLen, chunk: 3200,
		errKind: asr.KindQuota,
	}, {
		name: "inference-cancel", model: "fun-asr-realtime", path: inferencePath,
		audio: audioLen, chunk: 3200, cancelAfter: 1,
		errKind: asr.KindCancelled,
	}, {
		name: "inference-early-close", model: "fun-asr-realtime", path: inferencePath,
		audio: audioLen, chunk: 3200,
		errKind: asr.KindServer,
	}} {
		t.Run(sc.name, sc.run)
	}
}

func TestRealtimeGolden(t *testing.T) {
	for _, sc := range []scenario{{
		// Two VAD items with text + stash previews, events this package
		// does not use, the user's inference URL switched to realtime.
		name: "realtime-multi", model: "qwen3-asr-flash-realtime", path: inferencePath,
		req:   asr.StreamRequest{Lang: "zh-CN", Hints: []string{"Voxt"}, Prompt: "not sent"},
		audio: audioLen, chunk: 2000,
		final: "今天天气不错，阳光明媚。Let's go.",
	}, {
		name: "realtime-auth", model: "qwen3-asr-flash-realtime", path: realtimePath,
		openErr: asr.KindAuth,
	}, {
		name: "realtime-quota", model: "qwen3-asr-flash-realtime", path: realtimePath,
		audio: audioLen, chunk: 3200,
		errKind: asr.KindQuota,
	}, {
		// A bare "task failed" after session.finish, once the text is in,
		// ends the stream with the text (Voxt saw it in the field).
		name: "realtime-trailing-error", model: "qwen3-asr-flash-realtime", path: realtimePath,
		audio: 3200, chunk: 3200,
		final: "好的",
	}, {
		name: "realtime-cancel", model: "qwen3-asr-flash-realtime-2026-02-10", path: realtimePath,
		audio: audioLen, chunk: 3200, cancelAfter: 1,
		errKind: asr.KindCancelled,
	}, {
		// The service closes normally without session.finished.
		name: "realtime-early-close", model: "qwen3-asr-flash-realtime", path: realtimePath,
		audio: audioLen, chunk: 3200,
		errKind: asr.KindProtocol,
	}} {
		t.Run(sc.name, sc.run)
	}
}

// A refused handshake: 401 is KindAuth with DashScope's code, for both
// protocols, and nothing leaks the key.
func TestHandshakeRefused(t *testing.T) {
	for _, model := range []string{"fun-asr-realtime", "qwen3-asr-flash-realtime"} {
		t.Run(model, func(t *testing.T) {
			url := asrtest.RefuseServer(t, http.StatusUnauthorized,
				`{"code":"InvalidApiKey","message":"Invalid API-key provided.","request_id":"d6b2c4f0-0000"}`)
			c, err := New(asr.Config{Key: testKey, Model: model, Endpoint: url})
			if err != nil {
				t.Fatal(err)
			}
			st, err := c.Open(context.Background(), asr.StreamRequest{})
			if st != nil {
				t.Fatal("a stream with the error")
			}
			var e *asr.Error
			if !errors.As(err, &e) || e.Kind != asr.KindAuth || e.Status != 401 || e.Code != "InvalidApiKey" {
				t.Fatalf("err = %v", err)
			}
			if strings.Contains(err.Error(), testKey) {
				t.Fatal("the error carries the key")
			}
		})
	}
}

// Close is idempotent, delivers no final, and closes Events.
func TestCloseAbandons(t *testing.T) {
	rec := newRecorder()
	url := asrtest.WSServer(t, inferencePath, rec.handler(t, loadScript(t, "inference-cancel.jsonl")))
	c, _ := New(asr.Config{Key: testKey, Endpoint: url})
	st, err := c.Open(context.Background(), asr.StreamRequest{})
	if err != nil {
		t.Fatal(err)
	}
	st.Close()
	st.Close()
	for ev := range st.Events() {
		if ev.Kind != asr.EventPartial {
			t.Fatalf("event %v after Close", ev.Kind)
		}
	}
	if err := st.Send(make([]byte, 3200)); asr.KindOf(err) != asr.KindCancelled {
		t.Fatalf("Send after Close: %v", err)
	}
	<-rec.done
}

// A service that never answers run-task: Open gives up when ctx ends.
func TestOpenCancelled(t *testing.T) {
	url := asrtest.WSServer(t, inferencePath, func(c *asrtest.Conn) {
		for {
			if _, _, ok := c.TryRead(); !ok {
				return
			}
		}
	})
	c, _ := New(asr.Config{Key: testKey, Endpoint: url})
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	st, err := c.Open(ctx, asr.StreamRequest{})
	if st != nil || err == nil {
		t.Fatalf("Open = %v, %v", st, err)
	}
	if k := asr.KindOf(err); k != asr.KindNetwork && k != asr.KindCancelled {
		t.Fatalf("kind %q: %v", k, err)
	}
}
