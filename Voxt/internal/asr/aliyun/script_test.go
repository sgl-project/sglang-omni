package aliyun

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/coder/websocket"
	"github.com/sgl-project/sglang-omni/Voxt/internal/asr/asrtest"
)

// A script is a testdata/*.jsonl file the fake service plays: one JSON
// object per line. Lines with an "@" key are directives, every other
// line is a server message sent as a text frame, with "<task-id>"
// replaced by the task id the client chose:
//
//	{"@":"note","text":"…"}                 a comment (sources of the messages)
//	{"@":"wait","for":"start"}              read the client's frames up to run-task / session.update
//	{"@":"wait","for":"audio","n":3}        … until n audio frames came in all
//	{"@":"wait","for":"finish"}             … up to finish-task / session.finish
//	{"@":"hold"}                            read until the client goes away
//	{"@":"close","code":1011,"reason":"…"}  close the connection
//
// The end of the script closes the connection normally.

type directive struct {
	At     string `json:"@"`
	For    string `json:"for"`
	N      int    `json:"n"`
	Code   int    `json:"code"`
	Reason string `json:"reason"`
}

func loadScript(t *testing.T, name string) []json.RawMessage {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var out []json.RawMessage
	sc := bufio.NewScanner(f)
	sc.Buffer(nil, 1<<20)
	for n := 1; sc.Scan(); n++ {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		if !json.Valid([]byte(line)) {
			t.Fatalf("%s:%d: not JSON", name, n)
		}
		out = append(out, json.RawMessage(line))
	}
	return out
}

// recorder logs what the client sent, readably: the handshake, then one
// line per frame, JSON with sorted keys, ids replaced by placeholders,
// audio by its size; runs of the same line collapsed ("×n").
type recorder struct {
	mu      sync.Mutex
	lines   []string
	taskIDs map[string]bool
	events  map[string]int
	audio   int
	done    chan struct{}
}

func newRecorder() *recorder {
	return &recorder{taskIDs: map[string]bool{}, events: map[string]int{}, done: make(chan struct{})}
}

func (r *recorder) add(s string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.lines = append(r.lines, s)
}

func (r *recorder) text() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	var b strings.Builder
	for i := 0; i < len(r.lines); {
		j := i
		for j < len(r.lines) && r.lines[j] == r.lines[i] {
			j++
		}
		b.WriteString(r.lines[i])
		if j-i > 1 {
			fmt.Fprintf(&b, " ×%d", j-i)
		}
		b.WriteByte('\n')
		i = j
	}
	return b.String()
}

// frame logs one client frame and says what it was: "start", "audio",
// "finish" or "".
func (r *recorder) frame(typ websocket.MessageType, data []byte) string {
	if typ == websocket.MessageBinary {
		r.audio++
		r.add(fmt.Sprintf("→ binary %d bytes", len(data)))
		return "audio"
	}
	var v any
	if err := json.Unmarshal(data, &v); err != nil {
		r.add("→ text (not JSON) " + string(data))
		return ""
	}
	kind := ""
	if m, ok := v.(map[string]any); ok {
		if h, ok := m["header"].(map[string]any); ok {
			if id, _ := h["task_id"].(string); id != "" {
				r.taskIDs[id] = true
				h["task_id"] = "<task-id>"
			}
			switch h["action"] {
			case "run-task":
				kind = "start"
			case "finish-task":
				kind = "finish"
			}
		}
		if id, ok := m["event_id"].(string); ok {
			r.events[id]++
			m["event_id"] = "<event-id>"
		}
		switch m["type"] {
		case "session.update":
			kind = "start"
		case "session.finish":
			kind = "finish"
		case "input_audio_buffer.append":
			kind = "audio"
			r.audio++
			if a, ok := m["audio"].(string); ok {
				raw, err := base64.StdEncoding.DecodeString(a)
				if err != nil {
					m["audio"] = "<invalid base64>"
				} else {
					m["audio"] = fmt.Sprintf("<base64 of %d bytes>", len(raw))
				}
			}
		}
	}
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	enc.Encode(v)
	r.add("→ text " + strings.TrimSpace(b.String()))
	return kind
}

// check fails t when ids are missing or reused: one task id per
// connection, every event id distinct.
func (r *recorder) check(t *testing.T) {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.taskIDs) > 1 {
		t.Errorf("the client used %d task ids in one connection", len(r.taskIDs))
	}
	for id, n := range r.events {
		if id == "" || n > 1 {
			t.Errorf("event id %q used %d times", id, n)
		}
	}
}

func (r *recorder) taskID() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	for id := range r.taskIDs {
		return id
	}
	return ""
}

// handler plays script for one connection.
func (r *recorder) handler(t *testing.T, script []json.RawMessage) func(c *asrtest.Conn) {
	return func(c *asrtest.Conn) {
		defer close(r.done)
		r.add("GET " + c.Req.URL.RequestURI())
		for _, h := range []string{"Authorization", "OpenAI-Beta", "X-DashScope-DataInspection"} {
			if v := c.Req.Header.Get(h); v != "" {
				if h == "Authorization" {
					if v != "Bearer "+testKey {
						t.Errorf("Authorization = %q", v)
					}
					v = "Bearer <key>"
				}
				r.add(h + ": " + v)
			}
		}
		readUntil := func(done func(kind string) bool) bool {
			for {
				typ, data, ok := c.TryRead()
				if !ok {
					r.add("(the client went away)")
					return false
				}
				if done(r.frame(typ, data)) {
					return true
				}
			}
		}
		for _, raw := range script {
			var d directive
			json.Unmarshal(raw, &d)
			switch d.At {
			case "note":
			case "wait":
				var ok bool
				switch d.For {
				case "audio":
					if r.audio >= d.N {
						continue
					}
					ok = readUntil(func(string) bool { return r.audio >= d.N })
				default:
					ok = readUntil(func(k string) bool { return k == d.For })
				}
				if !ok {
					return
				}
			case "hold":
				readUntil(func(string) bool { return false })
				return
			case "close":
				r.add(fmt.Sprintf("(server closes: %d %s)", d.Code, d.Reason))
				c.Close(websocket.StatusCode(d.Code), d.Reason)
				return
			case "":
				msg := strings.ReplaceAll(string(raw), "<task-id>", r.taskID())
				if err := c.Write(c.Ctx, websocket.MessageText, []byte(msg)); err != nil {
					r.add("(server write failed)")
					return
				}
			default:
				t.Errorf("unknown directive %q", d.At)
				return
			}
		}
	}
}
