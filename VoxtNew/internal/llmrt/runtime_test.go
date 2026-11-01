package llmrt

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/llmrttest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/wire"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
)

func TestMain(m *testing.M) {
	if llmrttest.Enabled() {
		llmrttest.Main()
	}
	os.Exit(m.Run())
}

// logSink keeps the runtime's log lines.
type logSink struct {
	mu    sync.Mutex
	lines []core.LogEntry
}

func (l *logSink) add(e core.LogEntry) {
	l.mu.Lock()
	l.lines = append(l.lines, e)
	l.mu.Unlock()
}

func (l *logSink) has(sub string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	for _, e := range l.lines {
		if strings.Contains(e.Message, sub) {
			return true
		}
	}
	return false
}

// newRuntime is a Runtime over the fake process with opts; tune changes
// its config.
func newRuntime(t *testing.T, opts string, tune func(*Config)) (*Runtime, *logSink) {
	t.Helper()
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	logs := &logSink{}
	cfg := Config{Bin: self, AppVersion: "test", LogDir: t.TempDir(), Log: logs.add, Env: llmrttest.Env(opts), IdleTimeout: -1}
	if tune != nil {
		tune(&cfg)
	}
	rt := New(cfg)
	t.Cleanup(rt.Stop)
	return rt, logs
}

// mlxDir is a directory that looks like an MLX model.
func mlxDir(t *testing.T) string {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "config.json"), []byte(`{}`), 0o644)
	return dir
}

// chatLog reads the fake's chat-log.
func chatLog(t *testing.T, path string) []wire.ChatParams {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var out []wire.ChatParams
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		var p wire.ChatParams
		json.Unmarshal(sc.Bytes(), &p)
		out = append(out, p)
	}
	return out
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestChatStreamsOnTheCallersGoroutine(t *testing.T) {
	log := filepath.Join(t.TempDir(), "chats.jsonl")
	rt, logs := newRuntime(t, llmrttest.Opt("reply", "The meeting is at three.")+","+llmrttest.Opt("chat-log", log), nil)
	if st := rt.Status(); st.State != Stopped || st.Starts != 0 {
		t.Fatalf("a runtime starts nothing before its first call: %+v", st)
	}
	chat, err := Open(context.Background(), rt, Balanced, mlxDir(t))
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	var got strings.Builder
	caller := goroutineID()
	resp, err := chat.Chat(context.Background(), llm.Request{
		Messages: []llm.Message{{Role: llm.System, Content: "Clean it up."}, {Role: llm.User, Content: "<input>\num the meeting is at three\n</input>"}},
		Skill:    llm.SkillTranscribeEnhance,
		OnDelta: func(d string) {
			if goroutineID() != caller {
				t.Error("a delta on another goroutine")
			}
			mu.Lock()
			got.WriteString(d)
			mu.Unlock()
		},
	})
	if err != nil || resp.Text != "The meeting is at three." || got.String() != resp.Text || resp.Provider != Provider || resp.Usage.CompletionTokens == 0 {
		t.Fatalf("%+v %v (streamed %q)", resp, err, got.String())
	}
	st := rt.Status()
	if st.State != Ready || st.Starts != 1 || st.Model == nil || st.Model.ID != Balanced || st.Memory.FootprintBytes == 0 {
		t.Errorf("status %+v", st)
	}
	// The chat carries the profile's settings: greedy clean-up, no thinking.
	cs := chatLog(t, log)
	if len(cs) != 1 {
		t.Fatalf("%d chats", len(cs))
	}
	c := cs[0]
	if c.Model.Format != wire.FormatMLX || c.Sampling.Temperature != 0 || c.Sampling.RepetitionPenalty != 1.05 || c.Template["enable_thinking"] != false ||
		c.MaxTokens != 175 || c.PrefillStepSize != 256 || len(c.Messages) != 2 || c.Messages[0].Role != "system" {
		t.Errorf("chat params %+v", c)
	}
	if !logs.has("fake voxtnew-llm ready") || !logs.has("llm.balanced transcribe_enhance: 40+") {
		t.Errorf("log lines %+v", logs.lines)
	}
	// Rewriting samples.
	chat.Chat(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "write"}}, Skill: llm.SkillRewrite})
	if c := chatLog(t, log)[1]; c.Sampling.Temperature != 0.7 || c.Sampling.TopK != 20 || c.MaxTokens != 256 {
		t.Errorf("rewrite params %+v", c)
	}
}

// goroutineID tells goroutines apart (tests only).
func goroutineID() string {
	buf := make([]byte, 64)
	buf = buf[:runtime.Stack(buf, false)]
	f := strings.Fields(string(buf))
	if len(f) < 2 {
		return ""
	}
	return f[1]
}

func TestIdleStopAndRestart(t *testing.T) {
	rt, logs := newRuntime(t, "", func(c *Config) { c.IdleTimeout = 80 * time.Millisecond })
	b, err := NewModel(rt, Fast, mlxDir(t))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := b.Chat(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "hi"}}}); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "the idle stop", func() bool { return rt.Status().State == Stopped })
	if !logs.has("unused for") {
		t.Error("no idle line")
	}
	if _, err := b.Chat(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "hi"}}}); err != nil {
		t.Fatal(err)
	}
	if st := rt.Status(); st.State != Ready || st.Starts != 2 {
		t.Errorf("status %+v", st)
	}
}

// TestSetIdleTimeout: a change applies to the running process at once:
// kept (never) → stopped after the new time, counted from the change;
// back to never, it stays.
func TestSetIdleTimeout(t *testing.T) {
	rt, logs := newRuntime(t, "", nil) // IdleTimeout -1
	b, err := NewModel(rt, Fast, mlxDir(t))
	if err != nil {
		t.Fatal(err)
	}
	chat := func() {
		t.Helper()
		if _, err := b.Chat(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "hi"}}}); err != nil {
			t.Fatal(err)
		}
	}
	chat()
	time.Sleep(150 * time.Millisecond)
	if st := rt.Status(); st.State != Ready {
		t.Fatalf("kept: %+v", st)
	}
	changed := time.Now()
	rt.SetIdleTimeout(200 * time.Millisecond)
	if d := rt.IdleTimeout(); d != 200*time.Millisecond {
		t.Errorf("IdleTimeout %v", d)
	}
	waitFor(t, "the idle stop", func() bool { return rt.Status().State == Stopped })
	if took := time.Since(changed); took < 200*time.Millisecond {
		t.Errorf("stopped %v after the change", took)
	}
	if !logs.has("unused for 200ms") {
		t.Error("no idle line with the new time")
	}
	chat()
	rt.SetIdleTimeout(-1)
	time.Sleep(400 * time.Millisecond)
	if st := rt.Status(); st.State != Ready || st.Starts != 2 {
		t.Errorf("never: %+v", st)
	}
	rt.SetIdleTimeout(0)
	if d := rt.IdleTimeout(); d != DefaultIdleTimeout {
		t.Errorf("0 is %v", d)
	}
}

// A process that dies during a chat fails it; llm.Client starts another
// and tries again.
func TestCrashRetries(t *testing.T) {
	marker := filepath.Join(t.TempDir(), "crashed")
	rt, logs := newRuntime(t, llmrttest.Opt("crash-once", marker), nil)
	chat, err := Open(context.Background(), rt, Balanced, mlxDir(t))
	if err != nil {
		t.Fatal(err)
	}
	resp, err := chat.Chat(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "hi"}}})
	if err != nil || resp.Text != "Fake answer." || resp.Attempts != 2 {
		t.Fatalf("%+v %v", resp, err)
	}
	if st := rt.Status(); st.Starts != 2 || !logs.has("voxtnew-llm exited") {
		t.Errorf("status %+v", st)
	}
	// A crash every time: the error says the process failed.
	rt2, _ := newRuntime(t, "crash-chat", nil)
	b, _ := NewModel(rt2, Balanced, mlxDir(t))
	_, err = (&llm.Client{Backend: b, Retry: llm.RetryPolicy{MaxAttempts: 2}}).Chat(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "hi"}}})
	if llm.Code(err) != llm.CodeServer || !errors.Is(err, ErrExited) || rt2.Status().State != Failed {
		t.Errorf("%v, %+v", err, rt2.Status())
	}
}

func TestCancel(t *testing.T) {
	rt, _ := newRuntime(t, "piece-ms=30,"+llmrttest.Opt("reply", strings.Repeat("word ", 40)), nil)
	chat, _ := Open(context.Background(), rt, Balanced, mlxDir(t))
	ctx, cancel := context.WithCancel(context.Background())
	first := make(chan struct{})
	var once sync.Once
	go func() {
		<-first
		cancel()
	}()
	_, err := chat.Chat(ctx, llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "hi"}}, OnDelta: func(string) { once.Do(func() { close(first) }) }})
	if llm.Code(err) != llm.CodeCancelled || !errors.Is(err, core.ErrCancelled) {
		t.Fatalf("err %v", err)
	}
	// The process lives on.
	if st := rt.Status(); st.State != Ready || st.Starts != 1 {
		t.Errorf("status %+v", st)
	}
	// A deadline is a timeout, which llm.Client retries.
	resp, err := chat.Chat(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "hi"}}, Timeout: 100 * time.Millisecond})
	if llm.Code(err) != llm.CodeTimeout || resp.Attempts != 2 {
		t.Errorf("%+v %v", resp, err)
	}
}

func TestCancelThatHangsKills(t *testing.T) {
	rt, _ := newRuntime(t, "hang-cancel,piece-ms=200,"+llmrttest.Opt("reply", strings.Repeat("word ", 40)), func(c *Config) { c.CancelGrace = 100 * time.Millisecond })
	b, _ := NewModel(rt, Balanced, mlxDir(t))
	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	t0 := time.Now()
	_, err := b.Chat(ctx, llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "hi"}}})
	if !errors.Is(err, context.DeadlineExceeded) || time.Since(t0) > 2*time.Second {
		t.Fatalf("err %v after %v", err, time.Since(t0))
	}
	waitFor(t, "the kill", func() bool { return rt.Status().State == Failed })
	if st := rt.Status(); !strings.Contains(st.Err, "did not stop") {
		t.Errorf("status %+v", st)
	}
}

func TestStartFailures(t *testing.T) {
	msg := []llm.Message{{Role: llm.User, Content: "hi"}}
	// Another protocol: refused, and never started again.
	rt, _ := newRuntime(t, "mismatch", nil)
	b, _ := NewModel(rt, Balanced, mlxDir(t))
	for range 2 {
		if _, err := b.Chat(context.Background(), llm.Request{Messages: msg}); !errors.Is(err, ErrUnavailable) || llm.Code(err) != llm.CodeServer {
			t.Errorf("mismatch: %v", err)
		}
	}
	if st := rt.Status(); st.Starts != 1 || !strings.Contains(st.Err, "protocol 9") {
		t.Errorf("mismatch status %+v", st)
	}
	// No answer to llm.hello.
	rt, _ = newRuntime(t, "silent", func(c *Config) { c.HelloTimeout = 150 * time.Millisecond })
	b, _ = NewModel(rt, Balanced, mlxDir(t))
	if _, err := b.Chat(context.Background(), llm.Request{Messages: msg}); !errors.Is(err, ErrUnavailable) || !strings.Contains(rt.Status().Err, "did not answer") {
		t.Errorf("silent: %v %+v", err, rt.Status())
	}
	// No executable.
	rt = New(Config{})
	b, _ = NewModel(rt, Balanced, mlxDir(t))
	if _, err := b.Chat(context.Background(), llm.Request{Messages: msg}); !errors.Is(err, ErrUnavailable) || rt.Status().State != Failed {
		t.Errorf("no bin: %v", err)
	}
	// A model that does not load: not retried.
	rt, _ = newRuntime(t, "load-fail", nil)
	chat, _ := Open(context.Background(), rt, Balanced, mlxDir(t))
	resp, err := chat.Chat(context.Background(), llm.Request{Messages: msg})
	if llm.Code(err) != llm.CodeModelNotFound || resp.Attempts != 1 {
		t.Errorf("load-fail: %+v %v", resp, err)
	}
	// Stopped: calls fail at once.
	rt, _ = newRuntime(t, "", nil)
	b, _ = NewModel(rt, Balanced, mlxDir(t))
	b.Chat(context.Background(), llm.Request{Messages: msg})
	rt.Stop()
	if _, err := b.Chat(context.Background(), llm.Request{Messages: msg}); !errors.Is(err, ErrStopped) || rt.Status().State != Stopped {
		t.Errorf("stopped: %v %+v", err, rt.Status())
	}
}

func TestWarmAndMemory(t *testing.T) {
	rt, _ := newRuntime(t, "load-ms=20", nil)
	if res, err := rt.Memory(context.Background()); err != nil || res.Model != nil || rt.Status().Starts != 0 {
		t.Fatalf("Memory started the process: %+v %v", res, err)
	}
	if _, err := Open(context.Background(), rt, Fast, mlxDir(t)); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "the warm-up", func() bool { m := rt.Status().Model; return m != nil && m.ID == Fast })
	res, err := rt.Memory(context.Background())
	if err != nil || res.Model == nil || res.Model.ID != Fast || res.Memory.MLXActiveBytes == 0 {
		t.Errorf("memory %+v %v", res, err)
	}
	rt.Unload()
	if st := rt.Status(); st.State != Stopped || st.Model != nil {
		t.Errorf("after Unload %+v", st)
	}
}

// Prefill sends the prompt with maxTokens 0, and the chat after it with
// the same system prompt reports a prompt cache hit in its log line.
func TestPrefill(t *testing.T) {
	log := filepath.Join(t.TempDir(), "chats.jsonl")
	rt, logs := newRuntime(t, llmrttest.Opt("chat-log", log), nil)
	b, err := NewModel(rt, Fast, mlxDir(t))
	if err != nil {
		t.Fatal(err)
	}
	system := llm.Message{Role: llm.System, Content: "Clean it up."}
	req := llm.Request{Skill: llm.SkillTranscribeEnhance, Messages: []llm.Message{system, {Role: llm.User, Content: ""}}}
	if err := b.Prefill(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	req.Messages[1].Content = "<input>\num hello\n</input>"
	if _, err := b.Chat(context.Background(), req); err != nil {
		t.Fatal(err)
	}
	if cs := chatLog(t, log); len(cs) != 2 || cs[0].MaxTokens != 0 || cs[1].MaxTokens == 0 {
		t.Errorf("chats %+v", cs)
	}
	if !logs.has("llm.fast transcribe_enhance prefill: 40+0 tokens") || !logs.has("prompt cache miss (30 tokens kept)") || !logs.has("prompt cache hit (30 tokens)") {
		t.Errorf("log lines %+v", logs.lines)
	}
	// A GGUF model has nothing to warm: no chat.
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "Hy-MT2-1.8B-Q4_K_M.gguf"), []byte("GGUF"), 0o644)
	tr, _ := NewModel(rt, Translate, dir)
	if err := tr.Prefill(context.Background(), llm.Request{Skill: llm.SkillRewrite}); err != nil || len(chatLog(t, log)) != 2 {
		t.Errorf("GGUF prefill: %v", err)
	}
}

func TestOpenNotInstalled(t *testing.T) {
	rt := New(Config{})
	for _, c := range []struct{ id, dir string }{{"llm.nope", t.TempDir()}, {Balanced, t.TempDir()}, {Translate, t.TempDir()}} {
		if _, err := Open(context.Background(), rt, c.id, c.dir); !errors.Is(err, local.ErrNotInstalled) {
			t.Errorf("Open(%s): %v", c.id, err)
		}
	}
	if rt.Status().Starts != 0 {
		t.Error("a failed Open started the process")
	}
}
