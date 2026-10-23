package llmrt

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/llmrttest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/resident"
)

// heard records Config.OnModel's events (as "kind id", pids apart) and
// BeforeLoad's calls.
type heard struct {
	mu     sync.Mutex
	events []string
	pids   []int
	bytes  map[string]int64 // a Loaded event's, by id
	before []string
}

func (h *heard) tune(c *Config) {
	h.bytes = map[string]int64{}
	c.OnModel = func(ev resident.Event) {
		h.mu.Lock()
		defer h.mu.Unlock()
		h.events = append(h.events, fmt.Sprintf("%s %s", ev.Kind, ev.ID))
		h.pids = append(h.pids, ev.PID)
		if ev.Kind == resident.Loaded {
			h.bytes[ev.ID] = ev.Bytes
		}
	}
	c.BeforeLoad = func(_ context.Context, id string) { h.mu.Lock(); h.before = append(h.before, id); h.mu.Unlock() }
}

func (h *heard) wait(t *testing.T, want ...string) {
	t.Helper()
	for end := time.Now().Add(5 * time.Second); time.Now().Before(end); time.Sleep(5 * time.Millisecond) {
		h.mu.Lock()
		ok := slices.Equal(h.events, want)
		h.mu.Unlock()
		if ok {
			return
		}
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	t.Fatalf("events\n%q\nwant\n%q", h.events, want)
}

var hi = []llm.Message{{Role: llm.User, Content: "hi"}}

// TestModelEvents: the process's model is loaded after its load (with
// the memory the process reported; BeforeLoad first), busy during a chat,
// idle after; a chat of another model unloads it and loads that one;
// UnloadIdle never stops a chat in progress, nor another model, and
// unloads an idle one.
func TestModelEvents(t *testing.T) {
	var h heard
	rt, _ := newRuntime(t, "piece-ms=20,"+llmrttest.Opt("reply", strings.Repeat("word ", 20)), h.tune)
	ctx := context.Background()
	fast, _ := NewModel(rt, Fast, mlxDir(t))
	if err := fast.Warm(ctx); err != nil {
		t.Fatal(err)
	}
	h.wait(t, "loaded llm.fast", "idle llm.fast")
	if h.bytes[Fast] != 2<<30 || rt.Held() != Fast {
		t.Errorf("bytes %d, held %q", h.bytes[Fast], rt.Held())
	}
	// A chat in progress: busy, and UnloadIdle leaves it.
	done := make(chan error, 1)
	go func() { _, err := fast.Chat(ctx, llm.Request{Messages: hi}); done <- err }()
	h.wait(t, "loaded llm.fast", "idle llm.fast", "busy llm.fast")
	if rt.UnloadIdle(Fast) {
		t.Error("UnloadIdle stopped a chat in progress")
	}
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	h.wait(t, "loaded llm.fast", "idle llm.fast", "busy llm.fast", "idle llm.fast")

	// Another model: the process lets the first go.
	bal, _ := NewModel(rt, Balanced, mlxDir(t))
	if _, err := bal.Chat(ctx, llm.Request{Messages: hi}); err != nil {
		t.Fatal(err)
	}
	all := []string{"loaded llm.fast", "idle llm.fast", "busy llm.fast", "idle llm.fast",
		"unloaded llm.fast", "loaded llm.balanced", "idle llm.balanced"}
	h.wait(t, all...)
	h.mu.Lock()
	if !slices.Equal(h.before, []string{Fast, Balanced}) {
		t.Errorf("BeforeLoad %v", h.before)
	}
	h.mu.Unlock()
	if rt.UnloadIdle(Fast) {
		t.Error("UnloadIdle stopped another model")
	}
	if !rt.UnloadIdle(Balanced) || rt.Status().State != Stopped || rt.Held() != "" {
		t.Errorf("UnloadIdle: %+v", rt.Status())
	}
	h.wait(t, append(all, "unloaded llm.balanced")...)
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, pid := range h.pids {
		if pid != h.pids[0] || pid == 0 {
			t.Errorf("pids %v", h.pids)
			break
		}
	}
}

// TestModelEventsEnds: every end of the process unloads its model: the
// idle stop, a crash during a chat (the retry's process loads it again),
// Unload, Stop.
func TestModelEventsEnds(t *testing.T) {
	ctx := context.Background()
	t.Run("idle", func(t *testing.T) {
		var h heard
		rt, _ := newRuntime(t, "", func(c *Config) { h.tune(c); c.IdleTimeout = 80 * time.Millisecond })
		b, _ := NewModel(rt, Fast, mlxDir(t))
		if _, err := b.Chat(ctx, llm.Request{Messages: hi}); err != nil {
			t.Fatal(err)
		}
		h.wait(t, "loaded llm.fast", "idle llm.fast", "unloaded llm.fast")
	})
	t.Run("crash", func(t *testing.T) {
		var h heard
		marker := filepath.Join(t.TempDir(), "crashed")
		rt, _ := newRuntime(t, llmrttest.Opt("crash-once", marker), h.tune)
		b, _ := NewModel(rt, Fast, mlxDir(t))
		if err := b.Warm(ctx); err != nil {
			t.Fatal(err)
		}
		chat := &llm.Client{Backend: b, Retry: llm.RetryPolicy{MaxAttempts: 2}}
		if _, err := chat.Chat(ctx, llm.Request{Messages: hi}); err != nil {
			t.Fatal(err)
		}
		h.wait(t, "loaded llm.fast", "idle llm.fast", "busy llm.fast", "unloaded llm.fast", "loaded llm.fast", "idle llm.fast")
		h.mu.Lock()
		if h.pids[0] == h.pids[4] {
			t.Errorf("the same pid after the crash: %v", h.pids)
		}
		h.mu.Unlock()
	})
	t.Run("unload and stop", func(t *testing.T) {
		var h heard
		rt, _ := newRuntime(t, "", h.tune)
		b, _ := NewModel(rt, Translate, ggufDir(t))
		if _, err := b.Chat(ctx, llm.Request{Skill: llm.SkillTranslateText, Translate: &llm.TranslateHint{Text: "hi", Target: "zh"}}); err != nil {
			t.Fatal(err)
		}
		rt.Unload()
		h.wait(t, "loaded llm.translate", "idle llm.translate", "unloaded llm.translate")
		if _, err := b.Chat(ctx, llm.Request{Skill: llm.SkillTranslateText, Translate: &llm.TranslateHint{Text: "hi", Target: "zh"}}); err != nil {
			t.Fatal(err)
		}
		rt.Stop()
		h.wait(t, "loaded llm.translate", "idle llm.translate", "unloaded llm.translate",
			"loaded llm.translate", "idle llm.translate", "unloaded llm.translate")
	})
}

func ggufDir(t *testing.T) string {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "Hy-MT2-1.8B-Q4_K_M.gguf"), []byte("GGUF"), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}
