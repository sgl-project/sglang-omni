package actnow

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/registry"
	famruntime "github.com/hehehai/actnowkit/runtime"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

const testBundle = "dev.hehehai.voxtnew"

// noApps is a registry.System that finds no app.
type noApps struct{}

func (noApps) AppPaths(string) []string                { return nil }
func (noApps) Running(string) []registry.Process       { return nil }
func (noApps) BundleID(appPath string) (string, error) { return registry.BundleIDAt(appPath) }

// testFamily is VoxtNew in a family directory of its own, its runtime's
// clock at now and its Mac of 10 GB.
func testFamily(t *testing.T, now *time.Time) *Family {
	t.Helper()
	dir := t.TempDir()
	f := Open(Config{Dir: family.At(filepath.Join(dir, "ActNow")), Hub: filepath.Join(dir, "hub"), BundleID: testBundle,
		Version: "0.1.0-test", System: noApps{}, Logf: t.Logf}, settings.Settings{})
	f.runtime.now = func() time.Time { return *now }
	f.runtime.physical = func() uint64 { return 10 << 30 }
	f.runtime.othersWait = 200 * time.Millisecond
	t.Cleanup(f.Close)
	return f
}

func readFile(t *testing.T, f *Family) famruntime.File {
	t.Helper()
	file, err := famruntime.Read(f.dir, f.bundleID)
	if err != nil {
		t.Fatal(err)
	}
	return file
}

// TestRuntimeFile: the file follows the models (loaded busy, idle since
// when, busy again, unloaded) and the budget; quitting removes it, and
// nothing writes it again after.
func TestRuntimeFile(t *testing.T) {
	now := time.Date(2026, 10, 17, 8, 0, 0, 0, time.UTC)
	f := testFamily(t, &now)
	r := f.Runtime()
	r.Loaded("asr.fast", "omni", "mlx-community/Qwen3-ASR-0.6B-4bit", "abc123", 1<<30)
	file := readFile(t, f)
	if file.PID != os.Getpid() || file.BundleID != testBundle || file.BudgetBytes != 6<<30 || len(file.Loaded) != 1 {
		t.Fatalf("file %+v", file)
	}
	l := file.Loaded[0]
	if l.Repo != "mlx-community/Qwen3-ASR-0.6B-4bit" || l.Revision != "abc123" || l.Worker != "omni" || l.Bytes != 1<<30 || l.IdleSince != nil {
		t.Errorf("loaded %+v", l)
	}
	r.Idle("asr.fast")
	if l := readFile(t, f).Loaded[0]; l.IdleSince == nil || !l.IdleSince.Equal(now) {
		t.Errorf("idle %+v", l)
	}
	now = now.Add(time.Minute)
	r.Busy("asr.fast")
	if l := readFile(t, f).Loaded[0]; l.IdleSince != nil {
		t.Errorf("busy %+v", l)
	}
	r.Idle("unknown") // not loaded: nothing
	r.Loaded("llm.fast", "llm", "mlx-community/Qwen3.5-2B", "def456", 2<<30)
	if file := readFile(t, f); len(file.Loaded) != 2 || file.LoadedBytes() != 3<<30 {
		t.Errorf("two models %+v", file)
	}
	if r.LoadedBytes() != 3<<30 {
		t.Errorf("LoadedBytes %d", r.LoadedBytes())
	}
	r.Unloaded("asr.fast")
	if file := readFile(t, f); len(file.Loaded) != 1 || file.Loaded[0].Repo != "mlx-community/Qwen3.5-2B" {
		t.Errorf("after the unload %+v", file)
	}
	f.Close()
	if _, err := os.Stat(f.dir.RuntimeFile(testBundle)); !os.IsNotExist(err) {
		t.Fatalf("the file is left after quitting: %v", err)
	}
	r.Loaded("asr.fast", "omni", "x", "y", 1) // a runtime's late event
	if _, err := os.Stat(f.dir.RuntimeFile(testBundle)); !os.IsNotExist(err) {
		t.Errorf("written after quitting: %v", err)
	}
}

// unloads is a runtime that lets models go when asked (OnUnload), saying
// Unloaded as a runtime does, and the family's memory requests, in
// order.
type unloads struct {
	mu  sync.Mutex
	log []string
}

func (u *unloads) add(s string) {
	u.mu.Lock()
	u.log = append(u.log, s)
	u.mu.Unlock()
}

func (u *unloads) take() []string {
	u.mu.Lock()
	defer u.mu.Unlock()
	out := u.log
	u.log = nil
	return out
}

func (u *unloads) install(t *testing.T, f *Family) {
	r := f.Runtime()
	r.OnUnload(func(id string) {
		u.add("unload " + id)
		go r.Unloaded(id) // later, as a process exits
	})
	f.Watch(func(topic string) { u.add(topic) }, family.TopicMemory)
}

// TestOnMemory: another app's request unloads only the models idle for a
// minute; a busy one and one idle for less stay.
func TestOnMemory(t *testing.T) {
	now := time.Date(2026, 10, 17, 8, 0, 0, 0, time.UTC)
	f := testFamily(t, &now)
	r := f.Runtime()
	var u unloads
	u.install(t, f)
	r.OnMemory() // nothing loaded
	r.Loaded("asr.fast", "omni", "a", "1", 1<<30)
	r.Loaded("vad.silero", "omni", "b", "1", 2<<20)
	r.Loaded("llm.fast", "llm", "c", "1", 2<<30)
	r.Idle("asr.fast")
	now = now.Add(50 * time.Second)
	r.Idle("vad.silero")
	now = now.Add(10 * time.Second) // asr.fast idle 60 s, Silero 10 s, llm.fast busy
	r.OnMemory()
	if got := u.take(); !slices.Equal(got, []string{"unload asr.fast"}) {
		t.Errorf("unloaded %v", got)
	}
}

// TestReserve: a model that does not fit the budget (the family's 60 %
// of the Mac, less the other apps' models) unloads VoxtNew's idle models,
// the largest first and only as many as needed, and waits for them (idle
// for a minute: a session's models never unload each other); when
// that is not enough the other apps are asked (TopicMemory) after them;
// a model that fits does nothing; a busy model is never unloaded.
func TestReserve(t *testing.T) {
	now := time.Date(2026, 10, 17, 8, 0, 0, 0, time.UTC)
	f := testFamily(t, &now)
	r := f.Runtime()
	var u unloads
	u.install(t, f)
	// Another app holds 3 GB: VoxtNew's budget is 6 − 3 GB.
	if err := famruntime.Write(f.dir, famruntime.File{BundleID: "dev.hehehai.dicta", PID: os.Getpid(),
		Loaded: []famruntime.Loaded{{Repo: "other/model", Revision: "1", Bytes: 3 << 30}}}); err != nil {
		t.Fatal(err)
	}
	if b := r.Budget(); b != 3<<30 {
		t.Fatalf("budget %d", b)
	}
	ctx := context.Background()
	r.Loaded("llm.fast", "llm", "c", "1", 2<<30)      // idle, the largest
	r.Loaded("vad.silero", "omni", "b", "1", 100<<20) // idle
	r.Loaded("asr.fast", "omni", "a", "1", 512<<20)   // busy
	r.Idle("llm.fast")
	r.Idle("vad.silero")
	now = now.Add(30 * time.Second)
	r.Reserve(ctx, "asr.accurate", 1<<30) // idle for less than a minute: kept
	if got := u.take(); !slices.Equal(got, []string{family.TopicMemory}) {
		t.Errorf("models idle for 30 s: %v", got)
	}
	now = now.Add(30 * time.Second)

	r.Reserve(ctx, "asr.accurate", 300<<20) // 3 GB − 2.6 GB is enough
	if got := u.take(); len(got) != 0 {
		t.Errorf("a model that fits: %v", got)
	}
	r.Reserve(ctx, "asr.accurate", 1<<30)
	if got := u.take(); !slices.Equal(got, []string{"unload llm.fast"}) {
		t.Errorf("1 GB: %v", got)
	}
	if r.LoadedBytes() != 612<<20 {
		t.Errorf("Reserve did not wait for the unload: %d", r.LoadedBytes())
	}
	start := time.Now()
	r.Reserve(ctx, "llm.balanced", 3<<30)
	if got := u.take(); !slices.Equal(got, []string{"unload vad.silero", family.TopicMemory}) {
		t.Errorf("3 GB: %v", got)
	}
	if d := time.Since(start); d < 150*time.Millisecond || d > 2*time.Second {
		t.Errorf("the other apps were given %v", d)
	}
	if _, ok := r.loaded["asr.fast"]; !ok {
		t.Error("the busy model went")
	}
}

// TestPaste: the file says pasting while VoxtNew pastes; a paste waits
// for another app's (at most famruntime.PasteWait) and goes when it ends.
func TestPaste(t *testing.T) {
	now := time.Now()
	f := testFamily(t, &now)
	r := f.Runtime()
	ctx := context.Background()
	during := false
	if err := r.Paste(ctx, func() error { during = readFile(t, f).Pasting; return nil }); err != nil || !during {
		t.Errorf("pasting during the paste: %v %v", during, err)
	}
	if readFile(t, f).Pasting {
		t.Error("pasting after the paste")
	}
	other := famruntime.File{BundleID: "dev.hehehai.auralink", PID: os.Getpid(), Pasting: true}
	if err := famruntime.Write(f.dir, other); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	r.Paste(ctx, func() error { return nil })
	if d := time.Since(start); d < famruntime.PasteWait-50*time.Millisecond || d > 2*time.Second {
		t.Errorf("waited %v for a paste that does not end", d)
	}
	go func() {
		time.Sleep(100 * time.Millisecond)
		other.Pasting = false
		famruntime.Write(f.dir, other)
	}()
	start = time.Now()
	r.Paste(ctx, func() error { return nil })
	if d := time.Since(start); d < 80*time.Millisecond || d > 400*time.Millisecond {
		t.Errorf("waited %v for a paste of 100 ms", d)
	}
}
