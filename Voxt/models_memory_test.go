package main

import (
	"context"
	"errors"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni/omnitest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/resident"
)

// fakeProcMemory makes procMemory read mem (by pid; a pid not there
// cannot be measured) and records the pids it was asked for.
type fakeProcMemory struct {
	mu    sync.Mutex
	mem   map[int]int64
	asked []int
}

func (f *fakeProcMemory) install(t *testing.T) *fakeProcMemory {
	old := procMemory
	procMemory = func(pid int) (int64, error) {
		f.mu.Lock()
		defer f.mu.Unlock()
		f.asked = append(f.asked, pid)
		if b, ok := f.mem[pid]; ok {
			return b, nil
		}
		return 0, errors.New("no such process")
	}
	t.Cleanup(func() { procMemory = old })
	return f
}

func (f *fakeProcMemory) set(pid int, b int64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.mem[pid] = b
}

// omniLoads reports the omni runtime's models as their servers are
// loaded (resident.Loaded, with the pid), for the runtimes made from now
// on.
func omniLoads(t *testing.T) <-chan resident.Event {
	ch := make(chan resident.Event, 16)
	newRT := omniNewRuntime
	omniNewRuntime = func(c omni.Config) *omni.Runtime {
		on := c.OnModel
		c.OnModel = func(ev resident.Event) {
			if on != nil {
				on(ev)
			}
			if ev.Kind == resident.Loaded {
				ch <- ev
			}
		}
		return newRT(c)
	}
	t.Cleanup(func() { omniNewRuntime = newRT })
	return ch
}

// TestModelsMemory: the Speech models page says the memory of the model
// loaded in the fake Omni server, measured by its pid, every memPoll
// while the page shows; nothing for a model that is not loaded, and
// nothing after an unload.
func TestModelsMemory(t *testing.T) {
	fakeOmni(t, omnitest.Opts{})
	loads := omniLoads(t)
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1600) // every card on screen
	timers := new(fakeTimers).install(a)
	pm := (&fakeProcMemory{mem: map[int]int64{}}).install(t)
	seedModels(t, a, models.ASRFast)

	a.navigate(modelsPath)
	if has(tt, "Loaded") || len(a.modelMem) != 0 {
		t.Fatalf("nothing is loaded: %v %q", a.modelMem, tt.Texts())
	}

	// The speech model loads (a session opens it).
	if _, _, err := a.local.ASR(context.Background(), ""); err != nil {
		t.Fatal(err)
	}
	var pid int
	select { // the open's warm
	case ev := <-loads:
		pid = ev.PID
	case <-time.After(10 * time.Second):
		t.Fatal("the speech server did not load")
	}
	if ps := a.omniRT.Processes(); len(ps) != 1 || ps[0].PID != pid {
		t.Fatalf("processes %+v, loaded pid %d", ps, pid)
	}
	pm.set(pid, 1536<<20)
	timers.advance(memPoll)
	frames(tt)
	if !has(tt, "Loaded · 1.5 GB in memory") {
		t.Errorf("loaded models: %v %q", a.modelMem, tt.Texts())
	}
	if len(a.modelMem) != 1 {
		t.Errorf("memory %v", a.modelMem)
	}
	if !slices.Contains(pm.asked, pid) {
		t.Errorf("asked %v", pm.asked)
	}

	// Measured again on the next tick; zh-Hans says it in Chinese.
	pm.set(pid, 1<<30)
	timers.advance(memPoll)
	i18n.SetLocale("zh-Hans")
	frames(tt)
	if !has(tt, "已载入 · 占用内存 1.0 GB") {
		t.Errorf("zh-Hans: %q", tt.Texts())
	}
	i18n.SetLocale("en")

	// Another page stops the ticks; the next visit measures at once.
	a.navigate("/home")
	timers.advance(memPoll)
	if a.memPolling {
		t.Error("still measuring on Home")
	}
	asked := len(pm.asked)
	timers.advance(10 * memPoll)
	if len(pm.asked) != asked {
		t.Error("measured on Home")
	}
	a.unloadOmni(models.ASRFast)
	if ps := a.omniRT.Processes(); len(ps) != 0 { // Unload takes its servers off the list at once
		t.Errorf("processes after the unload %+v", ps)
	}
	a.navigate(modelsPath)
	frames(tt)
	if has(tt, "Loaded") || len(a.modelMem) != 0 {
		t.Errorf("after the unload: %v %q", a.modelMem, tt.Texts())
	}
	if !a.memPolling {
		t.Error("the visit did not start measuring")
	}
}
