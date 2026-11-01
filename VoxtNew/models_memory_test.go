package main

import (
	"context"
	"errors"
	"os"
	"slices"
	"sync"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llm"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/llmrttest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/omni/omnitest"
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

// TestModelsMemory: the Local models page says the memory of the models
// loaded in the fake Omni server and the fake voxtnew-llm, measured by
// their pids (the language model's own report when its process cannot
// be measured), every memPoll while the page shows; nothing for a model
// that is not loaded, and nothing after an unload.
func TestModelsMemory(t *testing.T) {
	fakeOmni(t, omnitest.Opts{})
	t.Setenv(llmrt.DevEnv, "")
	t.Setenv("VOXTNEW_LLM", "")
	a, _, tt := newTestApp(t)
	tt.SetSize(windowWidth, 1600) // every card on screen
	timers := new(fakeTimers).install(a)
	pm := (&fakeProcMemory{mem: map[int]int64{}}).install(t)
	self, _ := os.Executable()
	env := append(llmrttest.Env(llmrttest.Opt("reply", "Done.")), omnitest.EnvFake+"=") // this one is no Omni server
	rt := llmrt.New(llmrt.Config{Bin: self, Env: env, IdleTimeout: -1})
	a.llmrt = rt
	seedModels(t, a, models.ASRFast, models.LLMFast, models.LLMTranslate)

	a.navigate(modelsPath)
	if has(tt, "Loaded") || len(a.modelMem) != 0 {
		t.Fatalf("nothing is loaded: %v %q", a.modelMem, tt.Texts())
	}

	// The speech model loads (a session opens it), then the language model.
	if _, _, err := a.local.ASR(context.Background(), ""); err != nil {
		t.Fatal(err)
	}
	var pid int
	eventually(t, "the speech server", func() bool {
		ps := a.omniRT.Processes()
		if len(ps) == 1 {
			pid = ps[0].PID
		}
		return len(ps) == 1
	})
	pm.set(pid, 1536<<20)
	chat, _, err := a.local.LLM(context.Background(), models.LLMFast)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := chat.Chat(context.Background(), llm.Request{Messages: []llm.Message{{Role: llm.User, Content: "hi"}}, Skill: llm.SkillRewrite}); err != nil {
		t.Fatal(err)
	}
	timers.advance(memPoll)
	frames(tt)
	if !has(tt, "Loaded · 1.5 GB in memory") || !has(tt, "Loaded · 2.0 GB in memory") {
		t.Errorf("loaded models: %v %q", a.modelMem, tt.Texts())
	}
	if len(a.modelMem) != 2 || a.modelMem[models.LLMTranslate] != 0 {
		t.Errorf("memory %v", a.modelMem)
	}
	if !slices.Contains(pm.asked, pid) || !slices.Contains(pm.asked, rt.Status().PID) {
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
	a.unloadModel(models.ASRFast)
	rt.Unload()
	eventually(t, "the unload", func() bool { return len(a.omniRT.Processes()) == 0 })
	a.navigate(modelsPath)
	frames(tt)
	if has(tt, "Loaded") || len(a.modelMem) != 0 {
		t.Errorf("after the unload: %v %q", a.modelMem, tt.Texts())
	}
	if !a.memPolling {
		t.Error("the visit did not start measuring")
	}
}

// TestLLMRuntimeAfterUnload: an Unload before the language models'
// runtime was made leaves it to be made later (it was a sync.Once that
// the Unload used up), and after stopLLM the runtime is a stopped one.
func TestLLMRuntimeAfterUnload(t *testing.T) {
	a := &app{}
	a.unloadLLM(models.LLMFast)
	if a.llmMade() != nil {
		t.Fatal("the unload made a runtime")
	}
	a.llmrt = llmrt.New(llmrt.Config{})
	if a.llmRuntime() != a.llmrt {
		t.Error("another runtime")
	}
	b := &app{}
	b.stopLLM()
	if _, err := b.llmRuntime().Call(context.Background(), "llm.status", nil, nil); !errors.Is(err, llmrt.ErrStopped) {
		t.Errorf("after stopLLM: %v", err)
	}
}
