package main

// The language models' runtime (internal/llmrt): the voxtnew-llm process,
// made on first use and stopped on quit, behind local.Openers' LLM and
// Unload (models_runtimes.go).
//
// The developers' hook VOXTNEW_DEV_LLM=<id>=<dir>[,<id>=<dir>…] (llmrt's
// dev.go) lists those curated models as installed from those folders
// (an HF snapshot, say), ahead of the catalogue's:
//
//	VOXTNEW_DEV_LLM=llm.balanced=~/models/qwen open build/darwin-arm64/VoxtNew.app

import (
	"context"
	"log"
	"os"
	"sync"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
)

// llmState is the language models' runtime; goroutine-safe (the openers
// run on sessions' goroutines, the Local models page measures it on
// others).
type llmState struct {
	llmMu  sync.Mutex
	llmrt  *llmrt.Runtime
	llmEnd bool // stopped: the app quits
}

// llmRuntime is the runtime, made on first use (off the main thread: the
// openers run there; it starts no process). After stopLLM it is a stopped
// one, whose calls fail.
func (a *app) llmRuntime() *llmrt.Runtime {
	a.llmMu.Lock()
	defer a.llmMu.Unlock()
	switch {
	case a.llmrt != nil:
	case a.llmEnd:
		a.llmrt = llmrt.New(llmrt.Config{})
		a.llmrt.Stop()
	default:
		logDir, _ := mygo.App.Path(mygo.PathLogs)
		logs := a.logs
		_, idle := keepIdle(a.keepSetting()) // models_keep.go
		a.llmrt = llmrt.New(llmrt.Config{
			Bin: llmLocate(), AppVersion: mygo.App.Version(), LogDir: logDir, IdleTimeout: idle,
			Log: func(e core.LogEntry) {
				if logs != nil {
					logs.Append(e)
				}
			},
			OnModel: a.modelEvents(workerLLM), BeforeLoad: a.beforeLoad(), // family_runtime.go
			OnPace: a.onPace(), // memory_notice.go
		})
	}
	return a.llmrt
}

// llmMade is the runtime when one was made, else nil (it makes none).
func (a *app) llmMade() *llmrt.Runtime {
	a.llmMu.Lock()
	defer a.llmMu.Unlock()
	return a.llmrt
}

// openLLM is local.Openers' LLM: the model's client; the first open of a
// session starts the process and loads the model in the background, later
// ones find it loaded.
func (a *app) openLLM(ctx context.Context, id, dir string) (local.Chat, error) {
	return llmrt.Open(ctx, a.llmRuntime(), id, dir)
}

// unloadLLM is local.Openers' Unload: before model id is deleted, the
// process that holds it stops.
func (a *app) unloadLLM(id string) {
	if rt := a.llmMade(); rt != nil { // never made: nothing to unload
		if m := rt.Status().Model; m != nil && m.ID == id {
			rt.Unload()
		}
	}
}

// withDevLLM is base with VOXTNEW_DEV_LLM's models, when it is set.
func (a *app) withDevLLM(base local.Services) local.Services {
	spec := os.Getenv(llmrt.DevEnv)
	if spec == "" {
		return base
	}
	models, err := llmrt.ParseDev(spec)
	if err != nil {
		log.Printf("%v", err)
		return base
	}
	return llmrt.DevServices(base, a.llmRuntime(), models)
}

// stopLLM stops the language models' process (the app quits); a runtime
// asked for later is a stopped one.
func (a *app) stopLLM() {
	a.llmMu.Lock()
	a.llmEnd = true
	rt := a.llmrt
	a.llmMu.Unlock()
	if rt != nil {
		rt.Stop()
	}
}
