package main

// The memory the loaded models use, on the Speech models page: the speech
// models' servers (internal/omni) are processes of their own, measured by
// their process ids (macx.Memory: the footprint, or the resident memory when the
// weights are mapped from their file) off the main thread, when the page
// shows and every few seconds while it stays in front. Views read
// a.modelMem only.

import (
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/macx"
)

// memPoll is how often the page measures the loaded models again.
const memPoll = 3 * time.Second

// memoryState is what the Speech models page knows of the loaded models.
type memoryState struct {
	// modelMem is each loaded model's memory in bytes (0: loaded, not
	// measured); a model that is not loaded is absent.
	modelMem map[string]int64
	// memPolling: a measure is due (memTick), while the page shows.
	memPolling bool
}

// procMemory measures the process with pid (a test seam).
var procMemory = func(pid int) (int64, error) {
	m, err := macx.Memory(pid)
	return m.InUse(), err
}

// loadedProc is a loaded model's process.
type loadedProc struct {
	id  string
	pid int
}

// loadedModels lists the models loaded now and their processes (memory
// only; goroutine-safe). The speech models' list has Silero's server
// too, which no card shows.
func (a *app) loadedModels() []loadedProc {
	var out []loadedProc
	a.omniMu.Lock()
	ort := a.omniRT
	a.omniMu.Unlock()
	if ort != nil {
		for _, p := range ort.Processes() {
			out = append(out, loadedProc{id: p.ID, pid: p.PID})
		}
	}
	return out
}

// measureModels is each loaded model's memory (system calls: off the
// main thread).
func measureModels(ps []loadedProc) map[string]int64 {
	out := make(map[string]int64, len(ps))
	for _, p := range ps {
		b, err := procMemory(p.pid)
		if err != nil || b < 0 {
			b = 0 // loaded, not measured
		}
		out[p.id] = b
	}
	return out
}

// readModelMemory measures the loaded models off the main thread.
func (a *app) readModelMemory() {
	a.async(func() {
		mem := measureModels(a.loadedModels())
		a.post(func() { a.modelMem = mem })
	})
}

// modelsShown runs when the Speech models page shows: it measures now,
// then every memPoll while the page stays in front.
func (a *app) modelsShown() {
	a.readModelMemory()
	if !a.memPolling {
		a.memPolling = true
		a.later(memPoll, a.memTick)
	}
}

// memTick measures again while the page shows, when the window is
// visible, and stops on another page (the next visit starts it again).
func (a *app) memTick() {
	if a.currentPage() != modelsPath[1:] {
		a.memPolling = false
		return
	}
	if a.win == nil || a.win.IsVisible() {
		a.readModelMemory()
	}
	a.later(memPoll, a.memTick)
}
