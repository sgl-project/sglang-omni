package main

// The memory the loaded models use, on the Local models page: the speech
// models' servers (internal/omni) and the language models' process
// (internal/llmrt) are processes of their own, measured by their process
// ids (macx.Memory: the footprint, or the resident memory when the
// weights are mapped from their file) off the main thread, when the page
// shows and every few seconds while it stays in front. Views read
// a.modelMem only.

import (
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/macx"
)

// memPoll is how often the page measures the loaded models again.
const memPoll = 3 * time.Second

// memoryState is what the Local models page knows of the loaded models.
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
	// reported is the memory the process last reported itself (the
	// language models' process does, after a load or a chat), for when
	// it cannot be measured.
	reported int64
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
	if lrt := a.llmMade(); lrt != nil {
		// Status, not Memory(ctx): asking the process would start its
		// idle time again, and a page left open would keep the model
		// loaded.
		st := lrt.Status()
		if st.State == llmrt.Ready && st.Model != nil {
			out = append(out, loadedProc{id: st.Model.ID, pid: st.PID,
				reported: max(st.Memory.FootprintBytes, st.Memory.ResidentBytes)})
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
		if err != nil || b <= 0 {
			b = p.reported
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

// modelsShown runs when the Local models page shows: it measures now,
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
