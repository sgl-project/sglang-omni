package llmrt

// The model the process holds, as Config.OnModel hears it (the family's
// runtime file): loaded once a load or a chat of it succeeded, busy while
// a load, chat or prefill of it runs, idle after, unloaded when the
// process lets it go for another model or exits (idle stop, Unload,
// Stop, a crash).

import (
	"context"
	"encoding/json"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/wire"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/resident"
)

// modelCall is Call for a request that uses model m (model.load, chat):
// Config.BeforeLoad runs first when the process does not hold m.
func (r *Runtime) modelCall(ctx context.Context, m wire.Model, method string, params any, notify func(wire.Message)) (json.RawMessage, error) {
	r.mu.Lock()
	load := r.held != m.ID
	if load {
		r.held = "" // the process lets the other go for m
	}
	r.report()
	r.mu.Unlock()
	if load && r.cfg.BeforeLoad != nil {
		r.cfg.BeforeLoad(ctx, m.ID)
	}
	r.mu.Lock()
	r.work++
	r.report()
	r.mu.Unlock()
	raw, err := r.Call(ctx, method, params, notify)
	r.mu.Lock()
	defer r.mu.Unlock()
	r.work--
	if err == nil && r.proc != nil {
		var res struct {
			Memory wire.Memory `json:"memory"`
		}
		_ = json.Unmarshal(raw, &res)
		if r.held != m.ID {
			r.held, r.heldBytes = m.ID, max(res.Memory.FootprintBytes, res.Memory.ResidentBytes)
		}
	}
	r.report()
	return raw, err
}

// report tells Config.OnModel what changed (r.mu held).
func (r *Runtime) report() {
	id, pid := r.held, 0
	if r.proc == nil {
		id = ""
	} else if r.proc.cmd.Process != nil {
		pid = r.proc.cmd.Process.Pid
	}
	r.tracker.Set(r.events, id, pid, r.work > 0, r.heldBytes)
}

// gone notes that process p ended (r.mu held): its model left memory.
func (r *Runtime) gone(p *proc) {
	if r.proc == p {
		r.proc = nil
	}
	if r.proc == nil {
		r.held, r.heldBytes = "", 0
	}
	r.report()
}

// UnloadIdle stops the process when it holds model id and nothing uses
// it (another app asked for memory: a session is never interrupted). It
// reports whether it did.
func (r *Runtime) UnloadIdle(id string) bool {
	r.mu.Lock()
	p := r.proc
	if p == nil || r.held != id || r.work > 0 || r.inFlight > 0 || p.busy() {
		r.mu.Unlock()
		return false
	}
	r.mu.Unlock()
	r.shutdown(p, Stopped, "")
	return true
}

// Held is the model the process holds as far as the runtime knows ("":
// none; memory only).
func (r *Runtime) Held() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.proc == nil {
		return ""
	}
	return r.held
}

// newEvents is Config.OnModel's queue.
func newEvents(cfg Config) *resident.Queue { return resident.NewQueue(cfg.OnModel) }
