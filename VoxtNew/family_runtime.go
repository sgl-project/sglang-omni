package main

// VoxtNew's runtime file in the ActNow family (internal/actnow's Runtime,
// spec §11.1): the models the runtimes hold, from their events
// (internal/resident: omni's servers, worker "omni"; voxtnew-llm's
// model, worker "llm"), each with its repo, commit and memory; the
// family's memory budget before a model loads (Reserve); another app's
// memory request unloading VoxtNew's models idle for a minute (OnUnload);
// and the "pasting" flag around every paste through the clipboard
// (familyText). Quitting removes the file (closeFamily, after the
// runtimes stopped).

import (
	"context"
	"log"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/actnow"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/resident"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
)

// Workers of the runtime file: which runtime holds a model.
const (
	workerOmni = "omni"
	workerLLM  = "llm"
)

// famRuntime is the runtime file, nil without a family (some tests).
func (a *app) famRuntime() *actnow.Runtime {
	if a.fam == nil {
		return nil
	}
	return a.fam.Runtime()
}

// connectFamilyRuntime lets another app's memory request unload
// VoxtNew's idle models (openFamily).
func (a *app) connectFamilyRuntime() {
	if rt := a.famRuntime(); rt != nil {
		rt.OnUnload(a.unloadIdleModel)
	}
}

// modelEvents is a runtime's Config.OnModel: its events in the runtime
// file (on the runtime's event goroutine: the measure is a system call).
func (a *app) modelEvents(worker string) func(resident.Event) {
	rt, mgr := a.famRuntime(), a.models
	if rt == nil {
		return nil
	}
	return func(ev resident.Event) {
		switch ev.Kind {
		case resident.Loaded:
			m := lookupModel(mgr, ev.ID)
			rev := ""
			if mgr != nil {
				rev = mgr.Status(ev.ID).Revision
			}
			rt.Loaded(ev.ID, worker, m.Repo, rev, modelBytes(ev, m))
		case resident.Busy:
			rt.Busy(ev.ID)
		case resident.Idle:
			rt.Idle(ev.ID)
		case resident.Unloaded:
			rt.Unloaded(ev.ID)
		}
	}
}

// beforeLoad is a runtime's Config.BeforeLoad: room in the family's
// memory budget for the model's size.
func (a *app) beforeLoad() func(ctx context.Context, id string) {
	rt, mgr := a.famRuntime(), a.models
	if rt == nil {
		return nil
	}
	return func(ctx context.Context, id string) {
		rt.Reserve(ctx, id, lookupModel(mgr, id).Size())
	}
}

// lookupModel is model id of the app's catalogue (zero when unknown: a
// developer's model).
func lookupModel(mgr *models.Manager, id string) models.Model {
	var m models.Model
	var ok bool
	if mgr != nil {
		m, ok = mgr.Lookup(id)
	}
	if !ok {
		m, _ = models.Lookup(id)
	}
	return m
}

// modelBytes is the memory a model loaded in takes: the process's
// measure (macx: its footprint or resident memory), else what the
// runtime reported, else the size of its files.
func modelBytes(ev resident.Event, m models.Model) int64 {
	if ev.PID > 0 {
		if b, err := procMemory(ev.PID); err == nil && b > 0 {
			return b
		}
	}
	if ev.Bytes > 0 {
		return ev.Bytes
	}
	return m.Size()
}

// unloadIdleModel lets model id go when nothing uses it (the family's
// memory requests): the language model's process stops when idle, a
// speech model's server once its uses ended (never during one).
func (a *app) unloadIdleModel(id string) {
	if m, ok := models.Lookup(id); ok && m.Kind == models.LLM {
		if rt := a.llmMade(); rt != nil && !rt.UnloadIdle(id) {
			log.Printf("family: %s is in use: kept", id)
		}
		return
	}
	a.omniMu.Lock()
	rt := a.omniRT
	a.omniMu.Unlock()
	if rt != nil {
		rt.Unload(id)
	}
}

// familyText is the helper's text I/O with the family's paste rule: a
// paste waits for another voice-input app's (at most 500 ms) and says
// "pasting" in the runtime file while it runs (textio.paste answers once
// the clipboard is restored). Every paste goes through it: sessions, the
// answer card's Insert, paste-last.
type familyText struct {
	session.HelperText
	rt *actnow.Runtime // nil: no family
}

// textIO is the text I/O sessions and the pages paste with.
func (a *app) textIO() familyText {
	return familyText{HelperText: session.HelperText{H: a.host.sup}, rt: a.famRuntime()}
}

func (t familyText) Paste(ctx context.Context, p proto.PasteParams) (proto.PasteResult, error) {
	if t.rt == nil {
		return t.HelperText.Paste(ctx, p)
	}
	var res proto.PasteResult
	err := t.rt.Paste(ctx, func() error {
		var err error
		res, err = t.HelperText.Paste(ctx, p)
		return err
	})
	return res, err
}
