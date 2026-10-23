package main

// The speech models' runtime (internal/omni): the Omni native MLX
// servers, found in the app (Contents/Resources/omni) or at
// $VOXTNEW_OMNI_RUNTIME. localOpeners (models_runtimes.go) opens the
// speech models with openOmniASR and unloads a model about to be deleted
// with unloadOmni; quitting stops every server (closeOmni). The servers
// write to the app's log (source "models").

import (
	"context"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/local"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/omni"
)

// omniSource is the logs' source of the runtime's lines.
const omniSource = "models"

// omniState is the runtime and what it opened; goroutine-safe (the
// openers run on sessions' goroutines).
type omniState struct {
	omniMu  sync.Mutex
	omniRT  *omni.Runtime
	omniErr error                // Locate's, once: no runtime in this build
	omniASR map[string]*omni.ASR // by id and folder
	omniVAD *omni.VAD
	omniDir string // the VAD's folder
	omniEnd bool   // closed: quitting
}

// omniLocate finds the runtime (a test seam).
var omniLocate = func() (string, error) {
	exe, _ := os.Executable()
	return omni.Locate(exe)
}

// omniRuntime is the runtime, made on first use (a.omniMu held).
func (a *app) omniRuntime() (*omni.Runtime, error) {
	if a.omniRT != nil || a.omniErr != nil {
		return a.omniRT, a.omniErr
	}
	bin, err := omniLocate()
	if err != nil {
		a.omniErr = fmt.Errorf("%w: %v", local.ErrNoRuntime, err)
		a.omniLog("error", err.Error())
		return nil, a.omniErr
	}
	speech, _ := keepIdle(a.keepSetting()) // models_keep.go
	a.omniRT = omni.NewRuntime(omni.Config{Binary: bin, Log: a.omniLog, IdleTimeout: speech,
		OnModel: a.modelEvents(workerOmni), BeforeLoad: a.beforeLoad()}) // family_runtime.go
	return a.omniRT, nil
}

// omniLog writes a line of the runtime to the app's log.
func (a *app) omniLog(level, msg string) {
	// a.logs is set before the window opens (connectShell replaces the
	// first one, before any session).
	if lg := a.logs; lg != nil {
		lg.Append(core.LogEntry{Time: time.Now(), Level: core.Severity(level), Source: omniSource, Message: msg})
	}
}

// openOmniASR opens speech model id in dir (local.ASROpener), with
// Silero when it is installed, and starts loading both: the first
// session's model loads while the user speaks. Later calls only touch
// the servers (their idle time starts again).
func (a *app) openOmniASR(_ context.Context, id, dir string) (asr.Provider, error) {
	a.omniMu.Lock()
	defer a.omniMu.Unlock()
	if a.omniEnd {
		return nil, fmt.Errorf("%w: quitting", local.ErrNoRuntime)
	}
	rt, err := a.omniRuntime()
	if err != nil {
		return nil, err
	}
	v := a.omniSilero(rt)
	key := id + "\x00" + dir
	p := a.omniASR[key]
	if p == nil || (p.VAD() != nil) != (v != nil) {
		base, err := omni.NewASR(rt, id, dir)
		if err != nil {
			return nil, err
		}
		p = base.WithVAD(v)
		if a.omniASR == nil {
			a.omniASR = map[string]*omni.ASR{}
		}
		a.omniASR[key] = p
	}
	p.Warm()
	if v != nil {
		v.Warm()
	}
	return p, nil
}

// omniSilero is the VAD when it is installed, else nil (a.omniMu held).
func (a *app) omniSilero(rt *omni.Runtime) *omni.VAD {
	dir := ""
	if a.models != nil {
		dir = a.models.Dir(models.VADSilero)
	}
	if dir == "" {
		return nil
	}
	if a.omniVAD == nil || a.omniDir != dir {
		v, err := omni.NewVAD(rt, dir)
		if err != nil {
			a.omniLog("warning", "Silero VAD: "+err.Error())
			return nil
		}
		a.omniVAD, a.omniDir = v, dir
	}
	return a.omniVAD
}

// unloadOmni stops model id's server before its files are deleted.
func (a *app) unloadOmni(id string) {
	a.omniMu.Lock()
	rt := a.omniRT
	for k := range a.omniASR {
		if len(k) > len(id) && k[:len(id)+1] == id+"\x00" {
			delete(a.omniASR, k)
		}
	}
	if id == models.VADSilero {
		a.omniVAD, a.omniDir = nil, ""
		a.omniASR = nil // they carry the VAD
	}
	a.omniMu.Unlock()
	if rt != nil {
		rt.Unload(id)
	}
}

// closeOmni stops the servers (quit).
func (a *app) closeOmni() {
	a.omniMu.Lock()
	rt := a.omniRT
	a.omniEnd = true
	a.omniMu.Unlock()
	if rt != nil {
		rt.Close()
	}
}
