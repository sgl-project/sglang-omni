package main

// The speech models' runtime (internal/omni): the Omni native MLX
// servers, found in the app (Contents/Resources/omni) or at
// $VOXTNEW_OMNI_RUNTIME. localOpeners (models_runtimes.go) opens the
// speech models with openOmniASR and unloads a model about to be deleted
// with unloadOmni; quitting stops every server (closeOmni). The servers
// write to the app's log (source "models"). A Whisper model's server
// reads its tokenizer (an Aux of the catalogue, another repo) from the
// model's folder: it loads a folder of links to both (omniFolder) in the
// app's cache.

import (
	"cmp"
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
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
	// speechWarmed: Transcribe's speech model was warmed at the start
	// (warmSpeech; main thread only).
	speechWarmed bool
	// omniCache holds the folders of links omniFolder makes ("": the
	// system's temporary folder, in tests).
	omniCache string
}

// omniLocate finds the runtime (a test seam).
var omniLocate = func() (string, error) {
	exe, _ := os.Executable()
	return omni.Locate(exe)
}

// omniNewRuntime makes the runtime (a test seam).
var omniNewRuntime = omni.NewRuntime

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
	speech := keepIdle(a.keepSetting()) // models_keep.go
	a.omniRT = omniNewRuntime(omni.Config{Binary: bin, Log: a.omniLog, IdleTimeout: speech,
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
// Silero and the model's draft when they are installed, and starts
// loading both: the first session's model loads while the user speaks.
// Later calls only touch the servers (their idle time starts again).
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
	draft := a.omniDraft(id)
	key := id + "\x00" + dir + "\x00" + draft
	p := a.omniASR[key]
	if p == nil || (p.VAD() != nil) != (v != nil) {
		served, err := a.omniFolder(id, dir)
		if err != nil {
			return nil, err
		}
		base, err := omni.NewASR(rt, id, served)
		if err != nil {
			return nil, err
		}
		p = base.WithDraft(draft).WithVAD(v)
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

// warmSpeech starts loading Transcribe's speech model when it is a local
// one (a remote service and System dictation load nothing), so that the
// first session after the app opened, or after another model was chosen,
// finds its server ready. The server stops when idle, as after a session.
func (a *app) warmSpeech() {
	m, ok := a.featureModel(local.ASR, string(store.KindTranscribe))
	if cm, _ := models.Lookup(m.ID); !ok || cm.Runtime != models.Omni {
		return
	}
	lm := a.local
	a.async(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if _, _, err := lm.ASR(ctx, m.ID); err != nil { // openOmniASR warms it
			a.omniLog("warning", "warming "+m.ID+": "+err.Error())
		}
	})
}

// omniFolder is the folder model id's server loads: dir, its snapshot, or,
// when the model needs an Aux (Whisper's tokenizer), a folder of links to
// the files of both snapshots, made again only when its links changed (a
// server may be loading from it; a.omniMu held).
func (a *app) omniFolder(id, dir string) (string, error) {
	dirs := []string{dir}
	mod, _ := models.Lookup(id)
	for _, need := range mod.Needs {
		if n, _ := models.Lookup(need); n.Kind == models.Aux {
			d := a.models.Dir(need)
			if d == "" {
				return "", fmt.Errorf("%w: %s, which %s needs", local.ErrNotInstalled, n.Name, mod.Name)
			}
			dirs = append(dirs, d)
		}
	}
	if len(dirs) == 1 {
		return dir, nil
	}
	links := map[string]string{} // name: target
	for _, d := range dirs {
		files, err := os.ReadDir(d)
		if err != nil {
			return "", err
		}
		for _, f := range files {
			links[f.Name()] = filepath.Join(d, f.Name())
		}
	}
	out := filepath.Join(cmp.Or(a.omniCache, filepath.Join(os.TempDir(), "voxtnew-omni")), id)
	have, _ := os.ReadDir(out)
	same := len(have) == len(links)
	for _, f := range have {
		if target, err := os.Readlink(filepath.Join(out, f.Name())); err != nil || target != links[f.Name()] {
			same = false
		}
	}
	if same {
		return out, nil
	}
	if err := os.RemoveAll(out); err != nil {
		return "", err
	}
	if err := os.MkdirAll(out, 0o755); err != nil {
		return "", err
	}
	for name, target := range links {
		if err := os.Symlink(target, filepath.Join(out, name)); err != nil {
			return "", err
		}
	}
	return out, nil
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

// omniDraft is the folder of speech model id's installed draft (its
// catalogue Draft), "" when it has none, it is not installed or its
// Faster transcription is off: the server then decodes without one.
// Off the UI goroutine: the switch is read from the settings service.
func (a *app) omniDraft(id string) string {
	m, ok := models.Lookup(id)
	if !ok || m.Draft == "" || a.models == nil {
		return ""
	}
	dir := a.models.Dir(m.Draft)
	if a.settings != nil && !a.settings.Get().Models.FasterOn(id, dir != "") {
		return ""
	}
	return dir
}

// omniMeetingHold keeps the meeting's local omni ASR and Silero VAD loaded.
func (a *app) omniMeetingHold() (release func()) {
	a.omniMu.Lock()
	defer a.omniMu.Unlock()
	if a.omniEnd {
		return func() {}
	}
	rt, err := a.omniRuntime()
	if err != nil {
		return func() {}
	}
	asrID := a.prefs.Features.Meeting.ASR
	dir := ""
	if asrID != "" {
		m, ok := models.Lookup(asrID)
		if !ok || m.Runtime != models.Omni {
			return func() {}
		}
		dir = a.models.Dir(asrID)
	} else {
		asrID, dir = a.firstOmniASR()
	}
	if dir == "" {
		return func() {}
	}
	spec, ok := omni.Lookup(asrID)
	if !ok {
		return func() {}
	}
	ctx, cancel := context.WithTimeout(context.Background(), omni.DefaultStartTimeout+time.Second)
	defer cancel()
	targets := []omni.HoldTarget{{Spec: spec, Dir: dir, Draft: a.omniDraft(asrID)}}
	if vadDir := a.models.Dir(models.VADSilero); vadDir != "" {
		if vspec, ok := omni.Lookup(omni.VADSilero); ok {
			targets = append(targets, omni.HoldTarget{Spec: vspec, Dir: vadDir})
		}
	}
	rel, err := rt.Hold(ctx, targets...)
	if err != nil {
		a.omniLog("warning", "meeting hold: "+err.Error())
		return func() {}
	}
	return rel
}

// firstOmniASR is the speech model a meeting on Automatic uses: the
// first of Installed, the recommended one first.
func (a *app) firstOmniASR() (id, dir string) {
	if a.models == nil || a.local == nil {
		return "", ""
	}
	for _, m := range a.local.Installed(local.ASR) {
		if m.Stream { // an Omni model
			return m.ID, a.models.Dir(m.ID)
		}
	}
	return "", ""
}

// unloadOmni stops model id's server before its files are deleted; a
// draft's speech model holds the draft, so its server stops.
func (a *app) unloadOmni(id string) {
	for _, m := range models.Catalog() {
		if m.Draft == id {
			a.unloadOmni(m.ID)
		}
	}
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
