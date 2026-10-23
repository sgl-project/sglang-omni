package omni

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/resident"
)

// Provider is the name errors carry (asr.Error.Provider).
const Provider = "omni"

// EnvRuntime names a runtime directory (with bin/qwen3_asr_server) for
// development: it wins over the app's own.
const EnvRuntime = "VOXTNEW_OMNI_RUNTIME"

// ServerBinary is the server's path in a runtime directory.
const ServerBinary = "bin/qwen3_asr_server"

// Timeouts and limits of a Runtime (Config's zero values).
const (
	DefaultStartTimeout = 180 * time.Second // Voxt's: a cold first load compiles Metal kernels
	DefaultIdleTimeout  = 10 * time.Minute
	DefaultStallTimeout = 15 * time.Second
	shutdownGrace       = 5 * time.Second
	terminateGrace      = 3 * time.Second
)

// Config is how a Runtime starts its servers.
type Config struct {
	// Binary is qwen3_asr_server (Locate finds it).
	Binary string
	// Env is added to the servers' environment (VoxtNew's, without the
	// dynamic loader's DYLD_ settings).
	Env []string
	// StartTimeout bounds a server's start (its model's load); a server
	// that has not said it is ready by then is killed.
	StartTimeout time.Duration
	// IdleTimeout stops a server no one used for that long (< 0: never;
	// SetIdleTimeout changes it).
	IdleTimeout time.Duration
	// StallTimeout: a Final whose response stream (the server's heartbeat
	// is every 250 ms) is silent this long has hung its server, which is
	// killed and replaced at the next use.
	StallTimeout time.Duration
	// Log gets a line about the servers ("info", "warning", "error"); their
	// stderr goes there too (they never write audio or text). nil: none.
	Log func(level, msg string)
	// OnModel is told when a server's model is loaded (the server said it
	// is ready), busy (its first lease), idle (its last lease ended) and
	// unloaded (its process exited: idle stop, switch, Unload, Close or a
	// crash), Silero's too; in order, on a goroutine of the runtime's,
	// under none of its locks. nil: nobody.
	OnModel func(resident.Event)
	// BeforeLoad runs before a server starts loading its model (the
	// family's memory budget makes room); it may block a little. nil:
	// nothing.
	BeforeLoad func(ctx context.Context, id string)
}

// beforeLoadTimeout bounds Config.BeforeLoad.
const beforeLoadTimeout = 5 * time.Second

// Locate finds the runtime: $VOXTNEW_OMNI_RUNTIME (a directory with
// bin/qwen3_asr_server, or the binary itself), else the app bundle's
// Contents/Resources/omni beside exe (the running executable,
// Contents/MacOS/VoxtNew). It returns the server binary's path.
func Locate(exe string) (string, error) {
	var tried []string
	try := func(p string) (string, bool) {
		tried = append(tried, p)
		fi, err := os.Stat(p)
		if err == nil && fi.IsDir() {
			p = filepath.Join(p, ServerBinary)
			fi, err = os.Stat(p)
		}
		return p, err == nil && fi.Mode().IsRegular()
	}
	if env := strings.TrimSpace(os.Getenv(EnvRuntime)); env != "" {
		if p, ok := try(env); ok {
			return p, nil
		}
		return "", fmt.Errorf("omni: %s=%s has no %s", EnvRuntime, env, ServerBinary)
	}
	if exe != "" {
		if p, ok := try(filepath.Join(filepath.Dir(exe), "..", "Resources", "omni")); ok {
			return filepath.Clean(p), nil
		}
	}
	return "", fmt.Errorf("omni: no runtime (looked in %s; set %s)", strings.Join(tried, ", "), EnvRuntime)
}

// Runtime supervises the servers: one per model, one speech model at a
// time, and Silero's beside it. Goroutine-safe.
type Runtime struct {
	cfg    Config
	events *resident.Queue // Config.OnModel's
	idle   atomic.Int64    // the idle timeout (Config.IdleTimeout, SetIdleTimeout)

	mu      sync.Mutex
	servers map[string]*server // by model id and directory
	// draining are speech models' servers retired by a switch, until they
	// stopped: the next speech model starts after them.
	draining []*server
	closed   bool

	cacheLimitOnce  sync.Once
	cacheLimitKnown bool // the binary knows --cache-limit-mb (knowsCacheLimit)
}

// NewRuntime is a runtime that starts no server until one is used.
func NewRuntime(cfg Config) *Runtime {
	if cfg.StartTimeout <= 0 {
		cfg.StartTimeout = DefaultStartTimeout
	}
	if cfg.IdleTimeout == 0 {
		cfg.IdleTimeout = DefaultIdleTimeout
	}
	if cfg.StallTimeout <= 0 {
		cfg.StallTimeout = DefaultStallTimeout
	}
	if cfg.Log == nil {
		cfg.Log = func(string, string) {}
	}
	rt := &Runtime{cfg: cfg, events: resident.NewQueue(cfg.OnModel), servers: map[string]*server{}}
	rt.idle.Store(int64(cfg.IdleTimeout))
	return rt
}

// cacheLimit reports whether spec's server gets --cache-limit-mb: the
// binary is asked once, as the app may run an older runtime without it.
func (rt *Runtime) cacheLimit(spec Spec) bool {
	if spec.ASR == nil || spec.ASR.CacheMB <= 0 {
		return false
	}
	rt.cacheLimitOnce.Do(func() {
		rt.cacheLimitKnown = knowsCacheLimit(rt.cfg)
		if !rt.cacheLimitKnown {
			rt.logf("warning", "%s does not know --cache-limit-mb (an older runtime): MLX's cache stays unbounded", rt.cfg.Binary)
		}
	})
	return rt.cacheLimitKnown
}

// IdleTimeout is how long an unused server stays (< 0: until Close).
func (rt *Runtime) IdleTimeout() time.Duration { return time.Duration(rt.idle.Load()) }

// SetIdleTimeout changes how long an unused server stays: d < 0 until
// Close, 0 DefaultIdleTimeout. The servers unused now count it from now;
// the ones in use, from the end of their use.
func (rt *Runtime) SetIdleTimeout(d time.Duration) {
	if d == 0 {
		d = DefaultIdleTimeout
	}
	if time.Duration(rt.idle.Swap(int64(d))) == d {
		return
	}
	rt.mu.Lock()
	all := make([]*server, 0, len(rt.servers))
	for _, s := range rt.servers {
		all = append(all, s)
	}
	rt.mu.Unlock()
	for _, s := range all {
		s.rearm()
	}
}

func (rt *Runtime) logf(level, format string, args ...any) {
	rt.cfg.Log(level, "omni: "+fmt.Sprintf(format, args...))
}

// errClosed: the runtime was closed (VoxtNew is quitting).
var errClosed = errors.New("omni: the runtime is closed")

// acquire leases spec's server for dir, starting it if it is not running
// (and waiting for it, at most until ctx ends). Release the lease when
// the use ends. Taking a speech model retires the other speech models'
// servers (they stop once their uses end).
func (rt *Runtime) acquire(ctx context.Context, spec Spec, dir string) (*lease, error) {
	key := spec.ID + "\x00" + dir
	rt.mu.Lock()
	if rt.closed {
		rt.mu.Unlock()
		return nil, &asr.Error{Kind: asr.KindServer, Provider: Provider, Detail: "the speech runtime has stopped", Err: errClosed}
	}
	s := rt.servers[key]
	if s == nil {
		s = newServer(rt, spec, dir)
		rt.servers[key] = s
	}
	var retire []*server
	if spec.ASR != nil {
		for k, o := range rt.servers {
			if o != s && o.spec.ASR != nil {
				delete(rt.servers, k)
				retire = append(retire, o)
			}
		}
		kept := rt.draining[:0]
		for _, o := range rt.draining {
			select {
			case <-o.stopped:
			default:
				kept = append(kept, o)
			}
		}
		rt.draining = append(kept, retire...)
	}
	rt.mu.Unlock()
	for _, o := range retire {
		o.retire("another speech model is in use")
	}
	l, err := s.acquire(ctx)
	if errors.Is(err, errRetired) && ctx.Err() == nil {
		// Another switch retired s meanwhile: once more, with the map's.
		return rt.acquire(ctx, spec, dir)
	}
	return l, err
}

// Close stops every server and waits for them (VoxtNew's quit). Later
// uses fail.
func (rt *Runtime) Close() {
	rt.mu.Lock()
	rt.closed = true
	rt.draining = nil
	all := make([]*server, 0, len(rt.servers))
	for k, s := range rt.servers {
		all = append(all, s)
		delete(rt.servers, k)
	}
	rt.mu.Unlock()
	var wg sync.WaitGroup
	for _, s := range all {
		wg.Go(s.shutdown)
	}
	wg.Wait()
}

// Unload stops model id's servers (before its files are deleted): at
// once when unused, else when their uses end. The next use starts it
// again.
func (rt *Runtime) Unload(id string) {
	rt.mu.Lock()
	var out []*server
	for k, s := range rt.servers {
		if s.spec.ID == id {
			delete(rt.servers, k)
			out = append(out, s)
		}
	}
	rt.mu.Unlock()
	for _, s := range out {
		s.retire("unloaded")
	}
}

// Running lists the ids of the models whose servers run now (for the
// Local models page and Diagnostics: memory only).
func (rt *Runtime) Running() []string {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	var out []string
	for _, s := range rt.servers {
		if s.running() {
			out = append(out, s.spec.ID)
		}
	}
	return out
}

// Process is a running server: its model and process id.
type Process struct {
	ID  string
	PID int
}

// Processes lists the servers that run now with their process ids (the
// Local models page measures their memory: memory only).
func (rt *Runtime) Processes() []Process {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	var out []Process
	for _, s := range rt.servers {
		if pid := s.pid(); pid > 0 {
			out = append(out, Process{ID: s.spec.ID, PID: pid})
		}
	}
	return out
}

// forget drops s from the map when it is still there (it stopped).
func (rt *Runtime) forget(s *server) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	for k, o := range rt.servers {
		if o == s {
			delete(rt.servers, k)
		}
	}
}

// Warm starts spec's server for dir in the background, so that a first
// use does not wait for the model's load (a session starting). It is
// stopped again when idle.
func (rt *Runtime) Warm(spec Spec, dir string) {
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), rt.cfg.StartTimeout+time.Second)
		defer cancel()
		l, err := rt.acquire(ctx, spec, dir)
		if err != nil {
			rt.logf("warning", "warming %s: %v", spec.ID, err)
			return
		}
		l.release()
	}()
}
