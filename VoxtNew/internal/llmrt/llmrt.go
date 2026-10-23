// Package llmrt runs VoxtNew's language models: it supervises
// voxtnew-llm, the Swift process that runs one MLX or GGUF model at a
// time (llm/, docs/llm-protocol.md), and puts an llm.Backend over it.
//
//   - Runtime is the process: started on first use, handshaken
//     (llm.hello), its calls routed by id, its stderr written to the log,
//     started again on the next call after a crash, stopped after
//     Config.IdleTimeout unused (the one sure way to give a model's memory
//     back) and on quit (Stop). Config.OnModel hears the model it holds
//     loaded, busy, idle and unloaded (resident.go: the family's runtime
//     file); UnloadIdle lets an idle model go when another app asks.
//   - Open gives a model as a session uses it (local.Chat): an llm.Client
//     (retries, JSON repair, thinking stripped) over a backend that builds
//     each chat from the model's Profile (models.go: sampling, token
//     budgets, template variables, a translation model's own prompt) and
//     streams its answer on the caller's goroutine.
//
// Policy is here, not in the process: which model, its parameters, when
// to unload. The catalogue, the downloads and local.Services are M1b's;
// dev.go's VOXTNEW_DEV_LLM is a hook for trying a model without them.
//
// Imports: core, lang, llm, llmrt/wire, local, macx, resident.
package llmrt

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/llmrt/wire"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/macx"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/resident"
)

// BinName is the process's executable, in the app's resources under llm/
// beside mlx.metallib and llama.framework (scripts/build-llm.sh).
const BinName = "voxtnew-llm"

// LogSource is the source of the process's lines in the logs.
const LogSource = "llm"

// Find looks for voxtnew-llm: $VOXTNEW_LLM, then the app's resources
// (packaged, mygo dev, or go run from the project). It returns an
// absolute path, or "" when there is none.
func Find(resources string) string {
	if p := os.Getenv("VOXTNEW_LLM"); p != "" {
		if abs, err := filepath.Abs(p); err == nil {
			return abs
		}
		return p
	}
	for _, p := range []string{
		filepath.Join(resources, "llm", BinName),
		filepath.Join(resources, runtime.GOOS+"-"+runtime.GOARCH, "llm", BinName),
	} {
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			if abs, err := filepath.Abs(p); err == nil {
				return abs
			}
			return p
		}
	}
	return ""
}

// DefaultIdleTimeout is how long the process stays without a call.
const DefaultIdleTimeout = 5 * time.Minute

// Config configures a Runtime.
type Config struct {
	Bin        string // the executable (Find); "" fails every call
	AppVersion string // sent in llm.hello
	LogDir     string // stderr goes to LogDir/llm.log; "" = none
	// Log receives the process's log lines and stderr, and a line per
	// chat (Source "llm"); nil drops them.
	Log func(core.LogEntry)
	Env []string // added to the process's environment (tests)
	// HelloTimeout bounds the handshake (default 10 s).
	HelloTimeout time.Duration
	// IdleTimeout stops the process after so long without a call
	// (DefaultIdleTimeout; <0 never; SetIdleTimeout changes it).
	IdleTimeout time.Duration
	// CancelGrace is how long a cancelled chat may take to end before the
	// process is killed (default 3 s).
	CancelGrace time.Duration
	// CacheLimitMB bounds MLX's buffer cache, sent with each load (0:
	// MLX's default).
	CacheLimitMB int
	// OnModel is told when the process's model is loaded, busy, idle and
	// unloaded (resident.go), in order, on a goroutine of the runtime's,
	// under none of its locks. nil: nobody.
	OnModel func(resident.Event)
	// OnPace is told how each chat went, answered or stopped by a
	// deadline, with the Mac's memory at its end (zero when unread), on
	// the chat's goroutine: the overlay's memory notice (ShortOfMemory).
	// It must not block. nil: nobody.
	OnPace func(ChatPace, macx.SystemMemory)
	// BeforeLoad runs before a call that makes the process load a model
	// (the family's memory budget makes room); it may block a little.
	BeforeLoad func(ctx context.Context, id string)
}

func (c *Config) defaults() {
	if c.HelloTimeout == 0 {
		c.HelloTimeout = 10 * time.Second
	}
	if c.IdleTimeout == 0 {
		c.IdleTimeout = DefaultIdleTimeout
	}
	if c.CancelGrace == 0 {
		c.CancelGrace = 3 * time.Second
	}
}

// State is where the process is in its life.
type State string

const (
	Stopped  State = "stopped"  // not started, idle, or stopped on purpose
	Starting State = "starting" // spawned, llm.hello not answered yet
	Ready    State = "ready"    // handshake done
	Failed   State = "failed"   // exited or refused; Err says why; the next call starts it again
)

// Status is a snapshot of the runtime (memory only).
type Status struct {
	State State
	PID   int
	Hello *wire.HelloResult // nil until ready
	// Model is the model loaded as far as the runtime knows (the last
	// load or chat), nil when none.
	Model *wire.Model
	// Memory is the memory the process last reported.
	Memory wire.Memory
	Err    string    // why it failed, in English
	Starts int       // processes started
	Since  time.Time // when State last changed
}

// Errors wrapped by a call's error.
var (
	// ErrExited: the process died during the call.
	ErrExited = errors.New("llmrt: voxtnew-llm exited")
	// ErrUnavailable: the process could not be started or refused the
	// handshake (Status.Err says why).
	ErrUnavailable = errors.New("llmrt: voxtnew-llm is unavailable")
	// ErrStopped: the runtime was stopped (the app quits).
	ErrStopped = errors.New("llmrt: stopped")
)
