// Package helper runs voxtnew-helper, the Swift process that does what
// MyGo cannot: record the microphone, watch the keyboard with an event
// tap, read and write text in the app in front, and read browser URLs
// through Accessibility. A Supervisor starts it with the app, handshakes
// (helper.hello), routes responses and notifications by id, writes its
// stderr to the log, and restarts it when it crashes. It speaks the
// protocol in package proto (docs/helper-protocol.md).
//
// It is AudioSmith's engine supervisor (internal/engine) reduced to one
// process: no roles, no job scheduler, no memory budget. Unlike the
// engine, the helper sends notifications that belong to no request
// (hotkey.event): Config.OnNotify receives those.
//
// Imports: core, helper/proto.
package helper

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/core"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
)

// BinName is the helper's executable, in the app's resources under
// helper/ (mygo.json macos.helperEntitlements).
const BinName = "voxtnew-helper"

// Find looks for voxtnew-helper: $VOXTNEW_HELPER, then the app's
// resources (packaged, mygo dev, or go run from the project). It returns
// an absolute path, or "" when there is none.
func Find(resources string) string {
	if p := os.Getenv("VOXTNEW_HELPER"); p != "" {
		if abs, err := filepath.Abs(p); err == nil {
			return abs
		}
		return p
	}
	for _, p := range []string{
		filepath.Join(resources, "helper", BinName),
		filepath.Join(resources, runtime.GOOS+"-"+runtime.GOARCH, "helper", BinName),
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

// State is where the helper process is in its life.
type State string

const (
	Stopped  State = "stopped"  // not started, or stopped on purpose
	Starting State = "starting" // spawned, helper.hello not answered yet
	Ready    State = "ready"    // handshake done, accepting requests
	Failed   State = "failed"   // exited or refused; Err and StderrTail say why
)

// FailReason classifies a failure, so the UI can explain it in the
// user's language and offer the right fix.
type FailReason string

const (
	FailMissing   FailReason = "missing"   // no executable: build it, or set VOXTNEW_HELPER
	FailStart     FailReason = "start"     // the executable could not be started
	FailProtocol  FailReason = "protocol"  // it speaks another protocol version: rebuild it; not restarted
	FailHandshake FailReason = "handshake" // helper.hello was not answered in time
	FailCrashed   FailReason = "crashed"   // it exited without being asked to
	FailStuck     FailReason = "stuck"     // killed because a cancelled job did not stop within the grace period
	// FailUpdated: an update replaced the app on disk and the app has not
	// restarted (Config.Stale); the helper there is the new version's, so
	// it is not started (or, speaking another protocol, not restarted)
	// until the app restarts.
	FailUpdated FailReason = "updated"
)

// Status is a snapshot of the helper, for the status bar, the permissions
// page and Diagnostics.
type Status struct {
	State      State
	PID        int
	Hello      *proto.HelloResult // nil until ready
	Running    []string           // ids of jobs in flight
	Err        string             // why it failed, in English
	Reason     FailReason         // why it failed; "" unless failed
	StderrTail string             // last stderr lines after a crash
	Restarts   int                // automatic restarts since Start
	RestartAt  time.Time          // when the next automatic restart happens; zero = none planned
	Since      time.Time          // when State last changed
}

// Has reports whether the ready helper implements method.
func (s Status) Has(method string) bool {
	if s.State != Ready || s.Hello == nil {
		return false
	}
	for _, m := range s.Hello.Methods {
		if m == method {
			return true
		}
	}
	return false
}

// StatusChanged is published on the core.Bus whenever the helper's
// State, PID or Err change. Topic "helper.status".
type StatusChanged struct{ Status Status }

func (StatusChanged) Topic() string { return "helper.status" }

// Config configures a Supervisor.
type Config struct {
	Bin        string    // the executable (Find); "" fails with FailMissing
	AppVersion string    // sent in helper.hello
	LogDir     string    // stderr goes to LogDir/helper.log (rotated); "" = none
	Bus        *core.Bus // StatusChanged (and log lines when Log is nil)
	// Log receives the helper's log notifications and stderr lines, with
	// Source "helper" (the app's logs.Ring.Append). nil publishes them on
	// Bus as core.LogAppended.
	Log func(core.LogEntry)
	// OnNotify receives the notifications that name no request in flight
	// (hotkey.event, and anything a newer helper adds), on the reader
	// goroutine, in order; it must not block.
	OnNotify func(proto.Message)
	// HelloTimeout bounds the handshake (default 10 s).
	HelloTimeout time.Duration
	// MaxRestarts bounds automatic restarts within RestartWindow (default
	// 3 in 10 min); beyond it the helper stays failed until Restart.
	MaxRestarts   int
	RestartWindow time.Duration
	// CancelGrace is how long a cancelled job may take to end before the
	// helper is killed and restarted (default 3 s).
	CancelGrace time.Duration
	// OnTrace receives the Trace of every finished Call, on the caller's
	// goroutine (tests; diagnostics). Control requests slower than
	// SlowCall (default 500 ms) are logged with their phases.
	OnTrace  func(Trace)
	SlowCall time.Duration
	// Env is added to the helper's environment (tests select the fake
	// helper with it).
	Env  []string
	Logf func(format string, args ...any) // supervisor diagnostics; default log.Printf
	// Stale reports that the app on disk is no longer the one running: an
	// update replaced the bundle, Bin included, and the app has not
	// restarted. The supervisor then starts no helper (FailUpdated, no
	// restart): the new version's helper belongs to the new app. It is
	// asked before each start and when a helper speaks another protocol;
	// it may read the disk. nil: never.
	Stale func() bool
}

// Supervisor runs and talks to the helper. Goroutine-safe.
type Supervisor interface {
	// Start spawns the helper and returns; the handshake completes in the
	// background and is reported as StatusChanged.
	Start(ctx context.Context) error

	// Call sends a request and waits for its response, returning the raw
	// result or an error:
	//   - *proto.Error for an error response (use errors.Is with a code);
	//   - an error wrapping core.ErrHelper when the helper is not ready, or
	//     exits before answering.
	// id is the request id ("" numbers it); a job (capture.start) passes
	// the session's id so logs correlate. notify receives the job's
	// notifications (capture.level, capture.pcm, log lines naming it) on
	// the reader goroutine, in order, before Call returns; it must not
	// block. Cancelling ctx sends job.cancel for a job, and Call returns
	// once the helper answers (or after CancelGrace, killing it).
	Call(ctx context.Context, id, method string, params any, notify func(proto.Message)) (json.RawMessage, error)

	// Status returns a snapshot.
	Status() Status
	// Restart stops the helper (failing its calls) and starts a fresh
	// one; it resets the restart budget.
	Restart() error
	// Stop asks the helper to shut down (helper.shutdown), then kills it
	// after a grace period. Calls in flight fail with core.ErrHelper.
	Stop()
}

// Trace times one Call through its phases, to tell where a slow answer
// waited: for a ready helper, to write the request, for the helper's
// response line (the helper's own work, and its stdout), or for the
// caller to wake. Auralink's spikes saw 1–2 s round trips where the helper
// worked for tens of milliseconds; the helper logs its own side of a slow
// request (queued, ran) on stderr. Durations are since Start; a phase not
// reached is 0.
type Trace struct {
	ID, Method string
	Start      time.Time     // Call was entered
	Ready      time.Duration // a ready helper was found (waits while it starts or restarts)
	Written    time.Duration // the request was written to its stdin
	Answered   time.Duration // the reader read the response line from its stdout
	Returned   time.Duration // Call returned
	Err        string
}

// String is the trace for the log: "browser.url c12 1.32s (ready 0s,
// write 0.1ms, helper 1.31s, wake 0.2ms)".
func (t Trace) String() string {
	ms := func(d time.Duration) string { return d.Round(10 * time.Microsecond).String() }
	phase := func(from, to time.Duration) string {
		if to == 0 {
			return "-"
		}
		return ms(to - from)
	}
	s := fmt.Sprintf("%s %s %s (ready %s, write %s, helper %s, wake %s)", t.Method, t.ID, ms(t.Returned),
		ms(t.Ready), phase(t.Ready, t.Written), phase(t.Written, t.Answered), phase(t.Answered, t.Returned))
	if t.Err != "" {
		s += ": " + t.Err
	}
	return s
}

// answered records when the reader read c's response. Written is noted
// by the caller's goroutine after its write returns, and the reader's
// goroutine can read the response before that (the caller was not
// scheduled in between): the write had ended by then, so Answered is
// never before Written and the helper's phase never negative.
func (t *Trace) answered(c *call) {
	if !c.answered.IsZero() {
		t.Answered = max(c.answered.Sub(t.Start), t.Written)
	}
}

// ErrExited is wrapped by Call errors when the helper died mid-call.
var ErrExited = fmt.Errorf("the helper exited: %w", core.ErrHelper)
