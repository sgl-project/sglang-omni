// Package omnitest is a fake qwen3_asr_server for tests: a test binary
// runs it instead of its tests when OMNI_FAKE is set, so internal/omni's
// Runtime spawns a real process and talks to it over real pipes, HTTP and
// WebSockets, without a Mac or a model. In the test package:
//
//	func TestMain(m *testing.M) {
//		if omnitest.Enabled() {
//			omnitest.Main()
//		}
//		os.Exit(m.Run())
//	}
//
// and point omni.Config's Binary at os.Executable() with Env:
// omnitest.Env(opts). It speaks the launch protocol (the ready line, the
// shutdown command, the end of stdin) and serves /health,
// /v1/audio/transcriptions (SSE, with heartbeats), /v1/realtime (a
// partial per second of audio, the final segment on commit),
// /v1/vad/speech_timestamps and /v1/vad/stream (speech is audio louder
// than -34 dBFS). Every launch and request is appended to Opts.Log as a
// JSON line (Record), for the tests to check what was sent.
package omnitest

import (
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// EnvFake holds a fake's options (Env).
const EnvFake = "OMNI_FAKE"

// Modes of a fake.
const (
	ModeNormal = ""
	// ModeFail reports {"event":"failed"} instead of ready.
	ModeFail = "fail"
	// ModeHang never reports, never stops on stdin (only a kill ends it).
	ModeHang = "hang"
	// ModeStall answers a Final's headers, then nothing (no heartbeat).
	ModeStall = "stall"
	// ModeCrashFinal exits during a Final.
	ModeCrashFinal = "crash-final"
	// ModeError answers a Final with an error event.
	ModeError = "error"
	// ModeVADError fails the voice activity endpoints.
	ModeVADError = "vad-error"
)

// Opts are a fake's options.
type Opts struct {
	Mode string
	// Log is a file the fake appends its records to (JSON lines).
	Log string
	// Text is a Final's text (default "fake transcript"); "{n}" in it is
	// the request's number (1 for the server's first Final).
	Text string
	// FailFrom: the Finals from this one on (1 = the first) answer an
	// error event, as ModeError does; 0: none.
	FailFrom int
	// Delay is how long a Final takes (default 50 ms).
	Delay time.Duration
	// Load is how long the model "loads" before ready.
	Load time.Duration
	// Echo: a Final with a prompt ends with the prompt's lines, as
	// Qwen3-ASR echoes its context on silence ("text。A，B。").
	Echo bool
	// Older is a server from before --cache-limit-mb: it refuses the flag
	// as an unknown argument (a newer one only refuses a value below -1).
	Older bool
}

// Env is the environment that makes a test binary a fake with o.
func Env(o Opts) []string {
	v := url.Values{}
	v.Set("mode", o.Mode)
	v.Set("log", o.Log)
	v.Set("text", o.Text)
	v.Set("delay", o.Delay.String())
	v.Set("load", o.Load.String())
	v.Set("failfrom", strconv.Itoa(o.FailFrom))
	v.Set("older", strconv.FormatBool(o.Older))
	v.Set("echo", strconv.FormatBool(o.Echo))
	return []string{EnvFake + "=" + v.Encode()}
}

// Enabled reports whether this process is to be a fake.
func Enabled() bool { return os.Getenv(EnvFake) != "" }

func opts() Opts {
	v, _ := url.ParseQuery(os.Getenv(EnvFake))
	o := Opts{Mode: v.Get("mode"), Log: v.Get("log"), Text: v.Get("text")}
	o.Delay, _ = time.ParseDuration(v.Get("delay"))
	o.Load, _ = time.ParseDuration(v.Get("load"))
	o.FailFrom, _ = strconv.Atoi(v.Get("failfrom"))
	o.Older, _ = strconv.ParseBool(v.Get("older"))
	o.Echo, _ = strconv.ParseBool(v.Get("echo"))
	if o.Text == "" {
		o.Text = "fake transcript"
	}
	if o.Delay <= 0 {
		o.Delay = 50 * time.Millisecond
	}
	return o
}

// Record is one line of the log.
type Record struct {
	Event  string            `json:"event"` // launch, final, live, vad-timestamps, vad-stream, shutdown
	Args   []string          `json:"args,omitempty"`
	Fields map[string]string `json:"fields,omitempty"`
	// Samples is the audio a request carried (a live session: all it got).
	Samples int `json:"samples,omitempty"`
	PID     int `json:"pid"`
}

var logMu sync.Mutex

func record(o Opts, r Record) {
	if o.Log == "" {
		return
	}
	r.PID = os.Getpid()
	b, _ := json.Marshal(r)
	logMu.Lock()
	defer logMu.Unlock()
	f, err := os.OpenFile(o.Log, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return
	}
	defer f.Close()
	fmt.Fprintf(f, "%s\n", b)
}

// Records reads a fake's log.
func Records(path string) []Record {
	logMu.Lock()
	defer logMu.Unlock()
	raw, _ := os.ReadFile(path)
	var out []Record
	for _, line := range strings.Split(string(raw), "\n") {
		var r Record
		if json.Unmarshal([]byte(line), &r) == nil && r.Event != "" {
			out = append(out, r)
		}
	}
	return out
}

// Of are the records of one event.
func Of(rs []Record, event string) []Record {
	var out []Record
	for _, r := range rs {
		if r.Event == event {
			out = append(out, r)
		}
	}
	return out
}

// arg is the value of flag in args.
func arg(args []string, flag string) string {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag {
			return args[i+1]
		}
	}
	return ""
}
