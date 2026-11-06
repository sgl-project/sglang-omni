package omni

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// endpoint is a ready server's address and name.
type endpoint struct {
	host  string
	port  int
	model string // the model name requests carry
	pid   int
}

func (e endpoint) base() string { return fmt.Sprintf("http://%s:%d", e.host, e.port) }

// process is one server process.
type process struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	ep     endpoint
	exited chan struct{} // closed when it exited (exitErr set)
	// exitErr is the process's end (cmd.Wait), set before exited closes.
	exitErr error
	tail    *tail
	halting atomic.Bool // stop or kill asked for it: its exit is expected
	once    sync.Once
}

// readyEvent is the server's first line on stdout.
type readyEvent struct {
	Event  string `json:"event"`
	Host   string `json:"host"`
	Port   int    `json:"port"`
	Model  string `json:"model_name"`
	PID    int    `json:"server_pid"`
	Reason string `json:"reason"`
}

// spawn starts the server for spec's model in dir; the channel gets its
// first event (or closes when stdout ends without one).
// cacheLimit: pass spec's --cache-limit-mb (the binary knows it).
func spawn(cfg Config, spec Spec, dir string, cacheLimit bool) (*process, <-chan readyEvent, error) {
	if fi, err := os.Stat(dir); err != nil || !fi.IsDir() {
		return nil, nil, fmt.Errorf("no model directory %s", dir)
	}
	args := []string{"--supervised", "--host", "127.0.0.1", "--port", "0",
		"--model-directory", dir, "--startup-timeout-s", ftoa(cfg.StartTimeout.Seconds())}
	cmd := exec.Command(cfg.Binary, append(args, spec.launchArgs(cacheLimit)...)...)
	cmd.Env = append(serverEnv(os.Environ()), cfg.Env...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return nil, nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, nil, err
	}
	p := &process{cmd: cmd, stdin: stdin, exited: make(chan struct{}), tail: &tail{limit: 2048}}
	ready := make(chan readyEvent, 1)
	var pipes sync.WaitGroup
	pipes.Go(func() {
		sc := bufio.NewScanner(stdout)
		sc.Buffer(make([]byte, 64<<10), 1<<20)
		first := true
		for sc.Scan() {
			var ev readyEvent
			if json.Unmarshal(sc.Bytes(), &ev) != nil || ev.Event == "" {
				continue
			}
			if first {
				first = false
				ready <- ev
			}
		}
		close(ready)
	})
	pipes.Go(func() {
		sc := bufio.NewScanner(stderr)
		for sc.Scan() {
			line := strings.TrimSpace(sc.Text())
			if line == "" {
				continue
			}
			p.tail.add(line)
			cfg.Log("warning", "omni "+spec.ID+": "+line)
		}
	})
	go func() {
		pipes.Wait() // Wait closes the pipes: read them to the end first
		p.exitErr = cmd.Wait()
		close(p.exited)
	}()
	return p, ready, nil
}

// waitReady waits for the ready event; a server that sends none within
// timeout is killed (a stalled load).
func (p *process) waitReady(ready <-chan readyEvent, timeout time.Duration) error {
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case ev, ok := <-ready:
		switch {
		case !ok:
			<-p.exited
			return fmt.Errorf("it exited before it was ready (%v): %s", p.exitErr, p.tail.String())
		case ev.Event == "ready" && ev.Port > 0:
			p.ep = endpoint{host: ev.Host, port: ev.Port, model: ev.Model, pid: ev.PID}
			if p.ep.host == "" {
				p.ep.host = "127.0.0.1"
			}
			return nil
		case ev.Event == "failed":
			p.kill()
			return errors.New(ev.Reason)
		default:
			p.kill()
			return fmt.Errorf("unexpected first event %q", ev.Event)
		}
	case <-timer.C:
		p.kill()
		return fmt.Errorf("not ready within %v", timeout)
	}
}

// pidOf is its process id.
func (p *process) pidOf() int {
	if p.cmd.Process == nil {
		return 0
	}
	return p.cmd.Process.Pid
}

func (p *process) alive() bool {
	select {
	case <-p.exited:
		return false
	default:
		return true
	}
}

// stopping reports whether its end was asked for.
func (p *process) stopping() bool { return p.halting.Load() }

// stop asks for an orderly shutdown ({"command":"shutdown"} and the end
// of stdin), then escalates to SIGTERM and SIGKILL; it returns once the
// process exited.
func (p *process) stop() {
	p.halting.Store(true)
	p.once.Do(func() {
		_, _ = io.WriteString(p.stdin, "{\"command\": \"shutdown\"}\n")
		_ = p.stdin.Close()
	})
	if p.wait(shutdownGrace) {
		return
	}
	_ = p.cmd.Process.Signal(syscall.SIGTERM)
	if p.wait(terminateGrace) {
		return
	}
	p.kill()
	<-p.exited
}

// kill ends it at once.
func (p *process) kill() {
	p.halting.Store(true)
	_ = p.cmd.Process.Kill()
}

func (p *process) wait(d time.Duration) bool {
	select {
	case <-p.exited:
		return true
	case <-time.After(d):
		return false
	}
}

// serverEnv is VoxtNew's environment without the dynamic loader's
// settings a debugger or test runner injects (Voxt's runtimeEnvironment).
func serverEnv(env []string) []string {
	out := make([]string, 0, len(env))
	for _, kv := range env {
		if strings.HasPrefix(kv, "DYLD_") || strings.HasPrefix(kv, "__XPC_DYLD_") {
			continue
		}
		out = append(out, kv)
	}
	return out
}

// knowsCacheLimit runs cfg.Binary with an invalid --cache-limit-mb: a
// server that knows the flag refuses its value ("--cache-limit-mb must
// …"), an older one the flag itself ("unknown argument"), both before
// loading anything. Anything else (no answer in 10 s, a binary that does
// not run) counts as not knowing it: the launch says what is wrong.
func knowsCacheLimit(cfg Config) bool {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, cfg.Binary, "--model-kind", kindQwen3ASR,
		"--model-directory", os.DevNull, "--cache-limit-mb", "-2")
	cmd.Env = append(serverEnv(os.Environ()), cfg.Env...)
	out, _ := cmd.CombinedOutput()
	return strings.Contains(string(out), "--cache-limit-mb must")
}

// tail keeps the last lines a process wrote on stderr, to say why it
// failed.
type tail struct {
	mu    sync.Mutex
	limit int
	lines []string
	size  int
}

func (t *tail) add(line string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.lines = append(t.lines, line)
	t.size += len(line) + 1
	for t.size > t.limit && len(t.lines) > 1 {
		t.size -= len(t.lines[0]) + 1
		t.lines = t.lines[1:]
	}
}

func (t *tail) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	if len(t.lines) == 0 {
		return "(no output)"
	}
	return strings.Join(t.lines, " | ")
}
