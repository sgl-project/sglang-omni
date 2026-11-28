// Package speakerrt supervises the local Core ML speaker process. It never
// shares an inference queue with transcription or the capture helper.
package speakerrt

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"sync"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/speaker"
)

// Model is a verified model directory and the packaged inference executable.
type Model struct {
	Binary, Dir                  string
	StartTimeout, RequestTimeout time.Duration
	OnReady                      func(int)
	OnClose                      func()
	// Memory measures actual process memory, including mapped model pages.
	Memory    func(int) (int64, error)
	MaxMemory int64
}

type client struct {
	cmd       *exec.Cmd
	in        io.WriteCloser
	reader    *bufio.Scanner
	mu        sync.Mutex
	stop      sync.Once
	done      chan struct{}
	timeout   time.Duration
	onClose   func()
	memory    func(int) (int64, error)
	maxMemory int64
}

type response struct {
	Ready  bool         `json:"ready"`
	Error  string       `json:"error"`
	Start  int          `json:"start_frame"`
	Frames [][8]float32 `json:"frames"`
}

func (m Model) Open(ctx context.Context) (speaker.Stream, error) {
	cmd := exec.CommandContext(ctx, m.Binary, "--model", m.Dir)
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		_ = in.Close() // The process did not start; only the unused pipe remains.
		return nil, err
	}
	if err = cmd.Start(); err != nil {
		_ = in.Close() // The process did not start; only the unused pipe remains.
		return nil, err
	}
	c := &client{cmd: cmd, in: in, reader: bufio.NewScanner(out), done: make(chan struct{}), timeout: m.RequestTimeout}
	if c.timeout <= 0 {
		c.timeout = 2 * time.Second
	}
	c.reader.Buffer(make([]byte, 4096), 128*1024)
	go func() {
		_ = cmd.Wait() // Exit is reported by the bounded protocol read; kill is normal teardown.
		close(c.done)
	}()
	start := m.StartTimeout
	if start <= 0 {
		start = 60 * time.Second
	}
	readyCtx, cancel := context.WithTimeout(ctx, start)
	defer cancel()
	r, err := c.exchange(readyCtx, nil)
	if err == nil && !r.Ready {
		err = errors.New("speaker: missing ready handshake")
	}
	if err != nil {
		c.Close()
		return nil, err
	}
	c.onClose = m.OnClose
	c.memory, c.maxMemory = m.Memory, m.MaxMemory
	if err := c.checkMemory(); err != nil {
		c.Close()
		return nil, err
	}
	if m.OnReady != nil {
		m.OnReady(cmd.Process.Pid)
	}
	return c, nil
}

func (c *client) exchange(ctx context.Context, req any) (response, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	done := make(chan struct{})
	var r response
	var err error
	go func() {
		defer close(done)
		if req != nil {
			if err = json.NewEncoder(c.in).Encode(req); err != nil {
				return
			}
		}
		if !c.reader.Scan() {
			err = c.reader.Err()
			if err == nil {
				err = io.ErrUnexpectedEOF
			}
			return
		}
		err = json.Unmarshal(c.reader.Bytes(), &r)
		if err == nil && r.Error != "" {
			err = fmt.Errorf("speaker: %s", r.Error)
		}
	}()
	select {
	case <-done:
		return r, err
	case <-ctx.Done():
		c.Close()
		<-done // closing the pipes and owned process unblocks the I/O
		return response{}, ctx.Err()
	}
}

func (c *client) request(ctx context.Context, op string, pcm []byte) ([]speaker.Frame, error) {
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	r, err := c.exchange(ctx, struct {
		Op    string `json:"op"`
		Audio []byte `json:"audio,omitempty"`
	}{op, pcm})
	if err != nil {
		return nil, err
	}
	if err := c.checkMemory(); err != nil {
		c.Close()
		return nil, err
	}
	frames := make([]speaker.Frame, len(r.Frames))
	for i, scores := range r.Frames {
		at := time.Duration(r.Start+i) * 10 * time.Millisecond
		frames[i] = speaker.Frame{Start: at, End: at + 10*time.Millisecond, Scores: scores}
	}
	return frames, nil
}

func (c *client) checkMemory() error {
	if c.memory == nil || c.maxMemory <= 0 {
		return nil
	}
	if used, err := c.memory(c.cmd.Process.Pid); err == nil && used > c.maxMemory {
		return fmt.Errorf("speaker: process memory %d exceeds budget %d", used, c.maxMemory)
	}
	return nil
}

func (c *client) Push(ctx context.Context, pcm []byte) ([]speaker.Frame, error) {
	return c.request(ctx, "audio", pcm)
}
func (c *client) Close() error {
	c.stop.Do(func() {
		// Closing both ends also interrupts a peer that never reads stdin.
		_ = c.in.Close() // Closing stdin interrupts a peer blocked on input.
		if c.cmd.Process != nil {
			_ = c.cmd.Process.Kill() // It may already have exited.
		}
		<-c.done
		if c.onClose != nil {
			c.onClose()
		}
	})
	return nil // forced termination is the expected stream teardown
}
