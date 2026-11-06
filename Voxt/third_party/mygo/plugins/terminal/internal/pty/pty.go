// Package pty runs processes in pseudo-terminals, in pure Go: /dev/ptmx on
// macOS and Linux, ConPTY on Windows.
package pty

import (
	"errors"
	"sync"
)

// Options describe a process to start in a pseudo-terminal.
type Options struct {
	// Path is the program; Args its arguments, with Args[0] what the
	// program sees as its name ("-zsh" starts a login shell).
	Path string
	Args []string
	// Dir is the working directory, Env the environment ("KEY=value").
	Dir string
	Env []string
	// Cols and Rows are the size of the terminal in cells, Width and
	// Height in pixels.
	Cols, Rows, Width, Height int
}

// ErrClosed is returned by Write after Close.
var ErrClosed = errors.New("pty: closed")

// PTY is a process running in a pseudo-terminal: Read returns its output,
// Write types into it.
type PTY struct {
	sys
	waitOnce sync.Once
	code     int
	waitErr  error
	exited   chan struct{}
}

// Start starts a process in a pseudo-terminal.
func Start(o Options) (*PTY, error) {
	if o.Cols <= 0 || o.Rows <= 0 {
		o.Cols, o.Rows = 80, 24
	}
	if len(o.Args) == 0 {
		o.Args = []string{o.Path}
	}
	p := &PTY{exited: make(chan struct{})}
	if err := p.start(o); err != nil {
		return nil, err
	}
	go func() {
		p.code, p.waitErr = p.wait()
		close(p.exited)
	}()
	return p, nil
}

// Wait waits for the process to exit and returns its exit code.
func (p *PTY) Wait() (int, error) {
	<-p.exited
	return p.code, p.waitErr
}

// Exited is closed when the process exited.
func (p *PTY) Exited() <-chan struct{} { return p.exited }
