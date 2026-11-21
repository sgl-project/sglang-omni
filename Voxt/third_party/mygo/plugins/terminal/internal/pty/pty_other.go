//go:build !darwin && !linux && !windows

package pty

import "errors"

type sys struct{}

func (p *PTY) start(Options) error               { return errors.ErrUnsupported }
func (p *PTY) wait() (int, error)                { return 0, errors.ErrUnsupported }
func (p *PTY) Read([]byte) (int, error)          { return 0, errors.ErrUnsupported }
func (p *PTY) Write([]byte) (int, error)         { return 0, errors.ErrUnsupported }
func (p *PTY) Resize(cols, rows, w, h int) error { return errors.ErrUnsupported }
func (p *PTY) Pid() int                          { return 0 }
func (p *PTY) Close() error                      { return nil }
func (p *PTY) Kill() error                       { return errors.ErrUnsupported }
