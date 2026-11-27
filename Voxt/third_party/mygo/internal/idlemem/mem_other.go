//go:build !darwin && !linux && !windows

package main

import (
	"errors"
	"os"
)

type state struct{}

func before() state { return state{} }

func usage(int, state) (app, total uint64, procs []int, err error) {
	return 0, 0, nil, errors.New("not measured on this platform")
}

func stop(p *os.Process, _ []int) { p.Kill() }
