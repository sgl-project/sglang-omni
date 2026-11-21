//go:build !windows

package omni

import (
	"os"
	"syscall"
)

// terminateSignal is the gentle shutdown signal, before the hard kill.
var terminateSignal os.Signal = syscall.SIGTERM
