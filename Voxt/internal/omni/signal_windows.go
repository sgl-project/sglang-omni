//go:build windows

package omni

import "os"

// terminateSignal is the gentle shutdown signal, before the hard kill.
// Windows has no SIGTERM: os.Kill is the only signal it supports, so the
// orderly shutdown falls straight through to the hard kill.
var terminateSignal os.Signal = os.Kill
