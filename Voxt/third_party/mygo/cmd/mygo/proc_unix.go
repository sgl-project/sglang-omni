//go:build !windows

package main

import (
	"os/exec"
	"os/signal"
	"syscall"
)

// shell runs a command line with the shell.
func shell(line string) *exec.Cmd { return exec.Command("sh", "-c", line) }

// setProcessGroup starts the command in its own process group so the whole
// tree (e.g. a dev server and its workers) can be stopped.
func setProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// terminate asks a command started with setProcessGroup, and the processes
// it started, to exit.
func terminate(cmd *exec.Cmd) { signalGroup(cmd, syscall.SIGTERM) }

// interrupt asks the command alone to exit, as quitting it would: an app
// then ends the processes it started itself.
func interrupt(cmd *exec.Cmd) {
	if cmd != nil && cmd.Process != nil {
		_ = cmd.Process.Signal(syscall.SIGTERM)
	}
}

// kill stops them right away.
func kill(cmd *exec.Cmd) { signalGroup(cmd, syscall.SIGKILL) }

func signalGroup(cmd *exec.Cmd, sig syscall.Signal) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	_ = syscall.Kill(-cmd.Process.Pid, sig)
}

// ignoreTTIN lets reading the terminal fail, rather than stop mygo, when it
// runs in the background.
func ignoreTTIN() { signal.Ignore(syscall.SIGTTIN) }
