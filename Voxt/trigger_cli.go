package main

// `Voxt toggle FEATURE` and `Voxt cancel` (docs/trigger.md): what a tap of
// a feature's shortcut, or of the cancel shortcut, does, from a launcher, a
// key remapper or a script. The command sends the action to the running app
// over internal/triggerapi's socket and returns once the app has acted. Its
// messages are English, as the app's other developer commands.

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"

	"github.com/sgl-project/sglang-omni/Voxt/internal/triggerapi"
)

const triggerUsage = `usage:
  Voxt toggle (transcribe | translate | rewrite | note)
  Voxt cancel
toggle does what one tap of the feature's shortcut does: it starts the
feature, or finishes the recording that is on. cancel does what the cancel
shortcut does.`

func runTriggerCLI(args []string) int {
	return triggerCLI(args, os.Stderr, filepath.Join(filepath.Dir(historySocketPath()), triggerapi.SocketName))
}

// triggerCLI takes the command's words from "toggle" or "cancel" on.
func triggerCLI(args []string, stderr io.Writer, socket string) int {
	var action string
	switch {
	case len(args) == 2 && args[0] == "toggle" && slices.Contains(triggerapi.ToggleActions, args[1]):
		action = args[1]
	case len(args) == 1 && args[0] == "cancel":
		action = triggerapi.ActionCancel
	default:
		fmt.Fprintln(stderr, triggerUsage)
		return exitUsage
	}
	err := triggerapi.Send(socket, action)
	switch {
	case errors.Is(err, triggerapi.ErrUnavailable):
		fmt.Fprintln(stderr, "Voxt is not running. Open Voxt, then run the command again.")
		return exitUnavailable
	case err != nil:
		fmt.Fprintf(stderr, "Voxt did nothing: %v\n", err)
		return exitInterrupted
	}
	return exitOK
}
