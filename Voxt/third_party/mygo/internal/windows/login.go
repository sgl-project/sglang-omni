//go:build windows && (amd64 || arm64)

package windows

import (
	"os"
	"path/filepath"
	"strings"
)

// Starting at login is a value in the Run key of the user. Task Manager's
// Startup page turns it off without removing it, in StartupApproved.

const (
	runKey      = `Software\Microsoft\Windows\CurrentVersion\Run`
	approvedKey = `Software\Microsoft\Windows\CurrentVersion\Explorer\StartupApproved\Run`
)

func loginCommand(arg string) (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	if real, err := filepath.EvalSymlinks(exe); err == nil {
		exe = real
	}
	return `"` + exe + `" ` + arg, nil
}

func (a appController) SetOpenAtLogin(open bool, id, name, arg string) error {
	if !open {
		if err := regDeleteValue(hkeyCurrentUser, runKey, id); err != nil {
			return err
		}
		return regDeleteValue(hkeyCurrentUser, approvedKey, id)
	}
	cmd, err := loginCommand(arg)
	if err != nil {
		return err
	}
	if err := regSet(hkeyCurrentUser, runKey, id, cmd); err != nil {
		return err
	}
	// Turned off in Task Manager earlier: the app asks for it again now.
	return regDeleteValue(hkeyCurrentUser, approvedKey, id)
}

func (a appController) OpenAtLogin(id, name, arg string) bool {
	cmd, err := loginCommand(arg)
	if err != nil {
		return false
	}
	if got, ok := regGet(hkeyCurrentUser, runKey, id); !ok || !strings.EqualFold(got, cmd) {
		return false
	}
	// The first byte is odd when the entry is turned off.
	state, ok := regGetBytes(hkeyCurrentUser, approvedKey, id)
	return !ok || len(state) == 0 || state[0]&1 == 0
}

// OpenedAtLogin reports false: the Run entry passes the login argument
// instead.
func (a appController) OpenedAtLogin() bool { return false }
