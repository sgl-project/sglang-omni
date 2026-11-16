//go:build linux && (amd64 || arm64)

package linux

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// Starting at login follows the XDG autostart specification: a desktop
// entry in $XDG_CONFIG_HOME/autostart.

func autostartEntry(id string) string {
	return filepath.Join(xdgDir("XDG_CONFIG_HOME", ".config"), "autostart", desktopName(id)+".desktop")
}

func (a appController) SetOpenAtLogin(open bool, id, name, arg string) error {
	path := autostartEntry(id)
	if !open {
		if err := os.Remove(path); err != nil && !errors.Is(err, fs.ErrNotExist) {
			return err
		}
		return nil
	}
	exe, err := executable()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	entry := "[Desktop Entry]\nType=Application\nName=" + entryString(name) + "\nExec=" + execArg(exe) + " " + arg +
		"\nTerminal=false\nX-GNOME-Autostart-enabled=true\n"
	return writeFileAtomic(path, []byte(entry))
}

func (a appController) OpenAtLogin(id, name, arg string) bool {
	data, err := os.ReadFile(autostartEntry(id))
	if err != nil {
		return false
	}
	for _, line := range strings.Split(string(data), "\n") {
		switch strings.TrimSpace(line) {
		case "Hidden=true", "X-GNOME-Autostart-enabled=false":
			return false // turned off in the desktop's settings
		}
	}
	return true
}

// OpenedAtLogin reports false: the autostart entry passes the login
// argument instead.
func (a appController) OpenedAtLogin() bool { return false }
