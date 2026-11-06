package update

import (
	"bytes"
	"errors"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Linux apps that install.sh of `mygo build` installed have their desktop
// entry, and the MIME package of the file types they define, registered for
// the user in $XDG_DATA_HOME: the files <name>.desktop and <name>.xml of
// the app, the entry running the executable and showing the icon of the
// install by their absolute paths. RefreshDesktopEntry registers them again
// after an update, which may change them.

// RefreshDesktopEntry registers again the desktop entry and MIME package of
// the Linux app in dir, whose executable is name, as install.sh does, when
// install.sh registered them: when the user's entry runs the app in dir.
func RefreshDesktopEntry(dir, name string) error {
	data := os.Getenv("XDG_DATA_HOME")
	if !filepath.IsAbs(data) {
		home, err := os.UserHomeDir()
		if err != nil {
			return err
		}
		data = filepath.Join(home, ".local", "share")
	}
	entryPath := filepath.Join(data, "applications", name+".desktop")
	registered, err := os.ReadFile(entryPath)
	if err != nil {
		return nil
	}
	// The directory as the entry has it, which may go through a link, such
	// as /home to /var/home.
	exe := registeredExec(string(registered))
	if exe == "" || filepath.Base(exe) != name || !sameFile(exe, filepath.Join(dir, name)) {
		return nil
	}
	dir = filepath.Dir(exe)
	packaged, err := os.ReadFile(filepath.Join(dir, name+".desktop"))
	if err != nil {
		return err
	}
	if entry := desktopEntry(string(packaged), dir, name); entry != string(registered) {
		if err := os.WriteFile(entryPath, []byte(entry), 0o644); err != nil {
			return err
		}
		runTool("update-desktop-database", filepath.Join(data, "applications"))
	}

	mimePath := filepath.Join(data, "mime", "packages", name+".xml")
	mime, err := os.ReadFile(filepath.Join(dir, name+".xml"))
	old, oldErr := os.ReadFile(mimePath)
	switch {
	case err == nil && (oldErr != nil || !bytes.Equal(mime, old)):
		if err := os.MkdirAll(filepath.Dir(mimePath), 0o755); err != nil {
			return err
		}
		if err := os.WriteFile(mimePath, mime, 0o644); err != nil {
			return err
		}
	case errors.Is(err, fs.ErrNotExist) && oldErr == nil:
		// The version defines no file types any more.
		if err := os.Remove(mimePath); err != nil {
			return err
		}
	case err != nil && !errors.Is(err, fs.ErrNotExist):
		return err
	default:
		return nil
	}
	runTool("update-mime-database", filepath.Join(data, "mime"))
	return nil
}

// desktopEntry returns packaged, the desktop entry of the app in dir whose
// executable is name, running the executable and showing the icon by their
// paths, as install.sh makes it.
func desktopEntry(packaged, dir, name string) string {
	icon := dir + "/" + name + ".png"
	_, err := os.Stat(icon)
	var b strings.Builder
	for line := range strings.Lines(packaged) {
		line = strings.TrimSuffix(line, "\n")
		if rest, ok := strings.CutPrefix(line, "Exec="+name); ok && (rest == "" || rest[0] == ' ') {
			line = `Exec="` + dir + "/" + name + `"` + rest
		} else if line == "Icon="+name && err == nil {
			line = "Icon=" + icon
		}
		b.WriteString(line + "\n")
	}
	return b.String()
}

// registeredExec returns the executable of a desktop entry install.sh
// registered: the quoted path of its Exec key.
func registeredExec(entry string) string {
	for line := range strings.SplitSeq(entry, "\n") {
		if rest, ok := strings.CutPrefix(line, `Exec="`); ok {
			if exe, _, ok := strings.Cut(rest, `"`); ok && filepath.IsAbs(exe) {
				return exe
			}
		}
	}
	return ""
}

func sameFile(a, b string) bool {
	x, err := os.Stat(a)
	if err != nil {
		return false
	}
	y, err := os.Stat(b)
	return err == nil && os.SameFile(x, y)
}

// runTool runs a tool that updates the desktop's caches, when it is
// installed; desktops also notice the files change.
func runTool(name string, args ...string) {
	if path, err := exec.LookPath(name); err == nil {
		_ = exec.Command(path, args...).Run()
	}
}
