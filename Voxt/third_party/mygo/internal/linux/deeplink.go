//go:build linux && (amd64 || arm64)

package linux

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
)

// URL scheme handlers follow the XDG conventions: a desktop entry that
// opens URLs with the app, hidden from menus, is made the default handler
// of the scheme in the user's mimeapps.list, which is what `xdg-mime
// default` does.

func (a appController) RegisterURLScheme(scheme, id, name string) error {
	dir, file := handlerEntry(id)
	path := filepath.Join(dir, file)
	schemes := handlerSchemes(path)
	if !slices.Contains(schemes, scheme) {
		schemes = append(schemes, scheme)
	}
	exe, err := executable()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := writeFileAtomic(path, []byte(handlerDesktopEntry(name, exe, schemes))); err != nil {
		return err
	}
	return setMimeDefault("x-scheme-handler/"+scheme, file, "")
}

func (a appController) UnregisterURLScheme(scheme, id, name string) error {
	dir, file := handlerEntry(id)
	path := filepath.Join(dir, file)
	if schemes := handlerSchemes(path); slices.Contains(schemes, scheme) {
		schemes = slices.DeleteFunc(schemes, func(s string) bool { return s == scheme })
		var err error
		if len(schemes) == 0 {
			err = os.Remove(path)
		} else if exe, e := executable(); e != nil {
			err = e
		} else {
			err = writeFileAtomic(path, []byte(handlerDesktopEntry(name, exe, schemes)))
		}
		if err != nil {
			return err
		}
	}
	return setMimeDefault("x-scheme-handler/"+scheme, "", file)
}

func (a appController) IsURLSchemeRegistered(scheme, id, name string) bool {
	dir, file := handlerEntry(id)
	return slices.Contains(handlerSchemes(filepath.Join(dir, file)), scheme) &&
		mimeDefault("x-scheme-handler/"+scheme) == file
}

func xdgDir(env, fallback string) string {
	if d := os.Getenv(env); filepath.IsAbs(d) {
		return d
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, fallback)
}

// handlerEntry returns where the desktop entry of the app's URL handler
// goes, named after the app's identifier.
func handlerEntry(id string) (dir, file string) {
	return filepath.Join(xdgDir("XDG_DATA_HOME", ".local/share"), "applications"), desktopName(id) + ".url-handler.desktop"
}

// desktopName makes an application id usable in a desktop file name.
func desktopName(id string) string {
	return strings.Map(func(r rune) rune {
		if r < 0x80 && (r == '.' || r == '-' || r == '_' || r >= '0' && r <= '9' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z') {
			return r
		}
		return '-'
	}, id)
}

// executable is the file to run for a URL: the AppImage rather than the
// directory it is mounted at while running.
func executable() (string, error) {
	if p := os.Getenv("APPIMAGE"); p != "" {
		return p, nil
	}
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	if real, err := filepath.EvalSymlinks(exe); err == nil {
		exe = real
	}
	return exe, nil
}

func handlerDesktopEntry(name, exe string, schemes []string) string {
	var mime strings.Builder
	for _, s := range schemes {
		mime.WriteString("x-scheme-handler/" + s + ";")
	}
	return "[Desktop Entry]\nType=Application\nName=" + entryString(name) + "\nExec=" + execArg(exe) +
		" %u\nTerminal=false\nNoDisplay=true\nMimeType=" + mime.String() + "\n"
}

// entryString drops what a desktop entry string value cannot hold.
func entryString(s string) string {
	return strings.Map(func(r rune) rune {
		if r < ' ' {
			return -1
		}
		return r
	}, s)
}

// execArg quotes an argument of the Exec key of a desktop entry: quoting
// escapes ", `, $ and \ with a backslash, and backslashes are escaped again
// because the value is a string; % is doubled for field codes.
func execArg(arg string) string {
	var b strings.Builder
	b.WriteByte('"')
	for _, r := range arg {
		switch r {
		case '"', '`', '$':
			b.WriteString(`\\`)
			b.WriteRune(r)
		case '\\':
			b.WriteString(`\\\\`)
		case '%':
			b.WriteString("%%")
		default:
			b.WriteRune(r)
		}
	}
	b.WriteByte('"')
	return b.String()
}

// handlerSchemes returns the schemes a handler entry declares.
func handlerSchemes(path string) []string {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var schemes []string
	for _, line := range strings.Split(string(data), "\n") {
		if v, ok := strings.CutPrefix(strings.TrimSpace(line), "MimeType="); ok {
			for _, m := range strings.Split(v, ";") {
				if s, ok := strings.CutPrefix(m, "x-scheme-handler/"); ok && s != "" {
					schemes = append(schemes, s)
				}
			}
		}
	}
	return schemes
}

const defaultApps = "[Default Applications]"

func mimeappsList() string {
	return filepath.Join(xdgDir("XDG_CONFIG_HOME", ".config"), "mimeapps.list")
}

// mimeDefault returns the default application of a MIME type in the
// user's mimeapps.list.
func mimeDefault(mime string) string {
	data, _ := os.ReadFile(mimeappsList())
	section := ""
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "[") {
			section = line
			continue
		}
		if k, v, ok := strings.Cut(line, "="); ok && section == defaultApps && strings.TrimSpace(k) == mime {
			first, _, _ := strings.Cut(strings.TrimSpace(v), ";")
			return first
		}
	}
	return ""
}

// setMimeDefault makes file the default application of a MIME type in the
// user's mimeapps.list or, when file is "", removes the default if it is
// owner.
func setMimeDefault(mime, file, owner string) error {
	path := mimeappsList()
	data, err := os.ReadFile(path)
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return err
	}
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	if len(data) == 0 {
		lines = nil
	}
	start, end := -1, len(lines) // the Default Applications section
	for i, line := range lines {
		t := strings.TrimSpace(line)
		if start >= 0 && strings.HasPrefix(t, "[") {
			end = i
			break
		}
		if t == defaultApps {
			start = i
		}
	}
	entry := mime + "=" + file
	found := -1
	if start >= 0 {
		for i := start + 1; i < end; i++ {
			if k, _, ok := strings.Cut(lines[i], "="); ok && strings.TrimSpace(k) == mime {
				found = i
			}
		}
	}
	switch {
	case file == "":
		if found < 0 {
			return nil
		}
		if _, v, _ := strings.Cut(lines[found], "="); strings.TrimSuffix(strings.TrimSpace(v), ";") != owner {
			return nil // another application's
		}
		lines = slices.Delete(lines, found, found+1)
	case found >= 0:
		lines[found] = entry
	case start >= 0:
		lines = slices.Insert(lines, start+1, entry)
	default:
		if len(lines) > 0 {
			lines = append(lines, "")
		}
		lines = append(lines, defaultApps, entry)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return writeFileAtomic(path, []byte(strings.Join(lines, "\n")+"\n"))
}

func writeFileAtomic(path string, data []byte) error {
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
