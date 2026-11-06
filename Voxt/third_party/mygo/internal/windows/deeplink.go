//go:build windows && (amd64 || arm64)

package windows

import (
	"os"
	"path/filepath"
	"strings"
)

// URL schemes are registered for the current user under
// HKEY_CURRENT_USER\Software\Classes\<scheme>, whose shell\open\command
// runs the executable with the URL.

func schemeKey(scheme string) string { return `Software\Classes\` + scheme }

// openCommand is the command that opens URLs with this executable.
func openCommand() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	if real, err := filepath.EvalSymlinks(exe); err == nil {
		exe = real
	}
	return `"` + exe + `" "%1"`, nil
}

func (a appController) RegisterURLScheme(scheme, id, name string) error {
	cmd, err := openCommand()
	if err != nil {
		return err
	}
	key := schemeKey(scheme)
	exe := strings.TrimSuffix(strings.TrimPrefix(cmd, `"`), `" "%1"`)
	for _, v := range []struct{ path, name, value string }{
		{key, "", "URL:" + name},
		{key, "URL Protocol", ""},
		{key + `\DefaultIcon`, "", `"` + exe + `",0`},
		{key + `\shell\open\command`, "", cmd},
	} {
		if err := regSet(hkeyCurrentUser, v.path, v.name, v.value); err != nil {
			return err
		}
	}
	return nil
}

func (a appController) UnregisterURLScheme(scheme, id, name string) error {
	if !a.IsURLSchemeRegistered(scheme, id, name) {
		return nil // another application's, or none
	}
	return regDeleteTree(hkeyCurrentUser, schemeKey(scheme))
}

func (a appController) IsURLSchemeRegistered(scheme, id, name string) bool {
	cmd, err := openCommand()
	if err != nil {
		return false
	}
	got, ok := regGet(hkeyCurrentUser, schemeKey(scheme)+`\shell\open\command`, "")
	return ok && strings.EqualFold(got, cmd)
}
