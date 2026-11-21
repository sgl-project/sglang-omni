//go:build !js

package mygo

import (
	"os"
	"path/filepath"
	"strings"
)

// localeFile is the file in PathUserData that keeps the language the app
// chose (SetLocale) where the system keeps none, as on Windows and Linux.
const localeFile = "locale"

// localeFilePath returns the path of the file, without making its
// directory, which only saving one does.
func localeFilePath() (string, error) {
	App.mu.Lock()
	dir := App.paths[PathUserData]
	App.mu.Unlock()
	if dir == "" {
		d, _, err := defaultPath(PathUserData)
		if err != nil {
			return "", err
		}
		dir = d
	}
	return filepath.Join(dir, localeFile), nil
}

func readLocaleFile() string {
	path, err := localeFilePath()
	if err != nil {
		return ""
	}
	data, err := readAppFile(path)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

// writeLocaleFile writes tag, or removes the file for "". A failure keeps
// what was saved: the app shows the language anyway.
func writeLocaleFile(tag string) {
	path, err := localeFilePath()
	if err != nil {
		return
	}
	if tag == "" {
		_ = os.Remove(path)
		return
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return
	}
	if err := os.WriteFile(path+".tmp", []byte(tag+"\n"), 0o644); err == nil {
		_ = rename(path+".tmp", path)
	}
}
