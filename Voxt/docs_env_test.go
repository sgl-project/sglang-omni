package main

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestDocsEnvironmentVariables(t *testing.T) {
	content, err := os.ReadFile("docs/development.md")
	if err != nil {
		t.Fatal(err)
	}
	documented := make(map[string]bool)
	for _, match := range regexp.MustCompile("(?m)^\\| `(VOXTNEW_[A-Z0-9_]+)` \\|").FindAllStringSubmatch(string(content), -1) {
		documented[match[1]] = true
	}

	name := regexp.MustCompile(`\bVOXTNEW_[A-Z0-9_]+\b`)
	used := make(map[string]string)
	err = filepath.WalkDir(".", func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			if path != "." && (strings.HasPrefix(entry.Name(), ".") || entry.Name() == "build" || entry.Name() == "third_party") {
				return filepath.SkipDir
			}
			return nil
		}
		switch filepath.Ext(path) {
		case ".go", ".sh", ".swift":
		default:
			return nil
		}
		source, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		for _, found := range name.FindAllString(string(source), -1) {
			if _, seen := used[found]; !seen {
				used[found] = path
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(used) == 0 {
		t.Fatal("found no VOXTNEW_ variables in the code")
	}
	for found, path := range used {
		if !documented[found] {
			t.Errorf("%s (used in %s) has no row in the docs/development.md environment variable table", found, path)
		}
	}
}
