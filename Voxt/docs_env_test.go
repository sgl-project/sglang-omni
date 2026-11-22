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
	testFunc := regexp.MustCompile(`(?m)^func (Test\w+)\(`)
	tests := make(map[string]bool)
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
		for _, match := range testFunc.FindAllStringSubmatch(string(source), -1) {
			tests[match[1]] = true
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
	// a renamed test leaves the page pointing at a command that runs nothing
	for _, match := range regexp.MustCompile("`(Test\\w+)`").FindAllStringSubmatch(string(content), -1) {
		if !tests[match[1]] {
			t.Errorf("docs/development.md names %s, but no test function has that name", match[1])
		}
	}
	for found, path := range used {
		if !documented[found] {
			t.Errorf("%s (used in %s) has no row in the docs/development.md environment variable table", found, path)
		}
	}
	for found := range documented {
		if _, ok := used[found]; !ok {
			t.Errorf("%s is documented in docs/development.md but no longer used in the code", found)
		}
	}
}
