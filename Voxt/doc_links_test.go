package main

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestDocLinksResolve: every relative link in Voxt's Markdown names a file or
// directory that exists. A page about removed code names the old path as
// text, not as a link.
// shortcut: #anchors are not checked, add that when one breaks.
func TestDocLinksResolve(t *testing.T) {
	link := regexp.MustCompile(`\]\(([^)\s]+)\)`)
	err := filepath.WalkDir(".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		name := d.Name()
		if d.IsDir() {
			if path != "." && (strings.HasPrefix(name, ".") || name == "vendor" || name == "third_party" || name == "node_modules" || name == "testdata") {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(name, ".md") {
			return nil
		}
		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		for _, m := range link.FindAllStringSubmatch(string(b), -1) {
			target, _, _ := strings.Cut(m[1], "#")
			if target == "" || strings.Contains(target, ":") {
				continue
			}
			if _, err := os.Stat(filepath.Join(filepath.Dir(path), target)); err != nil {
				t.Errorf("%s links to %s, which does not exist", path, m[1])
			}
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
}
