package main

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestBehaviours: testdata/behaviours.tsv lists what a user can do with Voxt
// and the check that covers each. Levels: L runs on every pull request with
// fake models and a fake helper, M runs on a Mac by hand, E is an evaluation
// with real models, manual is a page of steps. An L or E check is a test; an
// M check is a test or a script; a manual check is the page.
func TestBehaviours(t *testing.T) {
	tests := map[string]bool{}
	fn := regexp.MustCompile(`(?m)^func (Test\w*)\(`)
	err := filepath.WalkDir(".", func(path string, d fs.DirEntry, err error) error {
		if err != nil || !strings.HasSuffix(path, "_test.go") {
			return err
		}
		b, err := os.ReadFile(path)
		for _, m := range fn.FindAllSubmatch(b, -1) {
			tests[string(m[1])] = true
		}
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile("testdata/behaviours.tsv")
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSuffix(string(b), "\n"), "\n")
	if lines[0] != "feature\tlevel\tcheck\tgap" {
		t.Fatalf("header = %q", lines[0])
	}
	features := map[string]bool{}
	for i, line := range lines[1:] {
		f := strings.Split(line, "\t")
		if len(f) != 4 || f[0] == "" || f[3] == "" {
			t.Errorf("line %d: want feature, level, check, gap: %q", i+2, line)
			continue
		}
		features[f[0]] = true
		_, statErr := os.Stat(f[2])
		switch f[1] {
		case "L", "E":
			if !tests[f[2]] {
				t.Errorf("line %d: %s: no test named %s", i+2, f[0], f[2])
			}
		case "M":
			if !tests[f[2]] && statErr != nil {
				t.Errorf("line %d: %s: %s is neither a test nor a file", i+2, f[0], f[2])
			}
		case "manual":
			if statErr != nil {
				t.Errorf("line %d: %s: no page %s", i+2, f[0], f[2])
			}
		default:
			t.Errorf("line %d: level %q, want L, M, E or manual", i+2, f[1])
		}
	}
	if len(features) < 32 {
		t.Errorf("%d features listed, want at least 32", len(features))
	}
}
