package main

import (
	"os"
	"strings"
	"testing"
)

// TestChangelogHeadingsOncePerVersion: a version's notes go out as they
// are, so each "### " heading (Added, Fixed, 中文…) appears once in it.
// A second one puts the entries after it under the wrong heading.
func TestChangelogHeadingsOncePerVersion(t *testing.T) {
	data, err := os.ReadFile("CHANGELOG.md")
	if err != nil {
		t.Fatal(err)
	}
	version, seen := "", map[string]bool{}
	for _, line := range strings.Split(string(data), "\n") {
		switch {
		case strings.HasPrefix(line, "## "):
			version, seen = line, map[string]bool{}
		case strings.HasPrefix(line, "### "):
			if seen[line] {
				t.Errorf("%s: %q appears twice", version, line)
			}
			seen[line] = true
		default:
		}
	}
}
