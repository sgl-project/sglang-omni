package main

import (
	"encoding/json"
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

// TestChangelogEntryTitlesOncePerVersion: one entry per change in the
// release being prepared. The 2.1.3 section listed the same clean-up entry
// twice, once for the landed #4459 and once for its closed-unmerged twin
// #4383.
func TestChangelogEntryTitlesOncePerVersion(t *testing.T) {
	var cfg struct{ Version string }
	b, err := os.ReadFile("mygo.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("CHANGELOG.md")
	if err != nil {
		t.Fatal(err)
	}
	inVersion, inSection, seen := false, "", map[string]bool{}
	for _, line := range strings.Split(string(data), "\n") {
		switch {
		case strings.HasPrefix(line, "## "):
			inVersion = strings.HasPrefix(line, "## ["+cfg.Version+"]")
		case !inVersion:
		case strings.HasPrefix(line, "### "), strings.HasPrefix(line, "#### "):
			inSection, seen = line, map[string]bool{}
		case strings.HasPrefix(line, "- **"):
			title, _, _ := strings.Cut(line[len("- **"):], "**")
			key := inSection + "|" + title
			if seen[key] {
				t.Errorf("%s: entry %q appears twice", inSection, title)
			}
			seen[key] = true
		}
	}
}

func TestChangelogChineseCoversRewritePipeAndQueuedEnter(t *testing.T) {
	content, err := os.ReadFile("CHANGELOG.md")
	if err != nil {
		t.Fatal(err)
	}
	want := "改写保留代码末尾的竖线，旧输入框排队的 Enter 不会发送新录音"
	if !strings.Contains(string(content), want) {
		t.Errorf("Chinese changelog is missing %q", want)
	}
}
