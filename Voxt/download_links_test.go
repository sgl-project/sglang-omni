package main

import (
	"encoding/json"
	"os"
	"regexp"
	"testing"
)

// TestDownloadLinks: every Voxt 2 download link in the docs downloads the
// disk image of the version in mygo.json in one click, not a release page or
// a release search (GitHub's search did not list the newest releases).
func TestDownloadLinks(t *testing.T) {
	var cfg struct{ Version string }
	b, err := os.ReadFile("mygo.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &cfg); err != nil {
		t.Fatal(err)
	}
	link := regexp.MustCompile(`https://github\.com/sgl-project/sglang-omni/releases[^\s)\]>"]*`)
	dmg := regexp.MustCompile(`^https://github\.com/sgl-project/sglang-omni/releases/download/voxt-v([0-9.]+)/Voxt-([0-9.]+?)(-arm64)?\.dmg$`)
	want := "https://github.com/sgl-project/sglang-omni/releases/download/voxt-v" + cfg.Version + "/Voxt-" + cfg.Version + "-arm64.dmg"
	for _, file := range []string{"../README.md", "README.md", "README.zh-CN.md", "../docs/get_started/voxt.md"} {
		b, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, u := range link.FindAllString(string(b), -1) {
			m := dmg.FindStringSubmatch(u)
			switch {
			case m == nil:
				t.Errorf("%s: %s is not a disk image download", file, u)
			case m[1] != m[2]:
				t.Errorf("%s: %s: the file's version is not the tag's", file, u)
			case m[1][0] == '1': // Voxt 1.x, for going back
			case u != want:
				t.Errorf("%s: %s, want %s", file, u, want)
			default:
				found = true
			}
		}
		if !found {
			t.Errorf("%s: no download link for Voxt %s", file, cfg.Version)
		}
	}
}
