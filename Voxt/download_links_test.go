package main

import (
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"testing"
)

// TestDownloadLinks: every Voxt 2 download link in the docs downloads the
// disk image of the version in mygo.json in one click, not a release page or
// a release search (GitHub's search did not list the newest releases). The
// tutorials' install steps may also name the version's release page, for
// a download that does not start.
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
	tag := regexp.MustCompile(`^https://github\.com/sgl-project/sglang-omni/releases/tag/voxt-v([0-9.]+)$`)
	want := "https://github.com/sgl-project/sglang-omni/releases/download/voxt-v" + cfg.Version + "/Voxt-" + cfg.Version + "-arm64.dmg"
	files := []struct {
		path    string
		tagPage bool
	}{
		{"../README.md", false}, {"README.md", false}, {"README.zh-CN.md", false}, {"../docs/get_started/voxt.md", false},
		{"../docs/get_started/voxt-tutorials/start.md", true}, {"../docs/get_started/voxt-tutorials/en/start.md", true},
	}
	// A pull request that changes only a tutorial page still runs this test.
	ci, err := os.ReadFile("../.github/workflows/voxtnew-ci.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(ci), `- "docs/get_started/voxt-tutorials/**"`) {
		t.Error("Voxt CI does not run for docs/get_started/voxt-tutorials/**")
	}
	for _, f := range files {
		file := f.path
		b, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, u := range link.FindAllString(string(b), -1) {
			if tm := tag.FindStringSubmatch(u); f.tagPage && tm != nil {
				if tm[1] != cfg.Version {
					t.Errorf("%s: %s, want the release page of %s", file, u, cfg.Version)
				}
				continue
			}
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
