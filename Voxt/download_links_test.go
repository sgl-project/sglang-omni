package main

import (
	"encoding/json"
	"os"
	"regexp"
	"strings"
	"testing"
)

func TestStableBehaviorTestsRunBeforeVersionRewrite(t *testing.T) {
	b, err := os.ReadFile("../.github/workflows/voxt-release.yaml")
	if err != nil {
		t.Fatal(err)
	}
	workflow := string(b)
	resolveStep := strings.Index(workflow, "      - name: Resolve the version\n")
	behaviorStep := strings.Index(workflow, "      - name: Stable behavior regression tests\n")
	if resolveStep < 0 || behaviorStep < 0 {
		t.Fatal("stable release workflow must keep its version and behavior test steps")
	}
	if behaviorStep > resolveStep {
		t.Error("stable behavior tests must run before Resolve the version rewrites mygo.json")
	}
	behavior := workflow[behaviorStep:]
	behavior = behavior[:strings.Index(behavior[1:], "      - name: ")+1]
	if !strings.Contains(behavior, "TMPDIR: ${{ runner.temp }}") {
		t.Error("stable behavior tests must run with a short TMPDIR: the runner's leaves the socket tests' paths a few bytes under macOS's limit")
	}
	if !strings.Contains(behavior, "github.event_name == 'pull_request'") {
		t.Error("release pull requests must run the behavior tests before their first unsigned build")
	}
}

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
