package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestReleasePreflight runs scripts/release-preflight.sh on a small copy of
// the files it reads: a good tree passes, and each thing a release got wrong
// before leaves its FAIL row and a non-zero exit.
func TestReleasePreflight(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("needs git")
	}
	script, err := filepath.Abs("scripts/release-preflight.sh")
	if err != nil {
		t.Fatal(err)
	}
	const dmg = "https://github.com/sgl-project/sglang-omni/releases/download/voxt-v9.1.0/Voxt-9.1.0-arm64.dmg\n"
	good := map[string]string{
		"Voxt/mygo.json":                 "{\n  \"name\": \"Voxt\",\n  \"version\": \"9.1.0\"\n}\n",
		"Voxt/helper/Support/Info.plist": "<key>CFBundleShortVersionString</key>\n<string>9.1.0</string>\n",
		"Voxt/CHANGELOG.md":              "# Changelog\n\n## [9.1.0] - 2026-01-02\n\n### Added\n\n- A thing (#12)\n\n### 中文\n\n#### 新增\n\n- 一件事（#12）\n\n## [9.0.0] - 2026-01-01\n",
		"README.md":                      dmg, "Voxt/README.md": dmg, "Voxt/README.zh-CN.md": dmg,
		"docs/get_started/voxt.md":                    dmg,
		"docs/get_started/voxt-tutorials/start.md":    dmg,
		"docs/get_started/voxt-tutorials/en/start.md": dmg,
	}
	// A section far larger than a pipe buffer: grep -q closing the pipe
	// early kills sed with SIGPIPE, so a pipefail pipeline misreads a long
	// valid section as having no entry.
	var long strings.Builder
	long.WriteString("# Changelog\n\n## [9.1.0] - 2026-01-02\n\n### Added\n\n")
	for i := 1; i <= 3000; i++ {
		fmt.Fprintf(&long, "- English entry number %d of a long release section\n", i)
	}
	long.WriteString("\n### 中文\n\n#### 新增\n\n")
	for i := 1; i <= 3000; i++ {
		fmt.Fprintf(&long, "- 长版本说明的第 %d 条中文条目\n", i)
	}
	long.WriteString("\n## [9.0.0] - 2026-01-01\n")
	for _, c := range []struct {
		name, file, content string
		want                string // a row of the output
		fails               bool
	}{
		{"good", "", "", "pulls      ok", false},
		{"long section", "Voxt/CHANGELOG.md", long.String(), "changelog  ok", false},
		{"helper version", "Voxt/helper/Support/Info.plist", "<key>CFBundleShortVersionString</key>\n<string>9.0.0</string>\n", "version    FAIL", true},
		{"no section", "Voxt/CHANGELOG.md", "# Changelog\n\n## [Unreleased]\n\n- A thing (#12)\n", "changelog  FAIL", true},
		{"no Chinese", "Voxt/CHANGELOG.md", "## [9.1.0] - 2026-01-02\n\n- A thing (#12)\n", "changelog  FAIL", true},
		{"empty Chinese entry", "Voxt/CHANGELOG.md", strings.Replace(good["Voxt/CHANGELOG.md"], "- 一件事（#12）\n", "", 1), "changelog  FAIL", true},
		{"old link", "docs/get_started/voxt-tutorials/en/start.md", strings.ReplaceAll(dmg, "9.1.0", "9.0.0"), "links      FAIL", true},
		{"unnamed pull request", "Voxt/CHANGELOG.md", strings.ReplaceAll(good["Voxt/CHANGELOG.md"], "#12", "#120"), "pulls      note  since voxt-v9.0.0, not in CHANGELOG.md: #12", false},
	} {
		t.Run(c.name, func(t *testing.T) {
			repo := t.TempDir()
			git := func(args ...string) {
				t.Helper()
				cmd := exec.Command("git", append([]string{"-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null"}, args...)...)
				cmd.Dir = repo
				if out, err := cmd.CombinedOutput(); err != nil {
					t.Fatalf("git %v: %v\n%s", args, err, out)
				}
			}
			git("init", "-q")
			git("commit", "-q", "--allow-empty", "-m", "start")
			git("tag", "voxt-v9.0.0")
			git("tag", "voxt-v9.1.0-beta.1")
			for file, content := range good {
				if file == c.file {
					content = c.content
				}
				path := filepath.Join(repo, file)
				if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			git("add", "-A")
			git("commit", "-q", "-m", "[Voxt] A thing (#12)")

			// A security that finds no identity: the test must not read the keychain.
			bin := t.TempDir()
			if err := os.WriteFile(filepath.Join(bin, "security"), []byte("#!/bin/sh\nexit 1\n"), 0o755); err != nil {
				t.Fatal(err)
			}
			cmd := exec.Command("bash", script, "9.1.0")
			cmd.Env = append(os.Environ(), "VOXT_PREFLIGHT_ROOT="+filepath.Join(repo, "Voxt"), "PATH="+bin+string(os.PathListSeparator)+os.Getenv("PATH"))
			out, err := cmd.CombinedOutput()
			if !strings.Contains(string(out), c.want) {
				t.Errorf("no row %q in:\n%s", c.want, out)
			}
			if !strings.Contains(string(out), "signing    note") {
				t.Errorf("the signing row is not a note:\n%s", out)
			}
			if failed := err != nil; failed != c.fails {
				t.Errorf("exit error = %v, want failure %t:\n%s", err, c.fails, out)
			}
		})
	}
}
