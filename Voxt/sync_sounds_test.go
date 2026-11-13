package main

// Regression test for scripts/sync-sounds.sh deleting a folder the user
// names: when $ACTNOW_SOUNDS had no catalog.json, the script ran rm -rf on
// it before trying a fresh clone, so a mistyped or half-copied library
// (and everything else in that folder) was gone. Only the script's own
// cache may be replaced. Runs on Linux too: git and go are fakes at the
// front of PATH. A future diff that removes $ACTNOW_SOUNDS again, or that
// stops refreshing the cache, turns this red.

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestSyncSoundsOnlyReplacesItsOwnCache(t *testing.T) {
	cases := []struct {
		name        string
		userDir     bool // ACTNOW_SOUNDS names the library; else the cache is used
		catalog     bool // the library already has catalog.json
		wantSuccess bool
	}{
		{"user-dir-without-catalog-is-kept", true, false, false},
		{"user-dir-with-catalog-is-used", true, true, true},
		{"stale-cache-is-recloned", false, false, true},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			root := t.TempDir()
			script, err := os.ReadFile("scripts/sync-sounds.sh")
			if err != nil {
				t.Fatal(err)
			}
			if err := os.MkdirAll(filepath.Join(root, "scripts"), 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(filepath.Join(root, "scripts/sync-sounds.sh"), script, 0o755); err != nil {
				t.Fatal(err)
			}
			fakeBinDir := filepath.Join(root, "fakebin")
			if err := os.MkdirAll(fakeBinDir, 0o755); err != nil {
				t.Fatal(err)
			}
			goLogPath := filepath.Join(root, "go.log")
			fakes := map[string]string{
				// Not a repository; clone gives a library with a catalogue.
				"git": `#!/bin/sh
[ "$1" = clone ] || exit 1
for last; do :; done
mkdir -p "$last" && echo '{}' > "$last/catalog.json"
`,
				"go": "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$FAKE_GO_LOG\"\n",
			}
			for name, fake := range fakes {
				if err := os.WriteFile(filepath.Join(fakeBinDir, name), []byte(fake), 0o755); err != nil {
					t.Fatal(err)
				}
			}
			cacheHome := filepath.Join(root, "cache")
			library := filepath.Join(cacheHome, "actnow-sounds")
			environment := append(os.Environ(),
				"PATH="+bashPathWithFakeBin(fakeBinDir),
				"XDG_CACHE_HOME="+cacheHome,
				"FAKE_GO_LOG="+goLogPath)
			if testCase.userDir {
				library = filepath.Join(root, "my-sounds")
				environment = append(environment, "ACTNOW_SOUNDS="+library)
			}
			if err := os.MkdirAll(library, 0o755); err != nil {
				t.Fatal(err)
			}
			ownFile := filepath.Join(library, "notes.txt")
			if err := os.WriteFile(ownFile, []byte("mine"), 0o644); err != nil {
				t.Fatal(err)
			}
			if testCase.catalog {
				if err := os.WriteFile(filepath.Join(library, "catalog.json"), []byte("{}"), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			command := exec.Command("sh", filepath.Join(root, "scripts/sync-sounds.sh"))
			command.Env = environment
			output, err := command.CombinedOutput()
			_, statErr := os.Stat(ownFile)
			if testCase.userDir && statErr != nil {
				t.Fatalf("the ACTNOW_SOUNDS folder lost its files: %v\n%s", statErr, output)
			}
			if !testCase.userDir && statErr == nil {
				t.Fatalf("the stale cache was not replaced by a fresh clone\n%s", output)
			}
			if (err == nil) != testCase.wantSuccess {
				t.Fatalf("sync-sounds.sh: err=%v, want success=%v\n%s", err, testCase.wantSuccess, output)
			}
			goLog, _ := os.ReadFile(goLogPath)
			fromLibrary := library
			if !testCase.userDir && runtime.GOOS == "windows" {
				fromLibrary = filepath.Dir(library) + "/actnow-sounds"
			}
			if testCase.wantSuccess && !strings.Contains(string(goLog), "-from\n"+fromLibrary+"\n") {
				t.Fatalf("did not copy from %s; go got:\n%s", library, goLog)
			}
			if !testCase.wantSuccess && !strings.Contains(string(output), "catalog.json") {
				t.Fatalf("the error does not say what is missing:\n%s", output)
			}
		})
	}
}
