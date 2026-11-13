package main

// Regression test for scripts/build-helper.sh dying after a successful
// build: it copied .build/arm64-apple-macosx/release/voxtnew-helper, but
// Swift 6.4.2's SwiftPM puts the product in .build/out/Products/Release,
// so the script stopped with "No such file or directory". It must copy
// from wherever swift build --show-bin-path says. Runs on Linux too:
// uname, swift and codesign are fakes at the front of PATH. A future diff
// that hardcodes a product path again turns this red.

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

func TestBuildHelperCopiesFromSwiftBinPath(t *testing.T) {
	// The script reads the version with grep -A1: the plist must keep the
	// key and its string on adjacent lines (the fake product below uses the
	// same layout for the embedded-version check).
	plist, err := os.ReadFile("helper/Support/Info.plist")
	if err != nil {
		t.Fatal(err)
	}
	if !regexp.MustCompile(`<key>CFBundleShortVersionString</key>\n[ \t]*<string>`).Match(plist) {
		t.Fatal("helper/Support/Info.plist must keep CFBundleShortVersionString's key and string on adjacent lines: build-helper.sh reads it with grep -A1")
	}
	cases := []struct {
		name       string
		productDir string // where the fake swift puts the product, under helper/
	}{
		{"swift-6.4-layout", ".build/out/Products/Release"},
		{"legacy-layout", ".build/arm64-apple-macosx/release"},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			root := t.TempDir()
			for _, name := range []string{"scripts/build-helper.sh", "mygo.json", "helper/Support/Info.plist", "helper/Support/helper.entitlements"} {
				content, err := os.ReadFile(name)
				if err != nil {
					t.Fatal(err)
				}
				destination := filepath.Join(root, name)
				if err := os.MkdirAll(filepath.Dir(destination), 0o755); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(destination, content, 0o755); err != nil {
					t.Fatal(err)
				}
			}
			fakeBinDir := filepath.Join(root, "fakebin")
			if err := os.MkdirAll(fakeBinDir, 0o755); err != nil {
				t.Fatal(err)
			}
			productDir := filepath.Join(root, "helper", testCase.productDir)
			codesignLogPath := filepath.Join(root, "codesign.log")
			fakes := map[string]string{
				"uname": "#!/bin/sh\necho Darwin\n",
				"swift": `#!/bin/sh
case " $* " in
*" --show-bin-path "*)
	echo "$FAKE_SWIFT_PRODUCTS"
	exit 0
	;;
esac
mkdir -p "$FAKE_SWIFT_PRODUCTS"
version="$(sed -n 's/^  "version": *"\([^"]*\)".*/\1/p' ../mygo.json)"
printf '<key>CFBundleShortVersionString</key>\n<string>%s</string>\n' "$version" > "$FAKE_SWIFT_PRODUCTS/voxtnew-helper"
`,
				"codesign": "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$FAKE_CODESIGN_LOG\"\n",
			}
			for name, script := range fakes {
				if err := os.WriteFile(filepath.Join(fakeBinDir, name), []byte(script), 0o755); err != nil {
					t.Fatal(err)
				}
			}
			outDir := filepath.Join(root, "staged")
			command := shellWithFakeBin("bash", fakeBinDir, filepath.Join(root, "scripts/build-helper.sh"), "--out", outDir)
			command.Env = append(command.Env,
				"FAKE_SWIFT_PRODUCTS="+productDir,
				"FAKE_CODESIGN_LOG="+codesignLogPath)
			output, err := command.CombinedOutput()
			if err != nil {
				t.Fatalf("build-helper.sh failed: %v\n%s", err, output)
			}
			stagedBytes, err := os.ReadFile(filepath.Join(outDir, "voxtnew-helper"))
			if err != nil {
				t.Fatalf("no staged helper: %v\n%s", err, output)
			}
			productBytes, err := os.ReadFile(filepath.Join(productDir, "voxtnew-helper"))
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(stagedBytes, productBytes) {
				t.Fatalf("staged helper is not the product from %s", testCase.productDir)
			}
			codesignLog, err := os.ReadFile(codesignLogPath)
			if err != nil {
				t.Fatalf("codesign did not run: %v\n%s", err, output)
			}
			stagedForBash, err := exec.Command("bash", "-c", "cd \"$1\" && pwd", "bash", outDir).Output()
			if err != nil {
				t.Fatal(err)
			}
			staged := strings.TrimSpace(string(stagedForBash)) + "/voxtnew-helper"
			if !strings.Contains(string(codesignLog), staged) {
				t.Fatalf("codesign ran on something else:\n%s", codesignLog)
			}
		})
	}
}
