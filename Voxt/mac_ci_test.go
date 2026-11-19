package main

import (
	"os"
	"strings"
	"testing"
)

func TestMacCIVendorsMyGoOnMacOS(t *testing.T) {
	data, err := os.ReadFile("../.github/workflows/mac-ci.yaml")
	if err != nil {
		t.Fatal(err)
	}
	workflow := string(data)
	start := strings.Index(workflow, "  voxtnew-swift:\n")
	if start < 0 {
		t.Fatal("Mac CI has no Voxt job")
	}
	job := workflow[start:]
	for _, want := range []string{
		`hasChangesUnder("Voxt/third_party/mygo-patches/")`,
		`changedPaths.includes("Voxt/scripts/vendor-mygo.sh")`,
	} {
		if !strings.Contains(workflow, want) {
			t.Errorf("Mac CI does not select the Voxt job for %q", want)
		}
	}
	for _, want := range []string{
		"runs-on: macos-26",
		"go-version-file: Voxt/go.mod",
		"run: scripts/vendor-mygo.sh --check",
	} {
		if !strings.Contains(job, want) {
			t.Errorf("the macOS Voxt job does not contain %q", want)
		}
	}
}
