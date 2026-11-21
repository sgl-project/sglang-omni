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
	workflow := strings.ReplaceAll(string(data), "\r\n", "\n") // a CRLF checkout on Windows
	if !strings.Contains(workflow, "run_voxt_helper: ${{") || !strings.Contains(workflow, "steps.pr.outputs.has_voxt_helper_changes") {
		t.Error("Mac CI does not connect its Voxt-job output to preflight")
	}
	selection := ""
	if start := strings.Index(workflow, `core.setOutput("has_voxt_helper_changes",`); start >= 0 {
		selection = workflow[start:]
		if end := strings.Index(selection, ");"); end >= 0 {
			selection = selection[:end]
		}
	}
	for _, want := range []string{
		`hasChangesUnder("Voxt/helper/")`,
		`hasChangesUnder("Voxt/third_party/mygo-patches/")`,
		`changedPaths.includes("Voxt/scripts/vendor-mygo.sh")`,
	} {
		if !strings.Contains(selection, want) {
			t.Errorf("Mac CI does not select the Voxt job for %q", want)
		}
	}
	start := strings.Index(workflow, "  voxtnew-swift:\n")
	if start < 0 {
		t.Fatal("Mac CI has no Voxt job")
	}
	job := workflow[start:]
	for _, want := range []string{
		"needs: [preflight]",
		"if: needs.preflight.outputs.run_mac == 'true' && needs.preflight.outputs.run_voxt_helper == 'true'",
		"runs-on: macos-26",
		"working-directory: Voxt",
		"go-version-file: Voxt/go.mod",
		"run: scripts/vendor-mygo.sh --check",
	} {
		if !strings.Contains(job, want) {
			t.Errorf("the macOS Voxt job does not contain %q", want)
		}
	}
	linuxCI, err := os.ReadFile("../.github/workflows/voxtnew-ci.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if count := strings.Count(string(linuxCI), `".github/workflows/mac-ci.yaml"`); count != 2 {
		t.Errorf("Voxt CI has %d mac-ci.yaml path filters, want pull-request and push filters", count)
	}
}
