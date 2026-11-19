package main

// Voxt CI runs the fast checks through scripts/check.sh, so a developer
// running the script locally runs what CI runs. This fails if the workflow
// stops calling the script or the script drops one of its checks.

import (
	"os"
	"strings"
	"testing"
)

func TestCICallsCheckScript(t *testing.T) {
	ci, err := os.ReadFile("../.github/workflows/voxtnew-ci.yaml")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(ci), "run: scripts/check.sh\n") {
		t.Error("Voxt CI does not run scripts/check.sh")
	}
	script, err := os.ReadFile("scripts/check.sh")
	if err != nil {
		t.Fatal(err)
	}
	for _, check := range []string{"xargs -0 gofmt -l", "go tool mygo vet .", "go test -count=1 ./internal/archtest"} {
		if !strings.Contains(string(script), check) {
			t.Errorf("scripts/check.sh no longer runs %q", check)
		}
	}
}
