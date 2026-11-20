package archtest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestQualityRulesMatchTheirChecks keeps the documented commands and exceptions
// aligned with the checks that actually run.
func TestQualityRulesMatchTheirChecks(t *testing.T) {
	root, _ := moduleRoot(t)
	checks := []struct {
		path     string
		snippets []string
	}{
		{"AGENTS.md", []string{
			"deadcode -tags=meetingdemo,updatecheck ./...",
			"deadcode -test -tags=meetingdemo,updatecheck ./...",
			"functions only the first command lists",
			"keep shared fakes and helpers",
			"compare the changed file names",
			"names an automatic check when one exists",
			"check the helper build output for `warning:`",
			"add a file or raise its numeric line ceiling only for a fix that cannot wait",
		}},
		{"docs/architecture.md", []string{
			"raise its numeric line ceiling only for a fix that cannot wait",
		}},
	}
	for _, check := range checks {
		data, err := os.ReadFile(filepath.Join(root, check.path))
		if err != nil {
			t.Fatal(err)
		}
		for _, snippet := range check.snippets {
			if !strings.Contains(string(data), snippet) {
				t.Errorf("%s does not contain %q", check.path, snippet)
			}
		}
	}
}
