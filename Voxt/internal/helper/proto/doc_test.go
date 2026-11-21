package proto

import (
	"os"
	"slices"
	"strings"
	"testing"
)

// TestDocsNameEveryMethod: every method has a row in docs/helper-protocol.md §4.
func TestDocsNameEveryMethod(t *testing.T) {
	doc, err := os.ReadFile("../../../docs/helper-protocol.md")
	if err != nil {
		t.Fatal(err)
	}
	_, methods, _ := strings.Cut(string(doc), "\n## 4. Methods\n")
	methods, _, _ = strings.Cut(methods, "\n## 5.")
	if methods == "" {
		t.Fatal("docs/helper-protocol.md has no §4 Methods")
	}
	for _, m := range slices.Concat(Requests, Notifications) {
		row := "\n| `" + m + "`"
		if !strings.Contains(methods, row+" |") && !strings.Contains(methods, row+" **job** |") {
			t.Errorf("%s has no row in docs/helper-protocol.md §4", m)
		}
	}
}
