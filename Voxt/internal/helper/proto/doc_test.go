package proto

import (
	"os"
	"slices"
	"strings"
	"testing"
)

// TestDocsNameEveryMethod: every method has a row in docs/helper-protocol.md.
func TestDocsNameEveryMethod(t *testing.T) {
	doc, err := os.ReadFile("../../../docs/helper-protocol.md")
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range slices.Concat(Requests, Notifications) {
		if !strings.Contains(string(doc), "\n| `"+m+"` ") {
			t.Errorf("%s has no row in docs/helper-protocol.md", m)
		}
	}
}
