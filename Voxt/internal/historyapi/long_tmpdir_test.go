package historyapi

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestSocketsWithLongTMPDIR runs socket tests again with a TMPDIR longer
// than a socket path may be: their sockets must not follow it.
func TestSocketsWithLongTMPDIR(t *testing.T) {
	long := filepath.Join(t.TempDir(), strings.Repeat("x", 120))
	if err := os.Mkdir(long, 0o755); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(os.Args[0], "-test.count=1", "-test.v", "-test.run", "^(TestServeFileMode|TestNoEndLine)$")
	cmd.Env = append(os.Environ(), "TMPDIR="+long)
	out, err := cmd.CombinedOutput()
	if err != nil || !strings.Contains(string(out), "--- PASS") {
		t.Fatalf("%v\n%s", err, out)
	}
}
