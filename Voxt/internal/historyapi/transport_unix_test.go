//go:build !windows

package historyapi

import (
	"net"
	"os"
	"testing"
)

// TestServeFileMode: Listen makes the socket 0600, and replaces a stale
// one left by a crash.
func TestServeFileMode(t *testing.T) {
	path := socketPath(t)
	stale, err := net.Listen("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	stale.(*net.UnixListener).SetUnlinkOnClose(false)
	stale.Close()
	listener, err := Listen(path)
	if err != nil {
		t.Fatalf("stale socket not replaced: %v", err)
	}
	defer listener.Close()
	info, err := os.Stat(path)
	if err != nil || info.Mode().Perm() != 0o600 {
		t.Fatalf("mode %v, %v", info, err)
	}
}
