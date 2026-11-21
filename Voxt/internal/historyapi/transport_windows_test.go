//go:build windows

package historyapi

import (
	"errors"
	"io"
	"net"
	"os"
	"path/filepath"
	"testing"
)

func TestServeEndpointLifecycle(t *testing.T) {
	path := filepath.Join(t.TempDir(), SocketName)
	first, err := Listen(path)
	if err != nil {
		t.Fatal(err)
	}
	stale, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("endpoint remains after close: %v", err)
	}
	if err := os.WriteFile(path, stale, 0o600); err != nil {
		t.Fatal(err)
	}
	listener, err := Listen(path)
	if err != nil {
		t.Fatalf("stale endpoint not replaced: %v", err)
	}
	defer listener.Close()
	address, ok := listener.Addr().(*net.TCPAddr)
	if !ok || !address.IP.IsLoopback() {
		t.Fatalf("listener address %T %v", listener.Addr(), listener.Addr())
	}
}

func TestServeRejectsMissingToken(t *testing.T) {
	audited := false
	path, cancel := startServer(t, &Server{History: &fakeHistory{}, Audit: func(Read) { audited = true }})
	current, err := readEndpoint(path)
	if err != nil {
		t.Fatal(err)
	}
	connection, err := net.Dial("tcp4", current.Address)
	if err != nil {
		t.Fatal(err)
	}
	defer connection.Close()
	if _, err := connection.Write([]byte("wrong\n")); err != nil {
		t.Fatal(err)
	}
	if _, err := connection.Read(make([]byte, 1)); !errors.Is(err, io.EOF) {
		t.Fatalf("unauthenticated connection read: %v", err)
	}
	cancel()
	if audited {
		t.Fatal("unauthenticated connection reached the server")
	}
}
