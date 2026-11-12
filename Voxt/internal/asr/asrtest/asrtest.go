// Package asrtest helps test the remote speech services' clients on
// Linux: a loopback WebSocket server that a test scripts (read the
// client's frames, answer with recorded ones) and golden files of what a
// client sent. No test reaches a real service.
//
// Golden files: Golden compares text with testdata/<name>.golden;
// VOXTNEW_UPDATE_GOLDEN=1 rewrites them. Review the diff before
// committing one.
package asrtest

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/coder/websocket"
)

// Conn is the server's side of one test connection.
type Conn struct {
	*websocket.Conn
	Req *http.Request // the handshake request (its headers carry the auth)
	T   testing.TB
	Ctx context.Context
}

// Read reads the client's next message, failing the test on error.
func (c *Conn) Read() (websocket.MessageType, []byte) {
	c.T.Helper()
	typ, data, err := c.Conn.Read(c.Ctx)
	if err != nil {
		c.T.Errorf("server read: %v", err)
		return 0, nil
	}
	return typ, data
}

// TryRead reads the next message, returning ok false when the client
// closed or the test's time ran out.
func (c *Conn) TryRead() (websocket.MessageType, []byte, bool) {
	typ, data, err := c.Conn.Read(c.Ctx)
	return typ, data, err == nil
}

// WriteText sends a text frame.
func (c *Conn) WriteText(s string) {
	c.T.Helper()
	if err := c.Write(c.Ctx, websocket.MessageText, []byte(s)); err != nil {
		c.T.Errorf("server write: %v", err)
	}
}

// WriteBinary sends a binary frame.
func (c *Conn) WriteBinary(b []byte) {
	c.T.Helper()
	if err := c.Write(c.Ctx, websocket.MessageBinary, b); err != nil {
		c.T.Errorf("server write: %v", err)
	}
}

// WSServer starts a loopback server whose every WebSocket connection
// runs handle; it returns the ws:// URL of path on it. handle returning
// closes the connection normally. The server stops with the test.
func WSServer(t testing.TB, path string, handle func(c *Conn)) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ws, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Errorf("accept: %v", err)
			return
		}
		ws.SetReadLimit(16 << 20)
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		handle(&Conn{Conn: ws, Req: r, T: t, Ctx: ctx})
		ws.Close(websocket.StatusNormalClosure, "")
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http") + path
}

// RefuseServer starts a loopback server that refuses every handshake
// with status and body (an auth failure, say); it returns its ws:// URL.
func RefuseServer(t testing.TB, status int, body string) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return "ws" + strings.TrimPrefix(srv.URL, "http")
}

// Golden compares got with testdata/<name>.golden (relative to the test's
// package), or writes it when VOXTNEW_UPDATE_GOLDEN=1.
func Golden(t testing.TB, name, got string) {
	t.Helper()
	path := filepath.Join("testdata", name+".golden")
	if os.Getenv("VOXTNEW_UPDATE_GOLDEN") == "1" {
		os.MkdirAll("testdata", 0o755)
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("%v (run with VOXTNEW_UPDATE_GOLDEN=1 to create it)", err)
	}
	if string(want) != got {
		t.Errorf("%s differs from what the client sent (VOXTNEW_UPDATE_GOLDEN=1 rewrites it)\n--- got\n%s\n--- want\n%s", path, got, want)
	}
}
