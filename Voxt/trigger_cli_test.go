package main

import (
	"context"
	"errors"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/triggerapi"
)

// triggerServer serves fire at a short socket path and returns it.
func triggerServer(t *testing.T, fire func(ctx context.Context, action string) error) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "tcli")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	path := filepath.Join(dir, triggerapi.SocketName)
	listener, err := net.Listen("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go (&triggerapi.Server{Fire: fire}).Serve(ctx, listener)
	return path
}

func runTrigger(socket string, args ...string) (code int, stderr string) {
	var errorOutput strings.Builder
	code = triggerCLI(args, &errorOutput, socket)
	return code, errorOutput.String()
}

func TestTriggerCLISendsTheAction(t *testing.T) {
	var fired []string
	socket := triggerServer(t, func(_ context.Context, action string) error {
		fired = append(fired, action)
		return nil
	})
	for _, args := range [][]string{{"toggle", "transcribe"}, {"toggle", "translate"}, {"toggle", "rewrite"}, {"toggle", "note"}, {"cancel"}} {
		if code, stderr := runTrigger(socket, args...); code != exitOK || stderr != "" {
			t.Fatalf("%v: code %d stderr %q", args, code, stderr)
		}
	}
	if got := strings.Join(fired, ","); got != "transcribe,translate,rewrite,note,cancel" {
		t.Errorf("app fired %s", got)
	}
}

func TestTriggerCLIFailures(t *testing.T) {
	socket := triggerServer(t, func(_ context.Context, action string) error {
		if action == "transcribe" {
			t.Error("the app was called for a usage error")
		}
		return errors.New("Translate is hidden in Settings → General")
	})
	for _, args := range [][]string{{"toggle"}, {"toggle", "meeting"}, {"toggle", "transcribe", "now"}, {"cancel", "transcribe"}} {
		if code, stderr := runTrigger(socket, args...); code != exitUsage || !strings.HasPrefix(stderr, "usage:") {
			t.Errorf("%v: code %d stderr %q", args, code, stderr)
		}
	}
	code, stderr := runTrigger(socket, "toggle", "translate")
	if code != exitInterrupted || stderr != "Voxt did nothing: Translate is hidden in Settings → General\n" {
		t.Errorf("refused: code %d stderr %q", code, stderr)
	}
	code, stderr = runTrigger(filepath.Join(filepath.Dir(socket), "gone.sock"), "cancel")
	if code != exitUnavailable || !strings.HasPrefix(stderr, "Voxt is not running.") {
		t.Errorf("not running: code %d stderr %q", code, stderr)
	}
}
