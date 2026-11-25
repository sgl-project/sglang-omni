package triggerapi

import (
	"context"
	"errors"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// serve runs a Server with fire at a short socket path (macOS limits a
// socket's path to 104 bytes) and returns the path.
func serve(t *testing.T, fire func(ctx context.Context, action string) error) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "tapi")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	path := filepath.Join(dir, SocketName)
	listener, err := net.Listen("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go (&Server{Fire: fire}).Serve(ctx, listener)
	return path
}

func TestSendCarriesEveryAction(t *testing.T) {
	var fired []string
	path := serve(t, func(_ context.Context, action string) error {
		fired = append(fired, action)
		return nil
	})
	want := append(append([]string{}, ToggleActions...), ActionCancel)
	for _, action := range want {
		if err := Send(path, action); err != nil {
			t.Fatalf("%s: %v", action, err)
		}
	}
	if strings.Join(fired, ",") != strings.Join(want, ",") {
		t.Errorf("app fired %v, want %v", fired, want)
	}
}

// The app's reason for doing nothing reaches the command as it was given.
func TestSendReturnsTheAppsReason(t *testing.T) {
	path := serve(t, func(context.Context, string) error {
		return errors.New("Translate is hidden in Settings → General")
	})
	err := Send(path, ActionTranslate)
	if err == nil || err.Error() != "Translate is hidden in Settings → General" {
		t.Fatalf("err %v", err)
	}
}

// An unknown action never reaches the app: the command refuses it, and the
// server refuses a request another client makes up.
func TestUnknownActionNeverFires(t *testing.T) {
	path := serve(t, func(_ context.Context, action string) error {
		t.Errorf("fired %q", action)
		return nil
	})
	if err := Send(path, "paste-last"); err == nil {
		t.Error("Send took an unknown action")
	}
	conn, err := net.Dial("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if _, err := conn.Write([]byte(`{"action":"meeting"}` + "\n")); err != nil {
		t.Fatal(err)
	}
	answer := make([]byte, 64)
	n, _ := conn.Read(answer)
	if got := strings.TrimSpace(string(answer[:n])); got != `{"error":"bad request"}` {
		t.Errorf("answer %q", got)
	}
}

func TestSendWithoutTheAppIsUnavailable(t *testing.T) {
	dir, err := os.MkdirTemp("", "tapi")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	if err := Send(filepath.Join(dir, SocketName), ActionTranscribe); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("err %v, want ErrUnavailable", err)
	}
}
