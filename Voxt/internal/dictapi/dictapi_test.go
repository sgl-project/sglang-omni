package dictapi

import (
	"context"
	"errors"
	"net"
	"os"
	"path/filepath"
	"testing"
)

// serve runs a Server with imp at a short socket path (macOS limits a
// socket's path to 104 bytes) and returns the path.
func serve(t *testing.T, imp func(ctx context.Context, name string, data []byte, dryRun bool) (Result, error)) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "dapi")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	path := filepath.Join(dir, SocketName)
	ln, err := net.Listen("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go (&Server{Import: imp}).Serve(ctx, ln)
	return path
}

func TestImportCarriesTheFileAndTheResult(t *testing.T) {
	var gotName, gotData string
	var gotDry bool
	path := serve(t, func(_ context.Context, name string, data []byte, dryRun bool) (Result, error) {
		gotName, gotData, gotDry = name, string(data), dryRun
		return Result{Added: 2, Updated: 1, Skipped: 3}, nil
	})
	file := "term,aliases\nSol,Soul\n"
	for _, dry := range []bool{false, true} {
		res, err := Import(path, "terms.csv", []byte(file), dry)
		if err != nil {
			t.Fatal(err)
		}
		if gotName != "terms.csv" || gotData != file || gotDry != dry {
			t.Errorf("server got %q %q %v, want terms.csv, the file, %v", gotName, gotData, gotDry, dry)
		}
		if want := (Result{Added: 2, Updated: 1, Skipped: 3, DryRun: dry}); res != want {
			t.Errorf("result %+v, want %+v", res, want)
		}
	}
}

func TestImportEmptyFile(t *testing.T) {
	path := serve(t, func(_ context.Context, _ string, data []byte, _ bool) (Result, error) {
		if len(data) != 0 {
			t.Errorf("data %q, want none", data)
		}
		return Result{}, nil
	})
	if _, err := Import(path, "empty.csv", nil, false); err != nil {
		t.Fatal(err)
	}
}

func TestImportGivesBackTheAppsError(t *testing.T) {
	path := serve(t, func(context.Context, string, []byte, bool) (Result, error) {
		return Result{}, errors.New("not a dictionary file")
	})
	_, err := Import(path, "x.json", []byte("{"), false)
	if err == nil || err.Error() != "not a dictionary file" {
		t.Fatalf("err %v, want the app's message", err)
	}
	if errors.Is(err, ErrUnavailable) {
		t.Error("an app's error is not ErrUnavailable")
	}
}

func TestImportWithoutTheApp(t *testing.T) {
	_, err := Import(filepath.Join(t.TempDir(), SocketName), "terms.csv", []byte("Sol\n"), false)
	if !errors.Is(err, ErrUnavailable) {
		t.Fatalf("err %v, want ErrUnavailable", err)
	}
}

func TestImportRefusesAFileTooLarge(t *testing.T) {
	called := false
	path := serve(t, func(context.Context, string, []byte, bool) (Result, error) {
		called = true
		return Result{}, nil
	})
	if _, err := Import(path, "big.csv", make([]byte, MaxBytes+1), false); err == nil {
		t.Fatal("a file over MaxBytes imported")
	}
	if called {
		t.Error("the app was asked to import a file over MaxBytes")
	}
}

func TestServerRefusesABadRequest(t *testing.T) {
	path := serve(t, func(context.Context, string, []byte, bool) (Result, error) {
		t.Error("Import called for a bad request")
		return Result{}, nil
	})
	for _, line := range []string{"not json\n", `{"op":"export","bytes":0}` + "\n", `{"op":"import","bytes":-1}` + "\n", `{"op":"import","bytes":99999999999}` + "\n"} {
		conn, err := net.Dial("unix", path)
		if err != nil {
			t.Fatal(err)
		}
		conn.Write([]byte(line))
		buf := make([]byte, 256)
		n, _ := conn.Read(buf)
		conn.Close()
		if n == 0 || !contains(string(buf[:n]), `"error"`) {
			t.Errorf("%q: answer %q, want an error", line, buf[:n])
		}
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
