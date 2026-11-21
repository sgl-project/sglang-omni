package main

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/dictapi"
)

// dictionaryServer serves imp at a short socket path and returns it.
func dictionaryServer(t *testing.T, imp func(ctx context.Context, name string, data []byte, dryRun bool) (dictapi.Result, error)) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "dcli")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	path := filepath.Join(dir, dictapi.SocketName)
	ln, err := net.Listen("unix", path)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go (&dictapi.Server{Import: imp}).Serve(ctx, ln)
	return path
}

func runDictionary(socket string, args ...string) (code int, stdout, stderr string) {
	var o, e strings.Builder
	code = dictionaryCLI(args, &o, &e, socket, openDictionaryFile)
	return code, o.String(), e.String()
}

func writeTermsFile(t *testing.T, name, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestDictionaryCLIImports(t *testing.T) {
	var gotName, gotData string
	var gotDry bool
	socket := dictionaryServer(t, func(_ context.Context, name string, data []byte, dryRun bool) (dictapi.Result, error) {
		gotName, gotData, gotDry = name, string(data), dryRun
		return dictapi.Result{Added: 2, Updated: 1}, nil
	})
	file := writeTermsFile(t, "terms.csv", "term,aliases\nSol,Soul\n")

	code, out, errOut := runDictionary(socket, "import", file)
	if code != exitOK || out != "Imported: 2 new, 1 updated, 0 rows skipped.\n" || errOut != "" {
		t.Fatalf("code %d out %q err %q", code, out, errOut)
	}
	if gotName != "terms.csv" || gotData != "term,aliases\nSol,Soul\n" || gotDry {
		t.Errorf("app got %q %q dry=%v", gotName, gotData, gotDry)
	}

	// Options after FILE, a dry run, JSON.
	code, out, _ = runDictionary(socket, "import", file, "--dry-run", "--json")
	if code != exitOK || !gotDry {
		t.Fatalf("code %d dry=%v", code, gotDry)
	}
	var res dictapi.Result
	if err := json.Unmarshal([]byte(out), &res); err != nil || res != (dictapi.Result{Added: 2, Updated: 1, DryRun: true}) {
		t.Errorf("json %q: %+v %v", out, res, err)
	}
	code, out, _ = runDictionary(socket, "import", "--dry-run", file)
	if code != exitOK || !strings.HasPrefix(out, "Would import: 2 new") {
		t.Errorf("dry run: code %d out %q", code, out)
	}
}

type dictionaryCountingReader struct {
	remaining int64
	read      int64
}

func (r *dictionaryCountingReader) Read(p []byte) (int, error) {
	if r.remaining == 0 {
		return 0, io.EOF
	}
	if int64(len(p)) > r.remaining {
		p = p[:r.remaining]
	}
	for i := range p {
		p[i] = 'x'
	}
	r.remaining -= int64(len(p))
	r.read += int64(len(p))
	return len(p), nil
}

func (r *dictionaryCountingReader) Close() error { return nil }

func TestReadDictionaryDataStopsAtLimit(t *testing.T) {
	reader := &dictionaryCountingReader{remaining: 2 * dictapi.MaxBytes}
	var stdout, stderr strings.Builder
	code := dictionaryCLI([]string{"import", "terms.csv"}, &stdout, &stderr, "unused", func(string) (io.ReadCloser, error) {
		return reader, nil
	})
	if code != exitInterrupted || !strings.Contains(stderr.String(), "larger than 8 MB") {
		t.Fatalf("code %d, stderr %q, want the oversize-file error", code, stderr.String())
	}
	if reader.read != int64(dictapi.MaxBytes+1) {
		t.Fatalf("read %d bytes, want only the oversize sentinel %d", reader.read, dictapi.MaxBytes+1)
	}
}

func TestDictionaryCLIUsage(t *testing.T) {
	socket := dictionaryServer(t, func(context.Context, string, []byte, bool) (dictapi.Result, error) {
		t.Error("the app was called for a usage error")
		return dictapi.Result{}, nil
	})
	file := writeTermsFile(t, "terms.csv", "Sol\n")
	for _, args := range [][]string{nil, {"export"}, {"import"}, {"import", file, file}, {"import", "--force", file}} {
		if code, _, errOut := runDictionary(socket, args...); code != exitUsage || !strings.Contains(errOut, "usage:") {
			t.Errorf("%q: code %d err %q, want usage", args, code, errOut)
		}
	}
}

func TestDictionaryCLIFailures(t *testing.T) {
	file := writeTermsFile(t, "terms.json", "{")
	if code, _, errOut := runDictionary(filepath.Join(t.TempDir(), "none.sock"), "import", file); code != exitUnavailable || !strings.Contains(errOut, "not running") {
		t.Errorf("no app: code %d err %q", code, errOut)
	}
	if code, _, errOut := runDictionary("unused", "import", filepath.Join(t.TempDir(), "missing.csv")); code != exitUnreadable || !strings.Contains(errOut, "Cannot read") {
		t.Errorf("missing file: code %d err %q", code, errOut)
	}
	socket := dictionaryServer(t, func(context.Context, string, []byte, bool) (dictapi.Result, error) {
		return dictapi.Result{}, errors.New("terms.json is not a dictionary file")
	})
	if code, _, errOut := runDictionary(socket, "import", file); code != exitInterrupted || !strings.Contains(errOut, "not a dictionary file") {
		t.Errorf("app error: code %d err %q", code, errOut)
	}
}
