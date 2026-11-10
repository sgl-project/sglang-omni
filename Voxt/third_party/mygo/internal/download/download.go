// Package download fetches the files that MyGo keeps in the user's cache:
// the native libraries plugins name in their mygo-plugin.json, which the
// CLI and unpackaged programs download into <cache>/mygo/natives/<sha256>/,
// and the tools the CLI runs, as NSIS. Every download checks the file's
// SHA-256, tries again when the host fails in passing, and processes that
// need one file at once (two apps, `go test` running packages side by
// side) download it once, one after the other.
package download

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// File is a file to download.
type File struct {
	URL string
	// SHA256 is the file's SHA-256 in lowercase hex.
	SHA256 string
	// Client sends the requests; nil means http.DefaultClient.
	Client *http.Client
	// Header is added to every request.
	Header http.Header
	// Progress, when set, returns where each attempt copies what arrives
	// too, given the length the response announces (or -1).
	Progress func(total int64) io.Writer
}

// Backoff is how long Get waits before each new attempt after a transient
// failure: 5xx, 408 and 429 responses, and connections that fail or break
// off. Hosts like GitHub's release downloads answer 502 and 504 now and
// then, and the next request succeeds.
var Backoff = []time.Duration{time.Second, 2 * time.Second, 4 * time.Second}

// maxSize bounds what a download may write.
const maxSize = 64 << 20

// StatusError is a response other than 200 OK.
type StatusError struct {
	URL    string
	Code   int
	Status string
}

func (e *StatusError) Error() string { return fmt.Sprintf("GET %s: %s", e.URL, e.Status) }

// transient reports whether err is a failure that a new request may not
// have: an overloaded or restarting server, or a connection that failed or
// broke off. Timeouts are not: a host that did not answer in time is
// unlikely to answer the next request sooner, and waiting for it again
// would hold up the next copy of the file; nor is a name that does not
// resolve, or a file that is not the expected one.
func transient(err error) bool {
	if se, ok := errors.AsType[*StatusError](err); ok {
		return se.Code >= 500 || se.Code == http.StatusRequestTimeout || se.Code == http.StatusTooManyRequests
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	if ne, ok := errors.AsType[net.Error](err); ok && ne.Timeout() {
		return false
	}
	if de, ok := errors.AsType[*net.DNSError](err); ok && de.IsNotFound {
		return false
	}
	if _, ok := errors.AsType[*bodyError](err); ok {
		return true
	}
	if _, ok := errors.AsType[*net.OpError](err); ok {
		return true
	}
	return errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF)
}

// Get downloads f into the file path, which it creates or truncates, and
// checks its SHA-256. It tries again, after Backoff, when an attempt fails
// in passing (see transient); the error is the last attempt's.
func Get(ctx context.Context, f File, path string) error {
	var err error
	for attempt := 0; ; attempt++ {
		err = get(ctx, f, path)
		if err == nil || attempt == len(Backoff) || !transient(err) {
			return err
		}
		select {
		case <-ctx.Done():
			return err
		case <-time.After(Backoff[attempt]):
		}
	}
}

func get(ctx context.Context, f File, path string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, f.URL, nil)
	if err != nil {
		return err
	}
	for k, v := range f.Header {
		req.Header[k] = v
	}
	client := f.Client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return &StatusError{URL: f.URL, Code: resp.StatusCode, Status: resp.Status}
	}
	out, err := os.Create(path)
	if err != nil {
		return err
	}
	h := sha256.New()
	w := io.MultiWriter(out, h)
	if f.Progress != nil {
		w = io.MultiWriter(out, h, f.Progress(resp.ContentLength))
	}
	_, err = io.Copy(w, io.LimitReader(bodyReader{resp.Body}, maxSize))
	if cerr := out.Close(); err == nil {
		err = cerr
	}
	if be, ok := errors.AsType[*bodyError](err); ok {
		return fmt.Errorf("GET %s: %w", f.URL, be)
	} else if err != nil {
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != f.SHA256 {
		return fmt.Errorf("%s has the SHA-256 %s, not %s", f.URL, got, f.SHA256)
	}
	return nil
}

// bodyReader tells the errors of reading a response's body, which broke
// off, from those of writing the file.
type bodyReader struct{ r io.Reader }

func (b bodyReader) Read(p []byte) (int, error) {
	n, err := b.r.Read(p)
	if err != nil && err != io.EOF {
		err = &bodyError{err}
	}
	return n, err
}

type bodyError struct{ err error }

func (e *bodyError) Error() string { return e.err.Error() }
func (e *bodyError) Unwrap() error { return e.err }

// Cached makes the file path hold f, downloading it when it is missing or
// differs, executable, as native libraries are. It holds a lock on the
// directory of path while it checks and downloads, so that processes that
// need the file at the same time download it once: the first downloads it
// and renames it into place, the others wait for it, then find it there.
//
// Without the lock, each would rename its own download onto path, and on
// Windows renaming onto a file that another process has open or loaded,
// or is renaming itself, fails with "Access is denied".
func Cached(ctx context.Context, f File, path string) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	unlock, err := Lock(filepath.Join(dir, ".lock"), func() { waiting(path) })
	if err != nil {
		return err
	}
	defer unlock()
	if Check(path, f.SHA256) {
		return nil
	}
	tmp, err := os.CreateTemp(dir, ".download-*")
	if err != nil {
		return err
	}
	tmp.Close()
	defer os.Remove(tmp.Name())
	if err := Get(ctx, f, tmp.Name()); err != nil {
		return err
	}
	if err := os.Chmod(tmp.Name(), 0o755); err != nil {
		return err
	}
	if err := rename(tmp.Name(), path); err != nil {
		// Where processes do not take the lock (see Lock), another one may
		// have put the file there meanwhile, and may have it loaded, which
		// Windows won't replace.
		if Check(path, f.SHA256) {
			return nil
		}
		return err
	}
	return nil
}

// rename is os.Rename, and waiting is called when Cached waits for
// another process to download the file, both for tests to replace.
var (
	rename  = os.Rename
	waiting = func(path string) {}
)

// Check reports whether the file path exists and has the SHA-256 sum.
func Check(path, sum string) bool {
	got, err := FileSHA256(path)
	return err == nil && got == sum
}

// FileSHA256 returns the SHA-256 of a file, in hex.
func FileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
