package mirror

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

const asset = "/o/r/releases/download/v1/tool.zip"

func digest(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func payload(n int) []byte {
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(i*7 + i/251)
	}
	return b
}

// source is a fake GitHub or mirror.
type source struct {
	*httptest.Server
	body       []byte
	mode       string        // "ok", "tampered", "404", "stall" (after half), "drip", "delay"
	delay      time.Duration // "delay": before the headers
	latest     string        // where /o/r/releases/latest redirects ("" = 404)
	latestJSON string        // or what it answers with (200)
	hits       atomic.Int32
	release    chan struct{} // closed at the test's end: stalled handlers return
}

func newSource(t *testing.T, body []byte, mode string) *source {
	s := &source{body: body, mode: mode, release: make(chan struct{})}
	s.Server = httptest.NewServer(http.HandlerFunc(s.serve))
	t.Cleanup(func() {
		close(s.release)
		s.Server.Close()
	})
	return s
}

func (s *source) serve(w http.ResponseWriter, r *http.Request) {
	s.hits.Add(1)
	if r.URL.Path == "/o/r/releases/latest" {
		if s.latestJSON != "" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(s.latestJSON))
			return
		}
		if s.latest == "" {
			http.NotFound(w, r)
			return
		}
		http.Redirect(w, r, s.latest, http.StatusFound)
		return
	}
	if r.URL.Path != asset && r.URL.Path != "/o/r/releases/download/v1/SUMS" {
		http.NotFound(w, r)
		return
	}
	body := s.body
	switch s.mode {
	case "404":
		http.NotFound(w, r)
		return
	case "tampered":
		body = bytes.Clone(body)
		body[len(body)/2] ^= 0xff
	case "delay":
		select {
		case <-time.After(s.delay):
		case <-s.release:
			return
		case <-r.Context().Done():
			return
		}
	}
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(http.StatusOK)
	f := w.(http.Flusher)
	switch s.mode {
	case "stall":
		w.Write(body[:len(body)/2])
		f.Flush()
		select {
		case <-s.release:
		case <-r.Context().Done():
		}
		return
	case "drip": // 1 KB every 20 ms: 50 KB/s
		for i := 0; i < len(body); i += 1024 {
			w.Write(body[i:min(i+1024, len(body))])
			f.Flush()
			select {
			case <-time.After(20 * time.Millisecond):
			case <-s.release:
				return
			case <-r.Context().Done():
				return
			}
		}
		return
	}
	w.Write(body)
}

func fast(srcs ...string) *Client {
	return &Client{Sources: srcs, Lead: 64 << 10, Stall: 300 * time.Millisecond, RateWindow: 400 * time.Millisecond, MinRate: 200 << 10}
}

func downURL(t *testing.T) string {
	s := httptest.NewServer(http.NotFoundHandler())
	u := s.URL
	s.Close()
	return u
}

func get(t *testing.T, c *Client, sum string) (string, []byte, error) {
	t.Helper()
	dir := t.TempDir()
	f, from, err := c.File(context.Background(), asset, FileOptions{Dir: dir, SHA256: sum})
	if err != nil {
		left, _ := os.ReadDir(dir)
		if len(left) != 0 {
			t.Errorf("temporary files left after a failure: %v", left)
		}
		return from, nil, err
	}
	b, rerr := os.ReadFile(f)
	if rerr != nil {
		t.Fatal(rerr)
	}
	return from, b, nil
}

func TestSources(t *testing.T) {
	t.Setenv(EnvSources, "")
	for v, want := range map[string][]string{
		"":                       {GitHub, ActNow},
		"auto":                   {GitHub, ActNow},
		"official":               {GitHub},
		"mirror":                 {ActNow},
		"https://hf.example.org": {GitHub, ActNow},
	} {
		if got := Sources(v); !slices.Equal(got, want) {
			t.Errorf("Sources(%q) = %v, want %v", v, got, want)
		}
	}
	t.Setenv(EnvSources, " http://a/ http://b http://a ")
	if got := Sources("official"); !slices.Equal(got, []string{"http://a", "http://b"}) {
		t.Errorf("with $%s: %v", EnvSources, got)
	}
	if got := Hosts([]string{GitHub, ActNow, "https://mirror.actnow.dev/other"}); !slices.Equal(got, []string{"https://github.com", "https://mirror.actnow.dev"}) {
		t.Errorf("Hosts: %v", got)
	}
}

func TestFileGitHubDown(t *testing.T) {
	body := payload(300 << 10)
	m := newSource(t, body, "ok")
	from, got, err := get(t, fast(downURL(t), m.URL), digest(body))
	if err != nil || from != m.URL || !bytes.Equal(got, body) {
		t.Fatalf("from %q, %d bytes, err %v; want the mirror", from, len(got), err)
	}
}

func TestFileFasterWins(t *testing.T) {
	body := payload(300 << 10)
	slow := newSource(t, body, "delay")
	slow.delay = 2 * time.Second
	m := newSource(t, body, "ok")
	start := time.Now()
	from, got, err := get(t, fast(slow.URL, m.URL), digest(body))
	if err != nil || from != m.URL || !bytes.Equal(got, body) {
		t.Fatalf("from %q, err %v; want the fast mirror", from, err)
	}
	if d := time.Since(start); d > time.Second {
		t.Errorf("took %v: waited for the slow source", d)
	}
}

func TestFileStallSwitches(t *testing.T) {
	body := payload(512 << 10)
	gh := newSource(t, body, "stall") // wins the race (half the file at once), then stalls
	m := newSource(t, body, "delay")
	m.delay = 150 * time.Millisecond
	var reports []int64
	c := fast(gh.URL, m.URL)
	dir := t.TempDir()
	f, from, err := c.File(context.Background(), asset, FileOptions{Dir: dir, SHA256: digest(body), Progress: func(done, total int64) {
		reports = append(reports, done)
		if total != int64(len(body)) {
			t.Errorf("total %d", total)
		}
	}})
	if err != nil || from != m.URL {
		t.Fatalf("from %q, err %v; want the mirror after GitHub stalled", from, err)
	}
	if b, _ := os.ReadFile(f); !bytes.Equal(b, body) {
		t.Fatal("wrong content")
	}
	if len(reports) == 0 || reports[len(reports)-1] != int64(len(body)) {
		t.Errorf("progress %v", reports)
	}
}

func TestFileSlowSwitches(t *testing.T) {
	body := payload(400 << 10)
	gh := newSource(t, body, "drip") // 50 KB/s, under MinRate
	m := newSource(t, body, "delay")
	m.delay = 500 * time.Millisecond // loses the race (the drip reaches Lead first)
	c := fast(gh.URL, m.URL)
	c.Lead = 8 << 10
	from, got, err := get(t, c, digest(body))
	if err != nil || from != m.URL || !bytes.Equal(got, body) {
		t.Fatalf("from %q, err %v; want the mirror after GitHub crawled", from, err)
	}
}

func TestFileSlowSourceRetriedOnce(t *testing.T) {
	body := payload(30 << 10)
	gh := newSource(t, body, "drip") // slow and the only one: left once, then let finish
	c := fast(gh.URL)
	c.Lead = 8 << 10
	c.RateWindow = 200 * time.Millisecond
	from, got, err := get(t, c, digest(body))
	if err != nil || from != gh.URL || !bytes.Equal(got, body) {
		t.Fatalf("from %q, err %v", from, err)
	}
	if n := gh.hits.Load(); n != 2 {
		t.Errorf("asked %d times, want 2 (once more on a new connection)", n)
	}
}

func TestFileSlowMirrorRetriedBeforeSlowGitHub(t *testing.T) {
	body := payload(300 << 10)
	m := newSource(t, body, "drip")
	gh := newSource(t, body, "delay")
	gh.delay = 5 * time.Second
	c := fast(gh.URL, m.URL)
	c.Lead = 8 << 10
	c.RateWindow = 200 * time.Millisecond
	// The mirror crawls on its first connection only.
	var first atomic.Bool
	m.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if first.CompareAndSwap(false, true) {
			m.serve(w, r)
			return
		}
		m.hits.Add(1)
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		w.Write(body)
	})
	start := time.Now()
	from, got, err := get(t, c, digest(body))
	if err != nil || from != m.URL || !bytes.Equal(got, body) {
		t.Fatalf("from %q, err %v", from, err)
	}
	if d := time.Since(start); d > 3*time.Second {
		t.Errorf("took %v", d)
	}
}

func TestFileTamperedFallsBack(t *testing.T) {
	body := payload(300 << 10)
	gh := newSource(t, body, "delay")
	gh.delay = 300 * time.Millisecond // the tampered mirror wins the race
	m := newSource(t, body, "tampered")
	var logs []string
	c := fast(gh.URL, m.URL)
	c.Logf = func(f string, a ...any) { logs = append(logs, f) }
	from, got, err := get(t, c, digest(body))
	if err != nil || from != gh.URL || !bytes.Equal(got, body) {
		t.Fatalf("from %q, err %v; want GitHub after the mirror's file failed its check", from, err)
	}
	if gh.hits.Load() != 2 {
		t.Errorf("GitHub asked %d times, want 2 (the race, then again)", gh.hits.Load())
	}
}

func TestFileOnlyTampered(t *testing.T) {
	body := payload(100 << 10)
	m := newSource(t, body, "tampered")
	_, _, err := get(t, fast(downURL(t), m.URL), digest(body))
	if !errors.Is(err, ErrChecksum) {
		t.Fatalf("err %v, want ErrChecksum", err)
	}
	var ce *ChecksumError
	if !errors.As(err, &ce) || !strings.HasPrefix(ce.URL, m.URL) {
		t.Fatalf("err %v", err)
	}
}

func TestFileMissingOnMirror(t *testing.T) {
	body := payload(10 << 10)
	gh := newSource(t, body, "ok")
	m := newSource(t, body, "404")
	from, got, err := get(t, fast(m.URL, gh.URL), digest(body))
	if err != nil || from != gh.URL || !bytes.Equal(got, body) {
		t.Fatalf("from %q, err %v", from, err)
	}
}

func TestFileAllFail(t *testing.T) {
	m := newSource(t, nil, "404")
	_, _, err := get(t, fast(downURL(t), m.URL), digest([]byte("x")))
	var se *StatusError
	if err == nil || !errors.As(err, &se) || se.Code != 404 {
		t.Fatalf("err %v", err)
	}
}

func TestFileCancel(t *testing.T) {
	body := payload(100 << 10)
	gh := newSource(t, body, "stall")
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	c := fast(gh.URL)
	_, _, err := c.File(ctx, asset, FileOptions{Dir: t.TempDir(), SHA256: digest(body)})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err %v", err)
	}
}

func TestFileLimit(t *testing.T) {
	body := payload(100 << 10)
	m := newSource(t, body, "ok")
	c := fast(m.URL)
	_, _, err := c.File(context.Background(), asset, FileOptions{Dir: t.TempDir(), SHA256: digest(body), Limit: 50 << 10})
	if err == nil || !strings.Contains(err.Error(), "larger") {
		t.Fatalf("err %v", err)
	}
}

func TestLatest(t *testing.T) {
	ok := func(tag string) bool { return strings.HasPrefix(tag, "2026.") }
	m := newSource(t, nil, "ok")
	m.latest = "/github/o/r/releases/tag/2026.10.01"
	tag, from, err := fast(downURL(t), m.URL).Latest(context.Background(), "o/r", ok)
	if err != nil || tag != "2026.10.01" || from != m.URL {
		t.Fatalf("tag %q from %q, err %v", tag, from, err)
	}
	bad := newSource(t, nil, "ok")
	bad.latest = "/o/r/releases/tag/nonsense"
	if _, _, err := fast(bad.URL).Latest(context.Background(), "o/r", ok); err == nil {
		t.Fatal("an unacceptable tag was taken")
	}
	r2 := newSource(t, nil, "ok") // the mirror's R2: JSON, no redirect
	r2.latestJSON = `{"tag": "2026.10.02", "synced": "2026-10-18T02:00:00Z"}`
	if tag, _, err := fast(r2.URL).Latest(context.Background(), "o/r", ok); err != nil || tag != "2026.10.02" {
		t.Fatalf("JSON: tag %q, err %v", tag, err)
	}
	for _, body := range []string{`{"tag": "../x"}`, `{"tag": "nonsense"}`, `{}`, `<html>`} {
		r2.latestJSON = body
		if tag, _, err := fast(r2.URL).Latest(context.Background(), "o/r", ok); err == nil {
			t.Errorf("JSON %s: took %q", body, tag)
		}
	}
	elsewhere := newSource(t, nil, "ok")
	elsewhere.latest = "/o/r/pulls"
	if _, _, err := fast(elsewhere.URL).Latest(context.Background(), "o/r", nil); err == nil {
		t.Fatal("a redirect to no tag was taken")
	}
}

func TestTryAndBytes(t *testing.T) {
	sums := []byte("abc  tool.zip\n")
	gh := newSource(t, sums, "tampered")
	m := newSource(t, sums, "ok")
	c := fast(gh.URL, m.URL)
	b, from, err := c.Bytes(context.Background(), "/o/r/releases/download/v1/SUMS", 1<<10, func(b []byte) error {
		if !bytes.Equal(b, sums) {
			return errors.New("bad")
		}
		return nil
	})
	if err != nil || from != m.URL || !bytes.Equal(b, sums) {
		t.Fatalf("from %q, err %v", from, err)
	}
	_, _, err = Try(context.Background(), c, func(ctx context.Context, base string) (int, error) {
		return 0, errors.New("no")
	})
	if err == nil || !strings.Contains(err.Error(), gh.URL+": no") || !strings.Contains(err.Error(), m.URL+": no") {
		t.Fatalf("err %v", err)
	}
	if _, _, err := c.Bytes(context.Background(), "/o/r/releases/download/v1/SUMS", 4, nil); err == nil {
		t.Fatal("over the limit")
	}
}

func TestWordings(t *testing.T) {
	for _, w := range []struct{ EN, ZH string }{TextSourceAuto, TextSourceOfficial, TextSourceMirror, TextSourceNote} {
		if w.EN == "" || w.ZH == "" {
			t.Errorf("wording %+v", w)
		}
	}
}
