package hfhub

import (
	"bytes"
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/hehehai/actnowkit/models"
)

const (
	testRepo   = "acme/tiny-tts"
	testCommit = "0123456789abcdef0123456789abcdef01234567"
	testToken  = "hf_secret"
)

// fakeHub serves one repo as the Hugging Face API and resolve endpoints do,
// LFS files through a redirect to a "CDN" path, with ranges.
type fakeHub struct {
	repo  string
	files map[string][]byte
	lfs   map[string]bool
	gated bool
	// Faults.
	apiDelay atomic.Int64 // ns
	apiFail  atomic.Bool
	corrupt  map[string]bool
	cutPath  string       // abort the next response of this file…
	cutAfter atomic.Int64 // …after this many bytes (0 = off)
	// writeDelay slows every write of file content (ns).
	writeDelay atomic.Int64
	// The first response of gatePath stops after gateAt bytes: reached
	// is closed then, and it goes on when release is closed (holdAt).
	gatePath    string
	gateAt      int64
	gateOnce    sync.Once
	reached     chan struct{}
	release     chan struct{}
	releaseOnce sync.Once

	srv *httptest.Server

	mu       sync.Mutex
	served   int64
	ranges   []string
	apiCalls int
	auth     []string // Authorization headers seen, by path
}

func newFakeHub(t *testing.T) *fakeHub {
	t.Helper()
	r := rand.New(rand.NewPCG(1, 2))
	big := make([]byte, 10_000+123)
	for i := range big {
		big[i] = byte(r.Uint32())
	}
	h := &fakeHub{
		repo: testRepo,
		files: map[string][]byte{
			"config.json":            []byte(`{"model_type":"tiny"}`),
			"model.safetensors":      big,
			"voices/af_heart.pt":     bytes.Repeat([]byte("voice"), 300),
			"tokenizer/vocab.txt":    []byte("a\nb\nc\n"),
			"empty.txt":              {},
			"weights/part 2.npz":     bytes.Repeat([]byte{7}, 4000),
			"README.md":              []byte("# tiny"),
			"nested/deeper/x.json":   []byte("{}"),
			"model.safetensors.json": []byte(`{"index":true}`),
		},
		lfs:     map[string]bool{"model.safetensors": true, "voices/af_heart.pt": true, "weights/part 2.npz": true},
		corrupt: map[string]bool{},
	}
	h.srv = httptest.NewServer(http.HandlerFunc(h.serve))
	t.Cleanup(h.srv.Close)
	return h
}

// holdAt makes the first response of path stop after at bytes until
// letGo; it returns a channel closed when it stops.
func (h *fakeHub) holdAt(t *testing.T, path string, at int64) <-chan struct{} {
	h.gatePath, h.gateAt = path, at
	h.reached, h.release = make(chan struct{}), make(chan struct{})
	t.Cleanup(h.letGo) // before the server closes: it waits for its handlers
	return h.reached
}

func (h *fakeHub) letGo() { h.releaseOnce.Do(func() { close(h.release) }) }

func gitBlobID(b []byte) string {
	h := sha1.New()
	fmt.Fprintf(h, "blob %d\x00", len(b))
	h.Write(b)
	return hex.EncodeToString(h.Sum(nil))
}

func sha(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

// count reads a counter while requests may run.
func (h *fakeHub) count(what string) int64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	if what == "api" {
		return int64(h.apiCalls)
	}
	return h.served
}

func (h *fakeHub) authSeen() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.auth)
}

func (h *fakeHub) serve(w http.ResponseWriter, r *http.Request) {
	h.mu.Lock()
	h.auth = append(h.auth, r.URL.Path+" "+r.Header.Get("Authorization"))
	h.mu.Unlock()
	authed := r.Header.Get("Authorization") == "Bearer "+testToken
	p := r.URL.Path
	switch {
	case strings.HasPrefix(p, "/api/models/"+h.repo+"/revision/"):
		h.mu.Lock()
		h.apiCalls++
		h.mu.Unlock()
		time.Sleep(time.Duration(h.apiDelay.Load()))
		if h.apiFail.Load() {
			http.Error(w, "down", http.StatusServiceUnavailable)
			return
		}
		rev := strings.TrimPrefix(p, "/api/models/"+h.repo+"/revision/")
		if rev != "main" && rev != testCommit {
			http.NotFound(w, r)
			return
		}
		type lfs struct {
			SHA256 string `json:"sha256"`
			Size   int    `json:"size"`
		}
		type sib struct {
			Path   string `json:"rfilename"`
			BlobID string `json:"blobId"`
			Size   int    `json:"size"`
			LFS    *lfs   `json:"lfs,omitempty"`
		}
		var sibs []sib
		for _, path := range slices.Sorted(mapsKeys(h.files)) {
			b := h.files[path]
			s := sib{Path: path, BlobID: gitBlobID(b), Size: len(b)}
			if h.lfs[path] {
				s.LFS = &lfs{SHA256: sha(b), Size: len(b)}
				s.Size = len(b)
			}
			sibs = append(sibs, s)
		}
		gated := any(false)
		if h.gated {
			gated = "manual"
		}
		json.NewEncoder(w).Encode(map[string]any{"id": h.repo, "sha": testCommit, "gated": gated, "siblings": sibs})
	case strings.HasPrefix(p, "/"+h.repo+"/resolve/"+testCommit+"/"):
		if h.gated && !authed {
			http.Error(w, "gated", http.StatusUnauthorized)
			return
		}
		path := strings.TrimPrefix(p, "/"+h.repo+"/resolve/"+testCommit+"/")
		if h.lfs[path] {
			http.Redirect(w, r, "/cdn/"+h.repo+"/"+path, http.StatusFound)
			return
		}
		h.content(w, r, path)
	case strings.HasPrefix(p, "/cdn/"+h.repo+"/"):
		h.content(w, r, strings.TrimPrefix(p, "/cdn/"+h.repo+"/"))
	default:
		http.NotFound(w, r)
	}
}

func mapsKeys(m map[string][]byte) func(func(string) bool) {
	return func(yield func(string) bool) {
		for k := range m {
			if !yield(k) {
				return
			}
		}
	}
}

func (h *fakeHub) content(w http.ResponseWriter, r *http.Request, path string) {
	b, ok := h.files[path]
	if !ok {
		http.NotFound(w, r)
		return
	}
	h.mu.Lock()
	corrupt := h.corrupt[path]
	h.mu.Unlock()
	if corrupt {
		b = slices.Clone(b)
		b[len(b)/2] ^= 0xff
	}
	h.mu.Lock()
	if rg := r.Header.Get("Range"); rg != "" {
		h.ranges = append(h.ranges, path+" "+rg)
	}
	h.mu.Unlock()
	cw := &countingWriter{ResponseWriter: w, h: h}
	if path == h.cutPath {
		cw.cut = h.cutAfter.Swap(0)
	}
	if path == h.gatePath {
		h.gateOnce.Do(func() { cw.gate = true })
	}
	w.Header().Set("ETag", `"`+gitBlobID(b)+`"`)
	http.ServeContent(cw, r, "", time.Time{}, bytes.NewReader(b))
}

type countingWriter struct {
	http.ResponseWriter
	h    *fakeHub
	n    int64
	cut  int64
	gate bool
}

func (c *countingWriter) Write(p []byte) (int, error) {
	if d := c.h.writeDelay.Load(); d > 0 {
		time.Sleep(time.Duration(d))
	}
	if c.gate && c.n+int64(len(p)) > c.h.gateAt {
		c.gate = false
		k := c.h.gateAt - c.n
		if _, err := c.write(p[:k]); err != nil {
			return 0, err
		}
		c.ResponseWriter.(http.Flusher).Flush()
		close(c.h.reached)
		<-c.h.release
		n, err := c.write(p[k:])
		return int(k) + n, err
	}
	return c.write(p)
}

func (c *countingWriter) write(p []byte) (int, error) {
	if c.cut > 0 && c.n+int64(len(p)) > c.cut {
		p = p[:c.cut-c.n]
		c.ResponseWriter.Write(p)
		c.ResponseWriter.(http.Flusher).Flush()
		panic(http.ErrAbortHandler)
	}
	c.n += int64(len(p))
	c.h.mu.Lock()
	c.h.served += int64(len(p))
	c.h.mu.Unlock()
	return c.ResponseWriter.Write(p)
}

func (h *fakeHub) entry() Entry {
	return Entry{ID: "tiny", Repo: h.repo, Family: "tiny", Role: models.RoleTTS}
}

func (h *fakeHub) client(t *testing.T, mod func(*Options)) *Client {
	t.Helper()
	o := Options{CacheDir: t.TempDir(), Endpoints: []string{h.srv.URL}, Official: h.srv.URL, ChunkSize: 1024, ReserveBytes: -1}
	if mod != nil {
		mod(&o)
	}
	c := New(o)
	c.progressEvery = 0
	return c
}

func checkSnapshot(t *testing.T, c *Client, h *fakeHub) {
	t.Helper()
	l, err := c.Status(context.Background(), h.entry())
	if err != nil || l.State != Downloaded {
		t.Fatalf("status %+v %v", l, err)
	}
	for path, want := range h.files {
		got, err := os.ReadFile(filepath.Join(l.Snapshot, filepath.FromSlash(path)))
		if err != nil || !bytes.Equal(got, want) {
			t.Errorf("%s: %d bytes, %v", path, len(got), err)
		}
		if runtime.GOOS == "windows" {
			blob := filepath.Join(RepoDir(c.opts.CacheDir, testRepo), "blobs", sha(want))
			if !h.lfs[path] {
				blob = filepath.Join(RepoDir(c.opts.CacheDir, testRepo), "blobs", gitBlobID(want))
			}
			linkInfo, err := os.Stat(filepath.Join(l.Snapshot, filepath.FromSlash(path)))
			blobInfo, blobErr := os.Stat(blob)
			if err != nil || blobErr != nil || !os.SameFile(linkInfo, blobInfo) {
				t.Errorf("%s does not resolve to its blob", path)
			}
		} else {
			link, _ := os.Readlink(filepath.Join(l.Snapshot, filepath.FromSlash(path)))
			if !strings.HasPrefix(link, "../") || filepath.IsAbs(link) {
				t.Errorf("%s links to %q, want a relative link", path, link)
			}
		}
	}
	ref, err := os.ReadFile(filepath.Join(RepoDir(c.opts.CacheDir, testRepo), "refs", "main"))
	if err != nil || string(ref) != testCommit {
		t.Errorf("refs/main %q %v", ref, err)
	}
	blobs, _ := os.ReadDir(filepath.Join(RepoDir(c.opts.CacheDir, testRepo), "blobs"))
	for _, b := range blobs {
		if strings.Contains(b.Name(), ".incomplete") {
			t.Errorf("left %s", b.Name())
		}
	}
	if total := uniqueSizes(h.files); l.Bytes != total {
		t.Errorf("status bytes %d, want %d", l.Bytes, total)
	}
	// Every lock was released, and its owner file with it.
	owners, _ := filepath.Glob(filepath.Join(models.LocksDir(c.opts.CacheDir, testRepo), "*.owner"))
	if len(owners) > 0 {
		t.Errorf("owner files left: %v", owners)
	}
}

// uniqueSizes counts files with the same content once, as blobs do.
func uniqueSizes(files map[string][]byte) int64 {
	seen := map[string]bool{}
	var n int64
	for _, b := range files {
		if k := sha(b); !seen[k] {
			seen[k] = true
			n += int64(len(b))
		}
	}
	return n
}

func sumSizes(files map[string][]byte) int64 {
	var n int64
	for _, b := range files {
		n += int64(len(b))
	}
	return n
}

func TestDownload(t *testing.T) {
	h := newFakeHub(t)
	c := h.client(t, nil)
	ctx := context.Background()
	if l, _ := c.Status(ctx, h.entry()); l.State != Missing {
		t.Fatalf("before: %+v", l)
	}
	p, err := c.Plan(ctx, h.entry())
	if err != nil || p.Revision != testCommit || len(p.Files) != len(h.files) || p.Bytes != sumSizes(h.files) || p.Bytes != p.TotalBytes {
		t.Fatalf("plan %+v %v", p, err)
	}
	var last Progress
	calls := 0
	if err := c.Download(ctx, h.entry(), func(pr Progress) { last = pr; calls++ }); err != nil {
		t.Fatal(err)
	}
	if last.Bytes != last.TotalBytes || last.TotalBytes != sumSizes(h.files) || last.Mirror != h.srv.URL || calls < 2 {
		t.Errorf("progress %+v after %d calls", last, calls)
	}
	checkSnapshot(t, c, h)
	// The big file came in ranges.
	if n := len(h.ranges); n < 9 {
		t.Errorf("only %d ranged requests: %v", n, h.ranges)
	}
	// Again: nothing to fetch.
	served := h.count("served")
	if err := c.Download(ctx, h.entry(), nil); err != nil {
		t.Fatal(err)
	}
	if h.count("served") != served {
		t.Errorf("re-download fetched %d bytes", h.count("served")-served)
	}
	if p, _ := c.Plan(ctx, h.entry()); p.Bytes != 0 {
		t.Errorf("plan after download: %d bytes to go", p.Bytes)
	}
}

// Files with the same content share a blob: it is fetched once, and both
// paths of the snapshot link to it (Marvis ships two identical prompts).
func TestDownloadDuplicateContent(t *testing.T) {
	h := newFakeHub(t)
	h.files["prompts/a.txt"] = []byte("the same words")
	h.files["prompts/b.txt"] = []byte("the same words")
	h.files["copy.safetensors"] = h.files["model.safetensors"]
	h.lfs["copy.safetensors"] = true
	c := h.client(t, nil)
	ctx := context.Background()
	p, err := c.Plan(ctx, h.entry())
	if err != nil || p.TotalBytes != sumSizes(h.files)-int64(len(h.files["prompts/b.txt"])+len(h.files["copy.safetensors"])) {
		t.Fatalf("plan %+v %v", p, err)
	}
	if err := c.Download(ctx, h.entry(), nil); err != nil {
		t.Fatal(err)
	}
	checkSnapshot(t, c, h)
}

func TestResume(t *testing.T) {
	h := newFakeHub(t)
	ctx := context.Background()
	// Interrupted half way through, by cancelling.
	c := h.client(t, func(o *Options) { o.Parallel = 1 })
	cctx, cancel := context.WithCancel(ctx)
	err := c.Download(cctx, h.entry(), func(p Progress) {
		if p.Bytes > p.TotalBytes/2 {
			cancel()
		}
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled download: %v", err)
	}
	l, _ := c.Status(ctx, h.entry())
	if l.State != Partial || l.Bytes == 0 {
		t.Fatalf("after cancel: %+v", l)
	}
	p, _ := c.Plan(ctx, h.entry())
	if p.Bytes >= p.TotalBytes || p.Bytes == 0 {
		t.Fatalf("plan after cancel: %d of %d to go", p.Bytes, p.TotalBytes)
	}
	served := h.count("served")
	if err := c.Download(ctx, h.entry(), nil); err != nil {
		t.Fatal(err)
	}
	checkSnapshot(t, c, h)
	if again := h.count("served") - served; again > p.Bytes+1024 {
		t.Errorf("resumed download fetched %d bytes, plan said %d", again, p.Bytes)
	}
}

func TestConnectionDropsAreRetried(t *testing.T) {
	h := newFakeHub(t)
	c := h.client(t, func(o *Options) { o.Parallel = 1; o.ChunkSize = 1 << 20 }) // whole files
	h.cutPath = "model.safetensors"
	h.cutAfter.Store(3000)
	if err := c.Download(context.Background(), h.entry(), nil); err != nil {
		t.Fatal(err)
	}
	checkSnapshot(t, c, h)
	found := false
	for _, r := range h.ranges {
		found = found || strings.HasSuffix(r, "bytes=3000-10122") || strings.Contains(r, "bytes=3000-")
	}
	if !found {
		t.Errorf("no resumed range after the drop: %v", h.ranges)
	}
}

func TestSequentialPartialFromOtherTools(t *testing.T) {
	// huggingface_hub leaves blobs/<sha>.incomplete written in order.
	h := newFakeHub(t)
	c := h.client(t, nil)
	big := h.files["model.safetensors"]
	blobs := filepath.Join(RepoDir(c.opts.CacheDir, testRepo), "blobs")
	os.MkdirAll(blobs, 0o755)
	os.WriteFile(filepath.Join(blobs, sha(big)+".incomplete"), big[:5000], 0o644)
	p, _ := c.Plan(context.Background(), h.entry())
	if want := sumSizes(h.files) - 5000; p.Bytes != want {
		t.Errorf("plan %d, want %d", p.Bytes, want)
	}
	if err := c.Download(context.Background(), h.entry(), nil); err != nil {
		t.Fatal(err)
	}
	checkSnapshot(t, c, h)
	for _, r := range h.ranges {
		if strings.HasPrefix(r, "model.safetensors bytes=0-") {
			t.Errorf("refetched the first chunk: %v", h.ranges)
		}
	}
}

func TestChecksumMismatch(t *testing.T) {
	h := newFakeHub(t)
	h.corrupt["model.safetensors"] = true
	c := h.client(t, nil)
	err := c.Download(context.Background(), h.entry(), nil)
	if !errors.Is(err, ErrChecksum) {
		t.Fatalf("corrupt download: %v", err)
	}
	repo := RepoDir(c.opts.CacheDir, testRepo)
	if _, err := os.Stat(filepath.Join(repo, "refs", "main")); err == nil {
		t.Error("a ref was written for a bad download")
	}
	if l, _ := c.Status(context.Background(), h.entry()); l.State == Downloaded {
		t.Errorf("status %+v", l)
	}
	// Fixed upstream: the next download completes. (A request of the
	// failed download may still be in flight: change it under the lock.)
	h.mu.Lock()
	h.corrupt = map[string]bool{}
	h.mu.Unlock()
	if err := c.Download(context.Background(), h.entry(), nil); err != nil {
		t.Fatal(err)
	}
	checkSnapshot(t, c, h)
}

func TestVerifyAndRepair(t *testing.T) {
	h := newFakeHub(t)
	c := h.client(t, nil)
	ctx := context.Background()
	if err := c.Download(ctx, h.entry(), nil); err != nil {
		t.Fatal(err)
	}
	bad, err := c.Verify(ctx, h.entry())
	if err != nil || len(bad) != 0 {
		t.Fatalf("verify clean: %v %v", bad, err)
	}
	// Damage one LFS blob and one small file.
	l, _ := c.Status(ctx, h.entry())
	for _, p := range []string{"model.safetensors", "config.json"} {
		blob, _ := filepath.EvalSymlinks(filepath.Join(l.Snapshot, p))
		data, _ := os.ReadFile(blob)
		data[0] ^= 1
		os.WriteFile(blob, data, 0o644)
	}
	bad, err = c.Verify(ctx, h.entry())
	slices.Sort(bad)
	if err != nil || !slices.Equal(bad, []string{"config.json", "model.safetensors"}) {
		t.Fatalf("verify damaged: %v %v", bad, err)
	}
	if l, _ := c.Status(ctx, h.entry()); l.State != Partial || len(l.Missing) != 2 {
		t.Errorf("status after verify: %+v", l)
	}
	if err := c.Download(ctx, h.entry(), nil); err != nil {
		t.Fatal(err)
	}
	checkSnapshot(t, c, h)
}

func TestDelete(t *testing.T) {
	h := newFakeHub(t)
	c := h.client(t, nil)
	ctx := context.Background()
	e := h.entry()
	e.Revision = testCommit
	if err := c.Download(ctx, e, nil); err != nil {
		t.Fatal(err)
	}
	repo := RepoDir(c.opts.CacheDir, testRepo)
	// Another snapshot (an older revision) shares config.json's blob and
	// keeps it.
	l, _ := c.Status(ctx, e)
	shared, _ := models.ResolveBlob(RepoDir(c.opts.CacheDir, testRepo), filepath.Join(l.Snapshot, "config.json"))
	old := filepath.Join(repo, "snapshots", strings.Repeat("f", 40))
	os.MkdirAll(old, 0o755)
	if runtime.GOOS == "windows" {
		if err := os.Link(filepath.Join(l.Snapshot, "config.json"), filepath.Join(old, "config.json")); err != nil {
			t.Fatal(err)
		}
	} else {
		shared, _ = os.Readlink(filepath.Join(l.Snapshot, "config.json"))
		if err := os.Symlink(shared, filepath.Join(old, "config.json")); err != nil {
			t.Fatal(err)
		}
	}
	// The engine's copy of the repo stays while a snapshot does…
	cp := EngineCopy(c.opts.CacheDir, testRepo)
	os.MkdirAll(cp, 0o755)
	os.WriteFile(filepath.Join(cp, "config.json"), []byte("{}"), 0o644)
	res, err := c.Delete(ctx, e, models.ThisApp)
	if err != nil || len(res) != 1 || !res[0].Deleted || res[0].Freed != uniqueSizes(h.files)-int64(len(h.files["config.json"])) {
		t.Fatalf("delete: %+v %v", res, err)
	}
	if _, err := os.Stat(cp); err != nil {
		t.Errorf("engine copy removed with a snapshot left: %v", err)
	}
	blobs, _ := os.ReadDir(filepath.Join(repo, "blobs"))
	if len(blobs) != 1 || blobs[0].Name() != filepath.Base(shared) {
		t.Errorf("blobs left: %v", blobs)
	}
	if _, err := os.Stat(filepath.Join(repo, "refs", "main")); err == nil {
		t.Error("refs/main left")
	}
	if l, _ := c.Status(ctx, e); l.State != Partial {
		t.Errorf("with an older snapshot left: %+v", l)
	}
	// …and goes with the last one, and the repo directory too.
	os.RemoveAll(old)
	if _, err := c.Delete(ctx, e, models.ThisApp); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(repo); !os.IsNotExist(err) {
		t.Errorf("repo dir left: %v", err)
	}
	if _, err := os.Stat(cp); !os.IsNotExist(err) {
		t.Errorf("engine copy left: %v", err)
	}
	if l, _ := c.Status(ctx, e); l.State != Missing {
		t.Errorf("after delete: %+v", l)
	}
	// An entry that is not there: nothing to do.
	if res, err := c.Delete(ctx, h.entry(), models.ThisApp); err != nil || len(res) != 0 {
		t.Errorf("deleting nothing: %+v %v", res, err)
	}
}

func TestPinnedRevision(t *testing.T) {
	h := newFakeHub(t)
	c := h.client(t, nil)
	e := h.entry()
	e.Revision = testCommit
	if err := c.Download(context.Background(), e, nil); err != nil {
		t.Fatal(err)
	}
	repo := RepoDir(c.opts.CacheDir, testRepo)
	// refs/main points at it, for tools that resolve the repo by id…
	if ref, err := os.ReadFile(filepath.Join(repo, "refs", "main")); err != nil || string(ref) != testCommit {
		t.Errorf("refs/main %q %v", ref, err)
	}
	// …unless it names another snapshot that is there.
	other := strings.Repeat("b", 40)
	os.MkdirAll(filepath.Join(repo, "snapshots", other), 0o755)
	os.WriteFile(filepath.Join(repo, "refs", "main"), []byte(other), 0o644)
	if err := c.Download(context.Background(), e, nil); err != nil {
		t.Fatal(err)
	}
	if ref, _ := os.ReadFile(filepath.Join(repo, "refs", "main")); string(ref) != other {
		t.Errorf("refs/main moved to %q", ref)
	}
	if l, _ := c.Status(context.Background(), e); l.State != Downloaded || filepath.Base(l.Snapshot) != testCommit {
		t.Errorf("status %+v", l)
	}
	// The catalogue's file list restricts the download and pins hashes.
	e2 := h.entry()
	e2.Repo = testRepo
	e2.Files = []File{{Path: "config.json"}, {Path: "model.safetensors", SHA256: strings.Repeat("0", 64)}}
	if _, err := c.Plan(context.Background(), e2); !errors.Is(err, ErrChecksum) {
		t.Errorf("a pinned hash that differs: %v", err)
	}
	e2.Files[1].SHA256 = sha(h.files["model.safetensors"])
	p, err := c.Plan(context.Background(), e2)
	if err != nil || len(p.Files) != 2 {
		t.Errorf("listed files: %+v %v", p, err)
	}
	e2.Files = append(e2.Files, File{Path: "nope.bin"})
	if _, err := c.Plan(context.Background(), e2); !errors.Is(err, ErrNotFound) {
		t.Errorf("a listed file the repo lacks: %v", err)
	}
}

func TestMirrorRace(t *testing.T) {
	official := newFakeHub(t)
	mirror := newFakeHub(t)
	official.apiDelay.Store(int64(300 * time.Millisecond))
	c := official.client(t, func(o *Options) {
		o.Endpoints = []string{official.srv.URL, mirror.srv.URL}
		o.Token = func() string { return testToken }
	})
	var last Progress
	if err := c.Download(context.Background(), official.entry(), func(p Progress) { last = p }); err != nil {
		t.Fatal(err)
	}
	if last.Mirror != mirror.srv.URL {
		t.Errorf("used %s, want the faster mirror", last.Mirror)
	}
	for _, a := range mirror.authSeen() {
		if strings.Contains(a, "Bearer") {
			t.Errorf("the mirror saw the token: %s", a)
		}
	}
	if official.count("served") != 0 || mirror.count("served") == 0 {
		t.Errorf("served: official %d mirror %d", official.count("served"), mirror.count("served"))
	}
	// The winner is remembered: the official is not asked again.
	calls := official.count("api")
	c.Plan(context.Background(), official.entry())
	if official.count("api") != calls {
		t.Error("raced again")
	}
	// When the winner goes down, race again.
	mirror.apiFail.Store(true)
	official.apiDelay.Store(int64(0))
	p, err := c.Plan(context.Background(), official.entry())
	if err != nil || p.Endpoint != official.srv.URL {
		t.Errorf("after the mirror failed: %+v %v", p, err)
	}
	// Every endpoint down.
	official.apiFail.Store(true)
	if _, err := c.Plan(context.Background(), official.entry()); !errors.Is(err, ErrOffline) {
		t.Errorf("all down: %v", err)
	}
}

func TestGated(t *testing.T) {
	official := newFakeHub(t)
	mirror := newFakeHub(t)
	official.gated, mirror.gated = true, true
	token := ""
	c := official.client(t, func(o *Options) {
		o.Endpoints = []string{mirror.srv.URL, official.srv.URL}
		o.Token = func() string { return token }
	})
	mirror.apiDelay.Store(int64(0))
	official.apiDelay.Store(int64(200 * time.Millisecond))
	if _, err := c.Plan(context.Background(), official.entry()); !errors.Is(err, ErrGated) {
		t.Fatalf("without a token: %v", err)
	}
	token = testToken
	p, err := c.Plan(context.Background(), official.entry())
	if err != nil || !p.Gated || p.Endpoint != official.srv.URL {
		t.Fatalf("with a token: %+v %v", p, err)
	}
	if err := c.Download(context.Background(), official.entry(), nil); err != nil {
		t.Fatal(err)
	}
	checkSnapshot(t, c, official)
	if mirror.count("served") != 0 {
		t.Error("a gated repo came from the mirror")
	}
	for _, a := range mirror.authSeen() {
		if strings.Contains(a, "Bearer") {
			t.Errorf("the mirror saw the token: %s", a)
		}
	}
}

func TestDiskPreflight(t *testing.T) {
	h := newFakeHub(t)
	c := h.client(t, func(o *Options) { o.ReserveBytes = 1 << 62 })
	_, err := c.Plan(context.Background(), h.entry())
	if !errors.Is(err, ErrDiskSpace) {
		t.Fatalf("plan: %v", err)
	}
	if err := c.Download(context.Background(), h.entry(), nil); !errors.Is(err, ErrDiskSpace) {
		t.Fatalf("download: %v", err)
	}
	if h.count("served") != 0 {
		t.Error("downloaded despite the preflight")
	}
	if freeBytes(t.TempDir()) <= 0 || freeBytes(filepath.Join(t.TempDir(), "not", "there")) <= 0 {
		t.Error("freeBytes")
	}
}

func TestNotFound(t *testing.T) {
	h := newFakeHub(t)
	c := h.client(t, nil)
	e := h.entry()
	e.Revision = "nope"
	if _, err := c.Plan(context.Background(), e); !errors.Is(err, ErrNotFound) {
		t.Errorf("unknown revision: %v", err)
	}
}

func TestStatusOfOtherToolsCache(t *testing.T) {
	// A snapshot with only config.json (a tool fetched metadata) is not a
	// model yet.
	dir := t.TempDir()
	e := Entry{Repo: "a/b"}
	repo := RepoDir(dir, "a/b")
	os.MkdirAll(filepath.Join(repo, "blobs"), 0o755)
	os.MkdirAll(filepath.Join(repo, "snapshots", testCommit), 0o755)
	os.MkdirAll(filepath.Join(repo, "refs"), 0o755)
	os.WriteFile(filepath.Join(repo, "refs", "main"), []byte(testCommit), 0o644)
	os.WriteFile(filepath.Join(repo, "blobs", "abc"), []byte("{}"), 0o644)
	if runtime.GOOS == "windows" {
		if err := os.Link(filepath.Join(repo, "blobs", "abc"), filepath.Join(repo, "snapshots", testCommit, "config.json")); err != nil {
			t.Fatal(err)
		}
	} else if err := os.Symlink("../../blobs/abc", filepath.Join(repo, "snapshots", testCommit, "config.json")); err != nil {
		t.Fatal(err)
	}
	if l, _ := Status(dir, e); l.State != Partial {
		t.Errorf("config only: %+v", l)
	}
	os.WriteFile(filepath.Join(repo, "blobs", "w"), []byte("weights"), 0o644)
	if runtime.GOOS == "windows" {
		if err := os.Link(filepath.Join(repo, "blobs", "w"), filepath.Join(repo, "snapshots", testCommit, "model.safetensors")); err != nil {
			t.Fatal(err)
		}
	} else if err := os.Symlink("../../blobs/w", filepath.Join(repo, "snapshots", testCommit, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	if l, _ := Status(dir, e); l.State != Downloaded || l.Bytes != 9 {
		t.Errorf("with weights: %+v", l)
	}
}

func TestAuxRepos(t *testing.T) {
	h := newFakeHub(t)
	g2p := newFakeHub(t)
	g2p.repo = "acme/g2p"
	g2p.files = map[string][]byte{"config.json": []byte(`{"g2p":1}`), "g2p.safetensors": bytes.Repeat([]byte{3}, 3000)}
	g2p.lfs = map[string]bool{"g2p.safetensors": true}
	both := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/acme/g2p/") {
			g2p.serve(w, r)
		} else {
			h.serve(w, r)
		}
	}))
	defer both.Close()
	c := New(Options{CacheDir: t.TempDir(), Endpoints: []string{both.URL}, Official: both.URL, ChunkSize: 1024, ReserveBytes: -1})
	c.progressEvery = 0
	e := h.entry()
	e.Aux = []Aux{{Repo: "acme/g2p"}}
	ctx := context.Background()
	p, err := c.Plan(ctx, e)
	if err != nil || p.TotalBytes != sumSizes(h.files)+sumSizes(g2p.files) || len(p.Files) != len(h.files) {
		t.Fatalf("plan %+v %v", p, err)
	}
	var last Progress
	if err := c.Download(ctx, e, func(pr Progress) {
		if pr.Bytes < last.Bytes || pr.Repo != testRepo {
			t.Errorf("progress went from %+v to %+v", last, pr)
		}
		last = pr
	}); err != nil {
		t.Fatal(err)
	}
	if last.Bytes != p.TotalBytes {
		t.Errorf("last progress %+v", last)
	}
	l, _ := c.Status(ctx, e)
	if l.State != Downloaded || l.Bytes != p.TotalBytes {
		t.Errorf("status %+v", l)
	}
	if bad, err := c.Verify(ctx, e); err != nil || len(bad) != 0 {
		t.Errorf("verify %v %v", bad, err)
	}
	// Without its aux repo the model is not usable.
	os.RemoveAll(RepoDir(c.opts.CacheDir, "acme/g2p"))
	l, _ = c.Status(ctx, e)
	if l.State != Partial || !slices.Contains(l.Missing, "acme/g2p:") {
		t.Errorf("status without aux %+v", l)
	}
	if bad, _ := c.Verify(ctx, e); !slices.Equal(bad, []string{"acme/g2p:"}) {
		t.Errorf("verify without aux %v", bad)
	}
	// Deleting the model keeps aux repos without a Store (shared, small).
	if err := c.Download(ctx, e, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := c.Delete(ctx, e, models.ThisApp); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(RepoDir(c.opts.CacheDir, "acme/g2p")); err != nil {
		t.Error("aux repo deleted")
	}
}

func TestCleanEngineCopies(t *testing.T) {
	dir := t.TempDir()
	if got, err := CleanEngineCopies(dir); err != nil || got != nil {
		t.Fatalf("no dir: %v %v", got, err)
	}
	copies := EngineCopiesDir(dir)
	for _, d := range []string{"mlx-community_Empty", "mlx-community_Full", "mlx-community_New"} {
		os.MkdirAll(filepath.Join(copies, d), 0o755)
	}
	os.WriteFile(filepath.Join(copies, "mlx-community_Full", "config.json"), []byte("{}"), 0o644)
	long := time.Now().Add(-time.Hour)
	os.Chtimes(filepath.Join(copies, "mlx-community_Empty"), long, long)
	os.Chtimes(filepath.Join(copies, "mlx-community_Full"), long, long)
	got, err := CleanEngineCopies(dir)
	if err != nil || !slices.Equal(got, []string{"mlx-community_Empty"}) {
		t.Fatalf("removed %v %v", got, err)
	}
	if _, err := os.Stat(filepath.Join(copies, "mlx-community_Full", "config.json")); err != nil {
		t.Error("removed a full copy")
	}
	if _, err := os.Stat(filepath.Join(copies, "mlx-community_New")); err != nil {
		t.Error("removed a new empty copy: another app's engine may be filling it")
	}
}
