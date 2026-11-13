package conformance

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
)

// FakeHub is a local Hugging Face endpoint serving fixture repos: the
// model info API (/api/models/<repo>/revision/<rev>) and file downloads
// (/<repo>/resolve/<rev>/<path>, with Range). It counts the bytes served
// per file and can pause a file after some bytes.
type FakeHub struct {
	URL string
	srv *httptest.Server

	mu     sync.Mutex
	repos  map[string]Repo // "repo@rev" and "repo@main"
	served map[string]int64
	pause  map[string]*pausePoint
}

type pausePoint struct {
	after       int64
	reached     chan struct{}
	release     chan struct{}
	once        sync.Once // closes release
	reachedOnce sync.Once // closes reached
}

// NewFakeHub starts a fake endpoint serving repos; Close it when done.
func NewFakeHub(repos ...Repo) *FakeHub {
	h := &FakeHub{repos: map[string]Repo{}, served: map[string]int64{}, pause: map[string]*pausePoint{}}
	for _, r := range repos {
		h.repos[r.Ref.String()] = r
		h.repos[r.Ref.Repo+"@main"] = r
	}
	h.srv = httptest.NewServer(http.HandlerFunc(h.serve))
	h.URL = h.srv.URL
	return h
}

// Close stops the server (releasing paused requests).
func (h *FakeHub) Close() {
	h.mu.Lock()
	for _, p := range h.pause {
		p.once.Do(func() { close(p.release) })
	}
	h.mu.Unlock()
	h.srv.CloseClientConnections()
	h.srv.Close()
}

// Served is the bytes of file bodies served so far, all files.
func (h *FakeHub) Served() int64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	var n int64
	for _, v := range h.served {
		n += v
	}
	return n
}

// ServedOf is the bytes served of one file of a repo.
func (h *FakeHub) ServedOf(repo, path string) int64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.served[repo+"/"+path]
}

// Pause makes the next responses of repo/path stop after `after` bytes of
// the file have been served in total, until Release. reached is closed
// when the pause point is hit.
func (h *FakeHub) Pause(repo, path string, after int64) (reached <-chan struct{}, release func()) {
	p := &pausePoint{after: after, reached: make(chan struct{}), release: make(chan struct{})}
	h.mu.Lock()
	h.pause[repo+"/"+path] = p
	h.mu.Unlock()
	return p.reached, func() { p.once.Do(func() { close(p.release) }) }
}

func (h *FakeHub) serve(w http.ResponseWriter, r *http.Request) {
	path := strings.TrimPrefix(r.URL.Path, "/")
	if rest, ok := strings.CutPrefix(path, "api/models/"); ok {
		repo, rev, ok := strings.Cut(rest, "/revision/")
		if !ok {
			http.NotFound(w, r)
			return
		}
		h.info(w, repo, rev)
		return
	}
	// <owner>/<name>/resolve/<rev>/<path>
	parts := strings.SplitN(path, "/", 5)
	if len(parts) < 5 || parts[2] != "resolve" {
		http.NotFound(w, r)
		return
	}
	h.file(w, r, parts[0]+"/"+parts[1], parts[3], parts[4])
}

func (h *FakeHub) info(w http.ResponseWriter, repo, rev string) {
	h.mu.Lock()
	rp, ok := h.repos[repo+"@"+rev]
	h.mu.Unlock()
	if !ok {
		http.Error(w, `{"error":"Repository not found"}`, http.StatusNotFound)
		return
	}
	type lfs struct {
		SHA256 string `json:"sha256"`
		Size   int64  `json:"size"`
	}
	type sibling struct {
		Path   string `json:"rfilename"`
		BlobID string `json:"blobId"`
		Size   int64  `json:"size"`
		LFS    *lfs   `json:"lfs,omitempty"`
	}
	out := struct {
		ID       string    `json:"id"`
		SHA      string    `json:"sha"`
		Gated    bool      `json:"gated"`
		Siblings []sibling `json:"siblings"`
	}{ID: repo, SHA: rp.Ref.Revision}
	for _, f := range rp.Files {
		s := sibling{Path: f.Path, Size: int64(len(f.Content)), BlobID: f.BlobID()}
		if f.LFS {
			s.LFS = &lfs{SHA256: f.BlobID(), Size: int64(len(f.Content))}
			// The Hub's blobId of an LFS file is the git id of its pointer.
			s.BlobID = fmt.Sprintf("%040x", len(f.Content))
		}
		out.Siblings = append(out.Siblings, s)
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}

func (h *FakeHub) file(w http.ResponseWriter, r *http.Request, repo, rev, path string) {
	h.mu.Lock()
	rp, ok := h.repos[repo+"@"+rev]
	h.mu.Unlock()
	if !ok {
		http.NotFound(w, r)
		return
	}
	var content []byte
	found := false
	for _, f := range rp.Files {
		if f.Path == path {
			content, found = f.Content, true
		}
	}
	if !found {
		http.NotFound(w, r)
		return
	}
	from, to := int64(0), int64(len(content))
	status := http.StatusOK
	if rg := r.Header.Get("Range"); strings.HasPrefix(rg, "bytes=") {
		a, b, _ := strings.Cut(strings.TrimPrefix(rg, "bytes="), "-")
		f, err1 := strconv.ParseInt(a, 10, 64)
		if err1 != nil || f >= int64(len(content)) {
			w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
			return
		}
		from = f
		if b != "" {
			if t, err := strconv.ParseInt(b, 10, 64); err == nil && t+1 < to {
				to = t + 1
			}
		}
		status = http.StatusPartialContent
		w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", from, to-1, len(content)))
	}
	w.Header().Set("Content-Length", strconv.FormatInt(to-from, 10))
	w.Header().Set("Accept-Ranges", "bytes")
	w.WriteHeader(status)
	key := repo + "/" + path
	const step = 32 << 10
	for off := from; off < to; {
		n := min(int64(step), to-off)
		h.mu.Lock()
		p := h.pause[key]
		if p != nil && h.served[key]+n > p.after {
			n = max(p.after-h.served[key], 0)
		}
		h.mu.Unlock()
		if n == 0 && p != nil {
			p.reachedOnce.Do(func() { close(p.reached) })
			select {
			case <-p.release:
				h.mu.Lock()
				if h.pause[key] == p {
					delete(h.pause, key)
				}
				h.mu.Unlock()
				continue
			case <-r.Context().Done():
				return
			}
		}
		if _, err := w.Write(content[off : off+n]); err != nil {
			return
		}
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		h.mu.Lock()
		h.served[key] += n
		h.mu.Unlock()
		off += n
	}
}
