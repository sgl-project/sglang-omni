//go:build !js

package mygo

import (
	"fmt"
	"io/fs"
	"log"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path"
	"runtime"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"

	"github.com/egoist/mygo/internal/platform"
)

// ProtocolModule registers custom URL schemes served by Go.
type ProtocolModule struct {
	mu       sync.RWMutex
	handlers map[string]http.Handler
}

// Protocol serves custom URL schemes such as app:// from Go.
var Protocol = &ProtocolModule{handlers: map[string]http.Handler{}}

var reservedSchemes = map[string]bool{
	"http": true, "https": true, "file": true, "about": true, "data": true, "blob": true,
	"javascript": true, "ws": true, "wss": true, "ftp": true, "mailto": true, "webkit": true,
}

// Handle serves requests for scheme with an http.Handler, for example
// files generated at run time:
//
//	mygo.Protocol.Handle("thumbs", mygo.FileServer(os.DirFS(cacheDir)))
//	win.Page().LoadURL("thumbs://localhost/")
//
// Pages served this way are loaded like regular web pages: they can use
// fetch, ES modules and relative URLs. Schemes must be registered before
// the windows that use them are created. Handlers run on their own
// goroutines and responses are streamed to the page.
//
// The mygo scheme serves the app's frontend (see SetFrontend) unless it is
// handled here.
func (p *ProtocolModule) Handle(scheme string, handler http.Handler) error {
	scheme = strings.ToLower(scheme)
	if !validScheme(scheme) {
		return fmt.Errorf("mygo: invalid scheme %q", scheme)
	}
	if reservedSchemes[scheme] {
		return fmt.Errorf("mygo: scheme %q is reserved", scheme)
	}
	p.mu.Lock()
	p.handlers[scheme] = handler
	p.mu.Unlock()
	return nil
}

// HandleFunc is Handle for a handler function.
func (p *ProtocolModule) HandleFunc(scheme string, fn func(http.ResponseWriter, *http.Request)) error {
	return p.Handle(scheme, http.HandlerFunc(fn))
}

// Unhandle removes the handler of scheme. Pages requesting it get an error.
func (p *ProtocolModule) Unhandle(scheme string) {
	p.mu.Lock()
	delete(p.handlers, strings.ToLower(scheme))
	p.mu.Unlock()
}

// IsHandled reports whether scheme has a handler.
func (p *ProtocolModule) IsHandled(scheme string) bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	_, ok := p.handlers[strings.ToLower(scheme)]
	return ok
}

// schemes returns the schemes windows must handle: the registered ones and
// the frontend's.
func (p *ProtocolModule) schemes() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()
	out := make([]string, 0, len(p.handlers)+1)
	for s := range p.handlers {
		out = append(out, s)
	}
	if _, ok := p.handlers[frontendScheme]; !ok {
		out = append(out, frontendScheme)
	}
	return out
}

func validScheme(s string) bool {
	if s == "" || s[0] < 'a' || s[0] > 'z' {
		return false
	}
	for _, r := range s {
		if !(r >= 'a' && r <= 'z' || r >= '0' && r <= '9' || r == '+' || r == '-' || r == '.') {
			return false
		}
	}
	return true
}

func (h *windowHandler) SchemeRequest(req *platform.SchemeRequest) {
	Protocol.serve(h.w, req)
}

// serve runs on the main thread when the webview requests a custom scheme.
func (p *ProtocolModule) serve(w *Window, req *platform.SchemeRequest) {
	scheme, _, _ := strings.Cut(req.URL, ":")
	scheme = strings.ToLower(scheme)
	p.mu.RLock()
	h := p.handlers[scheme]
	p.mu.RUnlock()
	if h == nil && scheme == frontendScheme {
		h = frontendHandler()
	}
	go serveScheme(w, h, req)
}

func serveScheme(win *Window, h http.Handler, req *platform.SchemeRequest) {
	rw := &schemeWriter{req: req, header: http.Header{}}
	defer func() {
		if r := recover(); r != nil {
			log.Printf("mygo: panic serving %s: %v\n%s", req.URL, r, debug.Stack())
			if !rw.wroteHeader {
				rw.header = http.Header{}
				rw.WriteHeader(http.StatusInternalServerError)
			}
		}
		rw.finish()
	}()
	if h == nil {
		rw.WriteHeader(http.StatusNotFound)
		return
	}
	body := req.Body
	if body == nil {
		body = http.NoBody
	}
	r, err := http.NewRequestWithContext(req.Context, req.Method, req.URL, body)
	if err != nil {
		rw.WriteHeader(http.StatusBadRequest)
		return
	}
	r.Header = req.Header
	if r.Header == nil {
		r.Header = http.Header{}
	}
	r.RequestURI = r.URL.RequestURI()
	r.RemoteAddr = "mygo:" + strconv.Itoa(win.id)
	if cl := r.Header.Get("Content-Length"); cl != "" {
		r.ContentLength, _ = strconv.ParseInt(cl, 10, 64)
	}
	if allowCORS(win, rw.header, r) {
		rw.WriteHeader(http.StatusNoContent) // a preflight, answered here
		return
	}
	h.ServeHTTP(rw, r)
}

// allowCORS lets the app's own pages read responses of custom schemes,
// which are other origins than theirs: the frontend, another scheme, or
// the dev server fetching thumbs://localhost/ for example. It sets the
// CORS headers for those origins, which handlers may still override, and
// reports whether the request is a preflight it answered. Pages of other
// sites get no CORS headers, unless the window trusts them
// (PageOptions.TrustedOrigins).
func allowCORS(win *Window, header http.Header, r *http.Request) (preflight bool) {
	origin := r.Header.Get("Origin")
	if origin == "" || origin == "null" || !win.isTrusted(appOrigin(origin)) {
		return false
	}
	header.Set("Access-Control-Allow-Origin", origin)
	header.Set("Access-Control-Expose-Headers", "*")
	header.Add("Vary", "Origin")
	method := r.Header.Get("Access-Control-Request-Method")
	if r.Method != http.MethodOptions || method == "" {
		return false
	}
	header.Set("Access-Control-Allow-Methods", method)
	if h := r.Header.Get("Access-Control-Request-Headers"); h != "" {
		header.Set("Access-Control-Allow-Headers", h)
	}
	header.Set("Access-Control-Max-Age", "600")
	return true
}

// schemeHosts is set where the webview serves custom schemes at
// http://<scheme>.localhost (WebView2). Elsewhere, such an origin is a
// server on the loopback interface, not the app's.
var schemeHosts = runtime.GOOS == "windows"

// appOrigin maps the origin WebView2 gives pages of a custom scheme,
// http://<scheme>.localhost, back to <scheme>://localhost, as the rest of
// MyGo names it.
func appOrigin(origin string) string {
	if !schemeHosts {
		return origin
	}
	u, err := url.Parse(origin)
	if err != nil || u.Scheme != "http" || u.Port() != "" {
		return origin
	}
	scheme, ok := strings.CutSuffix(strings.ToLower(u.Hostname()), ".localhost")
	if !ok || !(scheme == frontendScheme || Protocol.IsHandled(scheme)) {
		return origin
	}
	return scheme + "://localhost"
}

// schemeWriter is the http.ResponseWriter for custom scheme requests. Body
// bytes are buffered and handed to the main thread in chunks, or written
// to backends that take them on this goroutine.
type schemeWriter struct {
	req         *platform.SchemeRequest
	header      http.Header
	status      int
	wroteHeader bool
	buf         []byte
	// chunked is set once the body filled a chunk: the next ones are
	// allocated whole.
	chunked    bool
	sentHeader bool
	finished   bool
	// body takes the body from this goroutine, if the backend can
	// (platform.SchemeBodyWriter); err is its first error.
	body platform.SchemeBodyWriter
	err  error
	// queued holds a token per chunk waiting for the main thread.
	queued chan struct{}
}

const (
	schemeChunk = 256 << 10
	// A handler faster than the main thread waits once this many chunks
	// wait for it, so a large body does not pile up in memory.
	schemeQueued = 4
)

func (w *schemeWriter) Header() http.Header { return w.header }

func (w *schemeWriter) WriteHeader(status int) {
	if w.wroteHeader {
		return
	}
	w.wroteHeader = true
	w.status = status
}

func (w *schemeWriter) Write(p []byte) (int, error) {
	if err := w.writeErr(); err != nil {
		return 0, err
	}
	if !w.wroteHeader {
		w.WriteHeader(http.StatusOK)
	}
	for n := 0; n < len(p); {
		if w.buf == nil && w.chunked {
			w.buf = make([]byte, 0, schemeChunk)
		}
		k := min(len(p)-n, schemeChunk-len(w.buf))
		w.buf = append(w.buf, p[n:n+k]...)
		n += k
		if len(w.buf) == schemeChunk {
			w.chunked = true
			w.Flush()
			if err := w.writeErr(); err != nil {
				return n, err
			}
		}
	}
	return len(p), nil
}

// writeErr reports why the body can no longer be written.
func (w *schemeWriter) writeErr() error {
	if w.err != nil {
		return w.err
	}
	return w.req.Context.Err()
}

// Flush implements http.Flusher, sending buffered data to the page.
func (w *schemeWriter) Flush() {
	if !w.wroteHeader {
		w.WriteHeader(http.StatusOK)
	}
	if !w.sentHeader {
		w.sentHeader = true
		if w.header.Get("Content-Type") == "" && w.status != http.StatusNoContent && w.status != http.StatusNotModified {
			w.header.Set("Content-Type", http.DetectContentType(w.buf))
		}
		status, header, resp := w.status, w.header.Clone(), w.req.Responder
		if body, ok := resp.(platform.SchemeBodyWriter); ok {
			// The body goes to the backend from here, once it responded.
			onMain(func() { resp.Respond(status, header) })
			w.body = body
		} else {
			postMain(func() { resp.Respond(status, header) })
		}
	}
	switch {
	case len(w.buf) == 0:
	case w.body != nil:
		if w.err == nil {
			w.err = w.body.WriteBody(w.buf)
		}
		w.buf = w.buf[:0] // written: the buffer is free again
	default:
		chunk, resp := w.buf, w.req.Responder
		w.buf = nil
		if w.queued == nil {
			w.queued = make(chan struct{}, schemeQueued)
		}
		w.queued <- struct{}{}
		if !postMain(func() { resp.Write(chunk); <-w.queued }) {
			<-w.queued
		}
	}
}

func (w *schemeWriter) finish() {
	if w.finished {
		return
	}
	w.finished = true
	w.Flush()
	resp := w.req.Responder
	postMain(resp.Finish)
}

// FileServer serves files from fsys and falls back to index.html for
// unknown paths without an extension, so client side routers work. Files
// are served with their content type and support range requests.
func FileServer(fsys fs.FS) http.Handler {
	files := http.FileServerFS(fsys)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		name := strings.TrimPrefix(path.Clean("/"+r.URL.Path), "/")
		if name == "" {
			name = "."
		}
		if _, err := fs.Stat(fsys, name); err != nil && path.Ext(name) == "" {
			index, err := fs.ReadFile(fsys, "index.html")
			if err != nil {
				http.NotFound(w, r)
				return
			}
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			_, _ = w.Write(index)
			return
		}
		files.ServeHTTP(w, r)
	})
}

// frontendHandler serves mygo:// when the app has no handler of its own.
func frontendHandler() http.Handler {
	frontend.Lock()
	fsys := frontend.fsys
	frontend.Unlock()
	if fsys == nil && IsDev() {
		// `mygo dev` without a dev server: the files on disk.
		if dir := os.Getenv("MYGO_FRONTEND_DIST"); dir != "" {
			fsys = os.DirFS(dir)
		}
	}
	if fsys == nil {
		return http.HandlerFunc(noFrontend)
	}
	return FileServer(fsys)
}

func noFrontend(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(http.StatusNotFound)
	_, _ = w.Write([]byte(`<!doctype html>
<meta charset="utf-8">
<title>No frontend</title>
<style>body{font:15px/1.5 system-ui,sans-serif;margin:3em;color:#444}code{font-size:90%}</style>
<h1>No frontend</h1>
<p>Run the app with <code>mygo dev</code>, which loads <code>devUrl</code> from mygo.config.ts,
or build it with <code>mygo build</code>, which embeds <code>frontendDist</code>.</p>
`))
}

func (h *windowHandler) SchemeDownload(rawURL string) { go downloadScheme(h, rawURL) }

// downloadScheme serves a URL of a custom scheme into a download.
func downloadScheme(h *windowHandler, rawURL string) {
	scheme, _, _ := strings.Cut(rawURL, ":")
	scheme = strings.ToLower(scheme)
	Protocol.mu.RLock()
	handler := Protocol.handlers[scheme]
	Protocol.mu.RUnlock()
	if handler == nil && scheme == frontendScheme {
		handler = frontendHandler()
	}
	finish := func(path string, err error) {
		postMain(func() { h.DownloadFinished(rawURL, path, err) })
	}
	req, err := http.NewRequest(http.MethodGet, rawURL, nil)
	if err != nil || handler == nil {
		finish("", fmt.Errorf("mygo: cannot download %s", rawURL))
		return
	}
	rw := &downloadWriter{h: h, url: rawURL, header: http.Header{}}
	handler.ServeHTTP(rw, req)
	rw.WriteHeader(http.StatusOK) // nothing written
	if rw.file == nil {
		if rw.err != nil {
			finish("", rw.err)
		}
		return // canceled
	}
	if err := rw.file.Close(); rw.err == nil {
		rw.err = err
	}
	if rw.err != nil {
		os.Remove(rw.path)
	}
	finish(rw.path, rw.err)
}

// downloadWriter writes a response into the file OnWillDownload chose.
type downloadWriter struct {
	h      *windowHandler
	url    string
	header http.Header
	wrote  bool
	path   string
	file   *os.File
	err    error
}

func (d *downloadWriter) Header() http.Header { return d.header }

func (d *downloadWriter) WriteHeader(status int) {
	if d.wrote {
		return
	}
	d.wrote = true
	if status >= 400 {
		d.err = fmt.Errorf("mygo: downloading %s: %d %s", d.url, status, http.StatusText(status))
		return
	}
	name := ""
	if _, params, err := mime.ParseMediaType(d.header.Get("Content-Disposition")); err == nil {
		name = params["filename"]
	}
	if name == "" {
		if u, err := url.Parse(d.url); err == nil {
			name = path.Base(u.Path)
		}
	}
	d.path = onMainValue(func() string { return d.h.DownloadStarted(d.url, name) })
	if d.path == "" {
		return
	}
	d.file, d.err = os.Create(d.path)
	if d.err != nil {
		d.file = nil
	}
}

func (d *downloadWriter) Write(b []byte) (int, error) {
	d.WriteHeader(http.StatusOK)
	if d.file == nil || d.err != nil {
		return len(b), nil // canceled: drain
	}
	n, err := d.file.Write(b)
	if err != nil {
		d.err = err
	}
	return n, err
}
