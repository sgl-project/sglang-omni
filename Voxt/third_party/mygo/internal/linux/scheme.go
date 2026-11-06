//go:build linux && (amd64 || arm64)

package linux

import (
	"bytes"
	"context"
	"errors"
	"io"
	"net/http"
	"strconv"
	"syscall"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/egoist/mygo/internal/platform"
)

var (
	cbScheme        ptr
	cbHeaderForeach ptr
	headerSinks     = map[ptr]http.Header{}
)

func init() {
	cbScheme = purego.NewCallback(func(req, data ptr) { theBackend.serveScheme(req) })
	cbHeaderForeach = purego.NewCallback(func(name, value, data ptr) {
		if h := headerSinks[data]; h != nil {
			h.Add(goStr(name), goStr(value))
		}
	})
}

func (b *Backend) registerScheme(scheme string) {
	if b.schemes[scheme] {
		return
	}
	b.schemes[scheme] = true
	ctx := webkitWebContextGetDefault()
	webkitWebContextRegisterURIScheme(ctx, cs(scheme), cbScheme, 0, 0)
	sec := webkitWebContextGetSecurityManager(ctx)
	webkitSecurityManagerRegisterSecure(sec, cs(scheme))
	webkitSecurityManagerRegisterCORS(sec, cs(scheme))
}

func (b *Backend) serveScheme(req ptr) {
	gObjectRef(req)
	ctx, cancel := context.WithCancel(context.Background())
	t := &schemeTask{req: req, cancel: cancel, fd: -1}
	w := b.byWebView[webkitURISchemeRequestGetWebView(req)]
	if w == nil || w.closed {
		t.Fail(context.Canceled)
		return
	}
	r := &platform.SchemeRequest{
		Context:   ctx,
		Method:    http.MethodGet,
		URL:       goStr(webkitURISchemeRequestGetURI(req)),
		Header:    http.Header{},
		Responder: t,
	}
	if webkitURISchemeRequestGetHTTPMethod != nil {
		if m := goStr(webkitURISchemeRequestGetHTTPMethod(req)); m != "" {
			r.Method = m
		}
	}
	if webkitURISchemeRequestGetHTTPHeaders != nil {
		if hdrs := webkitURISchemeRequestGetHTTPHeaders(req); hdrs != 0 {
			key := ptr(len(headerSinks) + 1)
			headerSinks[key] = r.Header
			soupMessageHeadersForeach(hdrs, cbHeaderForeach, key)
			delete(headerSinks, key)
		}
	}
	if webkitURISchemeRequestGetHTTPBody != nil {
		if stream := webkitURISchemeRequestGetHTTPBody(req); stream != 0 {
			r.Body = io.NopCloser(bytes.NewReader(readStream(stream)))
			gObjectUnref(stream)
		}
	}
	w.h.SchemeRequest(r)
}

func readStream(stream ptr) []byte {
	var out []byte
	buf := make([]byte, 64<<10)
	for {
		var n uintptr
		var gerr ptr
		ok := gInputStreamReadAll(stream, unsafe.Pointer(&buf[0]), uintptr(len(buf)), &n, 0, &gerr)
		out = append(out, buf[:n]...)
		if gerr != 0 {
			gErrorFree(gerr)
		}
		if !ok || n < uintptr(len(buf)) {
			return out
		}
	}
}

// schemeTask streams a response to WebKit through a pipe, which WebKit
// reads on the main loop. The goroutine serving the request writes the
// body into it (WriteBody), waiting while it is full, so the main loop
// never blocks and a response WebKit reads slowly does not pile up in
// memory. The other methods run on the main thread.
type schemeTask struct {
	req       ptr
	cancel    context.CancelFunc
	responded bool
	done      bool
	// fd is the write end of the pipe, -1 before Respond and once closed.
	fd int
}

var errNoResponse = errors.New("mygo: the response was not started")

func (t *schemeTask) Respond(status int, h map[string][]string) {
	header := http.Header(h)
	if t.done || t.responded {
		return
	}
	var fds [2]int
	if err := syscall.Pipe2(fds[:], syscall.O_CLOEXEC); err != nil {
		t.Fail(err)
		return
	}
	t.responded = true
	stream := gUnixInputStreamNew(int32(fds[0]), true)
	length := int64(-1)
	if cl, err := strconv.ParseInt(header.Get("Content-Length"), 10, 64); err == nil {
		length = cl
	}
	if webkitURISchemeRequestFinishWithResponse != nil {
		resp := webkitURISchemeResponseNew(stream, length)
		webkitURISchemeResponseSetStatus(resp, uint32(status), nil)
		if ct := header.Get("Content-Type"); ct != "" {
			webkitURISchemeResponseSetContentType(resp, cs(ct))
		}
		hdrs := soupMessageHeadersNew(1) // SOUP_MESSAGE_HEADERS_RESPONSE
		for k, vs := range header {
			for _, v := range vs {
				soupMessageHeadersAppend(hdrs, cs(k), cs(v))
			}
		}
		webkitURISchemeResponseSetHTTPHeaders(resp, hdrs)
		webkitURISchemeRequestFinishWithResponse(t.req, resp)
		gObjectUnref(resp)
	} else {
		// WebKitGTK before 2.36 cannot send status codes or headers.
		webkitURISchemeRequestFinish(t.req, stream, length, optCS(header.Get("Content-Type")))
	}
	gObjectUnref(stream)
	t.release()
	t.fd = fds[1]
}

// WriteBody writes to the pipe, blocking while it is full.
func (t *schemeTask) WriteBody(p []byte) error {
	if t.fd < 0 {
		return errNoResponse
	}
	for len(p) > 0 {
		n, err := syscall.Write(t.fd, p)
		if err == syscall.EINTR {
			continue
		}
		if err != nil {
			// The page no longer wants the response.
			t.cancel()
			return err
		}
		p = p[n:]
	}
	return nil
}

// Write is not used: the body goes through WriteBody.
func (t *schemeTask) Write([]byte) {}

func (t *schemeTask) Finish() {
	if t.done {
		return
	}
	if !t.responded {
		t.Respond(http.StatusOK, http.Header{})
	}
	t.done = true
	if t.fd >= 0 {
		syscall.Close(t.fd) // the end of the body
		t.fd = -1
	}
}

func (t *schemeTask) Fail(err error) {
	if t.done {
		return
	}
	if !t.responded {
		t.responded = true
		gerr := gErrorNewLiteral(gQuarkFromString(cs("mygo")), 1, cs(err.Error()))
		webkitURISchemeRequestFinishError(t.req, gerr)
		gErrorFree(gerr)
		t.release()
	}
	t.Finish()
}

func (t *schemeTask) release() {
	if t.req != 0 {
		gObjectUnref(t.req)
		t.req = 0
	}
}
