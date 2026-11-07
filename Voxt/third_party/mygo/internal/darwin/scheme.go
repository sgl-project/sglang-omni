//go:build darwin

package darwin

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"strings"
	"unsafe"

	"github.com/ebitengine/purego/objc"

	"github.com/egoist/mygo/internal/platform"
)

var theSchemeHandler id

func schemeHandler() id {
	if theSchemeHandler == 0 {
		theSchemeHandler = alloc("MyGoSchemeHandler")
	}
	return theSchemeHandler
}

// schemeTask is the platform.SchemeResponder of a WKURLSchemeTask. All
// methods run on the main thread.
type schemeTask struct {
	task      id
	cancel    context.CancelFunc
	stopped   bool
	responded bool
	done      bool
}

// schemeTasks maps live WKURLSchemeTask objects to their state.
var schemeTasks = map[id]*schemeTask{}

func registerSchemeHandler() {
	classDef("MyGoSchemeHandler", "NSObject", []string{"WKURLSchemeHandler"}, []objc.MethodDef{
		method("webView:startURLSchemeTask:", func(self id, _ objc.SEL, web, task id) {
			w := theBackend.byWebView[web]
			ctx, cancel := context.WithCancel(context.Background())
			st := &schemeTask{task: retain(task), cancel: cancel}
			schemeTasks[task] = st
			if w == nil || w.closed {
				st.Respond(http.StatusServiceUnavailable, nil)
				st.Finish()
				return
			}
			req := readRequest(task)
			req.Context = ctx
			req.Responder = st
			w.h.SchemeRequest(req)
		}),
		method("webView:stopURLSchemeTask:", func(self id, _ objc.SEL, web, task id) {
			if st := schemeTasks[task]; st != nil {
				st.stopped = true
				st.cancel()
				st.release()
			}
		}),
	})
}

func readRequest(task id) *platform.SchemeRequest {
	header := http.Header{}
	req := &platform.SchemeRequest{Header: header}
	withPool(func() {
		r := send(task, "request")
		req.URL = goString(send(send(r, "URL"), "absoluteString"))
		req.Method = goString(send(r, "HTTPMethod"))
		if req.Method == "" {
			req.Method = http.MethodGet
		}
		fields := send(r, "allHTTPHeaderFields")
		for _, k := range arrayItems(send(fields, "allKeys")) {
			header.Add(goString(k), goString(send(fields, "objectForKey:", uintptr(k))))
		}
		if data := send(r, "HTTPBody"); data != 0 {
			req.Body = io.NopCloser(bytes.NewReader(goBytes(data)))
		} else if stream := send(r, "HTTPBodyStream"); stream != 0 {
			req.Body = io.NopCloser(bytes.NewReader(readStream(stream)))
		}
	})
	return req
}

// readStream drains an NSInputStream.
func readStream(stream id) []byte {
	send(stream, "open")
	defer send(stream, "close")
	var out []byte
	buf := make([]byte, 64<<10)
	for {
		n := sendInt(stream, "read:maxLength:", uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
		if n <= 0 {
			return out
		}
		out = append(out, buf[:n]...)
	}
}

func (st *schemeTask) release() {
	if st.done {
		return
	}
	st.done = true
	delete(schemeTasks, st.task)
	release(st.task)
}

// schemeDispositions records the Content-Disposition of the latest custom
// scheme responses by URL, which the navigation policy needs to recognize
// attachments.
var schemeDispositions = map[string]string{}

func (st *schemeTask) Respond(status int, h map[string][]string) {
	header := http.Header(h)
	if st.stopped || st.done || st.responded {
		return
	}
	st.responded = true
	withPool(func() {
		if len(schemeDispositions) > 64 {
			clear(schemeDispositions)
		}
		schemeDispositions[goString(send(send(send(st.task, "request"), "URL"), "absoluteString"))] = header.Get("Content-Disposition")
		fields := send(class("NSMutableDictionary"), "dictionary")
		for k, vs := range header {
			send(fields, "setObject:forKey:", uintptr(nsString(strings.Join(vs, ", "))), uintptr(nsString(k)))
		}
		url := send(send(st.task, "request"), "URL")
		resp := send(send(class("NSHTTPURLResponse"), "alloc"), "initWithURL:statusCode:HTTPVersion:headerFields:",
			uintptr(url), uintptr(status), uintptr(nsString("HTTP/1.1")), uintptr(fields))
		send(st.task, "didReceiveResponse:", uintptr(resp))
		release(resp)
	})
}

func (st *schemeTask) Write(p []byte) {
	if st.stopped || st.done || len(p) == 0 {
		return
	}
	if !st.responded {
		st.Respond(http.StatusOK, nil)
	}
	withPool(func() { send(st.task, "didReceiveData:", uintptr(nsData(p))) })
}

func (st *schemeTask) Finish() {
	if st.stopped || st.done {
		return
	}
	if !st.responded {
		st.Respond(http.StatusOK, nil)
	}
	send(st.task, "didFinish")
	st.release()
}

func (st *schemeTask) Fail(err error) {
	if st.stopped || st.done {
		return
	}
	withPool(func() {
		info := send(class("NSMutableDictionary"), "dictionary")
		send(info, "setObject:forKey:", uintptr(nsString(err.Error())), uintptr(nsString("NSLocalizedDescription")))
		nserr := send(class("NSError"), "errorWithDomain:code:userInfo:", uintptr(nsString("mygo")), ^uintptr(0), uintptr(info))
		send(st.task, "didFailWithError:", uintptr(nserr))
	})
	st.release()
}
