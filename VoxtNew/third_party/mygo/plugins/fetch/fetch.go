// Package fetch is the fetch plugin of MyGo: pages make HTTP requests from
// Go with the fetch of its JavaScript package, @mygo-plugins/fetch, which
// takes and returns the standard Request options and Response. Requests
// made this way are not subject to CORS, may set any header (Origin,
// Cookie, User-Agent, ...) and read every response header but Set-Cookie,
// which a Response cannot hold.
//
//	mygo.Use(fetch.Plugin)
//
// and in the frontend:
//
//	import { fetch } from "@mygo-plugins/fetch";
//
//	const res = await fetch("https://api.example.com/items", {
//	  headers: { authorization: `Bearer ${token}` },
//	});
//	const items = await res.json();
//
// The response body streams to the page as Go reads it, and aborting the
// request's signal cancels it in Go. URLs that are not http or https, such
// as the app's own pages, go to the webview's fetch.
package fetch

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/egoist/mygo"
)

// Options configure the plugin.
type Options struct {
	// Client makes the requests; nil means http.DefaultClient. Its
	// CheckRedirect limits the redirects followed when the request's
	// redirect mode is "follow".
	Client *http.Client
	// Allow, when not nil, decides whether pages may make a request; a
	// request it refuses rejects with an error.
	Allow func(r *http.Request) bool
}

// Plugin is the plugin with the default options.
var Plugin = New(Options{})

// New returns the plugin with options.
func New(opts Options) mygo.Plugin {
	if opts.Client == nil {
		opts.Client = http.DefaultClient
	}
	return mygo.Plugin{Name: "fetch", Service: &service{opts}}
}

type service struct{ opts Options }

// request is what the page asks for.
type request struct {
	URL      string      `json:"url"`
	Method   string      `json:"method"`
	Headers  [][2]string `json:"headers"`
	Body     []byte      `json:"body"`
	Redirect string      `json:"redirect"`
}

// part is the response: its head first, then chunks of its body.
type part struct {
	Status     int         `json:"status,omitzero"`
	StatusText string      `json:"statusText,omitzero"`
	URL        string      `json:"url,omitzero"`
	Redirected bool        `json:"redirected,omitzero"`
	Headers    [][2]string `json:"headers,omitzero"`
	Chunk      []byte      `json:"chunk,omitzero"`
}

// Fetch makes the request and streams the response to the page.
func (s *service) Fetch(ctx context.Context, req request, parts *mygo.Channel[part]) error {
	return s.fetch(ctx, req, parts.Send)
}

// chunkSize is the most body bytes a part holds.
const chunkSize = 64 << 10

func (s *service) fetch(ctx context.Context, req request, send func(part) error) error {
	r, err := http.NewRequestWithContext(ctx, req.Method, req.URL, nil)
	if err != nil {
		return err
	}
	if r.URL.Scheme != "http" && r.URL.Scheme != "https" {
		return fmt.Errorf("unsupported URL scheme %q", r.URL.Scheme)
	}
	if req.Body != nil {
		r.Body = io.NopCloser(bytes.NewReader(req.Body))
		r.ContentLength = int64(len(req.Body))
		body := req.Body
		r.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(body)), nil }
	}
	for _, h := range req.Headers {
		switch http.CanonicalHeaderKey(h[0]) {
		case "Host":
			r.Host = h[1]
		case "Content-Length": // the body's
		default:
			r.Header.Add(h[0], h[1])
		}
	}
	if s.opts.Allow != nil && !s.opts.Allow(r) {
		return fmt.Errorf("the app does not allow requests to %s", r.URL.Redacted())
	}

	client := *s.opts.Client
	redirected := false
	check := client.CheckRedirect
	client.CheckRedirect = func(next *http.Request, via []*http.Request) error {
		switch req.Redirect {
		case "manual":
			return http.ErrUseLastResponse
		case "error":
			return errors.New("redirected with redirect mode \"error\"")
		}
		if check != nil {
			if err := check(next, via); err != nil {
				return err
			}
		} else if len(via) >= 20 { // as browsers
			return errors.New("too many redirects")
		}
		redirected = true
		return nil
	}
	res, err := client.Do(r)
	if err != nil {
		return err
	}
	defer res.Body.Close()

	head := part{
		Status:     res.StatusCode,
		StatusText: strings.TrimSpace(strings.TrimPrefix(res.Status, fmt.Sprint(res.StatusCode))),
		URL:        res.Request.URL.String(),
		Redirected: redirected,
		Headers:    [][2]string{},
	}
	for k, vs := range res.Header {
		k = strings.ToLower(k)
		for _, v := range vs {
			head.Headers = append(head.Headers, [2]string{k, v})
		}
	}
	if err := send(head); err != nil {
		return err
	}
	buf := make([]byte, chunkSize)
	for {
		// What arrived goes out at once, for streamed responses.
		n, err := res.Body.Read(buf)
		if n > 0 {
			// Send encodes the chunk before returning, so buf can be reused.
			if err := send(part{Chunk: buf[:n]}); err != nil {
				return err
			}
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
	}
}
