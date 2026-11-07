// Package websocket is the WebSocket plugin of MyGo: pages open WebSocket
// connections from Go with the WebSocket class of its JavaScript package,
// @mygo-plugins/websocket, which has the API of the browser's. Unlike the
// browser's, it may send any header with the opening handshake, such as
// Authorization or Cookie, and is not subject to the page's Content
// Security Policy.
//
//	mygo.Use(websocket.Plugin)
//
// and in the frontend:
//
//	import { WebSocket } from "@mygo-plugins/websocket";
//
//	const ws = new WebSocket("wss://example.com/live", [], {
//	  headers: { authorization: `Bearer ${token}` },
//	});
//	ws.onmessage = (e) => console.log(e.data);
//
// Connections close when their page navigates away or its window closes.
package websocket

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/egoist/mygo"
)

// Options configure the plugin.
type Options struct {
	// Client makes the opening handshakes; nil means http.DefaultClient.
	// Its Timeout and CheckRedirect are ignored: connections last, and
	// redirects fail them, as in browsers.
	Client *http.Client
	// Allow, when not nil, decides whether pages may open a connection
	// with the handshake request; a connection it refuses fails.
	Allow func(r *http.Request) bool
	// MaxMessageSize is the most bytes a received message may have, 64 MiB
	// when zero. A bigger message fails the connection.
	MaxMessageSize int64
	// CloseTimeout is how long a connection closed by the page waits for
	// the server to finish the closing handshake, 5 seconds when zero.
	CloseTimeout time.Duration
}

// Plugin is the plugin with the default options.
var Plugin = New(Options{})

// New returns the plugin with options.
func New(opts Options) mygo.Plugin {
	if opts.Client == nil {
		opts.Client = http.DefaultClient
	}
	if opts.MaxMessageSize <= 0 {
		opts.MaxMessageSize = 64 << 20
	}
	if opts.CloseTimeout <= 0 {
		opts.CloseTimeout = 5 * time.Second
	}
	return mygo.Plugin{Name: "websocket", Service: &service{opts: opts, conns: map[key]*conn{}}}
}

type service struct {
	opts  Options
	mu    sync.Mutex
	conns map[key]*conn
}

// key identifies a connection: the page (a *mygo.Page) chooses ids for
// itself. A window may show several pages: those of its web views.
type key struct {
	page any
	id   string
}

type connectOptions struct {
	ID        string      `json:"id"`
	URL       string      `json:"url"`
	Protocols []string    `json:"protocols"`
	Headers   [][2]string `json:"headers"`
}

// event is what happens to a connection, for the page.
type event struct {
	Type     string `json:"type"` // open, text, binary or close
	Protocol string `json:"protocol,omitzero"`
	Text     string `json:"text,omitzero"`
	Data     []byte `json:"data,omitzero"`
	Code     int    `json:"code,omitzero"`
	Reason   string `json:"reason,omitzero"`
	Clean    bool   `json:"clean,omitzero"`
	// Error tells why the connection failed.
	Error string `json:"error,omitzero"`
}

// message is what the page sends.
type message struct {
	Type   string `json:"type"` // text, binary, close or skip
	Text   string `json:"text"`
	Data   []byte `json:"data"`
	Code   int    `json:"code"`
	Reason string `json:"reason"`
}

// Connect opens the connection and streams its events to the page until
// it closes.
func (s *service) Connect(ctx context.Context, opts connectOptions, events *mygo.Channel[event]) {
	s.connect(ctx, mygo.CallerPage(ctx), opts, events.Send)
}

// Send sends the seq-th message of the page on the connection id, after
// those before it.
func (s *service) Send(ctx context.Context, id string, seq int64, m message) error {
	return s.send(mygo.CallerPage(ctx), id, seq, m)
}

func (s *service) connect(ctx context.Context, page any, opts connectOptions, send func(event) error) {
	k := key{page, opts.ID}
	c, protocol, err := dial(ctx, s.opts.Client, opts.URL, opts.Protocols, opts.Headers, s.opts.Allow, s.opts.MaxMessageSize)
	if err == nil {
		s.mu.Lock()
		if s.conns[k] != nil {
			err = fmt.Errorf("connection %q is already open", opts.ID)
			c.terminate()
		} else {
			s.conns[k] = c
		}
		s.mu.Unlock()
	}
	if err != nil {
		send(event{Type: "close", Code: closeAbnormal, Error: err.Error()})
		return
	}
	defer func() {
		s.mu.Lock()
		delete(s.conns, k)
		s.mu.Unlock()
		c.terminate()
	}()
	// The page going away closes the connection.
	stop := context.AfterFunc(ctx, func() {
		time.AfterFunc(time.Second, c.terminate)
		c.writeClose(closeGoingAway, "")
		c.terminate()
	})
	defer stop()

	if send(event{Type: "open", Protocol: protocol}) != nil {
		return
	}
	for {
		op, data, err := c.readMessage()
		if err != nil {
			ev := event{Type: "close", Code: closeAbnormal}
			if f, ok := errors.AsType[*failure](err); ok {
				c.writeClose(f.code, "")
				ev.Error = err.Error()
			} else if !c.closing() {
				// Not the server missing the closing handshake the page
				// started.
				ev.Error = err.Error()
			}
			send(ev)
			return
		}
		switch op {
		case opText:
			err = send(event{Type: "text", Text: string(data)})
		case opBinary:
			err = send(event{Type: "binary", Data: data})
		case opClose:
			code, reason, perr := parseClose(data)
			if perr != nil {
				c.writeClose(perr.(*failure).code, "")
				send(event{Type: "close", Code: closeAbnormal, Error: perr.Error()})
				return
			}
			// Echo the code, unless the page closed first.
			c.writeClose(code, "")
			send(event{Type: "close", Code: code, Reason: reason, Clean: true})
			return
		}
		if err != nil {
			return // the page went away
		}
	}
}

func (s *service) send(page any, id string, seq int64, m message) error {
	s.mu.Lock()
	c := s.conns[key{page, id}]
	s.mu.Unlock()
	if c == nil {
		return errClosed
	}
	return c.inTurn(seq, func() error {
		switch m.Type {
		case "text":
			return c.writeFrame(opText, []byte(m.Text))
		case "binary":
			return c.writeFrame(opBinary, m.Data)
		case "skip": // a message the page could not read
			return nil
		case "close":
			if m.Code != closeNoStatus && m.Code != closeNormal && (m.Code < 3000 || m.Code > 4999) {
				return fmt.Errorf("invalid close code %d", m.Code)
			}
			if len(m.Reason) > 123 {
				return errors.New("close reason over 123 bytes")
			}
			return c.close(m.Code, m.Reason, s.opts.CloseTimeout)
		}
		return fmt.Errorf("unknown message type %q", m.Type)
	})
}
