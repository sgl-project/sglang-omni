package websocket

import (
	"bufio"
	"context"
	"crypto/rand"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"slices"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

// Opcodes (RFC 6455, section 5.2).
const (
	opContinuation = 0x0
	opText         = 0x1
	opBinary       = 0x2
	opClose        = 0x8
	opPing         = 0x9
	opPong         = 0xa
)

// Close codes (RFC 6455, section 7.4.1).
const (
	closeNormal        = 1000
	closeGoingAway     = 1001
	closeProtocolError = 1002
	closeNoStatus      = 1005
	closeAbnormal      = 1006
	closeInvalidData   = 1007
	closeTooBig        = 1009
)

// failure fails the connection with a close code (RFC 6455, section 7.1.7).
type failure struct {
	code int
	msg  string
}

func (f *failure) Error() string { return f.msg }

func protocolError(format string, args ...any) error {
	return &failure{closeProtocolError, fmt.Sprintf(format, args...)}
}

var errClosed = errors.New("websocket: connection closed")

// conn is a WebSocket connection, of a client unless server is set (tests).
type conn struct {
	rwc    io.ReadWriteCloser
	br     *bufio.Reader
	server bool
	max    int64 // the most bytes a message may have

	wmu       sync.Mutex // writes
	closeSent bool

	// Messages the page sends are written in the order it sent them: next
	// is the sequence number of the next one.
	mu   sync.Mutex
	cond sync.Cond
	next int64
	done bool

	once sync.Once
}

func newConn(rwc io.ReadWriteCloser, br *bufio.Reader, server bool, max int64) *conn {
	if br == nil {
		br = bufio.NewReader(rwc)
	}
	c := &conn{rwc: rwc, br: br, server: server, max: max, next: 1}
	c.cond.L = &c.mu
	return c
}

// acceptKey is the Sec-WebSocket-Accept of key.
func acceptKey(key string) string {
	h := sha1.Sum([]byte(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"))
	return base64.StdEncoding.EncodeToString(h[:])
}

// dial opens a connection with the opening handshake, sent by client.
// It returns the connection and the subprotocol the server chose.
func dial(ctx context.Context, client *http.Client, rawURL string, protocols []string, headers [][2]string, allow func(*http.Request) bool, max int64) (*conn, string, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, "", err
	}
	switch u.Scheme {
	case "ws":
		u.Scheme = "http"
	case "wss":
		u.Scheme = "https"
	case "http", "https":
	default:
		return nil, "", fmt.Errorf("unsupported URL scheme %q", u.Scheme)
	}
	u.Fragment = ""
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, "", err
	}
	for _, h := range headers {
		if http.CanonicalHeaderKey(h[0]) == "Host" {
			req.Host = h[1]
		} else {
			req.Header.Add(h[0], h[1])
		}
	}
	var nonce [16]byte
	rand.Read(nonce[:])
	key := base64.StdEncoding.EncodeToString(nonce[:])
	req.Header.Set("Upgrade", "websocket")
	req.Header.Set("Connection", "Upgrade")
	req.Header.Set("Sec-WebSocket-Key", key)
	req.Header.Set("Sec-WebSocket-Version", "13")
	req.Header.Del("Sec-WebSocket-Extensions")
	if len(protocols) > 0 {
		req.Header.Set("Sec-WebSocket-Protocol", strings.Join(protocols, ", "))
	} else {
		req.Header.Del("Sec-WebSocket-Protocol")
	}
	if allow != nil && !allow(req) {
		return nil, "", fmt.Errorf("the app does not allow connections to %s", u.Redacted())
	}

	// The connection outlives the request: no timeout, and like browsers,
	// no redirects.
	c := *client
	c.Timeout = 0
	c.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	res, err := c.Do(req)
	if err != nil {
		return nil, "", err
	}
	fail := func(format string, args ...any) (*conn, string, error) {
		res.Body.Close()
		return nil, "", fmt.Errorf(format, args...)
	}
	if res.StatusCode != http.StatusSwitchingProtocols {
		return fail("the server answered the handshake with %s", res.Status)
	}
	if !strings.EqualFold(res.Header.Get("Upgrade"), "websocket") || !hasToken(res.Header.Get("Connection"), "upgrade") {
		return fail("the server did not upgrade the connection to WebSocket")
	}
	if res.Header.Get("Sec-WebSocket-Accept") != acceptKey(key) {
		return fail("the server answered with a wrong Sec-WebSocket-Accept")
	}
	protocol := res.Header.Get("Sec-WebSocket-Protocol")
	if protocol != "" && !slices.Contains(protocols, protocol) {
		return fail("the server chose subprotocol %q, which was not requested", protocol)
	}
	if res.Header.Get("Sec-WebSocket-Extensions") != "" {
		return fail("the server chose extensions, which were not requested")
	}
	rwc, ok := res.Body.(io.ReadWriteCloser)
	if !ok {
		return fail("the HTTP client cannot upgrade connections")
	}
	return newConn(rwc, nil, false, max), protocol, nil
}

func hasToken(header, token string) bool {
	for t := range strings.SplitSeq(header, ",") {
		if strings.EqualFold(strings.TrimSpace(t), token) {
			return true
		}
	}
	return false
}

// readFrame reads a frame.
func (c *conn) readFrame() (fin bool, op byte, payload []byte, err error) {
	var h [8]byte
	if _, err := io.ReadFull(c.br, h[:2]); err != nil {
		return false, 0, nil, err
	}
	fin, op = h[0]&0x80 != 0, h[0]&0x0f
	if h[0]&0x70 != 0 {
		return false, 0, nil, protocolError("reserved bits set in a frame")
	}
	if masked := h[1]&0x80 != 0; masked != c.server {
		return false, 0, nil, protocolError("wrong frame masking")
	}
	n := uint64(h[1] & 0x7f)
	switch n {
	case 126:
		if _, err := io.ReadFull(c.br, h[:2]); err != nil {
			return false, 0, nil, err
		}
		n = uint64(binary.BigEndian.Uint16(h[:2]))
	case 127:
		if _, err := io.ReadFull(c.br, h[:8]); err != nil {
			return false, 0, nil, err
		}
		n = binary.BigEndian.Uint64(h[:8])
		if n>>63 != 0 {
			return false, 0, nil, protocolError("invalid frame length")
		}
	}
	if op >= opClose && (!fin || n > 125) {
		return false, 0, nil, protocolError("invalid control frame")
	}
	if n > uint64(c.max) {
		return false, 0, nil, &failure{closeTooBig, fmt.Sprintf("message over %d bytes", c.max)}
	}
	var mask [4]byte
	if c.server {
		if _, err := io.ReadFull(c.br, mask[:]); err != nil {
			return false, 0, nil, err
		}
	}
	payload = make([]byte, n)
	if _, err := io.ReadFull(c.br, payload); err != nil {
		return false, 0, nil, err
	}
	if c.server {
		for i := range payload {
			payload[i] ^= mask[i%4]
		}
	}
	return fin, op, payload, nil
}

// readMessage reads a text, binary or close message, answering pings.
func (c *conn) readMessage() (op byte, data []byte, err error) {
	var msgOp byte
	for {
		fin, op, p, err := c.readFrame()
		if err != nil {
			return 0, nil, err
		}
		switch op {
		case opPing:
			if err := c.writeFrame(opPong, p); err != nil && err != errClosed {
				return 0, nil, err
			}
			continue
		case opPong:
			continue
		case opClose:
			return opClose, p, nil
		case opContinuation:
			if msgOp == 0 {
				return 0, nil, protocolError("unexpected continuation frame")
			}
		case opText, opBinary:
			if msgOp != 0 {
				return 0, nil, protocolError("new message before the last one ended")
			}
			msgOp = op
		default:
			return 0, nil, protocolError("unknown opcode %d", op)
		}
		if int64(len(data)+len(p)) > c.max {
			return 0, nil, &failure{closeTooBig, fmt.Sprintf("message over %d bytes", c.max)}
		}
		data = append(data, p...)
		if fin {
			if msgOp == opText && !utf8.Valid(data) {
				return 0, nil, &failure{closeInvalidData, "text message is not valid UTF-8"}
			}
			if data == nil {
				data = []byte{}
			}
			return msgOp, data, nil
		}
	}
}

// writeFrame writes a whole message in one frame. Nothing is written after
// a close frame.
func (c *conn) writeFrame(op byte, p []byte) error {
	c.wmu.Lock()
	defer c.wmu.Unlock()
	if c.closeSent {
		return errClosed
	}
	if op == opClose {
		c.closeSent = true
	}
	b := make([]byte, 0, 14+len(p))
	b = append(b, 0x80|op)
	var maskBit byte
	if !c.server {
		maskBit = 0x80
	}
	switch n := len(p); {
	case n <= 125:
		b = append(b, maskBit|byte(n))
	case n <= 0xffff:
		b = append(b, maskBit|126)
		b = binary.BigEndian.AppendUint16(b, uint16(n))
	default:
		b = append(b, maskBit|127)
		b = binary.BigEndian.AppendUint64(b, uint64(n))
	}
	if c.server {
		b = append(b, p...)
	} else {
		var mask [4]byte
		rand.Read(mask[:])
		b = append(b, mask[:]...)
		for i, x := range p {
			b = append(b, x^mask[i%4])
		}
	}
	_, err := c.rwc.Write(b)
	return err
}

// writeClose sends a close frame, unless one was sent. A code of
// closeNoStatus sends none.
func (c *conn) writeClose(code int, reason string) error {
	var p []byte
	if code != closeNoStatus {
		p = binary.BigEndian.AppendUint16(nil, uint16(code))
		p = append(p, reason...)
	}
	return c.writeFrame(opClose, p)
}

// closing reports whether a close frame was sent.
func (c *conn) closing() bool {
	c.wmu.Lock()
	defer c.wmu.Unlock()
	return c.closeSent
}

// parseClose returns the code and reason of a close frame's payload.
func parseClose(p []byte) (int, string, error) {
	if len(p) == 0 {
		return closeNoStatus, "", nil
	}
	if len(p) == 1 {
		return 0, "", protocolError("invalid close frame")
	}
	code := int(binary.BigEndian.Uint16(p))
	if !validCloseCode(code) {
		return 0, "", protocolError("invalid close code %d", code)
	}
	if !utf8.Valid(p[2:]) {
		return 0, "", &failure{closeInvalidData, "close reason is not valid UTF-8"}
	}
	return code, string(p[2:]), nil
}

// validCloseCode reports whether code may be sent in a close frame.
func validCloseCode(code int) bool {
	switch {
	case code >= 1000 && code <= 1003, code >= 1007 && code <= 1014:
		return true
	default:
		return code >= 3000 && code <= 4999
	}
}

// inTurn runs fn once the messages sent before the seq-th are written.
func (c *conn) inTurn(seq int64, fn func() error) error {
	c.mu.Lock()
	for c.next != seq && !c.done {
		c.cond.Wait()
	}
	if c.done {
		c.mu.Unlock()
		return errClosed
	}
	c.mu.Unlock()
	err := fn()
	c.mu.Lock()
	c.next++
	c.mu.Unlock()
	c.cond.Broadcast()
	return err
}

// close starts the closing handshake: the connection is dropped if the
// server does not finish it in time.
func (c *conn) close(code int, reason string, timeout time.Duration) error {
	err := c.writeClose(code, reason)
	time.AfterFunc(timeout, c.terminate)
	return err
}

// terminate drops the connection.
func (c *conn) terminate() {
	c.once.Do(func() {
		c.rwc.Close()
		c.mu.Lock()
		c.done = true
		c.mu.Unlock()
		c.cond.Broadcast()
	})
}
