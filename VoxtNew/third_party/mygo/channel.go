//go:build !js

package mygo

import (
	"context"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"sync"
)

// Channel streams values of type T from a bound method to the page that
// called it, like a response that arrives in parts: the lines a command
// prints, the tokens of a model's answer, the progress of a download. The
// method takes it as a parameter:
//
//	// Tail sends the lines of a command's output as it prints them.
//	func (Shell) Tail(ctx context.Context, command string, lines *mygo.Channel[string]) error {
//		cmd := exec.CommandContext(ctx, "sh", "-c", command)
//		out, err := cmd.StdoutPipe()
//		if err != nil {
//			return err
//		}
//		if err := cmd.Start(); err != nil {
//			return err
//		}
//		scanner := bufio.NewScanner(out)
//		for scanner.Scan() {
//			if err := lines.Send(scanner.Text()); err != nil {
//				return err
//			}
//		}
//		return cmd.Wait()
//	}
//
// and the page passes a Channel of mygo-runtime in its place, whose values
// it iterates or handles with onmessage:
//
//	const lines = new Channel<string>();
//	const done = Shell.tail("ping -c 3 example.com", lines);
//	for await (const line of lines) output.append(line + "\n");
//	await done;
//
// Values arrive in order, and before the call's result. The channel closes
// when the method returns, or earlier with Close, which ends the page's
// iteration. The page may close it too, for example by breaking out of the
// loop: Send then fails and the context of the call is canceled, which
// stops the work the method passed it to. Navigating away and closing the
// window do the same.
//
// Send waits while the page has more than a MiB of values yet to take, so
// a fast producer does not pile them up in memory, except on the main
// thread, which must not wait.
//
// Channels are parameters of bound methods only, never parts of other
// values.
type Channel[T any] struct {
	c *channel
}

// ErrChannelClosed is returned by Channel.Send once the channel is closed.
var ErrChannelClosed = errors.New("mygo: channel closed")

// Send sends v to the page. It fails once the channel is closed.
func (ch *Channel[T]) Send(v T) error {
	if ch.c == nil {
		return ErrChannelClosed
	}
	p, err := json.Marshal(v, jsonOptions)
	if err != nil {
		return fmt.Errorf("mygo: cannot encode channel value: %w", err)
	}
	return ch.c.send(p)
}

// Close closes the channel, which ends the page's iteration once it took
// the values sent before. It is called when the method returns.
func (ch *Channel[T]) Close() {
	if ch.c != nil {
		ch.c.close(closedByGo)
	}
}

func (ch *Channel[T]) init(c *channel)      { ch.c = c }
func (*Channel[T]) valueType() reflect.Type { return reflect.TypeFor[T]() }

// channelParam is implemented by *Channel[T] of every T.
type channelParam interface {
	init(c *channel)
	valueType() reflect.Type
}

var channelParamType = reflect.TypeFor[channelParam]()

// hasChannel reports whether values of type t hold a Channel.
func hasChannel(t reflect.Type, seen map[reflect.Type]bool) bool {
	if seen[t] {
		return false
	}
	seen[t] = true
	if t.Implements(channelParamType) || reflect.PointerTo(t).Implements(channelParamType) {
		return true
	}
	switch t.Kind() {
	case reflect.Pointer, reflect.Slice, reflect.Array:
		return hasChannel(t.Elem(), seen)
	case reflect.Map:
		return hasChannel(t.Key(), seen) || hasChannel(t.Elem(), seen)
	case reflect.Struct:
		for i := range t.NumField() {
			if f := t.Field(i); (f.IsExported() || f.Anonymous) && hasChannel(f.Type, seen) {
				return true
			}
		}
	}
	return false
}

// channelWindow is how many bytes of messages with values may wait for the
// page to take them. The page acknowledges what it took when a value asks
// for it, once every half window.
const channelWindow = 1 << 20

// Who closes a channel.
const (
	closedByGo   = iota // the method: the page is told
	closedByPage        // the page: the call is canceled
	closedByLoss        // the page went away
)

// channel is a Channel of any type: the stream of one call's argument.
type channel struct {
	w      *Window
	id     int64 // the page's
	token  string
	ctx    context.Context // the call's
	cancel context.CancelFunc
	// head and ackHead start its values' messages, the latter asking for
	// an acknowledgment.
	head, ackHead []byte

	mu     sync.Mutex
	cond   sync.Cond
	closed bool
	// stop stops closing it with ctx. It is set once the page can close the
	// channel, so a close may find it nil and leave it to newChannel.
	stop func() bool
	// Flow control, in bytes: sent, taken by the page as far as it said,
	// and sent when an acknowledgment was last asked for, by seq.
	seq, sent, taken, asked int64
	asks                    []channelAsk
}

type channelAsk struct{ seq, sent int64 }

// newChannel creates the channel id of the page with token for a call
// whose context is ctx, canceled by cancel. page is the context of the page
// that made the call: after a navigation, the channel starts closed.
func (w *Window) newChannel(page, ctx context.Context, cancel context.CancelFunc, id int64, token string) *channel {
	c := &channel{w: w, id: id, token: token, ctx: ctx, cancel: cancel}
	c.cond.L = &c.mu
	head := make([]byte, 0, 48+len(token))
	head = append(head, `{"t":"chan","c":`...)
	head = strconv.AppendInt(head, id, 10)
	head = append(head, `,"k":`...)
	head, _ = jsontext.AppendQuote(head, token)
	c.ackHead = append(head[:len(head):len(head)], `,"a":1,"p":`...)
	c.head = append(head, `,"p":`...)

	w.mu.Lock()
	early := false
	if w.pageCtx == page {
		if w.channels == nil {
			w.channels = map[int64]*channel{}
		}
		w.channels[id] = c
		if tok, ok := w.closedEarly[id]; ok && tok == token {
			delete(w.closedEarly, id)
			early = true
		}
	} else {
		c.closed = true
	}
	w.mu.Unlock()
	stop := context.AfterFunc(ctx, func() { c.close(closedByLoss) })
	c.mu.Lock()
	c.stop = stop
	closed := c.closed
	c.mu.Unlock()
	if closed { // the page closed it in the meantime
		stop()
	}
	if early {
		c.close(closedByPage)
	}
	return c
}

// maxClosedEarly bounds the channels a page may close before their calls
// made them.
const maxClosedEarly = 1024

// pageClosedChannel closes the channel id that the page with token closed.
// Calls make their channels on goroutines of their own, so the page may
// close one before its call made it, e.g. by aborting a request right
// after starting it: the call then closes it as soon as it makes it.
func (w *Window) pageClosedChannel(id int64, token string) {
	w.mu.Lock()
	c := w.channels[id]
	if c == nil {
		if len(w.closedEarly) < maxClosedEarly {
			if w.closedEarly == nil {
				w.closedEarly = map[int64]string{}
			}
			w.closedEarly[id] = token
		}
		w.mu.Unlock()
		return
	}
	w.mu.Unlock()
	if c.token == token {
		c.close(closedByPage)
	}
}

// channel returns the channel id of the current page, if token is the
// page's.
func (w *Window) channel(id int64, token string) *channel {
	w.mu.Lock()
	defer w.mu.Unlock()
	if c := w.channels[id]; c != nil && c.token == token {
		return c
	}
	return nil
}

func (c *channel) send(p []byte) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	// Wait while the page is behind, unless that would block the main
	// thread, which receives its acknowledgments.
	for !c.closed && c.sent-c.taken >= channelWindow && !isMainThread() {
		c.cond.Wait()
	}
	// The call's context may be canceled before it closed the channel.
	if c.closed || c.ctx.Err() != nil {
		return ErrChannelClosed
	}
	c.seq++
	c.sent += int64(len(c.head) + len(p) + 2) // the whole message
	head := c.head
	if c.sent-c.asked >= channelWindow/2 {
		c.asked = c.sent
		c.asks = append(c.asks, channelAsk{c.seq, c.sent})
		head = c.ackHead
	}
	// Under the lock, so values are queued in order.
	c.w.enqueue(message{head, p, closeBrace}, false)
	return nil
}

// ack records that the page took the values up to seq.
func (c *channel) ack(seq int64) {
	c.mu.Lock()
	for len(c.asks) > 0 && c.asks[0].seq <= seq {
		c.taken = c.asks[0].sent
		c.asks = c.asks[1:]
	}
	c.mu.Unlock()
	c.cond.Broadcast()
}

func (c *channel) close(by int) {
	c.mu.Lock()
	if by == closedByPage {
		// Under the lock, before the channel closes: the call's context is
		// canceled once Send fails, and a call that returns as soon as it is
		// finds the channel closed, not one whose end the page is told.
		c.cancel()
	}
	if c.closed {
		c.mu.Unlock()
		return
	}
	c.closed = true
	if by == closedByGo {
		// After the values sent so far.
		end := append(c.head[:len(c.head)-len(`,"p":`):len(c.head)-len(`,"p":`)], `,"end":true}`...)
		c.w.enqueue(message{head: end}, false)
	}
	stop := c.stop
	c.mu.Unlock()
	c.cond.Broadcast()
	if stop != nil {
		stop()
	}
	c.w.mu.Lock()
	if c.w.channels[c.id] == c {
		delete(c.w.channels, c.id)
	}
	c.w.mu.Unlock()
}
