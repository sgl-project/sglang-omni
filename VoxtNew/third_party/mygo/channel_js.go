//go:build js

package mygo

import "errors"

// Channel streams values of type T from a bound method to the page that
// called it. Browsers show no web pages (see ipc_js.go): Send fails.
type Channel[T any] struct{}

// ErrChannelClosed is returned by Channel.Send once the channel is closed.
var ErrChannelClosed = errors.New("mygo: channel closed")

// Send sends v to the page: there is none, and it fails.
func (ch *Channel[T]) Send(v T) error { return ErrChannelClosed }

// Close closes the channel.
func (ch *Channel[T]) Close() {}

// channel is the stream of one call's argument, of which there are none.
type channel struct{}
