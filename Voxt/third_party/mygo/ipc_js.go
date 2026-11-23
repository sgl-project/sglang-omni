//go:build js

package mygo

import (
	"context"
	"fmt"
	"strings"
)

// Browsers show no web pages (backend_js.go): no page calls Go or hears
// its events. Builds for them leave out what ipc.go needs for that, the
// reflection of bound methods and JSON, and keep its API: Bind, BindAs and
// Use bind nothing, and events reach no page.

// Bind exposes the exported methods of each service to the frontend, which
// browsers do not have: it binds nothing.
func Bind(services ...any) {}

// BindAs is Bind with an explicit service name: it binds nothing.
func BindAs(name string, service any) {}

func bind(name string, svc any, internal bool) error { return nil }

type callerKey struct{}

// CallerWindow returns the window whose page called a bound method, given
// the context.Context passed to that method; nil for other contexts.
func CallerWindow(ctx context.Context) *Window {
	w, _ := ctx.Value(callerKey{}).(*Window)
	return w
}

// pageKey holds the calling page in the context of a call.
type pageKey struct{}

// CallerPage returns the page that called a bound method, given the
// context.Context passed to that method; nil for other contexts.
func CallerPage(ctx context.Context) *Page {
	p, _ := ctx.Value(pageKey{}).(*Page)
	return p
}

// Event is a typed event the Go side sends to pages, of which browsers
// show none.
type Event[T any] struct {
	name string
}

// NewEvent declares an event with payload type T. Names starting with
// "mygo:" are reserved.
func NewEvent[T any](name string) *Event[T] {
	if strings.HasPrefix(name, "mygo:") {
		panic(fmt.Sprintf("mygo: event %q: names starting with \"mygo:\" are reserved", name))
	}
	return &Event[T]{name: name}
}

// Name returns the event name.
func (e *Event[T]) Name() string { return e.name }

// Emit sends the event to the pages of w: there are none.
func (e *Event[T]) Emit(w *Window, payload T) error {
	if w.IsDestroyed() {
		return errDestroyed
	}
	return nil
}

// EmitPage sends the event to a page, of which there are none.
func (e *Event[T]) EmitPage(p *Page, payload T) error {
	if p == nil || p.w.IsDestroyed() {
		return errDestroyed
	}
	return nil
}

// Broadcast sends the event to every page: there are none.
func (e *Event[T]) Broadcast(payload T) error { return nil }

func encodeEvent(name string, payload any) (message, error) { return message{}, errNoPage }
