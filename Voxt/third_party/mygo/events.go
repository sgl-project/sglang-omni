package mygo

import (
	"sync"
)

// Preventable is embedded in events whose default action listeners can
// cancel.
type Preventable struct {
	prevented bool
}

// PreventDefault cancels the default action of the event.
func (p *Preventable) PreventDefault() { p.prevented = true }

// DefaultPrevented reports whether PreventDefault was called.
func (p *Preventable) DefaultPrevented() bool { return p.prevented }

// CloseEvent is passed to Window.OnClose listeners. Preventing it keeps the
// window open.
type CloseEvent struct {
	Preventable
	Window *Window
}

// QuitEvent is passed to Application.OnBeforeQuit and OnWillQuit listeners.
// Preventing it cancels the quit.
type QuitEvent struct {
	Preventable
}

// NavigateEvent is passed to Page.OnWillNavigate listeners. Preventing it
// cancels the navigation.
type NavigateEvent struct {
	Preventable
	URL string
	// UserInitiated is true when the navigation was triggered by a user
	// gesture such as clicking a link.
	UserInitiated bool
}

// TitleEvent is passed to Page.OnPageTitleUpdated listeners. Preventing
// it keeps the native window title unchanged.
type TitleEvent struct {
	Preventable
	Title string
}

// FileDropEvent is passed to Window.OnFileDrop listeners, and to the
// page's onFileDrop listeners.
type FileDropEvent struct {
	// Paths of the dropped files and directories.
	Paths []string `json:"paths"`
	// X and Y are where the files were dropped, in CSS pixels from the
	// top-left corner of the page's viewport, like a DOM event's clientX
	// and clientY.
	X int `json:"x"`
	Y int `json:"y"`
}

// fileDropEvent is the name of the page event of dropped files. Names
// starting with "mygo:" are MyGo's own.
const fileDropEvent = "mygo:file-drop"

// listeners is a goroutine safe list of event listeners of type F.
type listeners[F any] struct {
	mu   sync.Mutex
	next uint64
	list []listener[F]
}

type listener[F any] struct {
	id   uint64
	fn   F
	once bool
}

// add registers fn and returns a function that removes it again.
func (l *listeners[F]) add(fn F, once bool) func() {
	l.mu.Lock()
	l.next++
	id := l.next
	l.list = append(l.list, listener[F]{id: id, fn: fn, once: once})
	l.mu.Unlock()
	return func() { l.remove(id) }
}

func (l *listeners[F]) remove(id uint64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	for i, e := range l.list {
		if e.id == id {
			l.list = append(l.list[:i:i], l.list[i+1:]...)
			return
		}
	}
}

// snapshot returns the current listeners, dropping once-listeners from the
// list, so they can be called without holding the lock.
func (l *listeners[F]) snapshot() []F {
	l.mu.Lock()
	defer l.mu.Unlock()
	if len(l.list) == 0 {
		return nil
	}
	fns := make([]F, len(l.list))
	kept := l.list[:0:0]
	for i, e := range l.list {
		fns[i] = e.fn
		if !e.once {
			kept = append(kept, e)
		}
	}
	l.list = kept
	return fns
}

func (l *listeners[F]) len() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return len(l.list)
}

func (l *listeners[F]) clear() {
	l.mu.Lock()
	l.list = nil
	l.mu.Unlock()
}

// fire calls listeners of an event without arguments.
func fire(l *listeners[func()]) {
	for _, fn := range l.snapshot() {
		fn()
	}
}

// fire1 calls listeners of an event with one argument.
func fire1[A any](l *listeners[func(A)], a A) {
	for _, fn := range l.snapshot() {
		fn(a)
	}
}
