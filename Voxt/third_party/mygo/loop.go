package mygo

import (
	"sync"
)

// mainLoop is a FIFO of functions waiting to run on the main thread. The
// backend drains it (AppHandler.Dispatch) whenever it has been signaled.
type mainLoop struct {
	mu     sync.Mutex
	queue  []func()
	closed bool
}

var loop mainLoop

// post schedules fn on the main thread. It reports false when the event
// loop has already shut down and fn will never run.
func (l *mainLoop) post(fn func()) bool {
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return false
	}
	l.queue = append(l.queue, fn)
	first := len(l.queue) == 1
	l.mu.Unlock()
	if first {
		backend().Signal()
	}
	return true
}

// drain runs queued functions, one at a time, until the queue is empty.
// Main thread only. A function running a loop of its own, as a context
// menu, a modal dialog or await does, runs those queued behind it, which
// it may be waiting for: the backend is signaled again while some wait.
func (l *mainLoop) drain() {
	for {
		l.mu.Lock()
		if len(l.queue) == 0 {
			l.mu.Unlock()
			return
		}
		fn := l.queue[0]
		l.queue[0] = nil
		l.queue = l.queue[1:]
		more := len(l.queue) > 0
		l.mu.Unlock()
		if more {
			backend().Signal()
		}
		fn()
	}
}

// stopped reports whether the loop has shut down.
func (l *mainLoop) stopped() bool {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.closed
}

// shutdown runs whatever is still queued and rejects later posts. The
// queue is closed in the same critical section that finds it empty, so no
// post is accepted without running.
func (l *mainLoop) shutdown() {
	for {
		l.drain()
		l.mu.Lock()
		if len(l.queue) == 0 {
			l.closed = true
			l.mu.Unlock()
			return
		}
		l.mu.Unlock()
	}
}

func isMainThread() bool { return backend().IsMainThread() }

// RunOnMain runs fn on the main (UI) thread and waits for it to return.
// Use it to call native APIs directly, e.g. through Window.NativeHandle.
// It runs fn right away when called on the main thread, and drops it after
// the application has quit.
func RunOnMain(fn func()) { onMain(fn) }

// postMain runs fn asynchronously on the main thread.
func postMain(fn func()) bool { return loop.post(fn) }

// onMain runs fn on the main thread and waits for it to return. It runs fn
// directly when called on the main thread. After the event loop has shut
// down fn is dropped.
func onMain(fn func()) {
	if isMainThread() {
		fn()
		return
	}
	done := make(chan struct{})
	if !loop.post(func() {
		defer close(done)
		fn()
	}) {
		return
	}
	<-done
}

// onMainValue is onMain for functions returning a value.
func onMainValue[T any](fn func() T) T {
	var v T
	onMain(func() { v = fn() })
	return v
}

// await blocks until a value arrives on ch. On the main thread it keeps
// processing native events so the UI stays responsive and the value can
// actually be produced. Whoever sends on ch must call wake afterwards.
func await[T any](ch <-chan T) T {
	if !isMainThread() {
		return <-ch
	}
	for {
		select {
		case v := <-ch:
			return v
		default:
		}
		backend().Step()
	}
}

// deliver sends v on a buffered channel consumed by await and wakes the main
// thread in case it is waiting.
func deliver[T any](ch chan<- T, v T) {
	ch <- v
	backend().Wake()
}
