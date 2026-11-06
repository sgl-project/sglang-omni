// Package watchdriver runs the native file watchers of the backends on one
// goroutine each: the goroutine owns the system's handles, adds and removes
// targets between waits, and reports what the system tells it.
package watchdriver

import (
	"errors"
	"sync"

	"github.com/egoist/mygo/internal/platform"
)

// Driver is a backend's watcher. Every method but Wake runs on the
// goroutine of its Watch. Wait blocks until the system reports changes or
// Wake is called; an error it returns ends the watch.
type Driver interface {
	Add(platform.FileWatchTarget) error
	Remove(id uint64)
	Wait() ([]platform.FileWatchNotice, error)
	Wake()
	Close()
}

// errClosed is what Add returns once the watch has stopped.
var errClosed = errors.New("file watch closed")

type command struct {
	target platform.FileWatchTarget
	remove uint64
	reply  chan error
}

// Watch is a platform.FileWatch running a Driver.
type Watch struct {
	driver Driver
	notify func(platform.FileWatchNotice)
	mu     sync.Mutex
	queue  []command
	closed bool
	done   chan struct{}
}

// New starts the goroutine of d, which calls notify with what d reports.
func New(d Driver, notify func(platform.FileWatchNotice)) *Watch {
	w := &Watch{driver: d, notify: notify, done: make(chan struct{})}
	go w.run()
	return w
}

// Add enrolls t and returns once the driver has.
func (w *Watch) Add(t platform.FileWatchTarget) error {
	reply := make(chan error, 1)
	if !w.send(command{target: t, reply: reply}) {
		return errClosed
	}
	select {
	case err := <-reply:
		return err
	case <-w.done:
		return errClosed
	}
}

func (w *Watch) Remove(id uint64) { w.send(command{remove: id}) }

// Close stops the goroutine and waits until the driver has closed.
func (w *Watch) Close() {
	w.mu.Lock()
	w.closed = true
	w.mu.Unlock()
	w.driver.Wake()
	<-w.done
}

func (w *Watch) send(c command) bool {
	w.mu.Lock()
	if w.closed {
		w.mu.Unlock()
		return false
	}
	w.queue = append(w.queue, c)
	w.mu.Unlock()
	w.driver.Wake()
	return true
}

func (w *Watch) run() {
	defer close(w.done)
	defer w.driver.Close()
	for {
		w.mu.Lock()
		queue, closed := w.queue, w.closed
		w.queue = nil
		w.mu.Unlock()
		if closed {
			return
		}
		for _, c := range queue {
			if c.reply != nil {
				c.reply <- w.driver.Add(c.target)
			} else {
				w.driver.Remove(c.remove)
			}
		}
		notices, err := w.driver.Wait()
		for _, n := range notices {
			w.notify(n)
		}
		if err != nil {
			w.mu.Lock()
			w.closed = true
			w.mu.Unlock()
			w.notify(platform.FileWatchNotice{Err: err})
			return
		}
	}
}
