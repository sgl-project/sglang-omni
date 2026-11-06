// Package resident is what a model runtime (internal/omni's servers,
// internal/llmrt's process) says about the models it holds in memory:
// one loaded, busy with a use, idle again, unloaded. The app writes them
// into the ActNow family's runtime file (spec §11.1).
//
// A runtime keeps a Tracker per process slot and Sets it after each
// change, under its own lock; the Tracker pushes what changed on a Queue,
// which hands the events to the app in order, on a goroutine of its own,
// under no lock of the runtime.
//
// Imports: none.
package resident

import "sync"

// Kind is what happened to a model.
type Kind string

const (
	// Loaded: the model is in memory (a server said it is ready, the
	// process answered a load or a chat); busy until Idle.
	Loaded Kind = "loaded"
	// Busy: a use took it.
	Busy Kind = "busy"
	// Idle: its last use ended.
	Idle Kind = "idle"
	// Unloaded: it left memory (its process exited for any reason, or
	// the process let it go for another).
	Unloaded Kind = "unloaded"
)

// Event is one change of a model.
type Event struct {
	Kind Kind
	// ID is the model's catalogue id ("asr.fast").
	ID string
	// PID is the process holding it.
	PID int
	// Bytes is the memory the runtime says the model takes, 0 when it
	// does not know (Loaded only).
	Bytes int64
}

// Queue hands events to a function in the order they were pushed, one at
// a time, on a goroutine of its own. The zero value and a nil Queue drop
// them.
type Queue struct {
	fn      func(Event)
	mu      sync.Mutex
	q       []Event
	running bool
	idle    *sync.Cond
}

// NewQueue is a queue to fn (nil: one that drops everything).
func NewQueue(fn func(Event)) *Queue {
	q := &Queue{fn: fn}
	q.idle = sync.NewCond(&q.mu)
	return q
}

// Push queues ev; it never blocks on fn (call it under a lock).
func (q *Queue) Push(ev Event) {
	if q == nil || q.fn == nil {
		return
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	q.q = append(q.q, ev)
	if !q.running {
		q.running = true
		go q.drain()
	}
}

func (q *Queue) drain() {
	for {
		q.mu.Lock()
		if len(q.q) == 0 {
			q.running = false
			q.idle.Broadcast()
			q.mu.Unlock()
			return
		}
		ev := q.q[0]
		q.q = q.q[1:]
		q.mu.Unlock()
		q.fn(ev)
	}
}

// Flush waits until every event pushed so far was handed over (a runtime
// that stops, before the app removes its file).
func (q *Queue) Flush() {
	if q == nil || q.fn == nil {
		return
	}
	q.mu.Lock()
	defer q.mu.Unlock()
	for q.running {
		q.idle.Wait()
	}
}

// Tracker turns the state of one process slot (the model its process
// holds, whether a use runs) into events: Set it after each change.
// Not goroutine-safe: the runtime's lock guards it.
type Tracker struct {
	id   string
	pid  int
	busy bool
}

// Set says the slot holds model id in process pid ("" or 0: nothing),
// busy or not, taking bytes (0: unknown); the differences with the last
// Set go to q. Another id or pid is the old model unloaded and the new
// one loaded.
func (t *Tracker) Set(q *Queue, id string, pid int, busy bool, bytes int64) {
	if id == "" || pid == 0 {
		id, pid = "", 0
	}
	if id != t.id || pid != t.pid {
		if t.id != "" {
			q.Push(Event{Kind: Unloaded, ID: t.id, PID: t.pid})
		}
		t.id, t.pid, t.busy = id, pid, true
		if id != "" {
			q.Push(Event{Kind: Loaded, ID: id, PID: pid, Bytes: bytes})
		}
	}
	if id != "" && busy != t.busy {
		t.busy = busy
		kind := Idle
		if busy {
			kind = Busy
		}
		q.Push(Event{Kind: kind, ID: id, PID: pid})
	}
}
