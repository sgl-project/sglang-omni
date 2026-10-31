package resident

import (
	"fmt"
	"slices"
	"sync"
	"testing"
)

type recorder struct {
	mu  sync.Mutex
	got []string
}

func (r *recorder) add(ev Event) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.got = append(r.got, fmt.Sprintf("%s %s %d %d", ev.Kind, ev.ID, ev.PID, ev.Bytes))
}

// TestTracker: a slot's states become loaded, busy, idle and unloaded
// events, in order; another process or model is an unload and a load.
func TestTracker(t *testing.T) {
	var r recorder
	q := NewQueue(r.add)
	var tr Tracker
	tr.Set(q, "", 0, false, 0)        // nothing
	tr.Set(q, "asr.fast", 0, true, 5) // no process: nothing
	tr.Set(q, "asr.fast", 7, false, 5)
	tr.Set(q, "asr.fast", 7, true, 5)
	tr.Set(q, "asr.fast", 7, true, 5) // no change
	tr.Set(q, "asr.fast", 7, false, 5)
	tr.Set(q, "asr.fast", 8, true, 6) // restarted
	tr.Set(q, "llm.fast", 8, true, 9) // another model in the process
	tr.Set(q, "", 0, false, 0)
	q.Flush()
	want := []string{
		"loaded asr.fast 7 5", "idle asr.fast 7 0", "busy asr.fast 7 0", "idle asr.fast 7 0",
		"unloaded asr.fast 7 0", "loaded asr.fast 8 6",
		"unloaded asr.fast 8 0", "loaded llm.fast 8 9",
		"unloaded llm.fast 8 0",
	}
	if !slices.Equal(r.got, want) {
		t.Errorf("events\n%q\nwant\n%q", r.got, want)
	}
}

// TestQueueOrder: events pushed from many goroutines in turn arrive in
// the order of their pushes; a nil queue drops them.
func TestQueueOrder(t *testing.T) {
	var r recorder
	q := NewQueue(r.add)
	var mu sync.Mutex
	var pushed []string
	var wg sync.WaitGroup
	for i := range 50 {
		wg.Go(func() {
			mu.Lock()
			defer mu.Unlock()
			ev := Event{Kind: Busy, ID: "m", PID: i}
			pushed = append(pushed, fmt.Sprintf("%s %s %d %d", ev.Kind, ev.ID, ev.PID, ev.Bytes))
			q.Push(ev)
		})
	}
	wg.Wait()
	q.Flush()
	if !slices.Equal(r.got, pushed) {
		t.Errorf("order\n%q\nwant\n%q", r.got, pushed)
	}
	var none *Queue
	none.Push(Event{Kind: Busy})
	none.Flush()
	NewQueue(nil).Push(Event{Kind: Busy})
}
