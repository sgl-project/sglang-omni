// Package wait is how MyGo's tests wait for what happens elsewhere: on
// the main thread, in another goroutine or process, in the window system
// or the file system. They wait until it happened, never for a time.
//
// A test that sleeps and then checks, or that gives up after a few
// seconds, decides on the speed of the machine: a busy CI runner makes it
// fail although the code is right, and a longer timeout only makes that
// rarer. Here a wait ends when its condition holds. Guard bounds it only
// so that a hang fails the test with what never happened rather than
// stalling the whole run, and it is not to be tuned: nothing a test
// waits for takes that long when the code is right. A wait that reaches
// it found a bug (an event lost to a race, a deadlock, a condition that
// can't become true), and the fix is in the code or the condition.
package wait

import (
	"context"
	"testing"
	"time"
)

// Guard is the longest a wait waits before it fails the test.
const Guard = time.Minute

// Until waits until cond returns true, calling it again and again: right
// away, then at intervals growing to 20 ms. what says what is waited
// for, as "the window to close".
func Until(t testing.TB, what string, cond func() bool) {
	t.Helper()
	if !Poll(cond) {
		t.Fatalf("%s never happened (waited %v)", what, Guard)
	}
}

// Poll is Until without a test: it reports whether cond became true
// before Guard, for helpers that say more when it did not.
func Poll(cond func() bool) bool {
	deadline := time.Now().Add(Guard)
	pause := time.Millisecond
	for !cond() {
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(pause)
		pause = min(2*pause, 20*time.Millisecond)
	}
	return true
}

// For waits for a value from ch.
func For[T any](t testing.TB, what string, ch <-chan T) T {
	t.Helper()
	select {
	case v := <-ch:
		return v
	case <-time.After(Guard):
		t.Fatalf("%s never happened (waited %v)", what, Guard)
		var zero T
		return zero
	}
}

// Context returns a context for calls that wait on one, as
// Page.EvalContext and FileWatcher.Next do, canceled after Guard or when
// the test ends.
func Context(t testing.TB) context.Context {
	ctx, cancel := context.WithTimeout(t.Context(), Guard)
	t.Cleanup(cancel)
	return ctx
}
