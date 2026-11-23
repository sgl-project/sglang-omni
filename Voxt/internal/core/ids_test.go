package core

import (
	"testing"
	"time"
)

func TestNewIDMonotonicWithinAMillisecond(t *testing.T) {
	at := time.Date(2026, 10, 13, 12, 0, 0, 0, time.UTC)
	prev := NewID(at)
	for range 1000 {
		id := NewID(at)
		if id <= prev {
			t.Fatalf("%s after %s", id, prev)
		}
		if id[:10] != prev[:10] {
			t.Fatalf("time part changed: %s → %s", prev, id)
		}
		prev = id
	}
	// An earlier time still encodes its own millisecond.
	if old := NewID(at.Add(-time.Hour)); old >= prev {
		t.Errorf("an ID of an hour earlier sorts after: %s ≥ %s", old, prev)
	}
}
