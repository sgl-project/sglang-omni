package core

import (
	"slices"
	"strings"
	"testing"
	"time"
)

func TestNewID(t *testing.T) {
	t0 := time.Date(2026, 10, 13, 12, 0, 0, 0, time.UTC)
	seen := map[string]bool{}
	prev := ""
	for i := range 1000 {
		id := NewID(t0.Add(time.Duration(i) * time.Millisecond))
		if len(id) != 26 || strings.Trim(id, crockford) != "" {
			t.Fatalf("bad id %q", id)
		}
		if seen[id] || id[:10] < prev {
			t.Fatalf("id %q is a duplicate or out of order after %q", id, prev)
		}
		seen[id], prev = true, id[:10]
	}
	if NewID(t0) == NewID(t0) {
		t.Error("ids collide within a millisecond")
	}
	if got := NewID(time.UnixMilli(0))[:10]; got != "0000000000" {
		t.Errorf("epoch prefix %q", got)
	}
}

type ping struct{ n int }
type pong struct{}

func (ping) Topic() string { return "test.ping" }
func (pong) Topic() string { return "other.pong" }

func TestBus(t *testing.T) {
	b := NewBus()
	var got []string
	unsubAll := b.Subscribe(func(ev Event) { got = append(got, "all:"+ev.Topic()) })
	b.Subscribe(func(ev Event) { got = append(got, "prefix:"+ev.Topic()) }, "test.")
	b.Subscribe(func(ev Event) { got = append(got, "exact:"+ev.Topic()) }, "other.pong")
	unsubTyped := On(b, func(p ping) {
		got = append(got, "typed")
		if p.n == 1 {
			b.Publish(pong{}) // nested publish is delivered depth first
		}
	})

	b.Publish(ping{n: 1})
	want := []string{"all:test.ping", "prefix:test.ping", "typed", "all:other.pong", "exact:other.pong"}
	if !slices.Equal(got, want) {
		t.Fatalf("got %q\nwant %q", got, want)
	}

	got = nil
	unsubAll()
	unsubTyped()
	b.Publish(ping{})
	if !slices.Equal(got, []string{"prefix:test.ping"}) {
		t.Fatalf("after unsubscribe: %q", got)
	}

	var nilBus *Bus
	nilBus.Publish(ping{})
	nilBus.Subscribe(func(Event) {})()
}
