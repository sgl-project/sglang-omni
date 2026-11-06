package core

import (
	"strings"
	"sync"
	"sync/atomic"
)

// Event is anything published on the Bus. Topic names are dotted, the
// first part naming the owner: "settings.changed", "helper.status",
// "notification.posted"… Event types are plain values: subscribers may
// read them from any goroutine.
type Event interface {
	Topic() string
}

// Bus fans events out from the services and the helper supervisor to the
// UI.
//
// Delivery is synchronous: Publish calls every matching handler in turn,
// in subscription order, on the publisher's goroutine, and returns when
// they have all returned. That keeps events in order and tests
// deterministic, at the price of one rule: handlers must not block. A
// MyGo window hands the event to Window.Update. Publishers must not hold
// their own locks while publishing.
//
// The zero Bus is ready to use. A nil *Bus discards events, so packages
// can take an optional bus.
type Bus struct {
	mu   sync.Mutex
	next int
	subs []*subscription // in subscription order
}

type subscription struct {
	id     int
	topics []string // nil: every topic; "x." matches "x.anything"
	fn     func(Event)
	dead   atomic.Bool
}

// NewBus returns an empty bus.
func NewBus() *Bus { return &Bus{} }

// Subscribe calls fn for every event whose topic matches one of topics:
// an exact name, or a prefix ending in "." ("helper." matches "helper.status").
// No topics means every event. The returned function unsubscribes; after
// it returns, fn is not called again (except by a Publish already running
// it on another goroutine).
func (b *Bus) Subscribe(fn func(Event), topics ...string) (unsubscribe func()) {
	if b == nil {
		return func() {}
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.next++
	s := &subscription{id: b.next, topics: topics, fn: fn}
	b.subs = append(b.subs, s)
	return func() { b.remove(s) }
}

func (b *Bus) remove(s *subscription) {
	s.dead.Store(true)
	b.mu.Lock()
	defer b.mu.Unlock()
	for i, x := range b.subs {
		if x == s {
			b.subs = append(b.subs[:i:i], b.subs[i+1:]...)
			return
		}
	}
}

// Publish delivers ev to the matching subscribers. Handlers may publish
// and (un)subscribe; events they publish are delivered before Publish
// returns, depth first.
func (b *Bus) Publish(ev Event) {
	if b == nil || ev == nil {
		return
	}
	topic := ev.Topic()
	b.mu.Lock()
	subs := append([]*subscription(nil), b.subs...)
	b.mu.Unlock()
	for _, s := range subs {
		if !s.dead.Load() && s.matches(topic) {
			s.fn(ev)
		}
	}
}

func (s *subscription) matches(topic string) bool {
	if len(s.topics) == 0 {
		return true
	}
	for _, t := range s.topics {
		if t == topic || (strings.HasSuffix(t, ".") && strings.HasPrefix(topic, t)) {
			return true
		}
	}
	return false
}

// On subscribes fn to the events of type T (any topic):
//
//	core.On(bus, func(ev settings.Changed) { win.Update(func() { … }) })
func On[T Event](b *Bus, fn func(T)) (unsubscribe func()) {
	return b.Subscribe(func(ev Event) {
		if t, ok := ev.(T); ok {
			fn(t)
		}
	})
}
