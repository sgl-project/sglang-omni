package ui

import (
	"fmt"
	"iter"
)

// A Selection is the rows chosen in a List or a Table that lets the user
// choose several (ListState.Selection), held by the keys of their items
// (ListState.Key), of type K: the choice stays with its items as rows come,
// go and move. Without a Key, rows are their indices, of type int. Its
// zero value is empty. The keys of items gone from the list stay until
// removed.
//
//	type app struct {
//		files  []File
//		chosen ui.Selection[string] // their paths
//		list   ui.ListState
//	}
//
//	app.list.Key = func(i int) any { return app.files[i].Path }
//	app.list.Selection = &app.chosen
type Selection[K comparable] struct {
	keys map[K]struct{}
}

// Has reports whether the row of key k is chosen.
func (s *Selection[K]) Has(k K) bool {
	_, ok := s.keys[k]
	return ok
}

// Len returns how many rows are chosen.
func (s *Selection[K]) Len() int { return len(s.keys) }

// Add chooses the row of key k.
func (s *Selection[K]) Add(k K) {
	if s.keys == nil {
		s.keys = map[K]struct{}{}
	}
	s.keys[k] = struct{}{}
}

// Remove takes the row of key k out of the choice.
func (s *Selection[K]) Remove(k K) { delete(s.keys, k) }

// Clear chooses no row.
func (s *Selection[K]) Clear() { clear(s.keys) }

// All returns the keys of the rows chosen, in no order.
func (s *Selection[K]) All() iter.Seq[K] {
	return func(yield func(K) bool) {
		for k := range s.keys {
			if !yield(k) {
				return
			}
		}
	}
}

// A Selector is what ListState.Selection takes: a *Selection.
type Selector interface {
	has(key any) bool
	// set chooses the row of key, or takes it out, and reports whether
	// that changed the choice.
	set(key any, on bool) bool
	// clear chooses no row, and reports whether one was.
	clear() bool
	size() int
}

func (s *Selection[K]) has(key any) bool { return s.Has(s.key(key)) }

func (s *Selection[K]) set(key any, on bool) bool {
	k := s.key(key)
	if s.Has(k) == on {
		return false
	}
	if on {
		s.Add(k)
	} else {
		s.Remove(k)
	}
	return true
}

func (s *Selection[K]) clear() bool {
	n := len(s.keys)
	s.Clear()
	return n > 0
}

func (s *Selection[K]) size() int { return len(s.keys) }

// key returns a row's key as the selection holds it.
func (s *Selection[K]) key(key any) K {
	k, ok := key.(K)
	if !ok {
		panic(fmt.Sprintf("ui: a row's key is a %T, not the %T of its Selection", key, *new(K)))
	}
	return k
}
