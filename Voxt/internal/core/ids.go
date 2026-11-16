package core

import (
	"crypto/rand"
	"sync"
	"time"
)

// crockford is Crockford's base32 alphabet, as ULIDs use: no I, L, O, U.
const crockford = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

// NewID returns a ULID for t: 26 characters, the first 10 encoding the
// millisecond and the rest 80 random bits. IDs sort by creation time as
// strings, are unique across goroutines and processes, and are safe in
// file names. Every new record (history entry, dictionary term,
// notification…) uses one; code must not parse them.
//
// Within one process, IDs of one millisecond are monotonic (ULID's
// monotonic mode): an ID made in the same millisecond as the previous one
// is the previous one plus one, so IDs made one after the other sort in
// that order.
func NewID(t time.Time) string {
	b := lastID.next(uint64(t.UnixMilli()))
	// 128 bits → 26 base32 digits (the first carries 3 bits).
	var out [26]byte
	var acc uint64
	bits := 0
	j := 25
	for i := 15; i >= 0; i-- {
		acc |= uint64(b[i]) << bits
		bits += 8
		for bits >= 5 {
			out[j] = crockford[acc&31]
			j--
			acc >>= 5
			bits -= 5
		}
	}
	out[0] = crockford[acc&31]
	return string(out[:])
}

// lastID is the last ID made, for monotonic IDs.
var lastID monotonic

type monotonic struct {
	mu   sync.Mutex
	ms   uint64
	b    [16]byte
	used bool
}

// next returns the bytes of a new ID for millisecond ms: the last ID plus
// one when it is of the same millisecond, else random after the time.
func (m *monotonic) next(ms uint64) [16]byte {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.used && ms == m.ms {
		for i := 15; i >= 6; i-- {
			m.b[i]++
			if m.b[i] != 0 {
				break
			}
		}
		return m.b
	}
	var b [16]byte
	for i := 5; i >= 0; i-- {
		b[i] = byte(ms >> (8 * (5 - i)))
	}
	if _, err := rand.Read(b[6:]); err != nil {
		panic(err) // crypto/rand does not fail on supported platforms
	}
	b[6] &= 0x7f // leaves room to count up within the millisecond
	m.ms, m.b, m.used = ms, b, true
	return b
}
