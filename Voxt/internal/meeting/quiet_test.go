package meeting

import (
	"math/rand/v2"
	"testing"
	"time"
)

// A MacBook microphone with nobody speaking (11-25: -32 to -42 dBFS, peaks
// to -7): five minutes of noise around -38 dBFS with a key click every two
// seconds is quiet; syllables 20 dB over the same noise are speech within a
// second.
func TestRoomFloor(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	const step = 100 * time.Millisecond
	var f roomFloor
	at := time.Duration(0)
	for i := range int(5 * time.Minute / step) {
		db := -38 + rng.NormFloat64()*3
		if i%20 == 0 {
			db = -12 // a click: one level
		}
		if f.heard(at, db) {
			t.Fatalf("noise heard as speech at %v (%.1f dBFS)", at, db)
		}
		at += step
	}
	for i := range 10 {
		db := -38.0
		if i%5 < 3 { // 300 ms of a syllable, 200 ms between
			db = -18
		}
		if f.heard(at, db) {
			if at-5*time.Minute > time.Second {
				t.Fatalf("speech heard only after %v", at-5*time.Minute)
			}
			return
		}
		at += step
	}
	t.Fatal("speech over the noise never heard")
}
