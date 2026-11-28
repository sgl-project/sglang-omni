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

// Soft speech, about 7 dB over the room's noise, is heard within a few
// seconds (at 10 dB it went unheard for over 8 s), and noise jittering
// 2 dB per level is not.
func TestRoomFloorHearsSoftSpeech(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	const step = 50 * time.Millisecond
	var f roomFloor
	at := time.Duration(0)
	for range int(time.Minute / step) {
		if f.heard(at, -35+rng.NormFloat64()*2) {
			t.Fatalf("noise heard as speech at %v", at)
		}
		at += step
	}
	start := at
	for i := range 200 { // 10 s of syllables 9 dB over the noise, with pauses
		db := -35 + rng.NormFloat64()*2
		if i%8 < 5 {
			db = -26 + rng.NormFloat64()*2
		}
		if f.heard(at, db) {
			if at-start > 3*time.Second {
				t.Fatalf("soft speech heard only after %v", at-start)
			}
			return
		}
		at += step
	}
	t.Fatal("soft speech never heard")
}
