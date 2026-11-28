package vad

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"
)

// A steady noise 15 dB over the starting floor is speech for good with
// dictation's tuning; with SpeechRiseMS the floor rises to it and the
// "speech" ends within a few seconds, while speech over the noise starts
// again (11-25: a MacBook microphone at -35 dBFS with nobody speaking).
func TestSpeechRise(t *testing.T) {
	const rate = 16000
	rng := rand.New(rand.NewPCG(1, 2))
	amp := math.Pow(10, -35.0/20) * math.Sqrt(3)
	noise := make([]byte, 2*20*rate)
	for i := range len(noise) / 2 {
		binary.LittleEndian.PutUint16(noise[2*i:], uint16(int16((rng.Float64()*2-1)*amp*32767)))
	}
	for _, rise := range []int{0, 1500} {
		p := Defaults()
		p.SpeechRiseMS = rise
		d := NewDetector(rate, p)
		d.Write(noise)
		if quiet := !d.Speaking(); quiet != (rise > 0) {
			t.Fatalf("rise %d ms: after 20 s of steady noise speaking %v", rise, d.Speaking())
		}
		if rise == 0 {
			continue
		}
		loud := make([]byte, 2*rate/2) // half a second 18 dB over the noise
		for i := range len(loud) / 2 {
			v := (rng.Float64()*2-1)*amp + 0.25*math.Sin(2*math.Pi*200*float64(i)/rate)
			binary.LittleEndian.PutUint16(loud[2*i:], uint16(int16(v*32767)))
		}
		d.Write(loud)
		if !d.Speaking() {
			t.Fatal("speech over the noise not heard")
		}
	}
}

// Voices with digital silence between them (a system track) are each
// heard: the floor a voice raised falls back in the silence.
func TestSpeechRiseSilenceBetweenVoices(t *testing.T) {
	const rate = 16000
	p := Defaults()
	p.SpeechRiseMS = 1500
	d := NewDetector(rate, p)
	voice := make([]byte, 2*rate*3/2) // 1.5 s at about -18 dBFS
	for i := range len(voice) / 2 {
		if i%80 < 40 {
			voice[2*i+1] = 0x10
		}
	}
	silence := make([]byte, 2*rate*19/10)
	for n := range 5 {
		d.Write(voice)
		if !d.Speaking() && len(d.Segments()) <= n {
			t.Fatalf("voice %d not heard (floor %.1f dBFS)", n+1, d.floor)
		}
		d.Write(silence)
	}
	if got := len(d.Segments()); got != 5 {
		t.Fatalf("%d segments, want 5", got)
	}
}
