package macx

import (
	"encoding/binary"
	"math"
	"os"
	"runtime"
	"testing"
	"time"
)

// TestFadingSound: off macOS a fading sound cannot be made; on a Mac
// (VOXTNEW_MAC=1) a real AVAudioPlayer plays a tone at -80 dB (not heard),
// a fade of an older start leaves the newer one playing, and a fade of the
// newest ends it within the fade.
func TestFadingSound(t *testing.T) {
	if runtime.GOOS != "darwin" {
		if _, err := NewFadingSound(toneWAV(0.1)); err == nil {
			t.Error("made a fading sound off macOS")
		}
		return
	}
	if os.Getenv("VOXTNEW_MAC") == "" {
		t.Skip("set VOXTNEW_MAC=1 on a Mac")
	}
	s, err := NewFadingSound(toneWAV(2))
	if err != nil {
		t.Fatal(err)
	}
	const quiet = 1e-4 // -80 dB
	old, ok := s.Play(quiet)
	if !ok {
		t.Fatal("did not play")
	}
	gen, ok := s.Play(quiet)
	if !ok || gen == old {
		t.Fatalf("restart: ok %v, gen %d after %d", ok, gen, old)
	}
	s.FadeOut(old, 50*time.Millisecond)
	if waitFor(400*time.Millisecond, func() bool { return !s.Playing() }) {
		t.Fatal("a fade of the older start stopped the newer one")
	}
	start := time.Now()
	s.FadeOut(gen, 100*time.Millisecond)
	if !waitFor(time.Second, func() bool { return !s.Playing() }) {
		t.Fatal("still playing a second after a 100 ms fade")
	}
	if d := time.Since(start); d < 100*time.Millisecond {
		t.Errorf("stopped %v after the fade began, before the fade ended", d)
	}
}

// waitFor reports whether cond held within d, checking every 5 ms.
func waitFor(d time.Duration, cond func() bool) bool {
	tick := time.NewTicker(5 * time.Millisecond)
	defer tick.Stop()
	deadline := time.After(d)
	for !cond() {
		select {
		case <-tick.C:
		case <-deadline:
			return false
		}
	}
	return true
}

// toneWAV is sec seconds of a 440 Hz tone, 44.1 kHz 16-bit mono.
func toneWAV(sec float64) []byte {
	const rate = 44100
	n := int(sec * rate)
	b := make([]byte, 44+2*n)
	copy(b, "RIFF")
	binary.LittleEndian.PutUint32(b[4:], uint32(36+2*n))
	copy(b[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(b[16:], 16)
	binary.LittleEndian.PutUint16(b[20:], 1)
	binary.LittleEndian.PutUint16(b[22:], 1)
	binary.LittleEndian.PutUint32(b[24:], rate)
	binary.LittleEndian.PutUint32(b[28:], 2*rate)
	binary.LittleEndian.PutUint16(b[32:], 2)
	binary.LittleEndian.PutUint16(b[34:], 16)
	copy(b[36:], "data")
	binary.LittleEndian.PutUint32(b[40:], uint32(2*n))
	for i := range n {
		v := int16(0.5 * 32767 * math.Sin(2*math.Pi*440*float64(i)/rate))
		binary.LittleEndian.PutUint16(b[44+2*i:], uint16(v))
	}
	return b
}
