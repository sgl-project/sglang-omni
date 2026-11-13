//go:build darwin

package sounds

import (
	"os"
	"testing"
	"time"
)

// TestMacOutput plays a quiet tone and a system sound through
// AVAudioPlayer and NSSound (ACTNOWKIT_MAC=1; volume 5%).
func TestMacOutput(t *testing.T) {
	if os.Getenv("ACTNOWKIT_MAC") == "" {
		t.Skip("ACTNOWKIT_MAC not set")
	}
	o := DefaultOutput()
	wav := ToneWAV(880, 0.12, 0.3)
	start := time.Now()
	if err := o.Load("tone", wav); err != nil {
		t.Fatal(err)
	}
	t.Logf("load %v", time.Since(start))
	if err := o.Load("tone", wav); err != nil {
		t.Fatal(err)
	}
	for i := range 6 { // quicker than the tone: every voice in use
		start = time.Now()
		v, err := o.Play("tone", 0.05)
		if err != nil {
			t.Fatalf("play %d: %v", i, err)
		}
		// The first play of the process starts the audio engine; a node's
		// first start takes about 4 ms (mac-air, macOS 26.7), later ones less.
		d := time.Since(start)
		t.Logf("play %d: %v", i, d)
		if limit := 10 * time.Millisecond; i > 0 && d > limit || d > 250*time.Millisecond {
			t.Errorf("play %d took %v", i, d)
		}
		if i == 5 {
			v.Stop()
		}
		time.Sleep(15 * time.Millisecond)
	}
	if _, err := o.Play("nope", 0.05); err == nil {
		t.Error("unloaded key played")
	}
	v, err := o.PlaySystem("Tink", 0.05)
	if err != nil {
		t.Fatal(err)
	}
	v.Stop()
	if _, err := o.PlaySystem("No Such Sound", 0.05); err == nil {
		t.Error("missing system sound played")
	}
	if len(SystemSounds("")) == 0 {
		t.Error("no system sounds")
	}
	// From goroutines at once (the pool's lock, the threads' pools).
	done := make(chan error, 8)
	for range 8 {
		go func() { _, err := o.Play("tone", 0.03); done <- err }()
	}
	for range 8 {
		if err := <-done; err != nil {
			t.Error(err)
		}
	}
	time.Sleep(300 * time.Millisecond)
	if err := o.Load("bad", []byte("not audio")); err == nil {
		t.Error("bad data loaded")
	}
	// A WAV the engine does not take (A-law): AVAudioPlayer plays it.
	alaw := ToneWAV(660, 0.1, 0.3)
	alaw[20] = 6
	alaw[34] = 8
	if err := o.Load("alaw", alaw); err != nil {
		t.Logf("a-law: %v", err)
	} else if _, err := o.Play("alaw", 0.03); err != nil {
		t.Errorf("a-law play: %v", err)
	}
	// The engine stops when idle and starts again.
	mo := o.(*macOutput)
	mo.engine.IdleStop = 200 * time.Millisecond
	o.Play("tone", 0.03)
	time.Sleep(600 * time.Millisecond)
	if mo.engine.Running() {
		t.Error("engine still runs when idle")
	}
	start = time.Now()
	if _, err := o.Play("tone", 0.03); err != nil {
		t.Fatal(err)
	}
	t.Logf("play after idle stop: %v", time.Since(start))
	time.Sleep(200 * time.Millisecond)
}
