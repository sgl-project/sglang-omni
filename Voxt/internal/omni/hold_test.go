package omni

import (
	"context"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/asr"
	"github.com/sgl-project/sglang-omni/Voxt/internal/omni/omnitest"
)

func TestHoldBlocksIdleUnload(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, func(c *Config) { c.IdleTimeout = 200 * time.Millisecond })
	clock := new(fakeClock).install(f.rt)
	spec, _ := Lookup(ASRFast)
	rel, err := f.rt.Hold(context.Background(), HoldTarget{Spec: spec, Dir: f.model(ASRFast)})
	if err != nil {
		t.Fatal(err)
	}
	clock.advance(500 * time.Millisecond)
	if n := len(f.records("shutdown")); n != 0 {
		t.Fatalf("hold active: %d shutdowns", n)
	}
	rel()
	clock.advance(200 * time.Millisecond)
	if n := len(f.records("shutdown")); n != 1 {
		t.Errorf("after release: %d shutdowns, want 1", n)
	}
}

func TestHoldAfterStreamClose(t *testing.T) {
	f := newFake(t, omnitest.Opts{}, func(c *Config) { c.IdleTimeout = 200 * time.Millisecond })
	clock := new(fakeClock).install(f.rt)
	a := f.asr(t, ASRFast)
	st, err := a.Open(context.Background(), asr.StreamRequest{SampleRate: sampleRate})
	if err != nil {
		t.Fatal(err)
	}
	st.Close()
	spec, _ := Lookup(ASRFast)
	rel, err := f.rt.Hold(context.Background(), HoldTarget{Spec: spec, Dir: f.model(ASRFast)})
	if err != nil {
		t.Fatal(err)
	}
	clock.advance(500 * time.Millisecond)
	if n := len(f.records("shutdown")); n != 0 {
		t.Fatalf("hold after close: %d shutdowns", n)
	}
	rel()
	clock.advance(200 * time.Millisecond)
	if n := len(f.records("shutdown")); n != 1 {
		t.Errorf("after hold release: %d shutdowns", n)
	}
}
