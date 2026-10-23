package omni

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/asr"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/omni/omnitest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/resident"
)

// heard records Config.OnModel's events and BeforeLoad's calls.
type heard struct {
	mu     sync.Mutex
	events []resident.Event
	before []string
}

func (h *heard) tune(c *Config) {
	c.OnModel = func(ev resident.Event) { h.mu.Lock(); h.events = append(h.events, ev); h.mu.Unlock() }
	c.BeforeLoad = func(_ context.Context, id string) { h.mu.Lock(); h.before = append(h.before, id); h.mu.Unlock() }
}

// of is the events of model id as "kind pid".
func (h *heard) of(id string) []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	var out []string
	for _, ev := range h.events {
		if ev.ID == id {
			out = append(out, fmt.Sprintf("%s %d", ev.Kind, ev.PID))
		}
	}
	return out
}

func (h *heard) wait(t *testing.T, id string, want ...string) {
	t.Helper()
	for end := time.Now().Add(10 * time.Second); time.Now().Before(end); time.Sleep(10 * time.Millisecond) {
		if slices.Equal(h.of(id), want) {
			return
		}
	}
	t.Fatalf("%s's events %q, want %q", id, h.of(id), want)
}

func (h *heard) beforeLoads() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.before)
}

// TestModelEvents: a speech model's server is loaded once ready (after
// BeforeLoad), idle without a lease, busy while leased (two at once are
// one busy), idle again, and unloaded when Unload stops it; the next use
// loads it again.
func TestModelEvents(t *testing.T) {
	var h heard
	f := newFake(t, omnitest.Opts{}, h.tune)
	ctx := context.Background()
	a := f.asr(t, ASRFast)
	l1, err := f.rt.acquire(ctx, a.spec, a.dir)
	if err != nil {
		t.Fatal(err)
	}
	pid := f.records("launch")[0].PID
	if got := h.beforeLoads(); !slices.Equal(got, []string{ASRFast}) {
		t.Errorf("BeforeLoad %v", got)
	}
	l2, err := f.rt.acquire(ctx, a.spec, a.dir)
	if err != nil {
		t.Fatal(err)
	}
	p := func(k resident.Kind) string { return fmt.Sprintf("%s %d", k, pid) }
	h.wait(t, ASRFast, p(resident.Loaded), p(resident.Idle), p(resident.Busy))
	l1.release()
	l2.release()
	l2.release() // once only
	h.wait(t, ASRFast, p(resident.Loaded), p(resident.Idle), p(resident.Busy), p(resident.Idle))
	f.rt.Unload(ASRFast)
	h.wait(t, ASRFast, p(resident.Loaded), p(resident.Idle), p(resident.Busy), p(resident.Idle), p(resident.Unloaded))

	if _, err := a.Transcribe(ctx, asr.FileRequest{Audio: tone(time.Second, 0.3)}); err != nil {
		t.Fatal(err)
	}
	pid2 := f.records("launch")[1].PID
	q := func(k resident.Kind) string { return fmt.Sprintf("%s %d", k, pid2) }
	h.wait(t, ASRFast, p(resident.Loaded), p(resident.Idle), p(resident.Busy), p(resident.Idle), p(resident.Unloaded),
		q(resident.Loaded), q(resident.Idle), q(resident.Busy), q(resident.Idle))
	if got := h.beforeLoads(); len(got) != 2 {
		t.Errorf("BeforeLoad %v", got)
	}
}

// TestModelEventsEnds: every end of a process is an unload: the idle
// stop, a crash, a switch of speech model, Close (Silero's too).
func TestModelEventsEnds(t *testing.T) {
	ctx := context.Background()
	pcm := tone(time.Second, 0.3)
	t.Run("idle", func(t *testing.T) {
		var h heard
		f := newFake(t, omnitest.Opts{}, func(c *Config) { h.tune(c); c.IdleTimeout = 200 * time.Millisecond })
		if _, err := f.asr(t, ASRFast).Transcribe(ctx, asr.FileRequest{Audio: pcm}); err != nil {
			t.Fatal(err)
		}
		pid := f.records("launch")[0].PID
		h.wait(t, ASRFast, "loaded "+itoa(pid), "idle "+itoa(pid), "busy "+itoa(pid), "idle "+itoa(pid), "unloaded "+itoa(pid))
	})
	t.Run("crash", func(t *testing.T) {
		var h heard
		f := newFake(t, omnitest.Opts{Mode: omnitest.ModeCrashFinal}, h.tune)
		if _, err := f.asr(t, ASRFast).Transcribe(ctx, asr.FileRequest{Audio: pcm}); err == nil {
			t.Fatal("no error from a crash")
		}
		pid := f.records("launch")[0].PID
		h.wait(t, ASRFast, "loaded "+itoa(pid), "idle "+itoa(pid), "busy "+itoa(pid), "unloaded "+itoa(pid)) // gone while used
	})
	t.Run("switch and close", func(t *testing.T) {
		var h heard
		f := newFake(t, omnitest.Opts{}, h.tune)
		v, err := NewVAD(f.rt, f.model(VADSilero))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := v.Speech(ctx, pcm, 16000); err != nil {
			t.Fatal(err)
		}
		if _, err := f.asr(t, ASRFast).Transcribe(ctx, asr.FileRequest{Audio: pcm}); err != nil {
			t.Fatal(err)
		}
		if _, err := f.asr(t, ASRAccurate).Transcribe(ctx, asr.FileRequest{Audio: pcm}); err != nil {
			t.Fatal(err)
		}
		unloaded := func(id string) bool {
			s := h.of(id)
			return len(s) > 0 && strings.HasPrefix(s[len(s)-1], "unloaded ")
		}
		eventually(t, "the switch's unload", func() bool { return unloaded(ASRFast) })
		if unloaded(VADSilero) || unloaded(ASRAccurate) {
			t.Errorf("unloaded at the switch: %q %q", h.of(VADSilero), h.of(ASRAccurate))
		}
		f.rt.Close()
		for _, id := range []string{VADSilero, ASRAccurate} {
			eventually(t, id+" unloaded", func() bool { return unloaded(id) })
		}
		if got := h.beforeLoads(); !slices.Equal(got, []string{VADSilero, ASRFast, ASRAccurate}) {
			t.Errorf("BeforeLoad %v", got)
		}
	})
}
