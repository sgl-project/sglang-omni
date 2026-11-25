package main

import (
	"errors"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/hehehai/actnowkit/sounds"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// fadeLog records what the stop cue's voices did, at the fake clock's time.
type fadeLog struct {
	mu    sync.Mutex
	clock func() time.Time
	t0    time.Time
	log   []string
}

func (l *fadeLog) add(s string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.log = append(l.log, s+"@"+l.clock().Sub(l.t0).String())
}

func (l *fadeLog) take() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	s := l.log
	l.log = nil
	return s
}

// fakeFading is a player of fakeCueOutput's: it plays until stopped or
// faded.
type fakeFading struct {
	key     string
	log     *fadeLog
	mu      sync.Mutex
	gen     uint64
	playing bool
}

func (p *fakeFading) Play(float32) (uint64, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.gen++
	p.playing = true
	p.log.add("play " + p.key)
	return p.gen, true
}

func (p *fakeFading) Playing() bool { p.mu.Lock(); defer p.mu.Unlock(); return p.playing }

func (p *fakeFading) Stop(gen uint64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if gen == p.gen {
		p.playing = false
	}
}

func (p *fakeFading) FadeOut(gen uint64, d time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if gen == p.gen {
		p.playing = false
		p.log.add("fade " + p.key + " " + d.String())
	}
}

// fakeCueOutput is cueOutput over a silent engine, with fakeFading players.
func fakeCueOutput(cat *sounds.Catalog, log *fadeLog) *cueOutput {
	o := newCueOutput(cat)
	o.Output = sounds.Silent{}
	o.newFading = func([]byte) (fadingPlayer, error) { return &fakeFading{log: log}, nil }
	return o
}

// stopCueRig is the mute rig whose cues play on fakeCueOutput, through
// playCue, with the stop cue's tail on the rig's fake clock.
func newStopCueRig(t *testing.T) (*muteRig, *fadeLog) {
	r := newMuteRig(t)
	log := &fadeLog{clock: r.ft.clock, t0: r.ft.clock()}
	out := fakeCueOutput(r.soundCat, log)
	r.cuePlayer = sounds.NewPlayer(r.soundCat, out)
	r.playSound = func(s sounds.Target) { r.playCue(s) }
	r.stopTail.now = r.ft.clock
	r.stopTail.after = func(d time.Duration, fn func()) { r.laterHook(d, fn) }
	if err := r.cuePlayer.Preload("soft-kalimba-stop"); err != nil {
		t.Fatal(err)
	}
	for _, p := range out.stops["soft-kalimba-stop"] {
		p.(*fakeFading).key = "soft-kalimba-stop" // the log names the cue
	}
	return r, log
}

// TestStopCueEndsWithTheWait: the stop cue starts as the recording stops;
// a result that comes at once fades it at stopCueMin, one in the middle of
// the cue fades it then, one slower than the cue lets it ring out, and a
// new recording fades it at once (docs/behavior.md 9.2).
func TestStopCueEndsWithTheWait(t *testing.T) {
	stopAt := func(r *muteRig, id string) {
		r.feed(id, session.Recording)
		r.ft.advance(time.Second) // the start cue played, the output is muted
		r.feed(id, session.Recognizing)
	}
	t.Run("a result at once", func(t *testing.T) {
		r, log := newStopCueRig(t)
		stopAt(r, "s1")
		r.ft.advance(20 * time.Millisecond)
		r.feed("s1", session.Delivering)
		r.ft.advance(time.Second)
		want := []string{"play soft-kalimba-stop@1s", "fade soft-kalimba-stop 200ms@1.2s"}
		if got := log.take(); !slices.Equal(got, want) {
			t.Errorf("got %q, want %q", got, want)
		}
	})
	t.Run("a result in the cue", func(t *testing.T) {
		r, log := newStopCueRig(t)
		stopAt(r, "s1")
		r.ft.advance(150 * time.Millisecond)
		r.feed("s1", session.Composing) // the clean-up is still the wait
		r.ft.advance(100 * time.Millisecond)
		r.feed("s1", session.Done)
		r.ft.advance(time.Second)
		want := []string{"play soft-kalimba-stop@1s", "fade soft-kalimba-stop 200ms@1.25s"}
		if got := log.take(); !slices.Equal(got, want) {
			t.Errorf("got %q, want %q", got, want)
		}
	})
	t.Run("a result after the cue", func(t *testing.T) {
		r, log := newStopCueRig(t)
		stopAt(r, "s1")
		r.ft.advance(500 * time.Millisecond) // Kalimba's stop is heard 0.56 s
		r.feed("s1", session.Done)
		r.ft.advance(time.Second)
		want := []string{"play soft-kalimba-stop@1s"}
		if got := log.take(); !slices.Equal(got, want) {
			t.Errorf("got %q, want %q", got, want)
		}
	})
	t.Run("a failure ends the wait too", func(t *testing.T) {
		r, log := newStopCueRig(t)
		stopAt(r, "s1")
		r.feed("s1", session.Failed)
		r.ft.advance(time.Second)
		want := []string{"play soft-kalimba-stop@1s", "fade soft-kalimba-stop 200ms@1.2s"}
		if got := log.take(); !slices.Equal(got, want) {
			t.Errorf("got %q, want %q", got, want)
		}
	})
	t.Run("a new recording", func(t *testing.T) {
		r, log := newStopCueRig(t)
		stopAt(r, "s1")
		r.ft.advance(100 * time.Millisecond)
		r.feed("s2", session.Recording)
		r.feed("s1", session.Done) // s1 ends after s2 started: nothing more
		r.ft.advance(time.Second)
		got := log.take()
		want := []string{"play soft-kalimba-stop@1s", "fade soft-kalimba-stop 200ms@1.1s"}
		if !slices.Equal(got, want) {
			t.Errorf("got %q, want %q", got, want)
		}
	})
	t.Run("sounds off", func(t *testing.T) {
		r, log := newStopCueRig(t)
		r.prefs.Audio.Sounds = false
		stopAt(r, "s1")
		r.feed("s1", session.Done)
		r.ft.advance(time.Second)
		if got := log.take(); len(got) != 0 {
			t.Errorf("got %q, want nothing", got)
		}
	})
}

// TestStopCueTail covers the tail's cases the session does not reach: a
// result before the cue started, another sound, a voice that cannot fade,
// another session's end, and a stop cue armed again before the fade.
func TestStopCueTail(t *testing.T) {
	ft := new(fakeTimers)
	log := &fadeLog{clock: ft.clock, t0: ft.clock()}
	var (
		timers []func()
		waits  []time.Duration
	)
	c := &stopCueTail{now: ft.clock, after: func(d time.Duration, fn func()) { waits, timers = append(waits, d), append(timers, fn) }}
	kal := sounds.Target{Kind: sounds.KindCatalog, Name: "soft-kalimba-stop"}
	voice := func() fadeVoice { p := &fakeFading{key: "k", log: log}; g, _ := p.Play(1); return fadeVoice{p, g} }
	const length = 560 * time.Millisecond

	c.arm("s1", kal, length)
	c.resultReady("s1") // the result before the cue started (the output was slow to come back)
	ft.now += 200 * time.Millisecond
	c.started(kal, voice())
	if !slices.Equal(waits, []time.Duration{stopCueMin}) {
		t.Fatalf("fades in %v, want one in %v", waits, stopCueMin)
	}
	timers[0]()
	if got := log.take(); !slices.Equal(got, []string{"play k@200ms", "fade k 200ms@200ms"}) {
		t.Errorf("a result before the start: %q", got)
	}

	timers = nil
	c.arm("s2", kal, length)
	c.started(sounds.Target{Kind: sounds.KindCatalog, Name: "soft-wood-stop"}, voice())
	c.started(kal, sounds.Voice(nil))
	c.resultReady("s2")
	if len(timers) != 0 {
		t.Errorf("another sound, or a voice that cannot fade: %d fades scheduled", len(timers))
	}

	c.arm("s3", kal, length)
	c.started(kal, voice())
	c.resultReady("s9")
	if len(timers) != 0 {
		t.Errorf("another session's end scheduled a fade")
	}
	c.resultReady("s3")
	c.arm("s4", kal, length) // the next stop, before s3's fade is due
	for _, fn := range timers {
		fn()
	}
	if got := log.take(); slices.ContainsFunc(got, func(s string) bool { return strings.HasPrefix(s, "fade") }) {
		t.Errorf("s3's fade ran after s4 armed: %q", got)
	}
}

// TestCueOutput: a stop cue plays on its fading players (the free one, or
// the first when both play), any other sound on the engine, and a stop cue
// whose players could not be made on the engine too.
func TestCueOutput(t *testing.T) {
	cat := testSounds()
	ft := new(fakeTimers)
	log := &fadeLog{clock: ft.clock, t0: ft.clock()}
	rec := &sounds.Recorder{}
	o := fakeCueOutput(cat, log)
	o.Output = rec
	for _, id := range []string{"soft-kalimba-stop", "soft-kalimba-start"} {
		if err := o.Load(id, []byte("RIFF")); err != nil {
			t.Fatal(err)
		}
	}
	if n := len(o.stops["soft-kalimba-stop"]); n != fadingVoices {
		t.Fatalf("the stop cue has %d players, want %d", n, fadingVoices)
	}
	if _, ok := o.stops["soft-kalimba-start"]; ok {
		t.Error("the start cue got fading players")
	}
	ps := o.stops["soft-kalimba-stop"]
	v1, _ := o.Play("soft-kalimba-stop", 0.7)
	v2, _ := o.Play("soft-kalimba-stop", 0.7)
	v3, _ := o.Play("soft-kalimba-stop", 0.7)
	f1, ok1 := v1.(fadeVoice)
	f2, ok2 := v2.(fadeVoice)
	f3, ok3 := v3.(fadeVoice)
	if !ok1 || !ok2 || !ok3 || f1.p != ps[0] || f2.p != ps[1] || f3.p != ps[0] {
		t.Errorf("stop cue voices: %#v %#v %#v", v1, v2, v3)
	}
	if v, _ := o.Play("soft-kalimba-start", 0.7); v == nil {
		t.Error("the start cue did not play")
	} else if _, fades := v.(fadeVoice); fades {
		t.Error("the start cue played on a fading player")
	}
	if got := rec.Loaded(); !slices.Equal(got, []string{"soft-kalimba-stop", "soft-kalimba-start"}) {
		t.Errorf("engine loaded %q", got)
	}
	if got := rec.Played(); !slices.Equal(got, []string{"soft-kalimba-start@0.7"}) {
		t.Errorf("engine played %q", got)
	}

	broken := fakeCueOutput(cat, log)
	broken.Output = &sounds.Recorder{}
	broken.newFading = func([]byte) (fadingPlayer, error) { return nil, errors.New("no AVFoundation") }
	if err := broken.Load("soft-kalimba-stop", []byte("RIFF")); err != nil {
		t.Fatal(err)
	}
	if v, err := broken.Play("soft-kalimba-stop", 0.7); err != nil {
		t.Fatal(err)
	} else if _, fades := v.(fadeVoice); fades {
		t.Error("a stop cue without players faded")
	}
}
