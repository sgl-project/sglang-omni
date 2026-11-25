package main

import (
	"slices"
	"sync"
	"time"

	"github.com/hehehai/actnowkit/sounds"
	"github.com/sgl-project/sglang-omni/Voxt/internal/macx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// The stop cue lasts as long as the wait it covers (docs/behavior.md 9.2,
// 11-24): it starts as the recording stops, as before; once the result is
// ready (the moment the overlay stops thinking) it fades out over
// stopCueFade, softly, though not before stopCueMin, so a result that
// comes at once still lets both notes of a two-note cue be heard
// (Kalimba's second starts at 0.13 s); a result slower than the cue lets
// it ring out. A new recording fades it at once.
const (
	stopCueMin  = 200 * time.Millisecond
	stopCueFade = 200 * time.Millisecond
)

// fadingVoice is a cue's sound that can end with a fade (cueOutput's
// stop cues on the Mac).
type fadingVoice interface {
	sounds.Voice
	FadeOut(d time.Duration)
}

// stopCueTail follows one session's stop cue: when it started and whether
// the session's result is ready. Safe for concurrent use: the cue starts on
// a goroutine (playCueWait), the session's changes come on the main thread.
type stopCueTail struct {
	mu     sync.Mutex
	now    func() time.Time            // time.Now; tests set a clock
	after  func(time.Duration, func()) // time.AfterFunc; tests run it themselves
	id     string                      // the session whose stop cue it is ("" none)
	sound  string                      // the cue's sound
	length time.Duration               // its audible part
	voice  fadingVoice                 // nil until it starts
	start  time.Time
	ready  bool
}

// followStopCue keeps the stop cue's tail with the session's change
// (recordingAudio): a new recording fades it, the stop arms it with the
// cue c plays (when it plays), and the end of the wait (s past recognition
// and the clean-up: delivering, or ended) lets it fade.
func (a *app) followStopCue(c cue, s session.Snapshot, t sounds.Target, plays bool) {
	switch c {
	case startCue:
		a.stopTail.cut()
		// The stop cue's players are made now, not as it should start.
		if stop, p := a.cueTarget(stopCue), a.cuePlayer; stop.Kind == sounds.KindCatalog && p != nil {
			go func() { _ = p.Preload(stop.Name) }() // a load that fails fails the play too
		}
	case stopCue:
		if plays {
			a.stopTail.arm(s.ID, t, t.Length(a.soundCat))
		}
	}
	switch s.State {
	case session.Recording, session.Recognizing, session.Composing:
	default:
		a.stopTail.resultReady(s.ID)
	}
}

// arm follows session id's stop cue t, about to play.
func (c *stopCueTail) arm(id string, t sounds.Target, length time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.id, c.sound, c.length, c.voice, c.ready = id, t.Name, length, nil, false
}

// started is told that the cue t plays as v (playCueWait): the armed stop
// cue's start, when t is that cue and v can fade.
func (c *stopCueTail) started(t sounds.Target, v sounds.Voice) {
	f, ok := v.(fadingVoice)
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.id == "" || c.voice != nil || t.Kind != sounds.KindCatalog || t.Name != c.sound {
		return
	}
	if !ok {
		c.id = "" // it cannot fade: it rings out
		return
	}
	c.voice, c.start = f, c.clock()
	c.schedule()
}

// resultReady is told that session id's wait ended.
func (c *stopCueTail) resultReady(id string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if id == "" || id != c.id || c.ready {
		return
	}
	c.ready = true
	c.schedule()
}

// cut fades the stop cue now: a new recording starts.
func (c *stopCueTail) cut() {
	c.mu.Lock()
	v := c.voice
	c.id, c.voice = "", nil
	c.mu.Unlock()
	if v != nil {
		v.FadeOut(stopCueFade)
	}
}

// schedule fades the started cue of a ready session at stopCueMin from its
// start, or now when that has passed; a cue that ends by then is left to
// end. The caller holds mu.
func (c *stopCueTail) schedule() {
	if c.voice == nil || !c.ready {
		return
	}
	now := c.clock()
	wait := max(0, c.start.Add(stopCueMin).Sub(now))
	v, id := c.voice, c.id
	if !now.Add(wait).Before(c.start.Add(c.length - stopCueFade)) {
		c.id, c.voice = "", nil
		return
	}
	after := c.after
	if after == nil {
		after = func(d time.Duration, f func()) { time.AfterFunc(d, f) }
	}
	after(wait, func() {
		c.mu.Lock()
		mine := c.voice == v && c.id == id
		if mine {
			c.id, c.voice = "", nil
		}
		c.mu.Unlock()
		if mine {
			v.FadeOut(stopCueFade)
		}
	})
}

func (c *stopCueTail) clock() time.Time {
	if c.now != nil {
		return c.now()
	}
	return time.Now()
}

// fadingPlayer is one player of a stop cue (macx.FadingSound).
type fadingPlayer interface {
	Play(volume float32) (gen uint64, ok bool)
	Playing() bool
	Stop(gen uint64)
	FadeOut(gen uint64, d time.Duration)
}

// fadingVoices is how many players a stop cue has: a cue and a preview of
// it may overlap.
const fadingVoices = 2

// cueOutput is the Mac's output for the cues: actnowkit's AVAudioEngine,
// which starts a sound within milliseconds but can only cut it off, and
// for each stop cue (a sound for record.stop) AVAudioPlayers, which can
// fade out (stopCueTail). A stop cue whose players could not be made plays
// on the engine.
type cueOutput struct {
	sounds.Output
	isStop    func(id string) bool
	newFading func(data []byte) (fadingPlayer, error)
	mu        sync.Mutex
	stops     map[string][]fadingPlayer
}

// newCueOutput is the app's output for cat's sounds.
func newCueOutput(cat *sounds.Catalog) *cueOutput {
	return &cueOutput{
		Output: sounds.DefaultOutput(),
		isStop: func(id string) bool {
			s, ok := cat.Sound(id)
			return ok && slices.Contains(s.Uses, recordStopEvent.ID)
		},
		newFading: func(data []byte) (fadingPlayer, error) {
			s, err := macx.NewFadingSound(data)
			if err != nil {
				return nil, err
			}
			return s, nil
		},
	}
}

func (o *cueOutput) Load(key string, data []byte) error {
	if err := o.Output.Load(key, data); err != nil {
		return err
	}
	if !o.isStop(key) {
		return nil
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if _, done := o.stops[key]; done {
		return nil
	}
	var ps []fadingPlayer
	for range fadingVoices {
		p, err := o.newFading(data)
		if err != nil {
			break
		}
		ps = append(ps, p)
	}
	if o.stops == nil {
		o.stops = map[string][]fadingPlayer{}
	}
	o.stops[key] = ps
	return nil
}

func (o *cueOutput) Play(key string, volume float64) (sounds.Voice, error) {
	o.mu.Lock()
	ps := o.stops[key]
	o.mu.Unlock()
	if len(ps) > 0 {
		p := ps[0]
		for _, q := range ps {
			if !q.Playing() {
				p = q
				break
			}
		}
		if gen, ok := p.Play(float32(min(1, max(0, volume)))); ok {
			return fadeVoice{p, gen}, nil
		}
	}
	return o.Output.Play(key, volume)
}

// fadeVoice is one start of a stop cue's player.
type fadeVoice struct {
	p   fadingPlayer
	gen uint64
}

func (v fadeVoice) Stop()                   { v.p.Stop(v.gen) }
func (v fadeVoice) FadeOut(d time.Duration) { v.p.FadeOut(v.gen, d) }
