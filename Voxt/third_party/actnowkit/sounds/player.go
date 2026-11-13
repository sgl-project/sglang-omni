package sounds

import (
	"fmt"
	"strconv"
	"sync"
	"time"
)

// Output plays sounds: the platform's (DefaultOutput: AVAudioPlayer and
// NSSound on macOS, silence elsewhere) or a test's (Recorder).
type Output interface {
	// Load prepares sound data (a WAV, CAF … file's bytes) under key, so
	// that Play starts it at once. Loading a key again is a no-op.
	Load(key string, data []byte) error
	// Play starts the sound loaded under key at volume (0–1) and returns
	// at once. Quick repeats overlap rather than cut each other off.
	Play(key string, volume float64) (Voice, error)
	// PlaySystem starts the macOS system sound name ("Pop").
	PlaySystem(name string, volume float64) (Voice, error)
}

// Voice is a sound that plays.
type Voice interface {
	// Stop ends the sound now; one that ended is left alone.
	Stop()
}

// Player plays the sounds of a catalogue (nil: none, every event plays its
// fallback) on an Output. It is safe for concurrent use; Play and its
// kin return at once, but the first play of a sound reads and decodes it
// (Preload does that ahead).
type Player struct {
	cat *Catalog
	out Output

	mu      sync.Mutex
	loaded  map[string]bool
	preview Voice
	// previewing is the choice the last preview plays, until previewEnd.
	previewing string
	previewEnd time.Time
	now        func() time.Time
}

// NewPlayer is a player of cat on out (nil: DefaultOutput()).
func NewPlayer(cat *Catalog, out Output) *Player {
	if out == nil {
		out = DefaultOutput()
	}
	return &Player{cat: cat, out: out, loaded: map[string]bool{}, now: time.Now}
}

// Catalog is the player's catalogue (nil when the build has none).
func (p *Player) Catalog() *Catalog { return p.cat }

// load makes sure the catalogue sound id is loaded on the output.
func (p *Player) load(id string) error {
	p.mu.Lock()
	done := p.loaded[id]
	p.mu.Unlock()
	if done {
		return nil
	}
	data, err := p.cat.Data(id)
	if err != nil {
		return err
	}
	if err := p.out.Load(id, data); err != nil {
		return fmt.Errorf("sounds: load %s: %w", id, err)
	}
	p.mu.Lock()
	p.loaded[id] = true
	p.mu.Unlock()
	return nil
}

// Preload loads the catalogue sounds among choices (others are skipped),
// so that their first play starts at once. It returns the first error.
func (p *Player) Preload(choices ...string) error {
	var first error
	for _, c := range choices {
		if _, ok := p.cat.Sound(c); !ok {
			continue
		}
		if err := p.load(c); err != nil && first == nil {
			first = err
		}
	}
	return first
}

// Play plays t; nothing for KindNone.
func (p *Player) Play(t Target) (Voice, error) {
	switch t.Kind {
	case KindCatalog:
		if err := p.load(t.Name); err != nil {
			return nil, err
		}
		return p.out.Play(t.Name, t.Volume)
	case KindSystem:
		return p.out.PlaySystem(t.Name, t.Volume)
	}
	return nil, nil
}

// PlayEvent plays what ev plays with prefs (nothing while they are off)
// and returns what that was.
func (p *Player) PlayEvent(ev Event, prefs Prefs) (Target, error) {
	t := prefs.Target(p.cat, ev)
	_, err := p.Play(t)
	return t, err
}

// Preview plays choice for a picker at volume (percent), stopping the
// preview before it: ▶ on a row, "Test" on an event (pass the event's
// choice and Fallback). Choosing None stops the last preview and plays
// nothing.
func (p *Player) Preview(choice, fallback string, volume int) (Target, error) {
	t := Resolve(p.cat, choice, fallback, volume)
	p.StopPreview()
	v, err := p.Play(t)
	if v != nil {
		p.mu.Lock()
		p.preview = v
		if t.Kind != KindNone && err == nil {
			d := t.Length(p.cat)
			if d <= 0 {
				d = UnknownLength
			}
			p.previewing, p.previewEnd = choice, p.now().Add(d)
		}
		p.mu.Unlock()
	}
	return t, err
}

// UnknownLength is how long a preview of a sound of unknown length (a
// system sound not measured) is taken to play.
const UnknownLength = 800 * time.Millisecond

// Previewing is the choice the last preview plays and how long it has
// left; "" when none plays: a picker's ▶ shows its state (and redraws
// after left).
func (p *Player) Previewing() (choice string, left time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.previewing == "" {
		return "", 0
	}
	left = p.previewEnd.Sub(p.now())
	if left <= 0 {
		p.previewing = ""
		return "", 0
	}
	return p.previewing, left
}

// SetClock replaces the player's clock (tests of Previewing).
func (p *Player) SetClock(now func() time.Time) {
	p.mu.Lock()
	p.now = now
	p.mu.Unlock()
}

// StopPreview stops the last preview if it still plays.
func (p *Player) StopPreview() {
	p.mu.Lock()
	v := p.preview
	p.preview = nil
	p.previewing = ""
	p.mu.Unlock()
	if v != nil {
		v.Stop()
	}
}

// Silent is an Output that plays nothing (DefaultOutput off macOS).
type Silent struct{}

func (Silent) Load(string, []byte) error                 { return nil }
func (Silent) Play(string, float64) (Voice, error)       { return silentVoice{}, nil }
func (Silent) PlaySystem(string, float64) (Voice, error) { return silentVoice{}, nil }

type silentVoice struct{}

func (silentVoice) Stop() {}

// Recorder is an Output for tests: it records what was loaded and played
// ("chime@0.7", "system:Pop@0.5", and "stop chime" for a stopped voice).
type Recorder struct {
	mu     sync.Mutex
	loaded []string
	played []string
	// Err, when set, is returned by Load, Play and PlaySystem.
	Err error
}

func (r *Recorder) Load(key string, data []byte) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.Err != nil {
		return r.Err
	}
	r.loaded = append(r.loaded, key)
	return nil
}

func (r *Recorder) Play(key string, volume float64) (Voice, error) {
	return r.play(key, volume)
}

func (r *Recorder) PlaySystem(name string, volume float64) (Voice, error) {
	return r.play(SystemPrefix+name, volume)
}

func (r *Recorder) play(key string, volume float64) (Voice, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.Err != nil {
		return nil, r.Err
	}
	r.played = append(r.played, key+"@"+strconv.FormatFloat(volume, 'f', -1, 64))
	return recVoice{r, key}, nil
}

// Played lists what was played and stopped, in order.
func (r *Recorder) Played() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.played...)
}

// Loaded lists the keys loaded, in order.
func (r *Recorder) Loaded() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.loaded...)
}

// Reset forgets what was recorded.
func (r *Recorder) Reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.played, r.loaded = nil, nil
}

type recVoice struct {
	r   *Recorder
	key string
}

func (v recVoice) Stop() {
	v.r.mu.Lock()
	defer v.r.mu.Unlock()
	v.r.played = append(v.r.played, "stop "+v.key)
}
