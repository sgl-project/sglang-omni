package main

import (
	"sync"
	"time"

	"github.com/hehehai/actnowkit/sounds"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// The recording's cues: the family's Pebble and Nib (a soft tap as
// recording starts, a short tick as it stops; Tally and Wrap's beeps
// until 11-10, which users found harsh), Voxt's Pop and Tink in a build
// without the family's sounds.
var (
	recordStartEvent = sounds.Event{ID: "record.start", Name: sounds.Text{En: "Recording starts", Zh: "开始录音"},
		Default: "pebble", Fallback: "Pop", Uses: []string{"record.start"}}
	recordStopEvent = sounds.Event{ID: "record.stop", Name: sounds.Text{En: "Recording stops", Zh: "结束录音"},
		Default: "nib", Fallback: "Tink", Uses: []string{"record.stop"}}
)

// soundsState is the cues' sounds and the previews of Settings → Audio →
// Sound.
type soundsState struct {
	sounds struct {
		previewing string // the cue whose Try shows the accent ("" none)
		seq        int    // numbers the previews: a preview's timer ends only its own
		// playing is the choice a preview plays until playingEnd (a
		// pair's ▶ in the accent).
		playing    string
		playingEnd time.Time
	}
	soundCat *sounds.Catalog // the family's sounds this build has (nil: none)
	// sysSounds are the Mac's system sounds (/System/Library/Sounds),
	// read once (sysRead); tests set them.
	sysSounds []string
	sysRead   bool
	// cuePlayer plays the cues and the previews (the Mac's output; a
	// silent one in tests).
	cuePlayer *sounds.Player
	// startSound starts a sound and returns what stops it (cuePlayer, on
	// a goroutine); tests record the sounds instead.
	startSound func(sounds.Target) (stop func())
	sndPlayer  previewPlayer
}

// openSounds loads the family's sounds the app ships and makes the
// player; tests replace soundCat (testSounds) and the player's output.
func (a *app) openSounds() {
	a.soundCat = loadCueSounds()
	a.cuePlayer = sounds.NewPlayer(a.soundCat, nil)
	a.startSound = a.startPreview
}

// systemSounds are the Mac's system sounds, for the macOS choice.
func (a *app) systemSounds() []string {
	if !a.sysRead {
		a.sysSounds, a.sysRead = sounds.SystemSounds(""), true
	}
	return a.sysSounds
}

// soundPrefs are Settings → Audio's cues as actnowkit's prefs.
func (a *app) soundPrefs() sounds.Prefs {
	au := a.prefs.Audio
	p := sounds.Prefs{On: au.Sounds, Volume: au.SoundVolumeOrDefault()}
	p.Choose(recordStartEvent, au.StartSound)
	p.Choose(recordStopEvent, au.StopSound)
	return p
}

// cueEvent is a cue's event.
func cueEvent(c cue) sounds.Event {
	if c == stopCue {
		return recordStopEvent
	}
	return recordStartEvent
}

// cueTarget is what a cue plays now (nothing while sounds are off).
func (a *app) cueTarget(c cue) sounds.Target {
	return a.soundPrefs().Target(a.soundCat, cueEvent(c))
}

// chooseSound keeps choice as the event's sound.
func (a *app) chooseSound(ev sounds.Event, choice string) {
	p := a.soundPrefs()
	p.Choose(ev, choice)
	a.keepSounds(p)
}

// chooseSounds keeps start and stop as the cues' sounds, at once.
func (a *app) chooseSounds(start, stop string) {
	p := a.soundPrefs()
	p.Choose(recordStartEvent, start)
	p.Choose(recordStopEvent, stop)
	a.keepSounds(p)
}

// keepSounds saves p's choices (a default is stored as "").
func (a *app) keepSounds(p sounds.Prefs) {
	a.updateSettings(func(s *settings.Settings) {
		s.Audio.StartSound, s.Audio.StopSound = p.Events[recordStartEvent.ID], p.Events[recordStopEvent.ID]
	})
}

// soundLabelWithTranslator names a choice: a family sound's name in the
// translator's language, a system sound's, None.
func soundLabelWithTranslator(translator *i18n.Translator, catalog *sounds.Catalog, choice string) string {
	switch {
	case choice == sounds.None:
		return translator.T("None")
	case sounds.IsSystem(choice):
		return sounds.SystemName(choice)
	}
	if s, ok := catalog.Sound(choice); ok {
		return s.Name.In(translator.Locale())
	}
	return choice
}

// playCueWait plays t and returns when it ended (the start cue plays
// before the system's output is muted): its file's length, or startCueMax
// for a system sound of unknown length.
func (a *app) playCueWait(t sounds.Target) {
	if _, err := a.cuePlayer.Play(t); err != nil {
		return
	}
	d := startCueMax
	if s, ok := a.soundCat.Sound(t.Name); ok && t.Kind == sounds.KindCatalog {
		d = time.Duration(s.Duration * float64(time.Second))
	} else if l := t.Length(a.soundCat); l > 0 {
		d = l
	}
	time.Sleep(d)
}

// startPreview is cuePlayer for previews: what stops the sound (nothing
// when it could not start).
func (a *app) startPreview(t sounds.Target) func() {
	v, err := a.cuePlayer.Play(t)
	if err != nil || v == nil {
		return nil
	}
	return v.Stop
}

// Times of the previews.
const (
	// cueMargin follows a sound's audible part (Voxt's 80 ms, and the
	// output's start).
	cueMargin = 150 * time.Millisecond
	// previewMin is the least Try shows the accent (Voxt's).
	previewMin = 650 * time.Millisecond
)

// previewPlayer plays one preview at a time, off the main thread: a new
// one stops the one before; one that comes after a newer one (goroutines
// run in any order) is dropped.
type previewPlayer struct {
	mu   sync.Mutex
	seq  int
	stop func()
}

func (p *previewPlayer) play(seq int, t sounds.Target, start func(sounds.Target) func()) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if seq <= p.seq {
		return
	}
	p.seq = seq
	if p.stop != nil {
		p.stop()
		p.stop = nil
	}
	if start != nil && t.Kind != sounds.KindNone {
		p.stop = start(t)
	}
}

// previewSound plays choice for ev (Try, a pair's ▶), at the cues'
// volume, the sounds on or off; key names the Try button that shows the
// accent meanwhile ("" none). It has nothing to do with a session: one
// that records goes on (its muted output may hide the preview).
func (a *app) previewSound(ev sounds.Event, choice, key string) {
	t := sounds.Resolve(a.soundCat, choice, ev.Fallback, a.prefs.Audio.SoundVolumeOrDefault())
	a.sounds.seq++
	seq := a.sounds.seq
	a.sounds.previewing = key
	start, player := a.startSound, &a.sndPlayer
	a.async(func() { player.play(seq, t, start) })
	if t.Kind != sounds.KindNone {
		d := t.Length(a.soundCat)
		if d <= 0 {
			d = sounds.UnknownLength
		}
		a.sounds.playing, a.sounds.playingEnd = choice, a.now().Add(d)
	}
	a.later(max(t.Length(a.soundCat)+cueMargin, previewMin), func() {
		if a.sounds.seq == seq {
			a.sounds.previewing = ""
		}
	})
}

// stopPreview stops the preview that plays.
func (a *app) stopPreview() {
	a.sounds.seq++
	seq := a.sounds.seq
	a.sounds.previewing, a.sounds.playing = "", ""
	start, player := a.startSound, &a.sndPlayer
	a.async(func() { player.play(seq, sounds.Target{}, start) })
}

// previewPlayingAt is the choice a preview plays at now and how long it has
// left (a pair's ▶).
func (a *app) previewPlayingAt(now time.Time) (string, time.Duration) {
	if a.sounds.playing == "" {
		return "", 0
	}
	left := a.sounds.playingEnd.Sub(now)
	if left <= 0 {
		a.sounds.playing = ""
		return "", 0
	}
	return a.sounds.playing, left
}

// trySound plays the cue's sound now (Try).
func (a *app) trySound(c cue) {
	ev := cueEvent(c)
	a.previewSound(ev, a.soundPrefs().Choice(ev), ev.ID)
}
