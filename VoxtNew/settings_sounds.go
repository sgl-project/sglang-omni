package main

// Settings → Audio → Play sounds (settings_audio.go): whether a recording
// plays its cues, which sound it plays when it starts and when it stops
// (the ActNow family's sounds, actnowkit's sounds package, shipped in
// resources/sounds; a macOS system sound; none), their volume, and trying
// them. While the switch is on, a line for each cue shows under the row:
// its sound picker (settings_soundpicker.go: a panel of sounds by group
// and tag, each with ▶) and Try. A preview cuts the one before it short;
// Try shows the accent while it is heard. The cues of a recording are
// recordingAudio's (mute_link.go). Without the family's sounds (a build
// without resources/sounds) the cues are Voxt's Pop and Tink.

import (
	"os"
	"sync"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/hehehai/actnowkit/sounds"
)

// The recording's cues: the family's Tally and Wrap (a clear double beep
// as recording starts, a falling pair as it stops), Voxt's Pop and Tink
// in a build without the family's sounds.
var (
	recordStartEvent = sounds.Event{ID: "record.start", Name: sounds.Text{En: "Recording starts", Zh: "开始录音"},
		Default: "tally", Fallback: "Pop", Uses: []string{"record.start"}}
	recordStopEvent = sounds.Event{ID: "record.stop", Name: sounds.Text{En: "Recording stops", Zh: "结束录音"},
		Default: "wrap", Fallback: "Tink", Uses: []string{"record.stop"}}
)

// soundsState is the cues' sounds and the previews of Settings → Audio →
// Play sounds.
type soundsState struct {
	sounds struct {
		previewing string // the cue whose Try shows the accent ("" none)
		seq        int    // numbers the previews: a preview's timer ends only its own
		// playing is the choice a preview plays until playingEnd (the
		// picker's ▶ in the accent).
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

// loadCueSounds reads resources/sounds (Contents/Resources/sounds); nil
// when the build has none.
func loadCueSounds() *sounds.Catalog {
	res, err := mygo.App.Path(mygo.PathResources)
	if err != nil {
		return nil
	}
	cat, err := sounds.Load(os.DirFS(res), "sounds")
	if err != nil {
		return nil
	}
	return cat
}

// systemSounds are the Mac's system sounds, for the picker.
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
	a.updateSettings(func(s *settings.Settings) {
		s.Audio.StartSound, s.Audio.StopSound = p.Events[recordStartEvent.ID], p.Events[recordStopEvent.ID]
	})
}

// soundLabel names a choice: a family sound's name in the interface's
// language, a system sound's, None.
func (a *app) soundLabel(choice string) string {
	switch {
	case choice == sounds.None:
		return i18n.T("None")
	case sounds.IsSystem(choice):
		return sounds.SystemName(choice)
	}
	if s, ok := a.soundCat.Sound(choice); ok {
		return s.Name.In(i18n.Locale())
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

// previewSound plays choice for ev (Try, ▶ in the picker), at the cues'
// volume, Play sounds on or off; key names the Try button that shows the
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
		a.sounds.playing, a.sounds.playingEnd = choice, time.Now().Add(d)
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

// previewPlaying is the choice a preview plays and how long it has left
// (the picker's ▶).
func (a *app) previewPlaying() (string, time.Duration) {
	if a.sounds.playing == "" {
		return "", 0
	}
	left := time.Until(a.sounds.playingEnd)
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

// soundsRow is the Play sounds row, and under it while on a line per cue.
func (a *app) soundsRow(c *ui.Context) {
	t := c.Theme()
	settingRow(c, "audio-sounds", i18n.T("Play sounds"), i18n.T("A short sound when recording starts, and another when it stops. Choosing a sound plays it."), func() {
		if ui.Switch(c.Key("audio-sounds-switch"), &a.settingsForm.sounds).Label(i18n.T("Play sounds")).Changed() {
			on := a.settingsForm.sounds
			a.updateSettings(func(s *settings.Settings) { s.Audio.Sounds = on })
		}
	})
	if !a.settingsForm.sounds {
		return
	}
	ui.Column(c.Key("audio-cues")).Gap(px(t, sp1)).Padding(0, 0, 0, px(t, sp4)).Children(func() {
		for _, cu := range []cue{startCue, stopCue} {
			ev := cueEvent(cu)
			label, name := i18n.T("When recording starts"), i18n.T("Start sound")
			if cu == stopCue {
				label, name = i18n.T("When recording stops"), i18n.T("Stop sound")
			}
			settingRow(c, "audio-cue-"+ev.ID, label, "", func() {
				choice := a.soundPrefs().Choice(ev)
				soundPicker(c.Key("pick-"+ev.ID), soundPickerSpec{
					Name:      name,
					Event:     ev,
					Choice:    choice,
					Volume:    a.prefs.Audio.SoundVolumeOrDefault(),
					Catalog:   a.soundCat,
					System:    a.systemSounds(),
					Label:     a.soundLabel,
					Choose:    func(ch string) { a.chooseSound(ev, ch) },
					Preview:   func(ch string) { a.previewSound(ev, ch, "") },
					Stop:      a.stopPreview,
					Playing:   a.previewPlaying,
					SetVolume: func(v int) { a.updateSettings(func(s *settings.Settings) { s.Audio.SetSoundVolume(v) }) },
				})
				if a.tryButton(c.Key("try-"+ev.ID), ev.ID).Clicked() {
					a.trySound(cu)
				}
			})
		}
	})
}

// tryButton is Try sound: iconButton's face, its note in the accent while
// the preview of key plays.
func (a *app) tryButton(c *ui.Context, key string) ui.Element {
	t := c.Theme()
	k := tok(t)
	tip := i18n.T("Try the start sound")
	if key == recordStopEvent.ID {
		tip = i18n.T("Try the stop sound")
	}
	b := ghostBase(c).Size(px(t, 28), px(t, 28)).Center().Label(tip).Tooltip(tip).Shrink(0)
	b.Children(func() {
		col := k.Text2
		switch {
		case a.sounds.previewing == key:
			col = k.Accent
		case b.Hovered():
			col = k.Text
		}
		ui.Icon(c, icons.TurntableMusicNoteLineDuotone.Linear).Size(px(t, 16), px(t, 16)).TextColor(col)
	})
	return b
}
