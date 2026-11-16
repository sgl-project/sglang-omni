package main

import (
	"slices"
	"time"

	"github.com/hehehai/actnowkit/sounds"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// soundSource is where the cues come from.
type soundSource int

const (
	soundBuiltIn soundSource = iota // the family's sounds
	soundSystem                     // macOS's system sounds
	soundNone                       // no sound
)

// soundPair is a start cue and a stop cue that go together.
type soundPair struct{ key, start, stop string }

// soundPairs are the built-in pairs, softest first: the family's sounds
// made for the start and the stop of a recording. The first is the
// events' default.
var soundPairs = []soundPair{
	{"soft", "pebble", "nib"},
	{"switch", "clasp", "unclasp"},
	{"clean", "dot", "settle"},
	{"beep", "tally", "wrap"},
	{"tape", "roll", "hold"},
	{"day", "dawn", "dusk"},
}

// systemPair is what macOS plays until the user picks: Voxt 1.x's cues.
var systemPair = soundPair{"system", sounds.System("Pop"), sounds.System("Tink")}

// pairNameWithTranslator is a pair's name in the translator's language.
func pairNameWithTranslator(translator *i18n.Translator, key string) string {
	switch key {
	case "soft":
		return translator.T("Soft")
	case "switch":
		return translator.T("Switch")
	case "clean":
		return translator.T("Clean")
	case "beep":
		return translator.T("Beep")
	case "tape":
		return translator.T("Tape")
	case "day":
		return translator.T("Sunrise")
	}
	return translator.T("Your choice")
}

// pairGap is the silence between a pair's two sounds when it plays.
const pairGap = 250 * time.Millisecond

// soundSource is where the cues come from now: None while the sounds are
// off or muted; Built-in when a cue plays one of the family's sounds;
// macOS when they play system sounds (as a build without the family's
// sounds plays for them).
func (a *app) soundSource() soundSource {
	au := a.prefs.Audio
	if !au.Sounds || au.SoundVolumeOrDefault() == 0 {
		return soundNone
	}
	kinds := []sounds.Kind{a.cueTarget(startCue).Kind, a.cueTarget(stopCue).Kind}
	switch {
	case slices.Contains(kinds, sounds.KindCatalog):
		return soundBuiltIn
	case slices.Contains(kinds, sounds.KindSystem):
		return soundSystem
	}
	return soundNone
}

// setSoundSource turns the cues to src and plays the start cue: Built-in
// keeps cues that are the family's sounds (else the default pair), macOS
// keeps system sounds (else Pop and Tink), None turns the sounds off and
// keeps the choices for later.
func (a *app) setSoundSource(src soundSource) {
	if src == soundSystem {
		a.systemSounds()
	}
	p := a.soundPrefs()
	start, stop := p.Choice(recordStartEvent), p.Choice(recordStopEvent)
	builtIn := func(ch string) bool { return ch != sounds.None && !sounds.IsSystem(ch) }
	switch src {
	case soundBuiltIn:
		if !builtIn(start) || !builtIn(stop) {
			start, stop = soundPairs[0].start, soundPairs[0].stop
		}
	case soundSystem:
		if !sounds.IsSystem(start) || !sounds.IsSystem(stop) {
			start, stop = systemPair.start, systemPair.stop
		}
	}
	a.chooseSounds(start, stop)
	a.updateSettings(func(s *settings.Settings) {
		s.Audio.Sounds = src != soundNone
		if s.Audio.Sounds && s.Audio.SoundVolumeOrDefault() == 0 {
			s.Audio.SetSoundVolume(settings.DefaultSoundVolume)
		}
	})
	if src != soundNone {
		a.previewSound(recordStartEvent, start, "")
	}
}

func (a *app) chooseSoundSource(sourceID string) {
	switch sourceID {
	case soundBuiltInID:
		a.setSoundSource(soundBuiltIn)
	case soundSystemID:
		a.setSoundSource(soundSystem)
	case soundNoneID:
		a.setSoundSource(soundNone)
	}
}

// prepareSoundChoices reads system sounds only when the macOS source is
// about to show. View-state construction stays free of filesystem reads.
func (a *app) prepareSoundChoices() {
	if a.soundSource() == soundSystem {
		a.systemSounds()
	}
}

// previewPair plays a pair's start cue, then its stop cue.
func (a *app) previewPair(p soundPair) {
	a.previewSound(recordStartEvent, p.start, "")
	seq := a.sounds.seq
	first := sounds.Resolve(a.soundCat, p.start, recordStartEvent.Fallback, 100).Length(a.soundCat)
	a.later(first+pairGap, func() {
		if a.sounds.seq == seq {
			a.previewSound(recordStopEvent, p.stop, "")
		}
	})
}

func (a *app) previewSoundPair(start, stop string) {
	a.previewPair(soundPair{start: start, stop: stop})
}

func (a *app) chooseSoundPair(start, stop string) {
	a.chooseSounds(start, stop)
	a.previewSoundPair(start, stop)
}

func cueForEvent(eventID string) (cue, bool) {
	switch eventID {
	case recordStartEvent.ID:
		return startCue, true
	case recordStopEvent.ID:
		return stopCue, true
	}
	return noCue, false
}

func (a *app) chooseSystemSound(eventID, choice string) {
	c, ok := cueForEvent(eventID)
	if !ok {
		return
	}
	event := cueEvent(c)
	a.chooseSound(event, choice)
	a.previewSound(event, choice, event.ID)
}

func (a *app) tryCueSound(eventID string) {
	c, ok := cueForEvent(eventID)
	if ok {
		a.trySound(c)
	}
}

// soundVolumes are the volumes offered, loudest first (None turns the
// sounds off).
var soundVolumes = []int{100, 75, 50, 25}

func (a *app) setSoundVolume(volume int) {
	a.updateSettings(func(s *settings.Settings) { s.Audio.SetSoundVolume(volume) })
	a.previewSound(recordStartEvent, a.soundPrefs().Choice(recordStartEvent), "")
}

func (a *app) soundsSettingsModel(playing string, remaining time.Duration) settingsSoundsViewState {
	return buildSettingsSoundsViewState(i18n.CurrentTranslator(), settingsSoundsStateInput{
		Prefs: a.soundPrefs(), Source: a.soundSource(), Catalog: a.soundCat,
		StartTarget: a.cueTarget(startCue), StopTarget: a.cueTarget(stopCue), SystemSounds: a.sysSounds,
		Previewing: a.sounds.previewing, Playing: playing, Remaining: remaining,
	})
}
