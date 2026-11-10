package main

// Settings → Audio → Sound (11-10): where the cues come from, as three
// segments: Built-in (the family's sounds, settings_sounds.go), macOS (its
// system sounds) and None. Built-in lists pairs of a start and a stop
// sound, each with ▶ to hear both and a click to choose them; a choice
// that is no pair (made in the picker of 10-20 to 11-09) shows as its own
// line. macOS has a line per cue with its sound and Try. Volume follows,
// unless the sounds are off. A build without the family's sounds plays
// macOS's for them, so it offers macOS and None only.
//
// The settings stay as they were: None is Sounds off, and the source is
// read from what the cues play (soundSource).

import (
	"fmt"
	"slices"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/hehehai/actnowkit/sounds"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
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

// pairName is a pair's name.
func pairName(key string) string {
	switch key {
	case "soft":
		return i18n.T("Soft")
	case "switch":
		return i18n.T("Switch")
	case "clean":
		return i18n.T("Clean")
	case "beep":
		return i18n.T("Beep")
	case "tape":
		return i18n.T("Tape")
	case "day":
		return i18n.T("Sunrise")
	}
	return i18n.T("Your choice")
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

// soundsRow is the Sound row, and under it what its source offers.
func (a *app) soundsRow(c *ui.Context) {
	t := c.Theme()
	srcs := []soundSource{soundBuiltIn, soundSystem, soundNone}
	labels := []string{i18n.X("sound source", "Built-in"), i18n.T("macOS"), i18n.T("None")}
	if a.soundCat == nil {
		srcs, labels = srcs[1:], labels[1:]
	}
	src := a.soundSource()
	settingRow(c, "audio-sounds", i18n.T("Sound"), "", func() {
		i := max(0, slices.Index(srcs, src))
		if ui.Segmented(c.Key("audio-sound-source"), &i, labels...).Label(i18n.T("Sound")).Changed() {
			a.setSoundSource(srcs[i])
		}
	})
	switch {
	case src == soundSystem:
		a.systemCueRows(c) // settingRows: the search finds them by their own names
	case src == soundBuiltIn && settingsTextMatches(settingsSearchQuery, i18n.T("Sound")):
		ui.Column(c.Key("audio-cues")).Gap(px(t, sp1)).Padding(0, 0, 0, px(t, sp4)).Children(func() { a.soundPairRows(c) })
	}
	if src != soundNone {
		a.soundVolumeRow(c)
	}
}

// soundPairRows are the built-in pairs this build has, the chosen one
// checked; a choice that is no pair comes first, as Your choice.
func (a *app) soundPairRows(c *ui.Context) {
	p := a.soundPrefs()
	cur := soundPair{"", p.Choice(recordStartEvent), p.Choice(recordStopEvent)}
	pairs := []soundPair{}
	for _, sp := range soundPairs {
		if _, ok := a.soundCat.Sound(sp.start); !ok {
			continue
		}
		if _, ok := a.soundCat.Sound(sp.stop); ok {
			pairs = append(pairs, sp)
		}
	}
	if !slices.ContainsFunc(pairs, func(sp soundPair) bool { return sp.start == cur.start && sp.stop == cur.stop }) {
		pairs = append([]soundPair{cur}, pairs...)
	}
	for _, sp := range pairs {
		a.pairRow(c, sp, sp.start == cur.start && sp.stop == cur.stop)
	}
}

// pairRow is one pair on one line: ▶ plays it (stops it while it plays);
// its name and sounds choose and play it; ✓ marks the chosen one.
func (a *app) pairRow(c *ui.Context, sp soundPair, chosen bool) {
	t := c.Theme()
	k := tok(t)
	name := pairName(sp.key)
	playing := false
	if ch, left := a.previewPlaying(); ch != "" {
		playing = ch == sp.start || ch == sp.stop
		c.After(left) // redraw when it ends
	}
	ui.Row(c.Key("pair-" + sp.start + "-" + sp.stop)).Gap(px(t, 2)).AlignItems(ui.Center).MaxWidth(px(t, 420)).Children(func() {
		tip := i18n.T("Play %s", name)
		if playing {
			tip = i18n.T("Stop %s", name)
		}
		pl := ghostBase(c.Key("play")).Size(px(t, 26), px(t, 26)).Center().Shrink(0).Label(tip).Tooltip(tip).Selected(playing)
		col := k.Text2
		switch {
		case playing:
			pl.Background(k.AccentSoft)
			col = k.AccentInk
		case pl.Hovered():
			col = k.Text
		}
		pl.Children(func() {
			ui.Icon(c, icons.TurntableMusicNoteLineDuotone.Linear).Size(px(t, 15), px(t, 15)).TextColor(col)
		})
		if pl.Clicked() {
			if playing {
				a.stopPreview()
			} else {
				a.previewPair(sp)
			}
		}
		r := ui.ButtonBase(c.Key("pick")).Grow(1).MinWidth(0).Height(px(t, 28)).Padding(0, px(t, 8)).Gap(px(t, 8)).AlignItems(ui.Center).
			Radius(px(t, radiusControl)).Label(name).Selected(chosen)
		switch {
		case chosen:
			r.Background(k.Pressed)
		case r.Hovered():
			r.Background(k.Hover)
		}
		if r.Clicked() && !chosen {
			a.chooseSounds(sp.start, sp.stop)
			a.previewPair(sp)
		}
		r.Children(func() {
			ui.Text(c, name).TextColor(k.Text).SingleLine().Shrink(0)
			ui.Text(c, i18n.T("%s → %s", a.soundLabel(sp.start), a.soundLabel(sp.stop))).FontSize(px(t, fsCaption)).TextColor(k.Text3).
				SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
			ui.Box(c).Grow(1)
			if chosen {
				ui.Icon(c, icons.Check.Linear).Size(px(t, 14), px(t, 14)).TextColor(k.AccentInk)
			}
		})
	})
}

// systemCueRows are a line per cue: its macOS sound (None first) and Try.
func (a *app) systemCueRows(c *ui.Context) {
	t := c.Theme()
	for _, cu := range []cue{startCue, stopCue} {
		ev := cueEvent(cu)
		label, name := i18n.T("When recording starts"), i18n.T("Start sound")
		if cu == stopCue {
			label, name = i18n.T("When recording stops"), i18n.T("Stop sound")
		}
		settingRow(c, "audio-cue-"+ev.ID, label, "", func() {
			choices, labels := []string{sounds.None}, []string{i18n.T("None")}
			for _, n := range a.systemSounds() {
				choices, labels = append(choices, sounds.System(n)), append(labels, n)
			}
			cur := sounds.None
			if tg := a.cueTarget(cu); tg.Kind == sounds.KindSystem {
				cur = sounds.System(tg.Name) // a family sound this build lacks plays its fallback
			}
			if !slices.Contains(choices, cur) {
				choices, labels = append(choices, cur), append(labels, sounds.SystemName(cur))
			}
			sel := labels[slices.Index(choices, cur)]
			if ui.Select(c.Key("pick-"+ev.ID), &sel, labels).Label(name).Width(px(t, 176)).Shrink(1).MinWidth(px(t, 120)).Changed() {
				ch := choices[max(0, slices.Index(labels, sel))]
				a.chooseSound(ev, ch)
				a.previewSound(ev, ch, ev.ID)
			}
			if a.tryButton(c.Key("try-"+ev.ID), ev.ID).Clicked() {
				a.trySound(cu)
			}
		})
	}
}

// soundVolumes are the volumes offered, loudest first (None turns the
// sounds off).
var soundVolumes = []int{100, 75, 50, 25}

// soundVolumeRow is Volume: the step nearest to the cues' volume.
func (a *app) soundVolumeRow(c *ui.Context) {
	t := c.Theme()
	settingRow(c, "audio-volume", i18n.T("Volume"), "", func() {
		labels := make([]string, len(soundVolumes))
		for j, v := range soundVolumes {
			labels[j] = fmt.Sprintf("%d%%", v)
		}
		sel := fmt.Sprintf("%d%%", max(soundVolumes[len(soundVolumes)-1], sounds.NearestVolumeStep(a.prefs.Audio.SoundVolumeOrDefault())))
		if ui.Select(c.Key("audio-volume-sel"), &sel, labels).Label(i18n.T("Volume")).Width(px(t, 96)).Changed() {
			v := soundVolumes[max(0, slices.Index(labels, sel))]
			a.updateSettings(func(s *settings.Settings) { s.Audio.SetSoundVolume(v) })
			a.previewSound(recordStartEvent, a.soundPrefs().Choice(recordStartEvent), "")
		}
	})
}
