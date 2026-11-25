package main

// The sounds of a recording, Settings → Audio → Sound (settings_audio.go;
// its controls are settings_soundchoice.go's): which sound plays when a
// recording starts and when it stops (the ActNow family's sounds,
// actnowkit's sounds package, shipped in resources/sounds; a macOS system
// sound; none), their volume, and trying them. A preview cuts the one
// before it short; Try shows the accent while it is heard. The cues of a
// recording are recordingAudio's (mute_link.go). Without the family's
// sounds (a build without resources/sounds) the cues are Voxt's Pop and
// Tink.

import (
	"slices"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// soundsRow is the Sound row, and under it what its source offers.
func soundsRow(c *ui.Context, state settingsSoundsViewState, actions settingsSoundsActions) {
	t := c.Theme()
	settingRow(c, "audio-sounds", i18n.T("Sound"), "", func() {
		sourceIndex := state.SourceIndex
		if ui.Segmented(c.Key("audio-sound-source"), &sourceIndex, state.SourceOptions...).Label(i18n.T("Sound")).Changed() {
			actions.chooseSoundSource(state.SourceValues[sourceIndex])
		}
	})
	switch {
	case state.SourceID == soundSystemID:
		systemCueRows(c, state.SystemCues, actions) // settingRows: the search finds them by their own names
	case state.SourceID == soundBuiltInID && settingsSearchShows(i18n.T("Sound")):
		ui.Column(c.Key("audio-cues")).Gap(px(t, sp1)).Padding(0, 0, 0, px(t, sp4)).Children(func() { soundPairRows(c, state.Pairs, actions) })
	}
	if state.SourceID != soundNoneID {
		soundVolumeRow(c, state, actions)
	}
}

// tryButton is Try sound: iconButton's face, its note in the accent while
// the cue's preview plays.
func tryButton(c *ui.Context, state settingsSystemCueViewState) ui.Element {
	t := c.Theme()
	k := tok(t)
	b := ghostBase(c).Size(px(t, 28), px(t, 28)).Center().Label(state.TryLabel).Tooltip(state.TryLabel).Shrink(0)
	b.Children(func() {
		col := k.Text2
		switch {
		case state.TryAccent:
			col = k.Accent
		case b.Hovered():
			col = k.Text
		}
		ui.Icon(c, icons.TurntableMusicNoteLineDuotone.Linear).Size(px(t, 16), px(t, 16)).TextColor(col)
	})
	return b
}

// soundVolumeRow is Volume: the step nearest to the cues' volume.
func soundVolumeRow(c *ui.Context, state settingsSoundsViewState, actions settingsSoundsActions) {
	t := c.Theme()
	settingRow(c, "audio-volume", i18n.T("Volume"), "", func() {
		selection := state.VolumeSelection
		if ui.Select(c.Key("audio-volume-sel"), &selection, state.VolumeOptions).Label(i18n.T("Volume")).Width(px(t, 96)).Changed() {
			volume := state.VolumeValues[max(0, slices.Index(state.VolumeOptions, selection))]
			actions.setSoundVolume(volume)
		}
	})
}
