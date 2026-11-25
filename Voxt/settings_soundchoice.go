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
	"slices"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

// soundPairRows are the built-in pairs this build has, the chosen one
// checked; a choice that is no pair comes first, as Your choice.
func soundPairRows(c *ui.Context, pairs []settingsSoundPairViewState, actions settingsSoundsActions) {
	for _, pair := range pairs {
		soundPairRow(c, pair, actions)
	}
}

// soundPairRow is one pair on one line: ▶ plays it (stops it while it
// plays); its name and sounds choose and play it; ✓ marks the chosen one.
func soundPairRow(c *ui.Context, pair settingsSoundPairViewState, actions settingsSoundsActions) {
	t := c.Theme()
	k := tok(t)
	if pair.Playing {
		c.After(pair.Remaining) // redraw when it ends
	}
	ui.Row(c.Key("pair-" + pair.Start + "-" + pair.Stop)).Gap(px(t, 2)).AlignItems(ui.Center).MaxWidth(px(t, 420)).Children(func() {
		pl := ghostBase(c.Key("play")).Size(px(t, 26), px(t, 26)).Center().Shrink(0).Label(pair.PlayLabel).Tooltip(pair.PlayLabel).Selected(pair.Playing)
		col := k.Text2
		switch {
		case pair.Playing:
			pl.Background(k.AccentSoft)
			col = k.AccentInk
		case pl.Hovered():
			col = k.Text
		}
		pl.Children(func() {
			ui.Icon(c, icons.TurntableMusicNoteLineDuotone.Linear).Size(px(t, 15), px(t, 15)).TextColor(col)
		})
		if pl.Clicked() {
			if pair.Playing {
				actions.stopPreview()
			} else {
				actions.previewSoundPair(pair.Start, pair.Stop)
			}
		}
		r := ui.ButtonBase(c.Key("pick")).Grow(1).MinWidth(0).Height(px(t, 28)).Padding(0, px(t, 8)).Gap(px(t, 8)).AlignItems(ui.Center).
			Radius(px(t, radiusControl)).Label(pair.Name).Selected(pair.Chosen)
		switch {
		case pair.Chosen:
			r.Background(k.Pressed)
		case r.Hovered():
			r.Background(k.Hover)
		}
		if r.Clicked() && !pair.Chosen {
			actions.chooseSoundPair(pair.Start, pair.Stop)
		}
		r.Children(func() {
			ui.Text(c, pair.Name).TextColor(k.Text).SingleLine().Shrink(0)
			ui.Text(c, pair.Sounds).FontSize(px(t, fsCaption)).TextColor(k.Text3).SingleLine().Ellipsis("…").MinWidth(0).Shrink(1)
			ui.Box(c).Grow(1)
			if pair.Chosen {
				ui.Icon(c, icons.Check.Linear).Size(px(t, 14), px(t, 14)).TextColor(k.AccentInk)
			}
		})
	})
}

// systemCueRows are a line per cue: its macOS sound (None first) and Try.
func systemCueRows(c *ui.Context, cues []settingsSystemCueViewState, actions settingsSoundsActions) {
	t := c.Theme()
	for _, cue := range cues {
		settingRow(c, "audio-cue-"+cue.EventID, cue.Label, "", func() {
			selection := cue.Selected
			if ui.Select(c.Key("pick-"+cue.EventID), &selection, cue.Options).Label(cue.SelectLabel).Width(px(t, 176)).Shrink(1).MinWidth(px(t, 120)).Changed() {
				choice := cue.Values[max(0, slices.Index(cue.Options, selection))]
				actions.chooseSystemSound(cue.EventID, choice)
			}
			if tryButton(c.Key("try-"+cue.EventID), cue).Clicked() {
				actions.tryCueSound(cue.EventID)
			}
		})
	}
}
