package main

// Meeting detail playback controls and clickable transcript rows.

import (
	"strconv"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

func meetingPlaybackBar(c *ui.Context, state meetingPlaybackViewState, meetingID string, actions meetingsActions) {
	if !state.Available {
		return
	}
	t := c.Theme()
	if state.PositionText != "" {
		ui.Text(c, state.PositionText).FontSize(px(t, fsCaption)).TextColor(tok(t).Text3).FontFeatures("tnum")
	}
	icon, label := icons.Play, i18n.T("Play")
	if state.Playing {
		icon, label = icons.Pause, i18n.T("Pause")
	}
	if iconButton(c.Key("mt-playback"), icon, label).Clicked() {
		actions.toggleMeetingPlayback(meetingID)
	}
}

// meetingSegment is a transcript line; a click seeks the audio to its start.
func meetingSegment(c *ui.Context, row meetingSegmentViewState, highlight int, meetingID string, actions meetingsActions) {
	t := c.Theme()
	colors := tok(t)
	button := ui.ButtonBase(c.Key("mt-seg-" + strconv.Itoa(row.Index))).Padding(px(t, sp2)).Gap(px(t, sp3)).AlignItems(ui.Start).Radius(px(t, radiusControl)).
		Label(row.TimeText + " " + row.Speaker)
	switch {
	case row.Index == highlight:
		button.Background(colors.AccentSoft)
	case button.Pressed():
		button.Background(colors.Pressed)
	case button.Hovered():
		button.Background(colors.Hover)
	}
	if button.Clicked() {
		actions.seekMeetingPlayback(meetingID, row.StartMS)
	}
	button.Children(func() {
		ui.Text(c, row.TimeText).FontSize(px(t, fsCaption)).TextColor(colors.Text3).FontFeatures("tnum").Width(px(t, 40)).Shrink(0)
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
			speaker := ui.Text(c, row.Speaker).FontSize(px(t, fsCaption)).FontWeight(600).TextColor(colors.Text2)
			if row.Me {
				speaker.TextColor(colors.AccentInk)
			}
			ui.Text(c, row.Text)
			if row.Translation != "" {
				ui.Text(c, row.Translation).TextColor(colors.Text2).FontSize(px(t, fsCaption))
			}
		})
	})
}
