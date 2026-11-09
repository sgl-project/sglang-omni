package main

// Meeting settings on the Meetings page (/meetings): capture mode,
// screen-sharing privacy for the overlay, shortcuts, and consent text.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// meetingSettingsUI is the Meetings page settings controls (not persisted
// in app; mode is read from settings each frame).
type meetingSettingsUI struct {
	mode string
	hide bool
	seen string
}

func (a *app) meetingSettingsKey() string {
	m := a.prefs.Features.Meeting
	return m.Mode + "|" + b2s(m.HideFromSharing)
}

func (a *app) meetingSettingsSync() {
	if a.meetingUI == nil {
		a.meetingUI = &meetingSettingsUI{}
	}
	u := a.meetingUI
	seen := a.meetingSettingsKey()
	if u.seen == seen {
		return
	}
	u.seen = seen
	u.mode = a.prefs.Features.Meeting.Mode
	u.hide = a.prefs.Features.Meeting.HideFromSharing
}

func (a *app) meetingSettings(c *ui.Context) {
	t := c.Theme()
	a.meetingSettingsSync()
	u := a.meetingUI
	sectionTitle(c, i18n.T("Meeting settings"))
	modeLabels := []string{
		i18n.T("Full meeting (microphone and system audio)"),
		i18n.T("Subtitles (system audio only)"),
		i18n.T("Recording (microphone only)"),
	}
	modeValues := []string{"", "subtitles", "recording"}
	modeSel := modeLabels[0]
	for i, v := range modeValues {
		if u.mode == v {
			modeSel = modeLabels[i]
			break
		}
	}
	if sel := ui.Select(c.Key("mt-mode"), &modeSel, modeLabels).Label(i18n.T("Capture mode")); sel.Changed() {
		for i, l := range modeLabels {
			if l == modeSel {
				u.mode = modeValues[i]
				mode := u.mode
				a.updateSettings(func(s *settings.Settings) { s.Features.Meeting.Mode = mode })
				break
			}
		}
	}
	settingRow(c, "mt-hide-share", i18n.T("Hide from screen sharing"),
		i18n.T("When on, the meeting overlay is hidden from screenshots and screen sharing."), func() {
			if ui.Switch(c.Key("mt-hide-sw"), &u.hide).Label(i18n.T("Hide the meeting overlay")).Changed() {
				on := u.hide
				a.updateSettings(func(s *settings.Settings) { s.Features.Meeting.HideFromSharing = on })
				a.applyMeetingOverlayProtection()
			}
		})
	a.featureShortcuts(c, hotkey.Meeting)
	ui.Text(c, i18n.T("Get the other person's consent before you record them speaking.")).
		FontSize(px(t, fsCaption)).TextColor(t.TextMuted)
}

func b2s(v bool) string {
	if v {
		return "1"
	}
	return "0"
}
