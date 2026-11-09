package main

// Meeting settings on the Meetings page (/meetings): capture mode,
// screen-sharing privacy for the overlay, shortcuts, and consent text.

import (
	"fmt"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
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
	a.meetingLLMSettings(c)
	ui.Text(c, i18n.T("Get the other person's consent before you record them speaking.")).
		FontSize(px(t, fsCaption)).TextColor(t.TextMuted)
}

func (a *app) meetingLLMSettings(c *ui.Context) {
	t := c.Theme()
	ready := a.installed(local.LLM)
	label := i18n.T("Automatic")
	if m, ok := local.Choose(ready, a.prefs.Features.Meeting.LLM); ok {
		label = m.Name
	}
	settingRow(c, "mt-llm", i18n.T("Language model for meetings"),
		i18n.T("Summaries, live translation, and follow-up questions send the full meeting text to this model."), func() {
			opts := []string{i18n.T("Automatic")}
			for _, m := range ready {
				opts = append(opts, m.Name)
			}
			sel := label
			if ui.Select(c.Key("mt-llm-sel"), &sel, opts).Label(i18n.T("Language model")).Changed() {
				id := ""
				if sel != i18n.T("Automatic") {
					for _, m := range ready {
						if m.Name == sel {
							id = m.ID
							break
						}
					}
				}
				a.updateSettings(func(s *settings.Settings) { s.Features.Meeting.LLM = id })
			}
			if len(ready) == 0 {
				ui.Text(c, i18n.T("No language model is set up: summaries and translation stay off.")).
					FontSize(px(t, fsCaption)).TextColor(t.TextMuted)
				return
			}
			if m, ok := local.Choose(ready, a.prefs.Features.Meeting.LLM); ok && m.Name != "" {
				ui.Text(c, fmt.Sprintf(i18n.T("Meeting text is sent to %s."), m.Name)).
					FontSize(px(t, fsCaption)).TextColor(t.TextMuted)
			}
		})
}

func b2s(v bool) string {
	if v {
		return "1"
	}
	return "0"
}
