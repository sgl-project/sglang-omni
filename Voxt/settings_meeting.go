package main

// Meeting settings on the Meetings page (/meetings): capture mode,
// screen-sharing privacy for the overlay, shortcuts, the speech model and
// its language, the language model, and consent text.

import (
	"slices"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// meetingSettingsUI is the Meetings page settings controls (not persisted
// in app; mode is read from settings each frame).
type meetingSettingsUI struct {
	mode       string
	hide       bool
	seen       string
	correction speechContextState
}

func (a *app) meetingSettingsKey() string {
	m := a.prefs.Features.Meeting
	return m.Mode + "|" + b2s(m.HideFromSharing)
}

func (a *app) meetingSettingsSync() {
	if a.meetingUI == nil {
		a.meetingUI = &meetingSettingsUI{correction: speechContextState{text: meeting.DefaultCorrectionStyle}}
	}
	u := a.meetingUI
	savedStyle := a.prefs.Features.Meeting.CorrectionStyle
	if !u.correction.dirty && savedStyle != u.correction.seen {
		u.correction.text, u.correction.seen = savedStyle, savedStyle
		if savedStyle == "" {
			u.correction.text = meeting.DefaultCorrectionStyle
		}
	}
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
	settingRow(c, "mt-mode-row", i18n.T("Capture mode"), "", func() {
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
	})
	settingRow(c, "mt-hide-share", i18n.T("Hide from screen sharing"),
		"", func() {
			if ui.Switch(c.Key("mt-hide-sw"), &u.hide).Label(i18n.T("Hide meeting overlay")).Changed() {
				on := u.hide
				a.updateSettings(func(s *settings.Settings) { s.Features.Meeting.HideFromSharing = on })
				a.applyMeetingOverlayProtection()
			}
		})
	a.featureShortcuts(c, hotkey.Meeting)
	a.meetingASRSettings(c)
	a.meetingLLMSettings(c)
	a.meetingCorrectionStyleSettings(c)
	ui.Text(c, i18n.T("Get the other person's consent before you record them speaking.")).
		FontSize(px(t, fsCaption)).TextColor(t.TextMuted)
}

// meetingASRSettings are the meeting's speech model (the menu Transcribe
// has, Automatic first) and the language it listens for.
func (a *app) meetingASRSettings(c *ui.Context) {
	t := c.Theme()
	m := a.prefs.Features.Meeting
	ready := a.installed(local.ASR)
	shown := slices.DeleteFunc(slices.Clone(ready), func(x local.Model) bool { return a.modelHidden(x.ID) }) // as featureModels
	opts, auto := modelOptions(ready, shown)
	sel := modelLabel(ready, m.ASR, auto)
	settingRow(c, "mt-asr", i18n.T("Speech recognition"), "", func() {
		if ui.Select(c.Key("mt-asr-sel"), &sel, opts).Label(i18n.T("Speech model for meetings")).Width(px(t, 260)).Shrink(1).MinWidth(px(t, 160)).
			Disabled(len(ready) == 0).Changed() {
			id := modelID(opts, shown, sel)
			a.updateSettings(func(s *settings.Settings) { s.Features.Meeting.ASR = id })
		}
	})
	langOpts := []string{i18n.T("Same as the main language")}
	for _, code := range mainLanguages {
		langOpts = append(langOpts, uitext.LangLabel(code))
	}
	langSel := langOpts[0]
	if i := slices.Index(mainLanguages, m.Lang); i >= 0 {
		langSel = langOpts[i+1]
	}
	settingRow(c, "mt-lang", i18n.T("Spoken language"), i18n.T("The language the speech model listens for. Choose one when the whole meeting is in it."), func() {
		if ui.Select(c.Key("mt-lang-sel"), &langSel, langOpts).Label(i18n.T("Spoken language in meetings")).Width(px(t, 260)).Shrink(1).MinWidth(px(t, 160)).Changed() {
			code := ""
			if i := slices.Index(langOpts, langSel); i > 0 {
				code = mainLanguages[i-1]
			}
			a.updateSettings(func(s *settings.Settings) { s.Features.Meeting.Lang = code })
		}
	})
}

func (a *app) meetingLLMSettings(c *ui.Context) {
	t := c.Theme()
	ready := a.installed(local.LLM)
	label := i18n.T("Automatic")
	if m, ok := local.Choose(ready, a.prefs.Features.Meeting.LLM); ok {
		label = m.Name
	}
	settingRow(c, "mt-llm", i18n.T("Language model for meetings"),
		i18n.T("Summaries, live translation and follow-up questions send the full meeting text to this model."), func() {
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
				ui.Text(c, i18n.T("No language model is set up, so summaries and translation are off.")).
					FontSize(px(t, fsCaption)).TextColor(t.TextMuted)
			}
		})
}

func (a *app) meetingCorrectionStyleSettings(c *ui.Context) {
	t := c.Theme()
	u := &a.meetingUI.correction
	sectionTitle(c, i18n.T("Transcript correction"))
	caption(c, i18n.T("Choose how the language model corrects the transcript. It still keeps every line and uses your dictionary and Speech context."))
	in := ui.TextArea(c.Key("mt-correction-style"), &u.text).Label(i18n.T("Correction style for meeting transcripts")).
		Placeholder(i18n.T("For example: keep the conversation's original tone; use concise sentences; keep English product names in English.")).MinHeight(px(t, 96))
	a.autosave(in, u, a.saveMeetingCorrectionStyle)
	empty := a.prefs.Features.Meeting.CorrectionStyle == "" && strings.TrimSpace(u.text) == strings.TrimSpace(meeting.DefaultCorrectionStyle)
	if ghostButton(c.Key("mt-correction-reset"), i18n.T("Reset to default prompt")).Disabled(empty).Clicked() {
		u.text = meeting.DefaultCorrectionStyle
		a.saveMeetingCorrectionStyle()
	}
}

func (a *app) saveMeetingCorrectionStyle() {
	u := &a.meetingUI.correction
	u.dirty = false
	v := strings.TrimSpace(u.text)
	if v == strings.TrimSpace(meeting.DefaultCorrectionStyle) {
		v = ""
	}
	u.text, u.seen = v, v
	if v == "" {
		u.text = meeting.DefaultCorrectionStyle
	}
	if v != a.prefs.Features.Meeting.CorrectionStyle {
		a.updateSettings(func(s *settings.Settings) { s.Features.Meeting.CorrectionStyle = v })
	}
}

func b2s(v bool) string {
	if v {
		return "1"
	}
	return "0"
}
