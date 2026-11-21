package main

// Meeting settings on the Meetings page (/meetings): capture mode,
// screen-sharing privacy for the overlay, the speech model and its language,
// the language model, and consent text.

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

// meetingSettingsUI is the Meetings page settings controls (not persisted
// in app; mode is read from settings each frame).
type meetingSettingsUI struct {
	mode       string
	hide       bool
	seen       string
	correction speechContextState
}

func meetingCaptureSettings(c *ui.Context, state settingsMeetingViewState, actions settingsMeetingActions, local *meetingSettingsUI) {
	captureMode := state.CaptureModeSelected
	settingRow(c, "mt-mode-row", i18n.T("Capture mode"), "", func() {
		if ui.Select(c.Key("mt-mode"), &captureMode, settingsLanguageLabels(state.CaptureModeOptions)).Label(i18n.T("Capture mode")).Changed() {
			actions.setMeetingCaptureMode(settingsLanguageCode(captureMode, state.CaptureModeOptions))
		}
	})
	settingRow(c, "mt-hide-share", i18n.T("Hide from screen sharing"),
		"", func() {
			if ui.Switch(c.Key("mt-hide-sw"), &local.hide).Label(i18n.T("Hide meeting overlay")).Changed() {
				actions.setMeetingHideFromSharing(local.hide)
			}
		})
}

// meetingASRSettings are the meeting's speech model (the menu Transcribe
// has, Automatic first) and the language it listens for.
func meetingASRSettings(c *ui.Context, state settingsMeetingViewState, actions settingsMeetingActions) {
	t := c.Theme()
	speechModel := state.ASRSelected
	settingRow(c, "mt-asr", i18n.T("Speech recognition"), "", func() {
		if ui.Select(c.Key("mt-asr-sel"), &speechModel, settingsLanguageLabels(state.ASROptions)).Label(i18n.T("Speech model for meetings")).Width(px(t, 260)).Shrink(1).MinWidth(px(t, 160)).
			Disabled(state.ASRDisabled).Changed() {
			actions.setMeetingASR(settingsLanguageCode(speechModel, state.ASROptions))
		}
	})
	spokenLanguage := state.SpokenLanguageSelected
	settingRow(c, "mt-lang", i18n.T("Spoken language"), i18n.T("The language the speech model listens for. Choose one when the whole meeting is in it."), func() {
		if ui.Select(c.Key("mt-lang-sel"), &spokenLanguage, settingsLanguageLabels(state.SpokenLanguageOptions)).Label(i18n.T("Spoken language in meetings")).Width(px(t, 260)).Shrink(1).MinWidth(px(t, 160)).Changed() {
			actions.setMeetingLanguage(settingsLanguageCode(spokenLanguage, state.SpokenLanguageOptions))
		}
	})
}

func meetingLLMSettings(c *ui.Context, state settingsMeetingViewState, actions settingsMeetingActions) {
	t := c.Theme()
	languageModel := state.LLMSelected
	settingRow(c, "mt-llm", i18n.T("Language model for meetings"),
		i18n.T("Summaries, live translation and follow-up questions send the full meeting text to this model."), func() {
			if ui.Select(c.Key("mt-llm-sel"), &languageModel, settingsLanguageLabels(state.LLMOptions)).Label(i18n.T("Language model")).Changed() {
				actions.setMeetingLLM(settingsLanguageCode(languageModel, state.LLMOptions))
			}
			if state.NoLLM {
				ui.Text(c, i18n.T("No language model is set up, so summaries and translation are off.")).
					FontSize(px(t, fsCaption)).TextColor(t.TextMuted)
			}
		})
}

func meetingCorrectionStyleSettings(c *ui.Context, state settingsMeetingViewState, actions settingsMeetingActions, local *meetingSettingsUI) {
	t := c.Theme()
	sectionTitle(c, i18n.T("Transcript correction"))
	caption(c, i18n.T("Choose how the language model corrects the transcript. It still keeps every line and uses your dictionary and Speech context."))
	input := ui.TextArea(c.Key("mt-correction-style"), &local.correction.text).Label(i18n.T("Correction style for meeting transcripts")).
		Placeholder(i18n.T("For example: keep the conversation's original tone; use concise sentences; keep English product names in English.")).MinHeight(px(t, 96))
	actions.autosaveMeetingCorrectionStyle(input.Changed(), input.Focused())
	if ghostButton(c.Key("mt-correction-reset"), i18n.T("Reset to default prompt")).Disabled(state.CorrectionStyleResetDisabled).Clicked() {
		actions.resetMeetingCorrectionStyle()
	}
}

func meetingConsent(c *ui.Context) {
	t := c.Theme()
	ui.Text(c, i18n.T("Get the other person's consent before you record them speaking.")).
		FontSize(px(t, fsCaption)).TextColor(t.TextMuted)
}
