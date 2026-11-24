package main

import (
	"slices"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func (a *app) meetingSettingsKey() string {
	meeting := a.prefs.Features.Meeting
	return meeting.Mode + "|" + b2s(meeting.HideFromSharing) + "|" + b2s(meeting.NoDetect) + "|" + b2s(meeting.AutoStart)
}

func (a *app) meetingSettingsSync() {
	if a.meetingUI == nil {
		a.meetingUI = &meetingSettingsUI{correction: speechContextState{text: meeting.DefaultCorrectionStyle}}
	}
	local := a.meetingUI
	savedStyle := a.prefs.Features.Meeting.CorrectionStyle
	if !local.correction.dirty && savedStyle != local.correction.seen {
		local.correction.text, local.correction.seen = savedStyle, savedStyle
		if savedStyle == "" {
			local.correction.text = meeting.DefaultCorrectionStyle
		}
	}
	seen := a.meetingSettingsKey()
	if local.seen == seen {
		return
	}
	local.seen = seen
	local.mode = a.prefs.Features.Meeting.Mode
	local.hide = a.prefs.Features.Meeting.HideFromSharing
	local.detect = !a.prefs.Features.Meeting.NoDetect
	local.auto = a.prefs.Features.Meeting.AutoStart
}

func (a *app) meetingSettingsModel() settingsMeetingViewState {
	a.meetingSettingsSync()
	meeting := a.prefs.Features.Meeting
	meeting.Mode = a.meetingUI.mode
	meeting.HideFromSharing = a.meetingUI.hide
	meeting.NoDetect = !a.meetingUI.detect
	meeting.AutoStart = a.meetingUI.auto
	readyASR := a.installed(local.ASR)
	shownASR := slices.DeleteFunc(slices.Clone(readyASR), func(model local.Model) bool { return a.modelHidden(model.ID) })
	return buildSettingsMeetingViewState(i18n.CurrentTranslator(), settingsMeetingStateInput{
		Meeting: meeting, ReadyASR: readyASR, ShownASR: shownASR, ReadyLLM: a.installed(local.LLM), CorrectionText: a.meetingUI.correction.text,
	})
}

func (a *app) setMeetingCaptureMode(captureMode string) {
	a.meetingSettingsSync()
	a.meetingUI.mode = captureMode
	a.updateSettings(func(current *settings.Settings) { current.Features.Meeting.Mode = captureMode })
}

func (a *app) setMeetingHideFromSharing(hideFromSharing bool) {
	a.meetingSettingsSync()
	a.meetingUI.hide = hideFromSharing
	a.updateSettings(func(current *settings.Settings) { current.Features.Meeting.HideFromSharing = hideFromSharing })
	a.applyMeetingOverlayProtection()
}

// setMeetingDetect turns the detection of meetings on or off; off, a
// question on screen goes too (meeting_detect.go).
func (a *app) setMeetingDetect(detect bool) {
	a.meetingSettingsSync()
	a.meetingUI.detect = detect
	a.updateSettings(func(current *settings.Settings) { current.Features.Meeting.NoDetect = !detect })
	if !detect {
		a.closeMeetingAsk()
	}
	a.syncMicWatch()
}

// setMeetingAutoStart chooses whether a detected meeting starts without
// asking.
func (a *app) setMeetingAutoStart(auto bool) {
	a.meetingSettingsSync()
	a.meetingUI.auto = auto
	a.updateSettings(func(current *settings.Settings) { current.Features.Meeting.AutoStart = auto })
	if auto {
		a.closeMeetingAsk() // the card asked what is now decided
	}
}

func (a *app) setMeetingASR(speechModelID string) {
	a.updateSettings(func(current *settings.Settings) { current.Features.Meeting.ASR = speechModelID })
}

func (a *app) setMeetingLanguage(languageCode string) {
	a.updateSettings(func(current *settings.Settings) { current.Features.Meeting.Lang = languageCode })
}

func (a *app) setMeetingLLM(languageModelID string) {
	a.updateSettings(func(current *settings.Settings) { current.Features.Meeting.LLM = languageModelID })
}

func (a *app) autosaveMeetingCorrectionStyle(changed, focused bool) {
	a.autosaveState(changed, focused, &a.meetingUI.correction, a.saveMeetingCorrectionStyle)
}

func (a *app) saveMeetingCorrectionStyle() {
	local := &a.meetingUI.correction
	local.dirty = false
	value := strings.TrimSpace(local.text)
	if value == strings.TrimSpace(meeting.DefaultCorrectionStyle) {
		value = ""
	}
	local.seen = value
	if value == "" {
		local.text = meeting.DefaultCorrectionStyle
	}
	if value != a.prefs.Features.Meeting.CorrectionStyle {
		a.updateSettings(func(current *settings.Settings) { current.Features.Meeting.CorrectionStyle = value })
	}
}

func (a *app) resetMeetingCorrectionStyle() {
	a.meetingUI.correction.text = meeting.DefaultCorrectionStyle
	a.saveMeetingCorrectionStyle()
}

func b2s(value bool) string {
	if value {
		return "1"
	}
	return "0"
}
