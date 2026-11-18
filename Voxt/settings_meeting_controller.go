package main

import (
	"slices"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func (a *app) meetingSettingsKey() string {
	meeting := a.prefs.Features.Meeting
	return meeting.Mode + "|" + b2s(meeting.HideFromSharing)
}

func (a *app) meetingSettingsSync() {
	if a.meetingUI == nil {
		a.meetingUI = &meetingSettingsUI{}
	}
	local := a.meetingUI
	seen := a.meetingSettingsKey()
	if local.seen == seen {
		return
	}
	local.seen = seen
	local.mode = a.prefs.Features.Meeting.Mode
	local.hide = a.prefs.Features.Meeting.HideFromSharing
}

func (a *app) meetingSettingsModel() settingsMeetingViewState {
	a.meetingSettingsSync()
	meeting := a.prefs.Features.Meeting
	meeting.Mode = a.meetingUI.mode
	meeting.HideFromSharing = a.meetingUI.hide
	readyASR := a.installed(local.ASR)
	shownASR := slices.DeleteFunc(slices.Clone(readyASR), func(model local.Model) bool { return a.modelHidden(model.ID) })
	return buildSettingsMeetingViewState(i18n.CurrentTranslator(), settingsMeetingStateInput{
		Meeting: meeting, ReadyASR: readyASR, ShownASR: shownASR, ReadyLLM: a.installed(local.LLM),
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

func (a *app) setMeetingASR(speechModelID string) {
	a.updateSettings(func(current *settings.Settings) { current.Features.Meeting.ASR = speechModelID })
}

func (a *app) setMeetingLanguage(languageCode string) {
	a.updateSettings(func(current *settings.Settings) { current.Features.Meeting.Lang = languageCode })
}

func (a *app) setMeetingLLM(languageModelID string) {
	a.updateSettings(func(current *settings.Settings) { current.Features.Meeting.LLM = languageModelID })
}

func b2s(value bool) string {
	if value {
		return "1"
	}
	return "0"
}
