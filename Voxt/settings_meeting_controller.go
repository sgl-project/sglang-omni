package main

import (
	"slices"
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"time"
)

func (a *app) meetingSettingsKey() string {
	meeting := a.prefs.Features.Meeting
	return meeting.Mode + "|" + b2s(meeting.HideFromSharing)
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
}

func (a *app) meetingSettingsModel() settingsMeetingViewState {
	a.meetingSettingsSync()
	meeting := a.prefs.Features.Meeting
	meeting.Mode = a.meetingUI.mode
	meeting.HideFromSharing = a.meetingUI.hide
	readyASR := a.installed(local.ASR)
	shownASR := slices.DeleteFunc(slices.Clone(readyASR), func(model local.Model) bool { return a.modelHidden(model.ID) })
	automaticASR, _ := local.Choose(automaticSpeechFor(a.meetingLang(), readyASR), "")
	speakerStatus := a.modelStatus[models.SpeakerDiarization]
	percent := 0
	if speakerStatus.Total > 0 {
		percent = int(min(speakerStatus.Done*100/speakerStatus.Total, 100))
	}
	return buildSettingsMeetingViewState(i18n.CurrentTranslator(), settingsMeetingStateInput{
		SpeakerReady: speakerStatus.Installed, SpeakerDownloading: speakerStatus.Downloading, SpeakerProgress: percent, SpeakerFailed: speakerStatus.Err != nil,
		Meeting: meeting, ReadyASR: readyASR, ShownASR: shownASR, AutomaticASR: automaticASR, ReadyLLM: a.installed(local.LLM), CorrectionText: a.meetingUI.correction.text,
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

func (a *app) setMeetingSpeakers(enabled bool) {
	if enabled && a.models != nil && !a.models.Installed(models.SpeakerDiarization) {
		a.updateSettings(func(current *settings.Settings) { current.Features.Meeting.Speakers = false })
		a.downloadModel(models.SpeakerDiarization)
		return // a missing model is a download action, not an enabled feature
	}
	a.updateSettings(func(current *settings.Settings) { current.Features.Meeting.Speakers = enabled })
	if enabled && a.hstatus.Has(proto.MethodMeetingWindow) {
		helperCall(a, proto.MethodPermissionsRequest, proto.PermissionsRequestParams{Permission: proto.PermissionScreenRecording}, 10*time.Second, func(result proto.PermissionsResult, err error) {
			if err == nil && !result.ScreenRecording {
				a.toast(i18n.T("To match speaker names, allow Voxt in System Settings → Privacy & Security → Screen & System Audio Recording."))
			}
		})
	}

}
