package main

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/permission"
)

// System Settings panes.
const (
	privacyMicrophone    = "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
	privacyAccessibility = "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
	privacySpeech        = "x-apple.systempreferences:com.apple.preference.security?Privacy_SpeechRecognition"
	privacyAudioCapture  = "x-apple.systempreferences:com.apple.preference.security?Privacy_AudioCapture"
)

type permissionRowViewState struct {
	PermissionID string `json:"permission_id"`
	SettingsURL  string `json:"settings_url"`
	Status       string `json:"status"`
	Allowed      bool   `json:"allowed"`
	Known        bool   `json:"known"`
	Ask          bool   `json:"ask"`
	Open         bool   `json:"open"`
}

type permissionsViewState struct {
	AskRequired       bool                   `json:"ask_required"`
	Microphone        permissionRowViewState `json:"microphone"`
	Accessibility     permissionRowViewState `json:"accessibility"`
	SystemAudio       permissionRowViewState `json:"system_audio"`
	SpeechRecognition permissionRowViewState `json:"speech_recognition"`
}

type permissionsStateInput struct {
	Microphone        permission.State
	Accessibility     bool
	SpeechRecognition permission.State
}

func buildPermissionsViewState(translator *i18n.Translator, input permissionsStateInput) permissionsViewState {
	microphoneText, microphoneAllowed := permissionStatusText(translator, input.Microphone)
	accessibilityText := translator.T("Not allowed")
	if input.Accessibility {
		accessibilityText = translator.T("Allowed")
	}
	speechText, speechAllowed := permissionStatusText(translator, input.SpeechRecognition)
	return permissionsViewState{
		AskRequired: input.Microphone == permission.NotDetermined || !input.Accessibility,
		Microphone: permissionRowViewState{
			PermissionID: proto.PermissionMicrophone, SettingsURL: privacyMicrophone,
			Status: microphoneText, Allowed: microphoneAllowed, Known: true,
			Ask: input.Microphone == permission.NotDetermined, Open: !microphoneAllowed,
		},
		Accessibility: permissionRowViewState{
			PermissionID: proto.PermissionAccessibility, SettingsURL: privacyAccessibility,
			Status: accessibilityText, Allowed: input.Accessibility, Known: true,
			Ask: !input.Accessibility, Open: !input.Accessibility,
		},
		SystemAudio: permissionRowViewState{
			PermissionID: proto.PermissionSystemAudio, SettingsURL: privacyAudioCapture,
			Status: translator.T("Unknown"), Known: false, Ask: true, Open: true,
		},
		SpeechRecognition: permissionRowViewState{
			PermissionID: proto.PermissionSpeechRecognition, SettingsURL: privacySpeech,
			Status: speechText, Allowed: speechAllowed, Known: true,
			Ask: input.SpeechRecognition == permission.NotDetermined, Open: !speechAllowed,
		},
	}
}

func permissionStatusText(translator *i18n.Translator, state permission.State) (string, bool) {
	switch state {
	case permission.Granted:
		return translator.T("Allowed"), true
	case permission.Denied:
		return translator.T("Not allowed"), false
	case permission.Restricted:
		return translator.T("Restricted by your organisation"), false
	case permission.NotDetermined:
		return translator.T("Not asked yet"), false
	default:
		return translator.T("Unknown"), false
	}
}
