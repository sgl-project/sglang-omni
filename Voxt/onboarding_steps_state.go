package main

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/permission"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func onboardingPermission(translator *i18n.Translator, state permission.State, askAction string) onboardingPermissionRowViewState {
	row := onboardingPermissionRowViewState{AskAction: askAction, CanOpen: state != permission.Granted}
	switch state {
	case permission.Granted:
		row.Code, row.Text, row.OK = "granted", translator.T("Allowed"), true
	case permission.Denied:
		row.Code, row.Text = "denied", translator.T("Not allowed")
	case permission.Restricted:
		row.Code, row.Text = "restricted", translator.T("Restricted by your organisation")
	case permission.NotDetermined:
		row.Code, row.Text, row.CanAsk = "notDetermined", translator.T("Not asked yet"), true
	default:
		row.Code, row.Text = "unknown", translator.T("Unknown")
	}
	return row
}

func buildOnboardingPermissions(translator *i18n.Translator, input onboardingViewStateInput) onboardingPermissionsViewState {
	microphone := onboardingPermission(translator, input.PermissionMic, proto.PermissionMicrophone)
	speech := onboardingPermission(translator, input.PermissionSpeech, proto.PermissionSpeechRecognition)
	access := onboardingPermissionRowViewState{Code: "denied", Text: translator.T("Not allowed"), CanAsk: true, CanOpen: true, AskAction: proto.PermissionAccessibility, AskTooltip: translator.T("Adds Voxt to the list in System Settings")}
	if input.PermissionAccess {
		access.Code, access.Text, access.OK, access.CanAsk, access.CanOpen = "granted", translator.T("Allowed"), true, false, false
	}
	return onboardingPermissionsViewState{
		AskRequired: input.PermissionMic == permission.NotDetermined || !input.PermissionAccess,
		Microphone:  microphone, Access: access, Speech: speech,
	}
}

func buildOnboardingMore(translator *i18n.Translator, input onboardingViewStateInput) onboardingMoreViewState {
	state := onboardingMoreViewState{}
	current := settings.PresetOf(input.Settings.Features)
	for _, preset := range settings.Presets {
		state.PresetOptions = append(state.PresetOptions, onboardingOptionViewState{ID: preset, Label: presetTitleFor(translator, preset)})
	}
	if current == "" {
		state.PresetOptions = append(state.PresetOptions, onboardingOptionViewState{Label: presetTitleFor(translator, current)})
	}
	state.PresetSelected = presetTitleFor(translator, current)
	state.PresetAbout = presetAboutFor(translator, current)
	if current == settings.PresetFn {
		state.PresetWarnings = []string{
			translator.T("Auralink uses fn too, so while it runs, both apps react to these keys."),
			translator.T("macOS may react to fn too (the Globe key). If it gets in the way, set it to Do Nothing in System Settings → Keyboard."),
		}
	} else if current == settings.PresetMouseMiddle && !input.HotkeysInstalled {
		state.PresetWarnings = []string{translator.T("Mouse-button shortcuts need Accessibility. Use Start in the guide, or grant it in Permissions.")}
	}
	for _, feature := range append(append([]string{}, hotkey.Features...), hotkey.PasteLast) {
		keys := input.MoreKeys[feature]
		if keys == "" {
			keys = translator.T("None")
		}
		state.Shortcuts = append(state.Shortcuts, onboardingShortcutRowViewState{
			Title: onboardingFeatureTitle(translator, feature), Keys: keys, Problems: onboardingConflictLines(translator, input.Conflicts, feature, 0),
		})
	}
	return state
}
