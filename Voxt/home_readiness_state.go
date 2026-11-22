package main

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/dictation"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
	"github.com/sgl-project/sglang-omni/Voxt/internal/permission"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type homeReadinessViewState struct {
	Key    string `json:"key"`
	Title  string `json:"title"`
	OK     bool   `json:"ok"`
	Warn   bool   `json:"warn"`
	Text   string `json:"text"`
	Kind   string `json:"kind"`
	Action string `json:"action"`
	Fix    string `json:"fix"`
}

type homeReadinessInput struct {
	SpeechModelID      string
	SpeechModelName    string
	SpeechReady        bool
	SpeechExplicit     bool
	SpeechStatus       models.Status
	SpeechDownloadLeft string
	LanguageModelName  string
	LanguageModelReady bool
	PermissionsChecked bool
	Microphone         permission.State
	Accessibility      bool
	HotkeysInstalled   bool
	Helper             helper.Status
}

// buildHomeReadinessList says what is ready and what is missing. A helper
// that is starting is neither: it is ready in a moment.
func buildHomeReadinessList(translator *i18n.Translator, features settings.Features, input homeReadinessInput) []homeReadinessViewState {
	var out []homeReadinessViewState
	out = append(out, buildHomeSpeechReadiness(translator, input))
	if !input.LanguageModelReady {
		out = append(out, homeReadinessViewState{Key: "home-llm", Title: translator.T("Language model"), Kind: "warn",
			Text:   translator.T("No language model is set up. Translate and Rewrite need one, and Transcribe types the text as recognised."),
			Action: translator.T("Open Language models"), Fix: homeFixLanguageModels})
	} else {
		out = append(out, homeReadinessViewState{Key: "home-llm", Title: translator.T("Language model"), OK: true, Text: input.LanguageModelName})
	}
	switch {
	case !input.PermissionsChecked:
	case input.Microphone == permission.Granted:
		out = append(out, homeReadinessViewState{Key: "home-mic", Title: translator.T("Microphone"), OK: true, Text: translator.T("Allowed")})
	case input.Microphone == permission.NotDetermined:
		out = append(out, homeReadinessViewState{Key: "home-mic", Title: translator.T("Microphone"), Kind: "info",
			Text: translator.T("macOS asks for the microphone the first time you record."), Action: translator.T("Ask now"), Fix: homeFixMicrophone})
	default:
		out = append(out, homeReadinessViewState{Key: "home-mic", Title: translator.T("Microphone"), Kind: "error",
			Text: translator.T("Voxt does not have microphone access. Allow it in System Settings."), Action: translator.T("Open Permissions"), Fix: homeFixPermissions})
	}
	switch {
	case !input.PermissionsChecked:
	case input.Accessibility:
		out = append(out, homeReadinessViewState{Key: "home-access", Title: translator.T("Accessibility"), OK: true, Text: translator.T("Allowed")})
	default:
		out = append(out, homeReadinessViewState{Key: "home-access", Title: translator.T("Accessibility"), Kind: "warn",
			Text:   translator.T("Without Accessibility access, results are copied for you to paste, and shortcuts such as Right ⌥ do not work."),
			Action: translator.T("Open Permissions"), Fix: homeFixPermissions})
	}
	out = append(out, buildHomeShortcutReadiness(translator, features, input.HotkeysInstalled))
	switch status := input.Helper; {
	case status.State == helper.Ready, status.State == helper.Starting:
	case status.Reason == helper.FailUpdated:
		out = append(out, homeReadinessViewState{Key: "home-helper", Title: translator.T("Helper"), Kind: "warn",
			Text: uitext.HelperFailTextFor(translator, status), Action: translator.T("Restart Voxt"), Fix: homeFixRelaunch})
	default:
		out = append(out, homeReadinessViewState{Key: "home-helper", Title: translator.T("Helper"), Kind: "error",
			Text: uitext.HelperFailTextFor(translator, status), Action: translator.T("Restart helper"), Fix: homeFixRestartHelper})
	}
	return out
}

// buildHomeSpeechReadiness is the speech model's line: the model Transcribe
// uses, or how to get the recommended one. Automatic System dictation is not
// ready; explicitly selected System dictation is.
func buildHomeSpeechReadiness(translator *i18n.Translator, input homeReadinessInput) homeReadinessViewState {
	state := homeReadinessViewState{Key: "home-asr", Title: translator.T("Speech recognition")}
	if input.SpeechReady && (input.SpeechModelID != dictation.ModelID || input.SpeechExplicit) {
		state.OK, state.Text = true, input.SpeechModelName
		return state
	}
	state.Kind, state.Text = "warn", translator.T("No speech model is installed. Install one to transcribe, translate and rewrite by voice.")
	if input.SpeechReady {
		state.Kind, state.Text = "info", translator.T("Using System dictation for now.")
	}
	switch {
	case input.SpeechStatus.Downloading:
		state.Kind = "info"
		state.Text = translator.T("The speech model is still downloading (%d%%).", downloadPercent(input.SpeechStatus))
		state.Action, state.Fix = translator.T("Open Speech models"), homeFixSpeechModels
	case input.SpeechStatus.OnDisk:
		state.Action, state.Fix = translator.T("Open Speech models"), homeFixSpeechModels
	case input.SpeechReady:
		state.Action = translator.T("Download the recommended model (%s)", input.SpeechDownloadLeft)
		state.Fix = homeFixSpeechDownload
	default:
		state.Action = translator.T("Download (%s)", input.SpeechDownloadLeft)
		state.Fix = homeFixSpeechDownload
	}
	return state
}

// buildHomeShortcutReadiness says whether Transcribe's first shortcut works.
func buildHomeShortcutReadiness(translator *i18n.Translator, features settings.Features, installed bool) homeReadinessViewState {
	state := homeReadinessViewState{Key: "home-keys", Title: translator.T("Shortcut")}
	hotkeys := hotkey.HotkeysOf(features, hotkey.Transcribe)
	if len(hotkeys) == 0 {
		state.Kind = "warn"
		state.Text = translator.T("Transcribe has no shortcut. Add one, or start it from the menu bar.")
		state.Action, state.Fix = translator.T("Open Transcribe"), homeFixTranscribe
		return state
	}
	// A clash is reported on the later binding, and Transcribe's first one wins.
	for _, problem := range hotkey.Check(features) {
		if problem.Feature == hotkey.Transcribe && problem.Index == 0 {
			state.Kind = "warn"
			state.Text = translator.T("Transcribe's shortcut %s: %s", uitext.HotkeyTextFor(translator, hotkeys[0].Keys), uitext.HotkeyProblemTextFor(translator, problem))
			state.Action, state.Fix = translator.T("Open Transcribe"), homeFixTranscribe
			return state
		}
	}
	state.OK, state.Text = true, uitext.HotkeyTextFor(translator, hotkeys[0].Keys)
	if hotkey.NeedsTap(hotkeys[0]) && !installed {
		state.Warn = true
		state.Text = translator.T("%s (once Accessibility is allowed)", uitext.HotkeyTextFor(translator, hotkeys[0].Keys))
	}
	return state
}
