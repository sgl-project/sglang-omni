package main

import (
	"slices"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type featureCustomKeysViewState struct {
	InputText string `json:"input_text"`
	KeysText  string `json:"keys_text"`
	Canonical string `json:"canonical"`
	Parsed    bool   `json:"parsed"`
	ErrorText string `json:"error_text"`
}

type featureKeyRowViewState struct {
	Keys               string                     `json:"keys"`
	Value              string                     `json:"value"`
	Options            []featureOptionViewState   `json:"options"`
	Custom             featureCustomKeysViewState `json:"custom"`
	NeedsAccessibility bool                       `json:"needs_accessibility"`
	Problems           []string                   `json:"problems"`
	ProblemCodes       []string                   `json:"problem_codes"`
	UsedBy             []string                   `json:"used_by"`
}

type featureShortcutsViewState struct {
	Prefix            string                   `json:"prefix"`
	HotkeysInstalled  bool                     `json:"hotkeys_installed"`
	HelperRecordsKeys bool                     `json:"helper_records_keys"`
	EmptyText         string                   `json:"empty_text"`
	RewriteHint       bool                     `json:"rewrite_hint"`
	Rows              []featureKeyRowViewState `json:"rows"`
	Paste             *featureKeyRowViewState  `json:"paste,omitempty"`
	Cancel            *featureKeyRowViewState  `json:"cancel,omitempty"`
	EndWord           *featureEndWordViewState `json:"end_word,omitempty"`
}

type featureShortcutsStateInput struct {
	Feature           string
	Settings          settings.Settings
	Local             *featureUI
	HotkeysInstalled  bool
	HelperRecordsKeys bool
	UsedBy            map[string][]string
	EndWord           *featureEndWordViewState
}

func buildFeatureShortcutsViewState(translator *i18n.Translator, input featureShortcutsStateInput) featureShortcutsViewState {
	feature := input.Feature
	state := featureShortcutsViewState{
		Prefix: uitext.FeaturePrefix(feature), HotkeysInstalled: input.HotkeysInstalled,
		HelperRecordsKeys: input.HelperRecordsKeys, RewriteHint: feature == hotkey.Rewrite,
	}
	local := input.Local
	if local == nil {
		local = &featureUI{}
	}
	hotkeys := hotkey.HotkeysOf(input.Settings.Features, feature)
	problems := uitext.HotkeyProblems(input.Settings.Features, feature)
	problemCodes := featureHotkeyProblemCodes(input.Settings.Features, feature)
	state.Rows = make([]featureKeyRowViewState, len(hotkeys))
	for index, hotkey := range hotkeys {
		var row keyRow
		if index < len(local.rows) {
			row = local.rows[index]
		}
		state.Rows[index] = buildFeatureKeyRowViewState(translator, featureKeyRowStateInput{
			Hotkey: hotkey, Local: row, Problems: problems[index], ProblemCodes: problemCodes[index], UsedBy: input.UsedBy,
		})
	}
	if len(hotkeys) == 0 {
		state.EmptyText = translator.T("No shortcut yet. Add one to use this feature.")
		if feature == hotkey.Transcribe {
			state.EmptyText = translator.T("No shortcut yet. Start from the menu bar, or add one.")
		}
	}
	if feature == hotkey.Transcribe {
		state.EndWord = input.EndWord
		paste := buildFeatureKeyRowViewState(translator, featureKeyRowStateInput{
			Hotkey: input.Settings.Features.PasteLast, Local: local.paste, Off: true,
			Problems:     uitext.HotkeyProblems(input.Settings.Features, hotkey.PasteLast)[0],
			ProblemCodes: featureHotkeyProblemCodes(input.Settings.Features, hotkey.PasteLast)[0], UsedBy: input.UsedBy,
		})
		cancel := buildFeatureKeyRowViewState(translator, featureKeyRowStateInput{
			Hotkey: input.Settings.Features.Cancel, Local: local.cancel, Off: true, First: []string{settings.DefaultCancel},
			Problems:     uitext.HotkeyProblems(input.Settings.Features, hotkey.CancelID)[0],
			ProblemCodes: featureHotkeyProblemCodes(input.Settings.Features, hotkey.CancelID)[0], UsedBy: input.UsedBy,
		})
		state.Paste, state.Cancel = &paste, &cancel
	}
	return state
}

type featureKeyRowStateInput struct {
	Hotkey       settings.Hotkey
	Local        keyRow
	Off          bool
	First        []string
	Problems     []string
	ProblemCodes []string
	UsedBy       map[string][]string
}

func buildFeatureKeyRowViewState(translator *i18n.Translator, input featureKeyRowStateInput) featureKeyRowViewState {
	keys := input.Hotkey.Keys
	value := chordLabel(keys)
	if input.Off && keys == "" {
		value = translator.T("Off")
	}
	state := featureKeyRowViewState{
		Keys: keys, Value: value, NeedsAccessibility: hotkey.NeedsTap(input.Hotkey),
		Problems: input.Problems, ProblemCodes: input.ProblemCodes,
		UsedBy: input.UsedBy[recordedKeys(input.Local, keys)], Custom: buildFeatureCustomKeysViewState(input.Local),
	}
	labels, byLabel := keyOptionsFor(translator, keys, input.Off, input.First...)
	for _, label := range labels {
		state.Options = append(state.Options, featureOptionViewState{Label: label, Value: byLabel[label], Custom: label == translator.T("Custom…")})
	}
	return state
}

func keyOptionsFor(translator *i18n.Translator, current string, off bool, first ...string) (labels []string, byLabel map[string]string) {
	byLabel = map[string]string{}
	if off {
		labels = append(labels, translator.T("Off"))
		byLabel[translator.T("Off")] = ""
	}
	for _, preset := range slices.Concat(first, keyPresets) {
		label := chordLabel(preset)
		labels = append(labels, label)
		byLabel[label] = preset
	}
	if current != "" && presetOf(current) == "" {
		label := chordLabel(current)
		if _, duplicate := byLabel[label]; !duplicate {
			labels = append(labels, label)
			byLabel[label] = current
		}
	}
	labels = append(labels, translator.T("Custom…"))
	return labels, byLabel
}

func buildFeatureCustomKeysViewState(local keyRow) featureCustomKeysViewState {
	state := featureCustomKeysViewState{InputText: local.text, KeysText: uitext.HotkeyText(local.text)}
	chord, err := hotkey.Parse(local.text)
	state.ErrorText = uitext.ParseErrorText(err)
	if err == nil {
		state.Parsed, state.Canonical = true, chord.String()
	}
	return state
}

func featureHotkeyProblemCodes(features settings.Features, feature string) map[int][]string {
	codes := map[int][]string{}
	for _, problem := range hotkey.Check(features) {
		switch {
		case problem.Feature == feature:
			codes[problem.Index] = append(codes[problem.Index], problem.Code)
		case problem.Code == hotkey.ProblemConflict && problem.Other == feature:
			codes[problem.OtherIndex] = append(codes[problem.OtherIndex], problem.Code)
		}
	}
	return codes
}
