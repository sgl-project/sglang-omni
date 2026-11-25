package main

import (
	"slices"

	"github.com/hehehai/actnowkit/shortcuts"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type settingsShortcutRowViewState struct {
	Feature            string   `json:"feature"`
	Index              int      `json:"index"`
	Path               string   `json:"path"`
	Title              string   `json:"title"`
	PageTitle          string   `json:"page_title"`
	Icon               string   `json:"icon"`
	KeyText            string   `json:"key_text"`
	TriggerText        string   `json:"trigger_text"`
	HasShortcut        bool     `json:"has_shortcut"`
	NeedsAccessibility bool     `json:"needs_accessibility"`
	Problems           []string `json:"problems"`
}

type shortcutPresetOptionViewState struct {
	ID    string `json:"id"`
	Title string `json:"title"`
	About string `json:"about"`
}

type shortcutPresetPickerViewState struct {
	Options  []shortcutPresetOptionViewState `json:"options"`
	Selected string                          `json:"selected"`
	About    string                          `json:"about"`
	Warnings []string                        `json:"warnings"`
}

type sessionSlotViewState struct {
	Title    string               `json:"title"`
	Recorder keyRecorderViewState `json:"recorder"`
	Problems []string             `json:"problems"`
}

type settingsShortcutsViewState struct {
	AccessibilityInstalled bool                           `json:"accessibility_installed"`
	StrictShortcuts        bool                           `json:"strict_shortcuts"`
	ReturnFinishes         bool                           `json:"return_finishes"`
	Preset                 shortcutPresetPickerViewState  `json:"preset"`
	Rows                   []settingsShortcutRowViewState `json:"rows"`
	Sessions               []sessionSlotViewState         `json:"sessions"`
	SessionsOn             bool                           `json:"sessions_on,omitempty"`
	// SessionUse says how often the user switched sessions, once they did.
	SessionUse string `json:"session_use,omitempty"`
}

type settingsShortcutsStateInput struct {
	Features               settings.Features
	FamilyConflicts        []shortcuts.Conflict
	AccessibilityInstalled bool
	HelperRecordsKeys      bool
	SessionSwitches        int
	SessionReturns         int
}

func buildSettingsShortcutsViewState(translator *i18n.Translator, input settingsShortcutsStateInput) settingsShortcutsViewState {
	state := settingsShortcutsViewState{
		AccessibilityInstalled: input.AccessibilityInstalled,
		StrictShortcuts:        input.Features.StrictShortcuts,
		ReturnFinishes:         input.Features.ReturnFinishes,
		SessionsOn:             input.Features.SessionsOn,
		Preset:                 buildShortcutPresetPickerViewState(translator, input.Features, input.AccessibilityInstalled, false),
	}
	for _, feature := range append(append([]string{}, hotkey.Features...), hotkey.PasteLast, hotkey.CancelID, hotkey.Note) {
		if input.Features.IsHidden(feature) || (feature == hotkey.Note && input.Features.IsHidden(settings.HideNotes)) {
			continue
		}
		bindings := hotkey.HotkeysOf(input.Features, feature)
		problems := uitext.HotkeyProblemsFor(translator, input.Features, feature)
		if len(bindings) == 0 {
			state.Rows = append(state.Rows, buildSettingsShortcutRow(translator, feature, 0, nil, nil))
			continue
		}
		for index := range bindings {
			lines := append(slices.Clone(problems[index]), conflictLinesFor(translator, input.FamilyConflicts, feature, index)...)
			state.Rows = append(state.Rows, buildSettingsShortcutRow(translator, feature, index, &bindings[index], lines))
		}
	}
	sessions := hotkey.Sessions
	if !input.Features.SessionsOn {
		sessions = sessions[:len(sessions)-1] // shortcut: the fixtures of before the switch list three; drop this when they are next regenerated
	}
	for i, id := range sessions {
		recorder := buildKeyRecorderViewState(translator, input.HelperRecordsKeys, input.Features.Sessions[i].Keys)
		if input.Features.Sessions[i].Keys == "" {
			recorder.DisplayedKeys = uitext.HotkeyGlyphs("") // "—": the slot is off
		}
		state.Sessions = append(state.Sessions, sessionSlotViewState{
			Title:    uitext.FeatureTitleFor(translator, id),
			Recorder: recorder,
			Problems: uitext.HotkeyProblemsFor(translator, input.Features, id)[0],
		})
	}
	if input.SessionSwitches > 0 {
		state.SessionUse = translator.T("Switched sessions %d times, %d of them back to a session with a card. Counted on this Mac only.", input.SessionSwitches, input.SessionReturns)
	}
	return state
}

func buildSettingsShortcutRow(translator *i18n.Translator, feature string, index int, binding *settings.Hotkey, problems []string) settingsShortcutRowViewState {
	icon := "transcribe"
	switch feature {
	case hotkey.Translate:
		icon = "translate"
	case hotkey.Rewrite:
		icon = "rewrite"
	case hotkey.PasteLast:
		icon = "paste"
	case hotkey.CancelID:
		icon = "cancel"
	case hotkey.Note:
		icon = "note"
	}
	pageFeature := feature
	if feature == hotkey.PasteLast || feature == hotkey.CancelID {
		pageFeature = hotkey.Transcribe
	}
	row := settingsShortcutRowViewState{
		Feature: feature, Index: index, Path: featurePath(feature),
		Title: uitext.FeatureTitleFor(translator, feature), PageTitle: uitext.FeatureTitleFor(translator, pageFeature),
		Icon: icon, KeyText: translator.T("None"), Problems: problems,
	}
	if binding == nil {
		return row
	}
	row.HasShortcut = true
	row.KeyText = uitext.HotkeyTextFor(translator, binding.Keys)
	trigger := binding.Trigger
	if feature == hotkey.PasteLast || feature == hotkey.CancelID || feature == hotkey.Note {
		trigger = settings.TriggerTap
	}
	row.TriggerText = uitext.TriggerTextFor(translator, trigger)
	row.NeedsAccessibility = feature == hotkey.CancelID || hotkey.NeedsTap(*binding)
	return row
}

// presetTitleFor is a shortcut preset's name ("": custom).
func presetTitleFor(translator *i18n.Translator, preset string) string {
	switch preset {
	case settings.PresetOption:
		return translator.T("Right ⌥ (default)")
	case settings.PresetFn:
		return translator.T("fn (Auralink's keys)")
	case settings.PresetCommand:
		return translator.T("⌘ combination")
	case settings.PresetMouseMiddle:
		return translator.T("Middle mouse + fn⇧")
	}
	return translator.T("Custom")
}

// presetAboutFor is one line on who a preset suits (setup guide).
func presetAboutFor(translator *i18n.Translator, preset string) string {
	switch preset {
	case settings.PresetOption:
		return translator.T("Right ⌥: one-handed and Voxt's default. Good when fn is the Globe key or shared with Auralink.")
	case settings.PresetFn:
		return translator.T("fn: matches Auralink and many Voxt 1.x setups. Watch for clashes with the Globe key and input sources.")
	case settings.PresetCommand:
		return translator.T("⌘ combination: right ⌘ only, away from the letter keys. Good if you keep Option for typing.")
	case settings.PresetMouseMiddle:
		return translator.T("Middle mouse + fn⇧: click the wheel to transcribe, double-click to rewrite. Mouse shortcuts need Accessibility.")
	}
	return ""
}

func buildShortcutPresetPickerViewState(translator *i18n.Translator, features settings.Features, accessibilityInstalled, onboarding bool) shortcutPresetPickerViewState {
	current := settings.PresetOf(features)
	state := shortcutPresetPickerViewState{
		Selected: presetTitleFor(translator, current),
		About:    translator.T("Shortcuts you changed yourself stay as they are."),
	}
	if onboarding {
		state.About = translator.T("You can change them later in Settings → Shortcuts.")
	}
	for _, preset := range settings.Presets {
		state.Options = append(state.Options, shortcutPresetOptionViewState{
			ID: preset, Title: presetTitleFor(translator, preset), About: presetAboutFor(translator, preset),
		})
	}
	if current == "" {
		state.Options = append(state.Options, shortcutPresetOptionViewState{Title: presetTitleFor(translator, current)})
	}
	switch current {
	case settings.PresetFn:
		state.Warnings = append(state.Warnings,
			translator.T("Auralink uses fn too, so while it runs, both apps react to these keys."),
			translator.T("macOS may react to fn too (the Globe key). If it gets in the way, set it to Do Nothing in System Settings → Keyboard."))
	case settings.PresetMouseMiddle:
		if !accessibilityInstalled {
			state.Warnings = append(state.Warnings, translator.T("Mouse-button shortcuts need Accessibility. Use Start in the guide, or grant it in Permissions."))
		}
	}
	return state
}
