package main

import (
	"slices"
	"strconv"
	"strings"

	"github.com/hehehai/actnowkit/registry"
	"github.com/hehehai/actnowkit/shortcuts"
	"github.com/sgl-project/sglang-omni/Voxt/internal/actnow"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type shortcutPromptViewState struct {
	Keys        string `json:"keys"`
	OtherApp    string `json:"other_app"`
	Function    string `json:"function"`
	Link        string `json:"link"`
	FeaturePath string `json:"feature_path"`
}

// shortcutID is the family's id of feature's i-th shortcut, as
// actnow.Shortcuts declares it ("transcribe", "transcribe.2",
// "pasteLast").
func shortcutID(feature string, index int) string {
	if feature == hotkey.PasteLast {
		return "pasteLast"
	}
	if index > 0 {
		return feature + "." + itoa(index+1)
	}
	return feature
}

// shortcutOf is the feature and index of a declared shortcut id.
func shortcutOf(id string) (feature string, index int) {
	if id == "pasteLast" {
		return hotkey.PasteLast, 0
	}
	feature, number, ok := strings.Cut(id, ".")
	if ok {
		if parsed, err := strconv.Atoi(number); err == nil && parsed > 1 {
			index = parsed - 1
		}
	}
	return feature, index
}

// familyAppName is another app's name for the UI.
func familyAppName(holder shortcuts.Holder) string {
	if holder.Name != "" {
		return holder.Name
	}
	return actnow.AppName(holder.BundleID)
}

// familyFunctionFor is the name of another app's function, translated when
// it is one of the family's known ones (its manifest has it in English).
func familyFunctionFor(translator *i18n.Translator, shortcut shortcuts.Shortcut) string {
	switch shortcut.ID {
	case "transcribe":
		return translator.T("Transcribe")
	case "translate":
		return translator.T("Translate")
	case "rewrite":
		return translator.T("Rewrite")
	case "pasteLast":
		return translator.T("Paste last")
	case "dictation":
		return translator.T("Dictation")
	}
	if shortcut.Title != "" {
		return shortcut.Title
	}
	return shortcut.ID
}

// alsoUsedByFor is the line of another app's shortcut on the same keys.
func alsoUsedByFor(translator *i18n.Translator, holder shortcuts.Holder, theirs shortcuts.Shortcut) string {
	return translator.T("Also used by %s (%s)", familyAppName(holder), familyFunctionFor(translator, theirs))
}

func conflictLinesFor(translator *i18n.Translator, conflicts []shortcuts.Conflict, feature string, index int) []string {
	id := shortcutID(feature, index)
	var lines []string
	for _, conflict := range conflicts {
		if conflict.Mine.ID != id {
			continue
		}
		if line := alsoUsedByFor(translator, conflict.Other, conflict.Theirs); !slices.Contains(lines, line) {
			lines = append(lines, line)
		}
	}
	return lines
}

// conflictKeysFor are the keys of VoxtNew's shortcut of conflict, as
// VoxtNew writes them.
func conflictKeysFor(translator *i18n.Translator, features settings.Features, conflict shortcuts.Conflict) string {
	feature, index := shortcutOf(conflict.Mine.ID)
	if feature == hotkey.PasteLast {
		return uitext.HotkeyTextFor(translator, features.PasteLast.Keys)
	}
	if bindings := hotkey.HotkeysOf(features, feature); index < len(bindings) {
		return uitext.HotkeyTextFor(translator, bindings[index].Keys)
	}
	return uitext.HotkeyTextFor(translator, conflict.Mine.Keys)
}

func buildShortcutPromptViewState(translator *i18n.Translator, features settings.Features, conflict shortcuts.Conflict) shortcutPromptViewState {
	feature, _ := shortcutOf(conflict.Mine.ID)
	return shortcutPromptViewState{
		Keys: conflictKeysFor(translator, features, conflict), OtherApp: familyAppName(conflict.Other),
		Function: familyFunctionFor(translator, conflict.Theirs), Link: conflict.Other.Links[registry.LinkShortcuts],
		FeaturePath: featurePath(feature),
	}
}
