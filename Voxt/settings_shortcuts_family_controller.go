package main

import (
	"slices"
	"strconv"
	"strings"

	"github.com/egoist/mygo"
	"github.com/hehehai/actnowkit/registry"
	"github.com/hehehai/actnowkit/shortcuts"
	"github.com/sgl-project/sglang-omni/Voxt/internal/actnow"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

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

// conflictLines are the lines of the other apps that use feature's i-th
// shortcut too (a.famConflicts).
func (a *app) conflictLines(feature string, index int) []string {
	return conflictLinesFor(i18n.CurrentTranslator(), a.famConflicts, feature, index)
}

// usedByLines are the lines of the other apps whose shortcuts are on keys
// (VoxtNew's notation): the recorder's, from the last discovery (memory
// only).
func (a *app) usedByLines(keys string) []string {
	if a.fam == nil || keys == "" {
		return nil
	}
	normalized, err := shortcuts.Normalize(keys)
	if err != nil {
		return nil
	}
	translator := i18n.CurrentTranslator()
	var lines []string
	for _, holder := range a.fam.UsedBy(normalized) {
		if line := alsoUsedByFor(translator, holder.App, holder.Shortcut); !slices.Contains(lines, line) {
			lines = append(lines, line)
		}
	}
	return lines
}

// scPromptState is the one-time prompt of a shortcut conflict.
type scPromptState struct {
	scPrompt *shortcuts.Conflict
}

type shortcutPromptViewState struct {
	Keys        string `json:"keys"`
	OtherApp    string `json:"other_app"`
	Function    string `json:"function"`
	Link        string `json:"link"`
	FeaturePath string `json:"feature_path"`
}

// shortcutPrompt shows the prompt of the first conflict VoxtNew is the
// later of (c.Prompt) whose pair was never prompted
// (prefs.Family.Prompted); it waits while the setup guide shows, and for
// the prompt shown to be answered.
func (a *app) shortcutPrompt() {
	if a.scPrompt != nil {
		if a.conflictNow(*a.scPrompt) {
			return
		}
		a.scPrompt = nil // the keys changed meanwhile
	}
	for _, conflict := range a.famConflicts {
		if conflict.Prompt && !slices.Contains(a.prefs.Family.Prompted, conflict.Key()) {
			a.scPrompt = &conflict
			return
		}
	}
}

// conflictNow says whether conflict is still one of the conflicts.
func (a *app) conflictNow(conflict shortcuts.Conflict) bool {
	return slices.ContainsFunc(a.famConflicts, func(current shortcuts.Conflict) bool { return current.Key() == conflict.Key() })
}

// answerShortcutPrompt records the prompt's conflict as prompted, whatever
// the answer, and shows the next one.
func (a *app) answerShortcutPrompt() {
	conflict := a.scPrompt
	a.scPrompt = nil
	if conflict == nil {
		return
	}
	key := conflict.Key()
	if !slices.Contains(a.prefs.Family.Prompted, key) {
		a.updateSettings(func(current *settings.Settings) {
			current.Family.Prompted = append(slices.Clone(current.Family.Prompted), key)
		})
	}
	a.shortcutPrompt()
}

func (a *app) openShortcutPromptLink(context *shortcutPromptUIContext, link string) {
	a.answerShortcutPrompt()
	a.openLink(context, link)
}

func (a *app) changeShortcutPrompt(path string) {
	a.answerShortcutPrompt()
	a.navigate(path)
}

// conflictsSeen marks as prompted the conflicts of the keys the user just
// gave a shortcut (old: the settings before): the recorder said which app
// uses them, the prompt would only repeat it.
func (a *app) conflictsSeen(old settings.Settings) {
	before := map[string]string{}
	for _, shortcut := range actnow.Shortcuts(old) {
		before[shortcut.ID] = shortcut.Keys
	}
	var seen []string
	for _, conflict := range a.famConflicts {
		if key := conflict.Key(); conflict.Prompt && before[conflict.Mine.ID] != conflict.Mine.Keys && !slices.Contains(a.prefs.Family.Prompted, key) && !slices.Contains(seen, key) {
			seen = append(seen, key)
		}
	}
	if len(seen) > 0 {
		a.updateSettings(func(current *settings.Settings) {
			current.Family.Prompted = append(slices.Clone(current.Family.Prompted), seen...)
		})
	}
	a.shortcutPrompt()
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

func (a *app) shortcutPromptModel() *shortcutPromptViewState {
	if a.scPrompt == nil || a.ob.open {
		return nil
	}
	conflict := *a.scPrompt
	feature, _ := shortcutOf(conflict.Mine.ID)
	translator := i18n.CurrentTranslator()
	return &shortcutPromptViewState{
		Keys: conflictKeysFor(translator, a.prefs.Features, conflict), OtherApp: familyAppName(conflict.Other),
		Function: familyFunctionFor(translator, conflict.Theirs), Link: conflict.Other.Links[registry.LinkShortcuts],
		FeaturePath: featurePath(feature),
	}
}

// openLink opens a URL with the system (another app's deep link).
func (a *app) openLink(context *shortcutPromptUIContext, link string) {
	if a.win == nil {
		context.Services().OpenURL(link)
		return
	}
	go mygo.Shell.OpenExternal(link)
}
