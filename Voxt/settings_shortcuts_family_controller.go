package main

import (
	"slices"

	"github.com/egoist/mygo"
	"github.com/hehehai/actnowkit/shortcuts"
	"github.com/sgl-project/sglang-omni/Voxt/internal/actnow"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

type shortcutPromptServices interface {
	OpenURL(url string)
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

func (a *app) openShortcutPromptLink(services shortcutPromptServices, link string) {
	a.answerShortcutPrompt()
	a.openLink(services, link)
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

func (a *app) shortcutPromptModel() *shortcutPromptViewState {
	if a.scPrompt == nil || a.ob.open {
		return nil
	}
	state := buildShortcutPromptViewState(i18n.CurrentTranslator(), a.prefs.Features, *a.scPrompt)
	return &state
}

// openLink opens a URL with the system (another app's deep link).
func (a *app) openLink(services shortcutPromptServices, link string) {
	if a.win == nil {
		services.OpenURL(link)
		return
	}
	go mygo.Shell.OpenExternal(link)
}
