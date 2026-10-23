package main

// The shortcuts other ActNow apps also use (family spec §10.4): a line
// "Also used by Auralink (Transcribe)" under each of VoxtNew's shortcuts
// an installed family app uses too (Settings → Shortcuts, the setup
// guide), the same under the recorder of a feature's page while it shows
// keys another app uses (saving stays allowed), and a prompt, once per
// pair, for the conflicts VoxtNew is the later app of (the one that got
// its keys last shows it, the other only warns).
//
// The conflicts are a.famConflicts (family.go, after each discovery);
// the recorder asks the last discovery (actnow.Family.UsedBy, memory
// only). Keys show as VoxtNew writes them (uitext.HotkeyText), not in
// the family's notation.

import (
	"slices"
	"strconv"
	"strings"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/hehehai/actnowkit/registry"
	"github.com/hehehai/actnowkit/shortcuts"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/actnow"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/hotkey"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// shortcutID is the family's id of feature's i-th shortcut, as
// actnow.Shortcuts declares it ("transcribe", "transcribe.2",
// "pasteLast").
func shortcutID(feature string, i int) string {
	if feature == hotkey.PasteLast {
		return "pasteLast"
	}
	if i > 0 {
		return feature + "." + itoa(i+1)
	}
	return feature
}

// shortcutOf is the feature and index of a declared shortcut id.
func shortcutOf(id string) (feature string, i int) {
	if id == "pasteLast" {
		return hotkey.PasteLast, 0
	}
	feature, n, ok := strings.Cut(id, ".")
	if ok {
		if x, err := strconv.Atoi(n); err == nil && x > 1 {
			i = x - 1
		}
	}
	return feature, i
}

// familyAppName is another app's name for the UI.
func familyAppName(h shortcuts.Holder) string {
	if h.Name != "" {
		return h.Name
	}
	return actnow.AppName(h.BundleID)
}

// familyFunction is the name of another app's function, translated when
// it is one of the family's known ones (its manifest has it in English).
func familyFunction(s shortcuts.Shortcut) string {
	switch s.ID {
	case "transcribe":
		return i18n.T("Transcribe")
	case "translate":
		return i18n.T("Translate")
	case "rewrite":
		return i18n.T("Rewrite")
	case "pasteLast":
		return i18n.T("Paste last")
	case "dictation":
		return i18n.T("Dictation")
	}
	if s.Title != "" {
		return s.Title
	}
	return s.ID
}

// alsoUsedBy is the line of another app's shortcut on the same keys.
func alsoUsedBy(app shortcuts.Holder, theirs shortcuts.Shortcut) string {
	return i18n.T("Also used by %s (%s)", familyAppName(app), familyFunction(theirs))
}

// conflictLines are the lines of the other apps that use feature's i-th
// shortcut too (a.famConflicts).
func (a *app) conflictLines(feature string, i int) []string {
	id := shortcutID(feature, i)
	var out []string
	for _, c := range a.famConflicts {
		if c.Mine.ID != id {
			continue
		}
		if s := alsoUsedBy(c.Other, c.Theirs); !slices.Contains(out, s) {
			out = append(out, s)
		}
	}
	return out
}

// usedByLines are the lines of the other apps whose shortcuts are on keys
// (VoxtNew's notation): the recorder's, from the last discovery (memory
// only).
func (a *app) usedByLines(keys string) []string {
	if a.fam == nil || keys == "" {
		return nil
	}
	norm, err := shortcuts.Normalize(keys)
	if err != nil {
		return nil
	}
	var out []string
	for _, h := range a.fam.UsedBy(norm) {
		if s := alsoUsedBy(h.App, h.Shortcut); !slices.Contains(out, s) {
			out = append(out, s)
		}
	}
	return out
}

// usedByView is the recorder's warning: the lines of usedByLines.
func (a *app) usedByView(c *ui.Context, keys string) {
	for _, s := range a.usedByLines(keys) {
		warnLine(c, s)
	}
}

// ---- The prompt ----

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
	for _, c := range a.famConflicts {
		if c.Prompt && !slices.Contains(a.prefs.Family.Prompted, c.Key()) {
			a.scPrompt = &c
			return
		}
	}
}

// conflictNow says whether c is still one of the conflicts.
func (a *app) conflictNow(c shortcuts.Conflict) bool {
	return slices.ContainsFunc(a.famConflicts, func(x shortcuts.Conflict) bool { return x.Key() == c.Key() })
}

// answerPrompt records the prompt's conflict as prompted, whatever the
// answer, and shows the next one.
func (a *app) answerPrompt() {
	c := a.scPrompt
	a.scPrompt = nil
	if c == nil {
		return
	}
	key := c.Key()
	if !slices.Contains(a.prefs.Family.Prompted, key) {
		a.updateSettings(func(s *settings.Settings) { s.Family.Prompted = append(slices.Clone(s.Family.Prompted), key) })
	}
	a.shortcutPrompt()
}

// conflictsSeen marks as prompted the conflicts of the keys the user just
// gave a shortcut (old: the settings before): the recorder said which app
// uses them, the prompt would only repeat it.
func (a *app) conflictsSeen(old settings.Settings) {
	before := map[string]string{}
	for _, s := range actnow.Shortcuts(old) {
		before[s.ID] = s.Keys
	}
	var seen []string
	for _, c := range a.famConflicts {
		if k := c.Key(); c.Prompt && before[c.Mine.ID] != c.Mine.Keys && !slices.Contains(a.prefs.Family.Prompted, k) && !slices.Contains(seen, k) {
			seen = append(seen, k)
		}
	}
	if len(seen) > 0 {
		a.updateSettings(func(s *settings.Settings) { s.Family.Prompted = append(slices.Clone(s.Family.Prompted), seen...) })
	}
	a.shortcutPrompt()
}

// conflictKeys are the keys of VoxtNew's shortcut of conflict c, as
// VoxtNew writes them.
func (a *app) conflictKeys(c shortcuts.Conflict) string {
	feature, i := shortcutOf(c.Mine.ID)
	if feature == hotkey.PasteLast {
		return uitext.HotkeyText(a.prefs.Features.PasteLast.Keys)
	}
	if hs := hotkey.HotkeysOf(a.prefs.Features, feature); i < len(hs) {
		return uitext.HotkeyText(hs[i].Keys)
	}
	return uitext.HotkeyText(c.Mine.Keys)
}

// shortcutPromptDialog is the prompt: the keys, the other app and its
// function; Change shortcut (the feature's page), Open <app> settings
// (its manifest's link, when it has one), Ignore.
func (a *app) shortcutPromptDialog(c *ui.Context) {
	if a.scPrompt == nil || a.ob.open {
		return
	}
	t := c.Theme()
	cf := *a.scPrompt
	other := familyAppName(cf.Other)
	link := cf.Other.Links[registry.LinkShortcuts]
	open := true
	ui.Modal(c.Key("sc-prompt"), &open, func() {
		ui.Text(c, i18n.T("Shortcut used by another app")).Bold()
		ui.Text(c, i18n.T("%s is also the %s shortcut in %s. Pressing it will trigger both apps.",
			a.conflictKeys(cf), familyFunction(cf.Theirs), other)).Width(px(t, 380))
		ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
			if ghostButton(c.Key("sc-prompt-ignore"), i18n.T("Ignore")).Clicked() {
				a.answerPrompt()
			}
			if link != "" && ui.Button(c.Key("sc-prompt-other"), i18n.T("Open %s settings", other)).Clicked() {
				a.answerPrompt()
				a.openLink(c, link)
			}
			if primaryButton(c.Key("sc-prompt-change"), i18n.T("Change shortcut")).AutoFocus().Clicked() {
				feature, _ := shortcutOf(cf.Mine.ID)
				a.answerPrompt()
				a.navigate(featurePath(feature))
			}
		})
	})
	if !open && a.scPrompt != nil { // Escape: as Ignore
		a.answerPrompt()
	}
}

// openLink opens a URL with the system (another app's deep link).
func (a *app) openLink(c *ui.Context, u string) {
	if a.win == nil {
		c.Services().OpenURL(u)
		return
	}
	go mygo.Shell.OpenExternal(u)
}
