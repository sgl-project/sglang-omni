package main

// A feature page's Shortcuts section (feature_page.go): its rows of
// shortcuts with their triggers, choosing or typing a chord, and Transcribe's
// paste-the-last-result row. Keys another ActNow app uses too say so
// under the row, as they are chosen or typed; saving stays allowed.

import (
	"fmt"
	"slices"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/hotkey"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// keyRow is a shortcut's controls.
type keyRow struct {
	sel    string // the select's label
	custom bool   // "Custom…" chosen: the keys are typed
	text   string // the typed keys
	trig   int    // 0 tap, 1 double tap, 2 long press
	edited bool   // the typed keys are not saved yet
}

// keyPresets are the shortcuts the selects offer, in the grammar of
// internal/hotkey: the defaults first (the ActNow family's, right
// Option), then fn's (Auralink's defaults, for those without it). No
// more than ten: a longer list scrolls in the popup. Right Control is
// left out: Mac laptops' keyboards have none.
var keyPresets = []string{
	"RightOption", "RightOption+Shift", "Control+RightOption",
	"RightCommand", "RightShift",
	"Fn", "Fn+Shift", "Fn+Control",
	"Option+Space", "Control+Option+Space",
}

// triggers are the settings' triggers in the segmented control's order.
var triggers = []string{settings.TriggerTap, settings.TriggerDouble, settings.TriggerLong}

func triggerIndex(t string) int {
	if i := slices.Index(triggers, t); i >= 0 {
		return i
	}
	return 0
}

// ---- Shortcuts ----

// featureSync sets a feature's shortcut controls from the settings when
// they changed elsewhere (or the language did: the labels are
// translated).
func (a *app) featureSync(feature string) {
	u := a.fui(feature)
	seen := a.featureKeysKey(feature)
	if seen == u.seen {
		return
	}
	u.seen = seen
	hs := hotkey.HotkeysOf(a.prefs.Features, feature)
	u.rows = make([]keyRow, len(hs))
	for i, h := range hs {
		u.rows[i] = keyRow{sel: chordLabel(h.Keys), text: h.Keys, trig: triggerIndex(h.Trigger)}
	}
	pl := a.prefs.Features.PasteLast.Keys
	u.paste = keyRow{sel: chordLabel(pl), text: pl}
	if pl == "" {
		u.paste.sel = i18n.T("Off")
	}
}

func (a *app) featureKeysKey(feature string) string {
	f := a.prefs.Features
	return fmt.Sprint(hotkey.HotkeysOf(f, feature), f.PasteLast, i18n.Locale())
}

// chordLabel is the select's label of keys: the preset they are, or their
// own glyphs.
func chordLabel(keys string) string {
	if p := presetOf(keys); p != "" {
		return uitext.HotkeyText(p)
	}
	return uitext.HotkeyText(keys)
}

// presetOf is the preset keys are, "" for none.
func presetOf(keys string) string {
	c, err := hotkey.Parse(keys)
	if err != nil {
		return ""
	}
	for _, p := range keyPresets {
		if pc, _ := hotkey.Parse(p); pc.Equal(c) {
			return p
		}
	}
	return ""
}

// keyOptions are a select's options for a binding whose keys are cur:
// the presets, cur when it is none of them, then "Custom…" (and "Off"
// first with off). byLabel maps each label to its keys.
func keyOptions(cur string, off bool) (opts []string, byLabel map[string]string) {
	byLabel = map[string]string{}
	if off {
		opts = append(opts, i18n.T("Off"))
		byLabel[i18n.T("Off")] = ""
	}
	for _, p := range keyPresets {
		l := uitext.HotkeyText(p)
		opts = append(opts, l)
		byLabel[l] = p
	}
	if cur != "" && presetOf(cur) == "" {
		l := uitext.HotkeyText(cur)
		if _, dup := byLabel[l]; !dup {
			opts = append(opts, l)
			byLabel[l] = cur
		}
	}
	opts = append(opts, i18n.T("Custom…"))
	return opts, byLabel
}

// setHotkeys changes a feature's shortcuts with fn, on the settings' own
// copy.
func (a *app) setHotkeys(feature string, resync bool, fn func(hs []settings.Hotkey) []settings.Hotkey) {
	a.updateSettings(func(s *settings.Settings) {
		hs := &s.Features.Transcribe.Hotkeys
		switch feature {
		case hotkey.Translate:
			hs = &s.Features.Translate.Hotkeys
		case hotkey.Rewrite:
			hs = &s.Features.Rewrite.Hotkeys
		}
		*hs = fn(slices.Clone(*hs))
	})
	u := a.fui(feature)
	if resync {
		u.seen = ""
	} else {
		u.seen = a.featureKeysKey(feature) // the controls already show it
	}
}

// newHotkey is the shortcut "Add a shortcut" adds: the first preset no
// feature uses.
func newHotkey(f settings.Features) settings.Hotkey {
	for _, p := range keyPresets {
		pc, _ := hotkey.Parse(p)
		used := false
		for _, feat := range append(slices.Clone(hotkey.Features), hotkey.PasteLast) {
			for _, h := range hotkey.HotkeysOf(f, feat) {
				if hc, err := hotkey.Parse(h.Keys); err == nil && hc.Overlaps(pc) {
					used = true
				}
			}
		}
		if !used {
			return settings.Hotkey{Keys: p, Trigger: settings.TriggerTap}
		}
	}
	return settings.Hotkey{Keys: keyPresets[0], Trigger: settings.TriggerDouble}
}

func (a *app) featureShortcuts(c *ui.Context, feature string) {
	t := c.Theme()
	p := uitext.FeaturePrefix(feature)
	u := a.fui(feature)
	a.featureSync(feature)
	if sectionTitleAdd(c, p+"-key-add", i18n.T("Shortcuts"), i18n.T("Tap: press to start, press again to stop. Double tap: press twice quickly. Long press: record while you hold the keys. Escape cancels."), i18n.T("Add a shortcut")) {
		h := newHotkey(a.prefs.Features)
		a.setHotkeys(feature, true, func(hs []settings.Hotkey) []settings.Hotkey { return append(hs, h) })
	}
	if !a.hotkeys.Installed {
		notice(c, p+"-no-tap", noticeWarn,
			i18n.T("Without Accessibility only plain shortcuts work (⌃⌥⇧⌘ with a key); fn, a single modifier, double taps and long presses need it."),
			"", func() {
				if ui.Button(c.Key(p+"-open-perms"), i18n.T("Open Permissions")).Clicked() {
					a.navigate("/settings/permissions")
				}
			}, nil)
	}
	problems := uitext.HotkeyProblems(a.prefs.Features, feature)
	card(c.Key(p + "-keys")).Gap(px(t, sp2)).Children(func() {
		hs := hotkey.HotkeysOf(a.prefs.Features, feature)
		if len(hs) == 0 {
			if feature == hotkey.Transcribe {
				caption(c, i18n.T("No shortcut: start from the menu bar, or add one."))
			} else {
				caption(c, i18n.T("No shortcut: add one to use this feature."))
			}
		}
		for i := range hs {
			if i >= len(u.rows) {
				break
			}
			if i > 0 {
				ui.Box(c).Height(1).Background(tok(t).Line).Shrink(0)
			}
			a.keyRowView(c, feature, i, hs[i], problems[i])
		}
	})
	switch feature {
	case hotkey.Transcribe:
		a.trPasteRow(c)
	case hotkey.Rewrite:
		caption(c, i18n.T("While the answer card shows an answer, the shortcut records a follow-up that changes it (\"shorter\", \"in English\")."))
	}
}

// keyRowView is the i-th shortcut: its keys, its trigger, a mark when it
// needs Accessibility, Remove; the typed keys when custom; its problems.
func (a *app) keyRowView(c *ui.Context, feature string, i int, h settings.Hotkey, problems []string) {
	t := c.Theme()
	k := tok(t)
	r := &a.fui(feature).rows[i]
	n := i + 1
	ui.Column(c.Key(fmt.Sprintf("%s-key-%d", uitext.FeaturePrefix(feature), i))).Gap(px(t, 6)).Children(func() {
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
			ui.Row(c).Wrap().Gap(px(t, sp2)).AlignItems(ui.Center).Grow(1).MinWidth(0).Children(func() {
				opts, byLabel := keyOptions(h.Keys, false)
				if !slices.Contains(opts, r.sel) {
					r.sel = chordLabel(h.Keys)
				}
				if r.custom {
					r.sel = i18n.T("Custom…")
				}
				sel := ui.Select(c.Key("sel"), &r.sel, opts).Label(i18n.T("Keys of shortcut %d", n)).Width(px(t, 180)).Shrink(1).MinWidth(px(t, 120))
				if sel.Changed() {
					a.chooseKeys(feature, i, r, byLabel)
				}
				tips := []string{
					i18n.T("Press to start, press again to stop."),
					i18n.T("Press twice quickly to start; once more to stop."),
					i18n.T("Record while you hold the keys; release to stop."),
				}
				seg := segmentedSome(c.Key("trig"), &r.trig, []string{i18n.T("Tap"), i18n.T("Double tap"), i18n.T("Long press")}, nil, tips).
					Label(i18n.T("Trigger of shortcut %d", n))
				if seg.Changed() {
					trig := triggers[r.trig]
					a.setHotkeys(feature, false, func(hs []settings.Hotkey) []settings.Hotkey {
						if i < len(hs) {
							hs[i].Trigger = trig
						}
						return hs
					})
				}
			})
			if hotkey.NeedsTap(h) {
				col := k.Text3
				if !a.hotkeys.Installed {
					col = k.Warning
				}
				iconMark(c, icons.Accessibility, col, i18n.T("Works only while VoxtNew may use Accessibility."))
			}
			if iconButton(c.Key("remove"), icons.TrashBinMinimalistic, i18n.T("Remove shortcut %d", n)).Clicked() {
				a.setHotkeys(feature, true, func(hs []settings.Hotkey) []settings.Hotkey {
					if i < len(hs) {
						hs = slices.Delete(hs, i, i+1)
					}
					return hs
				})
			}
		})
		if r.custom {
			a.customKeys(c, r, i18n.T("Custom keys of shortcut %d", n), func(keys string) {
				a.setHotkeys(feature, false, func(hs []settings.Hotkey) []settings.Hotkey {
					if i < len(hs) {
						hs[i].Keys = keys
					}
					return hs
				})
			})
		}
		problemLines(c, problems)
		a.usedByView(c, recordedKeys(*r, h.Keys)) // settings_shortcuts_family.go: another app's keys
	})
}

// recordedKeys are the keys a shortcut's controls show: the typed ones
// while they parse, else saved.
func recordedKeys(r keyRow, saved string) string {
	if r.custom && r.edited {
		if ch, err := hotkey.Parse(r.text); err == nil {
			return ch.String()
		}
	}
	return saved
}

// chooseKeys applies a choice of a feature's i-th shortcut select.
func (a *app) chooseKeys(feature string, i int, r *keyRow, byLabel map[string]string) {
	if r.sel == i18n.T("Custom…") {
		r.custom = true
		if hs := hotkey.HotkeysOf(a.prefs.Features, feature); i < len(hs) {
			r.text = hs[i].Keys
		}
		return
	}
	keys, ok := byLabel[r.sel]
	if !ok {
		return
	}
	r.custom, r.text = false, keys
	a.setHotkeys(feature, false, func(hs []settings.Hotkey) []settings.Hotkey {
		if i < len(hs) {
			hs[i].Keys = keys
		}
		return hs
	})
}

// customKeys is the field for typed keys: they are saved (in canonical
// spelling) on Return or when the field loses the focus, if they parse;
// until then it says why not. Not on every key: on the way to
// "Control+Option+K" the field says "Control", a shortcut of its own.
func (a *app) customKeys(c *ui.Context, r *keyRow, label string, save func(keys string)) {
	t := c.Theme()
	in := ui.TextInput(c.Key("custom"), &r.text).Placeholder("Control+Option+Space").Label(label).Font("monospace").MaxWidth(px(t, 360))
	if in.Changed() {
		r.edited = true
	}
	ch, err := hotkey.Parse(r.text)
	if r.edited && err == nil && (in.Submitted() || !in.Focused()) {
		r.edited = false
		r.text = ch.String()
		save(r.text)
	}
	switch {
	case err != nil:
		warnLine(c, uitext.ParseErrorText(err))
	case r.edited:
		caption(c, i18n.T("Press Return to use %s.", uitext.HotkeyText(r.text)))
	default:
		caption(c, i18n.T("Modifiers first, then one key: Fn, RightCommand, Control+Option+Space, Fn+F5."))
	}
}

// problemLines are a binding's problems, in the warning colour.
func problemLines(c *ui.Context, problems []string) {
	for _, p := range problems {
		warnLine(c, p)
	}
}

// warnLine is a caption in the warning colour, with its icon.
func warnLine(c *ui.Context, text string) {
	t := c.Theme()
	k := tok(t)
	ui.Row(c).Gap(px(t, 4)).AlignItems(ui.Start).Children(func() {
		ui.Icon(c, icons.DangerTriangle.Linear).Size(px(t, 14), px(t, 14)).TextColor(k.Warning).Shrink(0).Margin(px(t, 1), 0, 0, 0)
		ui.Text(c, text).FontSize(px(t, fsCaption)).TextColor(k.Warning).Grow(1).MinWidth(0)
	})
}

// trPasteRow is "Paste the last result": its keys only (always a tap).
func (a *app) trPasteRow(c *ui.Context) {
	t := c.Theme()
	r := &a.fui(hotkey.Transcribe).paste
	cur := a.prefs.Features.PasteLast.Keys
	problems := uitext.HotkeyProblems(a.prefs.Features, hotkey.PasteLast)[0]
	ui.Column(c.Key("tr-paste")).Gap(px(t, 6)).Children(func() {
		settingRow(c, "tr-paste-row", i18n.T("Paste the last result"), i18n.T("Pastes the latest result again where you type, whichever feature made it."), func() {
			opts, byLabel := keyOptions(cur, true)
			if !slices.Contains(opts, r.sel) {
				r.sel = chordLabel(cur)
			}
			if r.custom {
				r.sel = i18n.T("Custom…")
			}
			if ui.Select(c.Key("tr-paste-sel"), &r.sel, opts).Label(i18n.T("Keys to paste the last result")).Width(px(t, 180)).Changed() {
				if r.sel == i18n.T("Custom…") {
					r.custom, r.text = true, cur
					return
				}
				keys, ok := byLabel[r.sel]
				if !ok {
					return
				}
				r.custom, r.text = false, keys
				a.setPasteLast(keys)
			}
		})
		if r.custom {
			a.customKeys(c, r, i18n.T("Custom keys to paste the last result"), a.setPasteLast)
		}
		problemLines(c, problems)
		a.usedByView(c, recordedKeys(*r, cur))
	})
}

func (a *app) setPasteLast(keys string) {
	a.updateSettings(func(s *settings.Settings) {
		s.Features.PasteLast = settings.Hotkey{Keys: keys, Trigger: settings.TriggerTap}
	})
	u := a.fui(hotkey.Transcribe)
	u.seen = a.featureKeysKey(hotkey.Transcribe)
}
