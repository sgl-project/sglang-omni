package main

// Settings → Shortcuts (/settings/shortcuts): every global shortcut
// in one table (Transcribe's, Translate's and Rewrite's, Paste the
// last result, Cancel transcription and Notes'), with their trigger and what
// hotkey.Check finds wrong:
// clashes, keys macOS uses, keys that need Accessibility while the event
// tap is missing, keys another ActNow app uses too. A row opens its
// feature's page, where it changes. Above it, the preset sets several at
// once (internal/settings' Presets).

import (
	"slices"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// featurePath is the page a shortcut changes on.
func featurePath(feature string) string {
	switch feature {
	case hotkey.Translate:
		return "/translate"
	case hotkey.Rewrite:
		return "/rewrite"
	case hotkey.Note:
		return "/notes"
	}
	return "/transcribe" // Transcribe, Paste the last result, Cancel transcription
}

// shortcutsSettings is the Shortcuts section of the settings dialog.
func (a *app) shortcutsSettings(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	caption(c, i18n.T("The shortcuts work in every app. Click one to change it on its feature's page."))
	if !a.hotkeys.Installed {
		notice(c, "sc-no-tap", noticeWarn,
			i18n.T("Without Accessibility only plain shortcuts work (⌃⌥⇧⌘ with a key); fn, a single modifier, double taps and long presses need it."),
			"", func() {
				if ui.Button(c.Key("sc-open-perms"), i18n.T("Open Permissions")).Clicked() {
					a.navigate("/settings/permissions")
				}
			}, nil)
	}
	a.shortcutPresetRow(c)
	f := a.prefs.Features
	keysW := px(t, 96)
	card(c.Key("sc-table")).Gap(0).Padding(px(t, sp2)).Children(func() {
		// The head.
		ui.Row(c).Gap(px(t, sp2)).Padding(px(t, sp1), px(t, sp2)).AlignItems(ui.Center).Children(func() {
			head := func(s string) ui.Element {
				return ui.Text(c, s).FontSize(px(t, fsMicro)).FontWeight(600).TextColor(k.Text3).SingleLine().Ellipsis("…").MinWidth(0)
			}
			head(i18n.T("Feature")).Grow(1)
			head(i18n.T("Shortcut")).Width(keysW).Shrink(0)
			ui.Box(c).Width(px(t, shortcutMarksW+14+sp2)).Shrink(0) // the marks and the chevron
		})
		for _, feat := range append(append([]string{}, hotkey.Features...), hotkey.PasteLast, hotkey.CancelID, hotkey.Note) {
			hs := hotkey.HotkeysOf(f, feat)
			problems := uitext.HotkeyProblems(f, feat) // feature_keys_text.go
			if len(hs) == 0 {
				a.shortcutRow(c, feat, 0, nil, nil, keysW)
				continue
			}
			for i := range hs {
				// The other ActNow apps on the same keys (settings_shortcuts_family.go).
				lines := append(slices.Clone(problems[i]), a.conflictLines(feat, i)...)
				a.shortcutRow(c, feat, i, &hs[i], lines, keysW)
			}
		}
	})
}

// shortcutMarksW is the room of a row's marks (Accessibility, a problem).
const shortcutMarksW = 16 + sp2 + 16

// shortcutRow is a row of the table: feature's i-th shortcut h (nil: it
// has none), its keys over its trigger, its problems under it; a click
// opens the feature's page.
func (a *app) shortcutRow(c *ui.Context, feature string, i int, h *settings.Hotkey, problems []string, keysW float32) {
	t := c.Theme()
	k := tok(t)
	title := uitext.FeatureTitle(feature)
	keys := i18n.T("None")
	if h != nil {
		keys = uitext.HotkeyText(h.Keys)
	}
	b := ui.ButtonBase(c.Key("sc-"+feature+"-"+itoa(i))).Padding(px(t, 6), px(t, sp2)).Radius(px(t, radiusControl)).
		AlignItems(ui.Stretch).Label(title + " " + keys).Tooltip(i18n.T("Change on the %s page", pageTitle(featurePath(feature))))
	if b.Hovered() {
		b.Background(k.Hover)
	}
	b.Children(func() {
		ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 4)).Children(func() {
			ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
				ic := kindIcon(featureKind(feature))
				if feature == hotkey.PasteLast {
					ic = icons.Copy
				}
				if feature == hotkey.CancelID {
					ic = icons.CloseCircle
				}
				if feature == hotkey.Note {
					ic = notesIcon
				}
				ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Grow(1).Basis(0).MinWidth(0).Children(func() {
					if i == 0 {
						ui.Icon(c, ic.Linear).Size(px(t, 16), px(t, 16)).TextColor(k.Text2).Shrink(0)
						truncText(c, title)
					}
				})
				ui.Column(c).Width(keysW).Shrink(0).Gap(px(t, 2)).AlignItems(ui.Start).Children(func() {
					if h == nil {
						ui.Text(c, keys).FontSize(px(t, fsCaption)).TextColor(k.Text3).SingleLine()
						return
					}
					kbd(c, keys).MinWidth(0).Ellipsis("…") // shell_commands.go
					trig := uitext.TriggerText(h.Trigger)
					if feature == hotkey.PasteLast || feature == hotkey.CancelID || feature == hotkey.Note {
						trig = uitext.TriggerText(settings.TriggerTap) // always a tap
					}
					ui.Text(c, trig).FontSize(px(t, fsMicro)).TextColor(k.Text2).SingleLine().Ellipsis("…").MinWidth(0)
				})
				ui.Row(c).Width(px(t, shortcutMarksW)).Shrink(0).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
					if h != nil && (feature == hotkey.CancelID || hotkey.NeedsTap(*h)) { // cancel works only through the tap, while active
						col := k.Text3
						if !a.hotkeys.Installed {
							col = k.Warning
						}
						iconMark(c, icons.Accessibility, col, i18n.T("Works only while Voxt may use Accessibility."))
					}
					if len(problems) > 0 {
						iconMark(c, icons.DangerTriangle, k.Warning, strings.Join(problems, "\n"))
					}
				})
				ui.Icon(c, icons.AltArrowRight.Linear).Size(px(t, 14), px(t, 14)).TextColor(k.Text3).Shrink(0)
			})
			if len(problems) > 0 {
				ui.Column(c).Padding(0, 0, 0, px(t, 16+sp2)).Gap(px(t, 2)).Children(func() {
					problemLines(c, problems) // feature_keys.go
				})
			}
		})
	})
	if b.Clicked() {
		a.navigate(featurePath(feature))
	}
}

// presetTitle is a shortcut preset's name ("": custom).
func presetTitle(p string) string {
	switch p {
	case settings.PresetOption:
		return i18n.T("Right ⌥ (default)")
	case settings.PresetFn:
		return i18n.T("fn (Auralink's keys)")
	}
	return i18n.T("Custom")
}

// shortcutPresetRow is the preset: the one the shortcuts follow (Custom
// when none), and under fn's what else reacts to fn.
func (a *app) shortcutPresetRow(c *ui.Context) {
	t := c.Theme()
	cur := settings.PresetOf(a.prefs.Features)
	settingRow(c, "sc-preset", i18n.T("Preset"), i18n.T("Sets Transcribe, Translate, Rewrite and Paste the last result at once. A shortcut you changed yourself stays as it is."), func() {
		var opts []string
		for _, p := range settings.Presets {
			opts = append(opts, presetTitle(p))
		}
		if cur == "" {
			opts = append(opts, presetTitle(cur))
		}
		sel := presetTitle(cur)
		if ui.Select(c.Key("sc-preset-sel"), &sel, opts).Label(i18n.T("Shortcut preset")).Width(px(t, 220)).Changed() {
			for _, p := range settings.Presets {
				if presetTitle(p) == sel {
					a.applyShortcutPreset(p)
				}
			}
		}
	})
	if cur == settings.PresetFn {
		warnLine(c, i18n.T("Auralink uses fn too: while it runs, both apps react to these keys."))
		warnLine(c, i18n.T("macOS may react to fn too (the Globe key). If it gets in the way, set it to Do Nothing in System Settings → Keyboard."))
	}
}

// applyShortcutPreset gives the shortcuts the user did not change preset
// p's keys, or says that there are none.
func (a *app) applyShortcutPreset(p string) {
	if f := a.prefs.Features; !settings.ApplyPreset(&f, p) {
		a.toast(i18n.T("You changed every shortcut yourself: the preset keeps them."))
		return
	}
	a.updateSettings(func(s *settings.Settings) { settings.ApplyPreset(&s.Features, p) })
}
