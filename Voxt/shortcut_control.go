package main

import (
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// shortcutControl makes recording the primary action, rather than a preset menu.
func (a *app) shortcutControl(c *ui.Context, r *keyRow, label string, off bool, save func(string)) {
	t := c.Theme()
	k := tok(t)
	ui.Column(c.Key("shortcut-control")).Width(px(t, 180)).Shrink(1).MinWidth(px(t, 120)).Gap(px(t, 4)).Children(func() {
		if r.custom {
			a.shortcutShown = true
			a.keysField(c, r, label, func(keys string) {
				save(keys)
				r.custom, r.typing, r.recFocused = false, false, false
				a.shortcutEditing = nil
			})
			return
		}
		shown := uitext.HotkeyText(r.text)
		if shown == "" {
			shown = i18n.T("None")
		}
		ui.Row(c).Gap(px(t, 4)).AlignItems(ui.Center).Children(func() {
			b := ui.ButtonBase(c.Key("record-shortcut")).Grow(1).MinWidth(0).Height(px(t, 30)).Padding(0, px(t, sp2)).Radius(px(t, radiusControl)).
				Border(1, k.Line).Background(k.Subtle).Label(label).Tooltip(i18n.T("Click, then press the keys"))
			if b.Hovered() {
				b.Background(k.Hover)
			}
			b.Children(func() { ui.Text(c, shown).Font("monospace").TextColor(k.Text).SingleLine().Ellipsis("…").MinWidth(0) })
			if b.Clicked() {
				a.closeShortcutEditor()
				r.before = r.text
				a.shortcutEditing = r
				a.shortcutShown = true
				r.custom, r.typing, r.edited, r.unknown, r.recFocused = true, false, false, false, false
			}
			if off && r.text != "" && iconButton(c.Key("off"), icons.CloseCircle, i18n.T("Clear shortcut")).Clicked() {
				r.text = ""
				save("")
			}
		})
	})
}

func (a *app) shortcutKeyRow(feature string, i int) *keyRow {
	if feature == hotkey.Note {
		cur := a.prefs.Features.Note.Keys
		if seen := cur + "\x00" + i18n.Locale(); seen != a.nv.keysFor {
			a.nv.keysFor = seen
			a.nv.keys = keyRow{text: cur}
		}
		return &a.nv.keys
	}
	owner := feature
	if feature == hotkey.PasteLast || feature == hotkey.CancelID {
		owner = hotkey.Transcribe
	}
	a.featureSync(owner)
	u := a.fui(owner)
	switch feature {
	case hotkey.PasteLast:
		return &u.paste
	case hotkey.CancelID:
		return &u.cancel
	default:
		return &u.rows[i]
	}
}

func (a *app) setShortcutKeys(feature string, i int, keys string) {
	switch feature {
	case hotkey.Note:
		a.setNoteKeys(keys)
	case hotkey.PasteLast:
		a.setPasteLast(keys)
	case hotkey.CancelID:
		a.setCancelKeys(keys)
	default:
		a.setHotkeys(feature, false, func(hs []settings.Hotkey) []settings.Hotkey {
			if i < len(hs) {
				hs[i].Keys = keys
			} else if i == len(hs) {
				hs = append(hs, settings.Hotkey{Keys: keys, Trigger: settings.TriggerTap})
			}
			return hs
		})
	}
}

// A hidden editor must release capture instead of resuming when its page reopens.
func (a *app) closeShortcutEditor() {
	if r := a.shortcutEditing; r != nil {
		if r.custom {
			r.text = r.before
		}
		r.custom, r.typing, r.recFocused = false, false, false
		a.shortcutEditing = nil
	}
	a.cancelHotkeyRecord()
}
