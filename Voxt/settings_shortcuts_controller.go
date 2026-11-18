package main

import (
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
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

func (a *app) shortcutsSettingsModel() settingsShortcutsViewState {
	var editing map[string]string
	for key, row := range a.settingsForm.settingsShortcutsLocalUI.rows {
		if row.custom {
			if editing == nil {
				editing = map[string]string{}
			}
			editing[key] = row.text
		}
	}
	return buildSettingsShortcutsViewState(i18n.CurrentTranslator(), settingsShortcutsStateInput{
		Features: a.prefs.Features, FamilyConflicts: a.famConflicts, AccessibilityInstalled: a.hotkeys.Installed,
		HelperRecordsKeys: a.helperRecordsKeys(), Editing: editing,
	})
}

// setShortcutKeys saves the keys recorded for feature's index-th shortcut
// in Settings → Shortcuts; a feature without shortcuts gets its first.
func (a *app) setShortcutKeys(feature string, index int, keys string) {
	switch feature {
	case hotkey.Note:
		a.setNoteKeys(keys)
	case hotkey.PasteLast:
		a.setPasteLast(keys)
	case hotkey.CancelID:
		a.setCancelKeys(keys)
	default:
		a.setHotkeys(feature, false, func(hotkeys []settings.Hotkey) []settings.Hotkey {
			if index < len(hotkeys) {
				hotkeys[index].Keys = keys
			} else if index == len(hotkeys) && keys != "" {
				hotkeys = append(hotkeys, settings.Hotkey{Keys: keys, Trigger: settings.TriggerTap})
			}
			return hotkeys
		})
	}
}

func (a *app) openShortcutPermissions() {
	a.navigate("/settings/permissions")
}

func (a *app) openShortcutFeature(path string) {
	a.navigate(path)
}

func (a *app) setStrictShortcuts(on bool) {
	a.updateSettings(func(current *settings.Settings) { current.Features.StrictShortcuts = on })
}

// applyShortcutPreset gives the shortcuts the user did not change preset's
// keys, or says that there are none.
func (a *app) applyShortcutPreset(preset string) {
	if features := a.prefs.Features; !settings.ApplyPreset(&features, preset) {
		a.toast(i18n.T("You changed every shortcut yourself, so the preset leaves them as they are."))
		return
	}
	a.updateSettings(func(current *settings.Settings) { settings.ApplyPreset(&current.Features, preset) })
}
