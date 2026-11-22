package main

import (
	"fmt"
	"slices"

	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

var keyPresets = []string{
	"RightOption", "RightOption+Shift", "Control+RightOption",
	"RightCommand", "RightShift",
	"Fn", "Fn+Shift", "Fn+Control",
	"Option+Space", "Control+Option+Space",
}

var triggers = []string{settings.TriggerTap, settings.TriggerDouble, settings.TriggerLong}

func triggerIndex(trigger string) int {
	if index := slices.Index(triggers, trigger); index >= 0 {
		return index
	}
	return 0
}

func (a *app) featureSync(feature string) {
	local := a.fui(feature)
	seen := a.featureKeysKey(feature)
	if seen == local.seen {
		return
	}
	local.seen = seen
	hotkeys := hotkey.HotkeysOf(a.prefs.Features, feature)
	local.rows = make([]keyRow, len(hotkeys))
	for index, hotkey := range hotkeys {
		local.rows[index] = keyRow{sel: chordLabel(hotkey.Keys), text: hotkey.Keys, trig: triggerIndex(hotkey.Trigger)}
	}
	pasteLast := a.prefs.Features.PasteLast.Keys
	local.paste = keyRow{sel: chordLabel(pasteLast), text: pasteLast}
	if pasteLast == "" {
		local.paste.sel = i18n.T("Off")
	}
	cancel := a.prefs.Features.Cancel.Keys
	local.cancel = keyRow{sel: chordLabel(cancel), text: cancel}
	if cancel == "" {
		local.cancel.sel = i18n.T("Off")
	}
}

func (a *app) featureKeysKey(feature string) string {
	features := a.prefs.Features
	return fmt.Sprint(hotkey.HotkeysOf(features, feature), features.PasteLast, features.Cancel, i18n.Locale())
}

func chordLabel(keys string) string {
	return chordLabelFor(i18n.CurrentTranslator(), keys)
}

func chordLabelFor(translator *i18n.Translator, keys string) string {
	if preset := presetOf(keys); preset != "" {
		return uitext.HotkeyTextFor(translator, preset)
	}
	return uitext.HotkeyTextFor(translator, keys)
}

func presetOf(keys string) string {
	chord, err := hotkey.Parse(keys)
	if err != nil {
		return ""
	}
	for _, preset := range keyPresets {
		if presetChord, _ := hotkey.Parse(preset); presetChord.Equal(chord) {
			return preset
		}
	}
	return ""
}

func (a *app) setHotkeys(feature string, resync bool, change func([]settings.Hotkey) []settings.Hotkey) {
	a.updateSettings(func(settings *settings.Settings) {
		hotkeys := &settings.Features.Transcribe.Hotkeys
		switch feature {
		case hotkey.Translate:
			hotkeys = &settings.Features.Translate.Hotkeys
		case hotkey.Rewrite:
			hotkeys = &settings.Features.Rewrite.Hotkeys
		}
		*hotkeys = change(slices.Clone(*hotkeys))
	})
	local := a.fui(feature)
	if resync {
		local.seen = ""
	} else {
		local.seen = a.featureKeysKey(feature)
	}
}

func newHotkey(features settings.Features) settings.Hotkey {
	for _, preset := range keyPresets {
		presetChord, _ := hotkey.Parse(preset)
		used := false
		for _, feature := range append(slices.Clone(hotkey.Features), hotkey.PasteLast, hotkey.Note) {
			for _, binding := range hotkey.HotkeysOf(features, feature) {
				if chord, err := hotkey.Parse(binding.Keys); err == nil && chord.Overlaps(presetChord) {
					used = true
				}
			}
		}
		if !used {
			return settings.Hotkey{Keys: preset, Trigger: settings.TriggerTap}
		}
	}
	return settings.Hotkey{Keys: keyPresets[0], Trigger: settings.TriggerDouble}
}

func (a *app) featureShortcutsModel(feature string) featureShortcutsViewState {
	a.featureSync(feature)
	return buildFeatureShortcutsViewState(i18n.CurrentTranslator(), featureShortcutsStateInput{
		Feature: feature, Settings: a.prefs, Local: a.fui(feature), HotkeysInstalled: a.hotkeys.Installed,
		HelperRecordsKeys: a.helperRecordsKeys(), UsedBy: a.featureUsedBy(feature), EndWord: a.featureEndWordModel(),
	})
}

func (a *app) featureUsedBy(feature string) map[string][]string {
	local := a.fui(feature)
	usedBy := map[string][]string{}
	remember := func(row keyRow, saved string) {
		keys := recordedKeys(row, saved)
		usedBy[keys] = a.usedByLines(keys)
	}
	for index, hotkey := range hotkey.HotkeysOf(a.prefs.Features, feature) {
		var row keyRow
		if index < len(local.rows) {
			row = local.rows[index]
		}
		remember(row, hotkey.Keys)
	}
	remember(local.paste, a.prefs.Features.PasteLast.Keys)
	remember(local.cancel, a.prefs.Features.Cancel.Keys)
	return usedBy
}

func (a *app) openFeaturePermissions() {
	a.navigate("/settings/permissions")
}

func (a *app) addFeatureShortcut(feature string) {
	hotkey := newHotkey(a.prefs.Features)
	a.setHotkeys(feature, true, func(hotkeys []settings.Hotkey) []settings.Hotkey {
		return append(hotkeys, hotkey)
	})
}

func (a *app) chooseFeatureShortcut(feature string, index int, keys string) {
	local := a.fui(feature)
	if index < len(local.rows) {
		local.rows[index].custom, local.rows[index].text = false, keys
	}
	a.setHotkeys(feature, false, func(hotkeys []settings.Hotkey) []settings.Hotkey {
		if index < len(hotkeys) {
			hotkeys[index].Keys = keys
		}
		return hotkeys
	})
}

func (a *app) setFeatureShortcutTrigger(feature string, index, trigger int) {
	if trigger < 0 || trigger >= len(triggers) {
		return
	}
	a.setHotkeys(feature, false, func(hotkeys []settings.Hotkey) []settings.Hotkey {
		if index < len(hotkeys) {
			hotkeys[index].Trigger = triggers[trigger]
		}
		return hotkeys
	})
}

func (a *app) removeFeatureShortcut(feature string, index int) {
	a.setHotkeys(feature, true, func(hotkeys []settings.Hotkey) []settings.Hotkey {
		if index < len(hotkeys) {
			hotkeys = slices.Delete(hotkeys, index, index+1)
		}
		return hotkeys
	})
}

func (a *app) saveFeatureShortcutKeys(feature string, index int, keys string) {
	a.setHotkeys(feature, false, func(hotkeys []settings.Hotkey) []settings.Hotkey {
		if index < len(hotkeys) {
			hotkeys[index].Keys = keys
		}
		return hotkeys
	})
}

func (a *app) chooseFeaturePasteLast(keys string) {
	local := &a.fui(hotkey.Transcribe).paste
	local.custom, local.text = false, keys
	a.setPasteLast(keys)
}

func (a *app) saveFeaturePasteLast(keys string) {
	a.setPasteLast(keys)
}

func (a *app) chooseFeatureCancelKeys(keys string) {
	local := &a.fui(hotkey.Transcribe).cancel
	local.custom, local.text = false, keys
	a.setCancelKeys(keys)
}

func (a *app) saveFeatureCancelKeys(keys string) {
	a.setCancelKeys(keys)
}

func (a *app) setPasteLast(keys string) {
	a.updateSettings(func(s *settings.Settings) {
		s.Features.PasteLast = settings.Hotkey{Keys: keys, Trigger: settings.TriggerTap}
	})
	local := a.fui(hotkey.Transcribe)
	local.seen = a.featureKeysKey(hotkey.Transcribe)
}

func (a *app) setCancelKeys(keys string) {
	a.updateSettings(func(s *settings.Settings) {
		s.Features.Cancel = settings.Hotkey{Keys: keys, Trigger: settings.TriggerTap}
	})
	local := a.fui(hotkey.Transcribe)
	local.seen = a.featureKeysKey(hotkey.Transcribe)
}

func recordedKeys(row keyRow, saved string) string {
	if row.custom && row.edited {
		if chord, err := hotkey.Parse(row.text); err == nil {
			return chord.String()
		}
	}
	return saved
}

func (a *app) markFeatureKeysHeard() {
	a.keysHeard = true
}
