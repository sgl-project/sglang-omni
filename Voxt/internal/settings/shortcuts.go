package settings

// Shortcut presets (Voxt 1.x's hotkeyPreset): one choice gives
// Transcribe, Translate, Rewrite and Paste the last result a set of keys
// at once. Nothing stores the choice: the shortcuts themselves say which
// preset they follow (PresetOf).
const (
	// PresetOption is the ActNow family's set, the defaults: right Option.
	PresetOption = "option"
	// PresetFn is fn's set: Auralink's defaults, Voxt 1.x's fn combo
	// and settings version 3's defaults. Auralink reacts to the same keys.
	PresetFn = "fn"
	// PresetCommand is Voxt 1.x's commandCombo without meeting and note.
	PresetCommand = "command"
	// PresetMouseMiddle is Voxt 1.x's mouseMiddleFnShift: middle click
	// transcribe, fn⇧ translate, double middle click rewrite.
	PresetMouseMiddle = "mouseMiddle"
)

// Presets are the shortcut presets, in the order the settings offer them.
var Presets = []string{PresetOption, PresetFn, PresetCommand, PresetMouseMiddle}

// presetKeys are a preset's keys of Transcribe, Translate, Rewrite and
// Paste the last result (presetSlots' order).
var presetKeys = map[string][4]string{
	PresetOption:      {DefaultTranscribe, DefaultTranslate, DefaultRewrite, DefaultPasteLast},
	PresetFn:          {"Fn", "Fn+Shift", "Fn+Control", "Control+Command+V"},
	PresetCommand:     {"RightCommand", "RightCommand+Shift", "RightCommand+Option", "Control+Command+V"},
	PresetMouseMiddle: {"Mouse3", "Fn+Shift", "Mouse3", "Control+Command+V"},
}

// presetTriggers are non-tap triggers for a preset slot; missing entries
// use TriggerTap.
var presetTriggers = map[string][4]string{
	PresetMouseMiddle: {TriggerTap, TriggerTap, TriggerDouble, TriggerTap},
}

func presetTrigger(p string, i int) string {
	if trs, ok := presetTriggers[p]; ok && trs[i] != "" {
		return trs[i]
	}
	return TriggerTap
}

func hotkeyTrigger(h Hotkey) string {
	if h.Trigger == "" {
		return TriggerTap
	}
	return h.Trigger
}

// presetSlots are the shortcuts a preset sets: each feature's, and
// PasteLast's.
func presetSlots(f Features) [4][]Hotkey {
	return [4][]Hotkey{f.Transcribe.Hotkeys, f.Translate.Hotkeys, f.Rewrite.Hotkeys, {f.PasteLast}}
}

// presetAt is the preset slot i's shortcuts are, "" when they are the
// user's: other keys, another trigger, none or several.
func presetAt(i int, hs []Hotkey) string {
	if len(hs) != 1 {
		return ""
	}
	tr := hotkeyTrigger(hs[0])
	var match string
	for _, p := range Presets {
		if presetKeys[p][i] == hs[0].Keys && presetTrigger(p, i) == tr {
			if match != "" {
				return "" // paste last is shared between fn and command
			}
			match = p
		}
	}
	return match
}

// PresetOf is the preset f's shortcuts follow: the one of every shortcut
// still on a preset's keys; "" (custom) when the user changed them all, or
// they mix presets.
func PresetOf(f Features) string {
	slots := presetSlots(f)
	for _, p := range Presets {
		match := true
		for i, hs := range slots {
			if len(hs) != 1 {
				match = false
				break
			}
			if hs[0].Keys != presetKeys[p][i] || hotkeyTrigger(hs[0]) != presetTrigger(p, i) {
				match = false
				break
			}
		}
		if match {
			return p
		}
	}
	return ""
}

func slotMatchesPreset(i int, hs []Hotkey, preset string) bool {
	return len(hs) == 1 &&
		hs[0].Keys == presetKeys[preset][i] &&
		hotkeyTrigger(hs[0]) == presetTrigger(preset, i)
}

// ApplyPreset gives preset's keys, tapped, to each of f's shortcuts still
// on a preset's keys; a shortcut the user changed, added or removed
// stays. It reports whether a shortcut changed. It replaces the slices
// it changes, so that a copy of f keeps the old ones.
func ApplyPreset(f *Features, preset string) bool {
	keys, ok := presetKeys[preset]
	if !ok {
		return false
	}
	slots := presetSlots(*f)
	on := PresetOf(*f)
	changed := false
	for i, hs := range slots {
		at := presetAt(i, hs)
		stillOn := on != "" && slotMatchesPreset(i, hs, on)
		if len(hs) != 1 {
			continue
		}
		switch {
		case on != "":
			if at != "" && at != on {
				continue // mixed presets
			}
			if at == "" && !stillOn {
				continue // user's keys or another preset
			}
		default:
			if at == "" {
				continue
			}
			if at != preset && at != PresetOption && !(preset == PresetOption && at == PresetFn) {
				continue // mixed: leave other presets' keys, still refresh option leftovers
			}
		}
		if slotMatchesPreset(i, hs, preset) {
			continue
		}
		slots[i], changed = []Hotkey{{Keys: keys[i], Trigger: presetTrigger(preset, i)}}, true
	}
	f.Transcribe.Hotkeys, f.Translate.Hotkeys, f.Rewrite.Hotkeys, f.PasteLast = slots[0], slots[1], slots[2], slots[3][0]
	return changed
}
