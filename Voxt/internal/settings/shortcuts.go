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
)

// Presets are the shortcut presets, in the order the settings offer them.
var Presets = []string{PresetOption, PresetFn}

// presetKeys are a preset's keys of Transcribe, Translate, Rewrite and
// Paste the last result (presetSlots' order), each tapped.
var presetKeys = map[string][4]string{
	PresetOption: {DefaultTranscribe, DefaultTranslate, DefaultRewrite, DefaultPasteLast},
	PresetFn:     {"Fn", "Fn+Shift", "Fn+Control", "Control+Command+V"},
}

// presetSlots are the shortcuts a preset sets: each feature's, and
// PasteLast's.
func presetSlots(f Features) [4][]Hotkey {
	return [4][]Hotkey{f.Transcribe.Hotkeys, f.Translate.Hotkeys, f.Rewrite.Hotkeys, {f.PasteLast}}
}

// presetAt is the preset slot i's shortcuts are, "" when they are the
// user's: other keys, another trigger, none or several.
func presetAt(i int, hs []Hotkey) string {
	if len(hs) != 1 || hs[0].Trigger != "" && hs[0].Trigger != TriggerTap {
		return ""
	}
	for _, p := range Presets {
		if presetKeys[p][i] == hs[0].Keys {
			return p
		}
	}
	return ""
}

// PresetOf is the preset f's shortcuts follow: the one of every shortcut
// still on a preset's keys; "" (custom) when the user changed them all, or
// they mix presets.
func PresetOf(f Features) string {
	got := ""
	for i, hs := range presetSlots(f) {
		switch p := presetAt(i, hs); {
		case p == "":
		case got != "" && p != got:
			return ""
		default:
			got = p
		}
	}
	return got
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
	changed := false
	for i, hs := range slots {
		if presetAt(i, hs) == "" || hs[0].Keys == keys[i] {
			continue
		}
		slots[i], changed = []Hotkey{{Keys: keys[i], Trigger: TriggerTap}}, true
	}
	f.Transcribe.Hotkeys, f.Translate.Hotkeys, f.Rewrite.Hotkeys, f.PasteLast = slots[0], slots[1], slots[2], slots[3][0]
	return changed
}
