package shortcuts

// Default is a row of the family's default assignment (spec §10.2): new
// apps check it before choosing defaults; exclusive (event-swallowing)
// keys go to one app only — Fn belongs to Auralink.
type Default struct {
	App string // app id
	Shortcut
}

// Defaults is the family's default shortcut table. Disabled rows are
// reserved: they must not collide either, since users may turn them on.
var Defaults = []Default{
	{"auralink", Shortcut{ID: "transcribe", Title: "Transcribe", Keys: "Fn", Trigger: Tap, Enabled: true, Mechanism: EventTap, Exclusive: true}},
	{"auralink", Shortcut{ID: "translate", Title: "Translate", Keys: "Fn+Shift", Trigger: Tap, Enabled: true, Mechanism: EventTap}},
	{"auralink", Shortcut{ID: "rewrite", Title: "Rewrite", Keys: "Fn+Control", Trigger: Tap, Enabled: true, Mechanism: EventTap}},
	{"auralink", Shortcut{ID: "pasteLast", Title: "Paste last", Keys: "Control+Command+V", Trigger: Press, Enabled: true, Mechanism: EventTap, Exclusive: true}},

	// VoxtNew: the right Option family (spec §17 Q4, tentative).
	{"voxtnew", Shortcut{ID: "transcribe", Title: "Transcribe", Keys: "RightOption", Trigger: Tap, Enabled: true, Mechanism: EventTap}},
	{"voxtnew", Shortcut{ID: "translate", Title: "Translate", Keys: "RightOption+Shift", Trigger: Tap, Enabled: true, Mechanism: EventTap}},
	{"voxtnew", Shortcut{ID: "rewrite", Title: "Rewrite", Keys: "Control+RightOption", Trigger: Tap, Enabled: true, Mechanism: EventTap}},
	{"voxtnew", Shortcut{ID: "pasteLast", Title: "Paste last", Keys: "Control+Option+V", Trigger: Press, Enabled: true, Mechanism: EventTap, Exclusive: true}},

	{"audiosmith", Shortcut{ID: "dictation", Title: "Dictation", Keys: "Shift+Command+Space", Trigger: Press, Enabled: true, Mechanism: Carbon}},

	// Dicta: reserved, not registered.
	{"dicta", Shortcut{ID: "meeting", Title: "Start / stop meeting notes", Keys: "Control+Option+Command+R", Trigger: Press, Enabled: false, Mechanism: Carbon}},

	// Pik: Sip's defaults; only the picker is on.
	{"pik", Shortcut{ID: "openPicker", Title: "Open Picker", Keys: "Control+Option+Command+P", Trigger: Press, Enabled: true, Mechanism: Carbon}},
	{"pik", Shortcut{ID: "openColorPanel", Title: "Open Color Panel", Keys: "Control+Option+Command+C", Trigger: Press, Mechanism: Carbon}},
	{"pik", Shortcut{ID: "openStatusMenu", Title: "Open Status Menu", Keys: "Control+Option+Command+M", Trigger: Press, Mechanism: Carbon}},
	{"pik", Shortcut{ID: "openPikMenu", Title: "Open Pik Menu", Keys: "Control+Option+Shift+Command+M", Trigger: Press, Mechanism: Carbon}},
	{"pik", Shortcut{ID: "openContrastChecker", Title: "Open Contrast Checker", Keys: "Control+Option+Command+O", Trigger: Press, Mechanism: Carbon}},
	{"pik", Shortcut{ID: "toggleColorDock", Title: "Show/Hide Color Dock", Keys: "Control+Option+Command+D", Trigger: Press, Mechanism: Carbon}},
	{"pik", Shortcut{ID: "getPrimaryColor", Title: "Get Primary Color", Keys: "Control+Option+Command+F", Trigger: Press, Mechanism: Carbon}},
	{"pik", Shortcut{ID: "getSecondaryColor", Title: "Get Secondary Color", Keys: "Control+Option+Command+B", Trigger: Press, Mechanism: Carbon}},
	// Pik's own (Sip has none): Pick Area, off.
	{"pik", Shortcut{ID: "pickArea", Title: "Pick Area", Keys: "Control+Option+Command+A", Trigger: Press, Mechanism: Carbon}},

	// HereNow: opens the menu bar panel.
	{"herenow", Shortcut{ID: "openPanel", Title: "Open HereNow", Keys: "Control+Option+Command+H", Trigger: Press, Enabled: true, Mechanism: Carbon}},

	// Act: the quick collect panel, S for "save" (its screenshot
	// shortcuts have no default keys).
	{"act", Shortcut{ID: "quickCollect", Title: "Quick Collect", Keys: "Control+Option+Command+S", Trigger: Press, Enabled: true, Mechanism: Carbon}},
}

// DefaultsOf lists an app's default shortcuts.
func DefaultsOf(app string) []Shortcut {
	var out []Shortcut
	for _, d := range Defaults {
		if d.App == app {
			out = append(out, d.Shortcut)
		}
	}
	return out
}
