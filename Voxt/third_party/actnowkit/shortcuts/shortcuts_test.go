package shortcuts

import (
	"testing"
	"time"
)

func TestNormalize(t *testing.T) {
	cases := map[string]string{
		"CmdOrCtrl+Shift+Space":    "Shift+Command+Space",
		"Super+Ctrl+Alt+P":         "Control+Option+Command+P",
		"fn":                       "Fn",
		"globe+shift":              "Fn+Shift",
		"Fn+Control":               "Fn+Control",
		"rightoption":              "RightOption",
		"RightOption+Shift":        "RightOption+Shift",
		"ralt+ctrl":                "Control+RightOption",
		"LeftCmd+a":                "LeftCommand+A",
		"⌃⌘V":                      "Control+Command+V",
		"⇧⌃⌥⌘M":                    "Control+Option+Shift+Command+M",
		"Command++":                "Command++",
		"Option+RightOption":       "RightOption",
		"Ctrl+Return":              "Control+Enter",
		"control+option+command+r": "Control+Option+Command+R",
		"F5":                       "F5",
		"Shift+f12":                "Shift+F12",
	}
	for in, want := range cases {
		got, err := Normalize(in)
		if err != nil || got != want {
			t.Errorf("%q: %q %v", in, got, err)
		}
	}
	for _, bad := range []string{"", "Shift+Foo", "A+Shift", "LeftFn", "+"} {
		if got, err := Normalize(bad); err == nil && bad != "+" {
			t.Errorf("%q accepted as %q", bad, got)
		}
	}
	if NormalizeTrigger("long") != Hold || NormalizeTrigger("") != Press {
		t.Error("triggers")
	}
}

func TestConflicts(t *testing.T) {
	on := func(keys, trigger string) Shortcut { return Shortcut{Keys: keys, Trigger: trigger, Enabled: true} }
	yes := [][2]Shortcut{
		{on("Fn", Tap), on("fn", Hold)},
		{on("Option+Space", Press), on("RightOption+Space", Press)},
		{on("Cmd+Shift+Space", Press), on("Shift+Command+Space", Hold)},
		{on("RightOption", Tap), on("Option", Double)},
	}
	no := [][2]Shortcut{
		{on("LeftOption+Space", Press), on("RightOption+Space", Press)},
		{on("Fn", Tap), on("Fn+Shift", Tap)},
		{on("Command+Shift+Space", Press), on("Control+Shift+Space", Press)},
		{on("Fn", Tap), {Keys: "Fn", Trigger: Tap}},
	}
	for _, p := range yes {
		if !Conflicts(p[0], p[1]) || !Conflicts(p[1], p[0]) {
			t.Errorf("%q vs %q: no conflict", p[0].Keys, p[1].Keys)
		}
	}
	for _, p := range no {
		if Conflicts(p[0], p[1]) {
			t.Errorf("%q vs %q: conflict", p[0].Keys, p[1].Keys)
		}
	}
}

// The family defaults never collide, enabled or not (spec §10.2).
func TestDefaultsDisjoint(t *testing.T) {
	for i, a := range Defaults {
		if n, err := Normalize(a.Keys); err != nil || n != a.Keys {
			t.Errorf("%s/%s: %q is not canonical (%q, %v)", a.App, a.ID, a.Keys, n, err)
		}
		for _, b := range Defaults[i+1:] {
			x, y := a.Shortcut, b.Shortcut
			x.Enabled, y.Enabled = true, true
			if Conflicts(x, y) {
				t.Errorf("%s/%s %q collides with %s/%s %q", a.App, a.ID, a.Keys, b.App, b.ID, b.Keys)
			}
		}
	}
	exclusive := map[string]string{}
	for _, d := range Defaults {
		if d.Exclusive {
			if other, ok := exclusive[d.Keys]; ok {
				t.Errorf("%s exclusive in %s and %s", d.Keys, other, d.App)
			}
			exclusive[d.Keys] = d.App
		}
	}
	if len(DefaultsOf("voxtnew")) != 4 || DefaultsOf("voxtnew")[0].Keys != "RightOption" {
		t.Error("VoxtNew defaults")
	}
	if d := DefaultsOf("act"); len(d) != 1 || d[0].ID != "quickCollect" || d[0].Keys != "Control+Option+Command+S" || !d[0].Enabled || d[0].Mechanism != Carbon {
		t.Errorf("act defaults %+v", d)
	}
	if d := DefaultsOf("herenow"); len(d) != 1 || d[0].ID != "openPanel" || d[0].Keys != "Control+Option+Command+H" || !d[0].Enabled {
		t.Error("HereNow defaults")
	}
}

func TestFindConflicts(t *testing.T) {
	t0 := time.Date(2026, 10, 20, 8, 0, 0, 0, time.UTC)
	t1 := t0.Add(time.Hour)
	auralink := Holder{BundleID: "dev.hehehai.auralink", Name: "Auralink", Updated: t0, Shortcuts: DefaultsOf("auralink")}
	smith := Holder{BundleID: "dev.hehehai.audiosmith", Name: "AudioSmith", Updated: t0,
		Shortcuts: []Shortcut{{ID: "dictation", Keys: "Fn", Trigger: Hold, Enabled: true, EnabledAt: &t1}}}
	got := FindConflicts(smith, []Holder{auralink, smith})
	if len(got) != 1 || got[0].Theirs.ID != "transcribe" || !got[0].Prompt {
		t.Fatalf("%+v", got)
	}
	back := FindConflicts(auralink, []Holder{smith})
	if len(back) != 1 || back[0].Prompt {
		t.Fatalf("earlier app prompts: %+v", back)
	}
	if got[0].Key() == "" {
		t.Error("key")
	}
	hits := UsedBy("globe", smith.BundleID, []Holder{auralink})
	if len(hits) != 1 || hits[0].Shortcut.Title != "Transcribe" {
		t.Errorf("%+v", hits)
	}
}
