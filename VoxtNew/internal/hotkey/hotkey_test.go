package hotkey

import (
	"errors"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

func TestParse(t *testing.T) {
	for _, tc := range []struct {
		in, canon string
		code      int
		acc       string
		err       error
	}{
		{in: "Fn", canon: "Fn"},
		{in: "fn+shift", canon: "Fn+Shift"},
		{in: "Shift+Fn", canon: "Fn+Shift"},
		{in: "Fn+Control", canon: "Fn+Control"},
		{in: "RightCommand", canon: "RightCommand"},
		{in: "Right Command", canon: "RightCommand"},
		{in: "LeftShift+RightOption", canon: "RightOption+LeftShift"},
		{in: "Control+Command+V", canon: "Control+Command+V", code: 9, acc: "Control+Command+V"},
		{in: "cmd+ctrl+v", canon: "Control+Command+V", code: 9, acc: "Control+Command+V"},
		{in: "Option+Space", canon: "Option+Space", code: 49, acc: "Option+Space"},
		{in: "Command+Shift+k", canon: "Shift+Command+K", code: 40, acc: "Shift+Command+K"},
		{in: "Control+Option+Return", canon: "Control+Option+Return", code: 36, acc: "Control+Option+Enter"},
		{in: "F5", canon: "F5", code: 96},
		{in: "Fn+A", canon: "Fn+A", code: 0},
		{in: "RightCommand+Up", canon: "RightCommand+Up", code: 126},
		{in: "", err: ErrEmpty},
		{in: "Hyper", err: ErrUnknown},
		{in: "Control+A+B", err: ErrTwoKeys},
		{in: "RightFn", err: ErrFnSide},
		{in: "Control++", err: ErrUnknown},
	} {
		c, err := Parse(tc.in)
		if tc.err != nil {
			if !errors.Is(err, tc.err) {
				t.Errorf("Parse(%q) = %v, want %v", tc.in, err, tc.err)
			}
			continue
		}
		if err != nil {
			t.Errorf("Parse(%q): %v", tc.in, err)
			continue
		}
		if c.String() != tc.canon {
			t.Errorf("Parse(%q) = %q, want %q", tc.in, c.String(), tc.canon)
		}
		if !c.ModifierOnly() && c.Code != tc.code {
			t.Errorf("Parse(%q) code %d, want %d", tc.in, c.Code, tc.code)
		}
		acc, ok := c.Accelerator()
		if acc != tc.acc || ok != (tc.acc != "") {
			t.Errorf("Parse(%q).Accelerator() = %q %v, want %q", tc.in, acc, ok, tc.acc)
		}
		// The canonical spelling reads back to the same chord.
		again, err := Parse(c.String())
		if err != nil || !again.Equal(c) {
			t.Errorf("round trip of %q: %v %v", tc.in, again, err)
		}
	}
}

func TestOverlaps(t *testing.T) {
	p := func(s string) Chord { c, _ := Parse(s); return c }
	for _, tc := range []struct {
		a, b string
		want bool
	}{
		{"Fn", "Fn", true},
		{"Fn", "Fn+Shift", false},
		{"Command", "RightCommand", true},
		{"LeftCommand", "RightCommand", false},
		{"Control+Command+V", "Command+Control+V", true},
		{"Control+Command+V", "Control+Command+C", false},
	} {
		if got := p(tc.a).Overlaps(p(tc.b)); got != tc.want {
			t.Errorf("%s overlaps %s = %v", tc.a, tc.b, got)
		}
	}
}

func TestBuildDefaults(t *testing.T) {
	f := settings.Defaults().Features
	p := Build(f, []string{Transcribe})
	if len(p.Bindings) != 2 {
		t.Fatalf("bindings %+v", p.Bindings)
	}
	ropt, paste := p.Bindings[0], p.Bindings[1]
	if paste.ID != PasteLast || paste.Key == nil || *paste.Key != 9 || paste.Behavior != proto.BehaviorTap || len(paste.Modifiers) != 2 ||
		paste.Modifiers[proto.ModControl] != proto.SideAny || paste.Modifiers[proto.ModOption] != proto.SideAny {
		t.Errorf("paste-last %+v", paste)
	}
	// Right Option alone: the left one is no part of it.
	if ropt.ID != "transcribe/0" || ropt.Key != nil || ropt.Modifiers[proto.ModOption] != proto.SideRight || len(ropt.Modifiers) != 1 {
		t.Errorf("right option %+v", ropt)
	}
	if len(p.Carbon) != 1 || p.Carbon[0] != (Carbon{ID: PasteLast, Accelerator: "Control+Option+V"}) {
		t.Errorf("carbon %+v", p.Carbon)
	}
	// No fn in the defaults: fn and 🌐 are left alone (Auralink's).
	cfg := p.Configure(true)
	if cfg.SwallowFn || !cfg.Active || len(cfg.Bindings) != 2 || cfg.DoubleTapMS != 0 {
		t.Errorf("configure %+v", cfg)
	}
	all := Build(f, Features)
	if len(all.Bindings) != 4 || all.Bindings[1].ID != "translate/0" || all.Bindings[2].ID != "rewrite/0" || all.Bindings[3].ID != PasteLast {
		t.Errorf("all %+v", all.Bindings)
	}
	if got := Build(settings.Features{}, Features).Configure(false); got.Bindings == nil || len(got.Bindings) != 0 || got.SwallowFn {
		t.Errorf("empty %+v", got)
	}
}

func TestBuildSkipsBadAndDuplicates(t *testing.T) {
	f := settings.Features{Transcribe: settings.Transcribe{Feature: settings.Feature{Hotkeys: []settings.Hotkey{
		{Keys: "Fn"}, {Keys: "Hyper"}, {Keys: "Fn", Trigger: settings.TriggerTap}, {Keys: "Fn", Trigger: settings.TriggerDouble},
		{Keys: "RightCommand", Trigger: settings.TriggerLong},
	}}}}
	p := Build(f, Features)
	var ids []string
	for _, b := range p.Bindings {
		ids = append(ids, b.ID+":"+b.Behavior)
	}
	want := []string{"transcribe/0:tap", "transcribe/3:doubleTap", "transcribe/4:longPress"}
	if len(ids) != len(want) {
		t.Fatalf("ids %v", ids)
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Errorf("ids %v, want %v", ids, want)
		}
	}
	if len(p.Carbon) != 0 {
		t.Errorf("carbon %+v", p.Carbon)
	}
}

func TestParseID(t *testing.T) {
	for id, want := range map[string]struct {
		f  string
		i  int
		ok bool
	}{
		"transcribe/0": {"transcribe", 0, true}, "rewrite/12": {"rewrite", 12, true}, "paste-last": {PasteLast, 0, true},
		"escape": {}, "meeting/0": {}, "transcribe/x": {}, "transcribe/-1": {},
	} {
		f, i, ok := ParseID(id)
		if f != want.f || i != want.i || ok != want.ok {
			t.Errorf("ParseID(%q) = %q %d %v", id, f, i, ok)
		}
	}
}

func TestCheck(t *testing.T) {
	hk := func(keys, trig string) settings.Hotkey { return settings.Hotkey{Keys: keys, Trigger: trig} }
	f := settings.Features{
		Transcribe: settings.Transcribe{Feature: settings.Feature{Hotkeys: []settings.Hotkey{
			hk("Fn", ""), hk("Fn", settings.TriggerDouble), hk("Command+Space", ""), hk("K", ""), hk("F6", ""), hk("Bad+", ""),
		}}},
		Translate: settings.Translate{Feature: settings.Feature{Hotkeys: []settings.Hotkey{
			hk("Fn", settings.TriggerTap), hk("Fn", settings.TriggerLong),
		}}},
		Rewrite: settings.Rewrite{Feature: settings.Feature{Hotkeys: []settings.Hotkey{
			hk("Control+Command+V", ""),
		}}},
		PasteLast: hk("Command+Control+V", ""),
	}
	got := map[string]Problem{}
	for _, p := range Check(f) {
		got[ID(p.Feature, p.Index)+" "+p.Code] = p
	}
	for key, check := range map[string]func(Problem) bool{
		"transcribe/1 duplicate": func(p Problem) bool { return p.Other == Transcribe && p.OtherIndex == 0 },
		"transcribe/2 system":    func(p Problem) bool { return p.System == SystemSpotlight },
		"transcribe/3 bareKey":   func(Problem) bool { return true },
		"transcribe/5 invalid":   func(p Problem) bool { return p.Err != nil },
		"translate/0 conflict":   func(p Problem) bool { return p.Other == Transcribe && p.OtherIndex == 0 },
		"translate/1 duplicate":  func(p Problem) bool { return p.Other == Translate && p.OtherIndex == 0 },
		"paste-last/0 conflict":  func(p Problem) bool { return p.Other == Rewrite },
	} {
		p, ok := got[key]
		if !ok || !check(p) {
			t.Errorf("%s: %+v (all %v)", key, p, got)
		}
		delete(got, key)
	}
	// F6 alone is fine; Translate's fn long press is its own duplicate,
	// not a conflict with Transcribe's fn tap (other trigger).
	if len(got) != 0 {
		t.Errorf("unexpected problems %v", got)
	}
	if ps := Check(settings.Defaults().Features); len(ps) != 0 {
		t.Errorf("defaults have problems: %+v", ps)
	}
}

func TestNeedsTap(t *testing.T) {
	for _, tc := range []struct {
		h    settings.Hotkey
		want bool
	}{
		{settings.Hotkey{Keys: "Fn"}, true},
		{settings.Hotkey{Keys: "RightCommand"}, true},
		{settings.Hotkey{Keys: "Control+Command+V"}, false},
		{settings.Hotkey{Keys: "Control+Command+V", Trigger: settings.TriggerDouble}, true},
		{settings.Hotkey{Keys: "RightCommand+V"}, true},
		{settings.Hotkey{Keys: "Option+Space", Trigger: settings.TriggerLong}, true},
	} {
		if got := NeedsTap(tc.h); got != tc.want {
			t.Errorf("NeedsTap(%+v) = %v", tc.h, got)
		}
	}
}

func TestRoute(t *testing.T) {
	ev := func(binding, phase string) proto.HotkeyEventParams {
		return proto.HotkeyEventParams{Binding: binding, Phase: phase}
	}
	for _, tc := range []struct {
		name  string
		e     proto.HotkeyEventParams
		phase Phase
		hold  string
		want  Decision
	}{
		{"tap starts", ev("transcribe/0", proto.PhaseTrigger), Idle, "", Decision{Action: Start, Feature: Transcribe}},
		{"double tap starts", ev("transcribe/1", proto.PhaseTrigger), Idle, "", Decision{Action: Start, Feature: Transcribe}},
		{"long press down starts held", ev("transcribe/2", proto.PhaseDown), Idle, "", Decision{Action: Start, Feature: Transcribe, Hold: true}},
		{"up when idle", ev("transcribe/2", proto.PhaseUp), Idle, "", Decision{}},
		{"translate starts", ev("translate/0", proto.PhaseTrigger), Idle, "", Decision{Action: Start, Feature: Translate}},
		{"tap stops", ev("transcribe/0", proto.PhaseTrigger), Recording, "", Decision{Action: Stop}},
		{"another feature stops", ev("translate/0", proto.PhaseTrigger), Recording, "", Decision{Action: Stop}},
		{"release of the hold stops", ev("transcribe/2", proto.PhaseUp), Recording, "transcribe/2", Decision{Action: Stop}},
		{"other release does nothing", ev("transcribe/3", proto.PhaseUp), Recording, "transcribe/2", Decision{}},
		{"long press while recording stops", ev("transcribe/3", proto.PhaseDown), Recording, "", Decision{Action: Stop}},
		{"busy waits", ev("transcribe/0", proto.PhaseTrigger), Busy, "", Decision{}},
		{"escape cancels recording", ev(proto.HotkeyEscape, proto.PhaseTrigger), Recording, "", Decision{Action: Cancel}},
		{"escape cancels busy", ev(proto.HotkeyEscape, proto.PhaseTrigger), Busy, "", Decision{Action: Cancel}},
		{"escape idle", ev(proto.HotkeyEscape, proto.PhaseTrigger), Idle, "", Decision{}},
		{"paste last", ev(PasteLast, proto.PhaseTrigger), Idle, "", Decision{Action: Paste}},
		{"paste last while recording", ev(PasteLast, proto.PhaseTrigger), Recording, "", Decision{}},
		{"unknown binding", ev("try/x", proto.PhaseTrigger), Idle, "", Decision{}},
	} {
		if got := Route(tc.e, tc.phase, tc.hold); got != tc.want {
			t.Errorf("%s: %+v, want %+v", tc.name, got, tc.want)
		}
	}
}
