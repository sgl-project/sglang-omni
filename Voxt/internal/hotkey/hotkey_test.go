package hotkey

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
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
		{in: "Mouse3", canon: "Mouse3", code: 2},
		{in: "option+mouse4", canon: "Option+Mouse4", code: 3},
		{in: "Control+Command+Mouse32", canon: "Control+Command+Mouse32", code: 31},
		{in: "Mouse1", err: ErrUnknown},
		{in: "Mouse2", err: ErrUnknown},
		{in: "Mouse33", err: ErrUnknown},
		{in: "Mouse04", err: ErrUnknown},
		{in: "Mouse4+V", err: ErrTwoKeys},
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
		{"Mouse4", "Mouse4", true},
		{"Mouse4", "Mouse5", false},
		{"Option+Mouse4", "Mouse4", false},
	} {
		if got := p(tc.a).Overlaps(p(tc.b)); got != tc.want {
			t.Errorf("%s overlaps %s = %v", tc.a, tc.b, got)
		}
	}
}

func TestBuildDefaults(t *testing.T) {
	f := settings.Defaults().Features
	p := Build(f, []string{Transcribe})
	if len(p.Bindings) != 3 {
		t.Fatalf("bindings %+v", p.Bindings)
	}
	ropt, paste, note := p.Bindings[0], p.Bindings[1], p.Bindings[2]
	if paste.ID != PasteLast || paste.Key == nil || *paste.Key != 9 || paste.Behavior != proto.BehaviorTap || len(paste.Modifiers) != 2 ||
		paste.Modifiers[proto.ModControl] != proto.SideAny || paste.Modifiers[proto.ModOption] != proto.SideAny {
		t.Errorf("paste-last %+v", paste)
	}
	// Right Option alone: the left one is no part of it.
	if ropt.ID != "transcribe/0" || ropt.Key != nil || ropt.Modifiers[proto.ModOption] != proto.SideRight || len(ropt.Modifiers) != 1 {
		t.Errorf("right option %+v", ropt)
	}
	// The note shortcut, Voxt 1.x's right Command tapped alone, whatever
	// features are on.
	if note.ID != Note || note.Key != nil || note.Behavior != proto.BehaviorTap || note.Modifiers[proto.ModCommand] != proto.SideRight || len(note.Modifiers) != 1 {
		t.Errorf("note %+v", note)
	}
	if len(p.Carbon) != 1 || p.Carbon[0] != (Carbon{ID: PasteLast, Accelerator: "Control+Option+V"}) {
		t.Errorf("carbon %+v", p.Carbon)
	}
	// No fn in the defaults: fn and 🌐 are left alone (Auralink's).
	cfg := p.Configure(true)
	if cfg.SwallowFn || !cfg.Active || len(cfg.Bindings) != 3 || cfg.DoubleTapMS != 0 {
		t.Errorf("configure %+v", cfg)
	}
	all := Build(f, Features)
	if len(all.Bindings) != 5 || all.Bindings[1].ID != "translate/0" || all.Bindings[2].ID != "rewrite/0" || all.Bindings[3].ID != PasteLast ||
		all.Bindings[4].ID != Note {
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

// TestBuildMouseButtons: a mouse button goes to the helper as its button
// number (Mouse3, the middle button, is 2), with its modifiers and no key,
// and never to Carbon, which has no mouse buttons.
func TestBuildMouseButtons(t *testing.T) {
	f := settings.Features{
		Transcribe: settings.Transcribe{Feature: settings.Feature{Hotkeys: []settings.Hotkey{
			{Keys: "Mouse4", Trigger: settings.TriggerLong}, {Keys: "Option+Mouse3"},
		}}},
		Cancel: settings.Hotkey{Keys: "Mouse5"},
	}
	p := Build(f, Features)
	if len(p.Carbon) != 0 {
		t.Errorf("carbon %+v", p.Carbon)
	}
	// What hotkey.configure sends: the wire form is what the helper judges.
	got, _ := json.Marshal(map[string]any{"bindings": p.Bindings, "cancel": p.Cancel})
	want := `{"bindings":[{"id":"transcribe/0","modifiers":{},"button":3,"behavior":"longPress"},` +
		`{"id":"transcribe/1","modifiers":{"option":"any"},"button":2,"behavior":"tap"}],` +
		`"cancel":{"id":"cancel","modifiers":{},"button":4,"behavior":"tap"}}`
	if string(got) != want {
		t.Errorf("sent %s\nwant %s", got, want)
	}
}

// TestBuildCancel: the cancel shortcut becomes hotkey.configure's cancel
// (never a regular binding, so its keys are only taken while active);
// it can be changed, turned off, and a bare key is fine for it. A clash
// with an earlier binding drops it, as Check reports.
func TestBuildCancel(t *testing.T) {
	for _, tc := range []struct {
		name string
		keys string
		code int // -1: modifier-only; -2: no cancel in the plan
		mods map[string]string
	}{
		{"default escape", settings.DefaultCancel, 53, map[string]string{}},
		{"custom key", "Control+.", 47, map[string]string{proto.ModControl: proto.SideAny}},
		{"modifier only", "RightShift", -1, map[string]string{proto.ModShift: proto.SideRight}},
		{"clashes with the note", settings.DefaultNote, -2, nil},
		{"off", "", -2, nil},
		{"does not parse", "Hyper", -2, nil},
		{"clashes with transcribe", settings.DefaultTranscribe, -2, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := settings.Defaults().Features
			f.Cancel = settings.Hotkey{Keys: tc.keys}
			p := Build(f, Features)
			if tc.code == -2 {
				if p.Cancel != nil {
					t.Fatalf("cancel %+v, want none", p.Cancel)
				}
				return
			}
			if p.Cancel == nil {
				t.Fatal("no cancel in the plan")
			}
			c := *p.Cancel
			if c.ID != CancelID || c.Behavior != proto.BehaviorTap || len(c.Modifiers) != len(tc.mods) {
				t.Fatalf("cancel %+v", c)
			}
			for fam, side := range tc.mods {
				if c.Modifiers[fam] != side {
					t.Errorf("cancel %+v", c)
				}
			}
			if tc.code == -1 {
				if c.Key != nil {
					t.Errorf("modifier-only cancel has key %+v", c)
				}
			} else if c.Key == nil || *c.Key != tc.code {
				t.Errorf("cancel key %+v, want %d", c.Key, tc.code)
			}
			for _, b := range p.Bindings {
				if b.ID == CancelID {
					t.Error("cancel must not be a regular binding (it would fire while idle)")
				}
			}
			for _, cb := range p.Carbon {
				if cb.ID == CancelID {
					t.Error("cancel must not be a Carbon shortcut (it would take the keys while idle)")
				}
			}
			if cfg := p.Configure(true); cfg.Cancel != p.Cancel {
				t.Errorf("configure %+v", cfg)
			}
		})
	}
	// A bare key is fine for cancel (only taken while active); a system
	// shortcut still warns; the clash is reported on the cancel row.
	f := settings.Defaults().Features
	if ps := ProblemsOf(Check(f), CancelID); len(ps[0]) != 0 {
		t.Errorf("default cancel has problems: %+v", ps[0])
	}
	f.Cancel = settings.Hotkey{Keys: "Command+C"}
	if ps := ProblemsOf(Check(f), CancelID); len(ps[0]) != 1 || ps[0][0].Code != ProblemSystem {
		t.Errorf("system cancel: %+v", ps[0])
	}
	f.Cancel = settings.Hotkey{Keys: settings.DefaultTranscribe}
	if ps := ProblemsOf(Check(f), CancelID); len(ps[0]) != 1 || ps[0][0].Code != ProblemConflict || ps[0][0].Other != Transcribe {
		t.Errorf("clashing cancel: %+v", ps[0])
	}
}

func TestParseID(t *testing.T) {
	for id, want := range map[string]struct {
		f  string
		i  int
		ok bool
	}{
		"transcribe/0": {"transcribe", 0, true}, "rewrite/12": {"rewrite", 12, true}, "paste-last": {PasteLast, 0, true}, "note": {Note, 0, true},
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
			hk("Mouse4", ""),
		}}},
		Translate: settings.Translate{Feature: settings.Feature{Hotkeys: []settings.Hotkey{
			hk("Fn", settings.TriggerTap), hk("Fn", settings.TriggerLong),
		}}},
		Rewrite: settings.Rewrite{Feature: settings.Feature{Hotkeys: []settings.Hotkey{
			hk("Control+Command+V", ""),
		}}},
		PasteLast: hk("Command+Control+V", ""),
		Note:      hk("Fn", settings.TriggerDouble), // always a tap: Transcribe's fn tap has it
		Cancel:    hk("Fn", ""),
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
		"note/0 conflict":        func(p Problem) bool { return p.Other == Transcribe && p.OtherIndex == 0 },
		"cancel/0 conflict":      func(p Problem) bool { return p.Other == Transcribe && p.OtherIndex == 0 },
	} {
		p, ok := got[key]
		if !ok || !check(p) {
			t.Errorf("%s: %+v (all %v)", key, p, got)
		}
		delete(got, key)
	}
	// F6 alone is fine, so is a mouse button; Translate's fn long press is its own duplicate,
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
		{settings.Hotkey{Keys: "Control+Option+Mouse4"}, true},
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
		{"cancel shortcut cancels recording", ev(proto.HotkeyCancel, proto.PhaseTrigger), Recording, "", Decision{Action: Cancel}},
		{"cancel shortcut cancels busy", ev(proto.HotkeyCancel, proto.PhaseTrigger), Busy, "", Decision{Action: Cancel}},
		{"cancel shortcut idle", ev(proto.HotkeyCancel, proto.PhaseTrigger), Idle, "", Decision{}},
		{"paste last", ev(PasteLast, proto.PhaseTrigger), Idle, "", Decision{Action: Paste}},
		{"paste last while recording", ev(PasteLast, proto.PhaseTrigger), Recording, "", Decision{}},
		{"note starts", ev(Note, proto.PhaseTrigger), Idle, "", Decision{Action: Start, Feature: Note}},
		{"note stops", ev(Note, proto.PhaseTrigger), Recording, "", Decision{Action: Stop}},
		{"unknown binding", ev("try/x", proto.PhaseTrigger), Idle, "", Decision{}},
	} {
		if got := Route(tc.e, tc.phase, tc.hold); got != tc.want {
			t.Errorf("%s: %+v, want %+v", tc.name, got, tc.want)
		}
	}
}
