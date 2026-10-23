package main

import (
	"bytes"
	"image/png"
	"slices"
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/store"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/trayicon"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

func TestTrayIcons(t *testing.T) {
	if bytes.Equal(trayicon.Idle, trayicon.Recording) {
		t.Fatal("the recording icon is the same")
	}
	for _, b := range [][]byte{trayicon.Idle, trayicon.Recording} {
		img, err := png.Decode(bytes.NewReader(b))
		if err != nil {
			t.Fatal(err)
		}
		// 18 pt at 2×: MyGo shows any tray icon 18 pt tall.
		if r := img.Bounds(); r.Dx() != trayicon.Size || r.Dy() != trayicon.Size || trayicon.Size != 36 {
			t.Errorf("size %v", r)
		}
		// A template image: black, only its alpha counts.
		for y := 0; y < trayicon.Size; y++ {
			for x := 0; x < trayicon.Size; x++ {
				if r, g, b, a := img.At(x, y).RGBA(); a > 0 && r|g|b != 0 {
					t.Fatalf("pixel %d,%d is not black", x, y)
				}
			}
		}
	}
}

func TestTrayRecent(t *testing.T) {
	es := []store.Entry{
		{Status: store.StatusDone, Result: "newest"},
		{Status: store.StatusFailed, Result: "failed"},
		{Status: store.StatusDone, Result: "  "},
		{Status: store.StatusDone, Result: "second"},
		{Status: store.StatusDone, Result: "third"},
		{Status: store.StatusDone, Result: "fourth"},
	}
	if got := uitext.RecentResults(es, 3); !slices.Equal(got, []string{"newest", "second", "third"}) {
		t.Fatalf("recent %q", got)
	}
	// One short line: as wide as the menu's other items, whatever the
	// script; the whole result in the tooltip, and copied on a click.
	for _, c := range []struct{ in, want string }{
		{"two\nlines  here", "two lines here"},
		{"短句。", "短句。"},
		{strings.Repeat("一二三四五六七八九十", 5), "一二三四五六七八九十一二…"},
		{"Please send the meeting notes to the whole team before Friday.", "Please send the meeting…"},
		{"讨论一下 VoxtNew 的发布计划，然后把引导页做完", "讨论一下 VoxtNew 的发布计…"},
		{"今天下午三点开会，讨论发布计划，然后把引导页做完", "今天下午三点开会，讨论发…"},
		{"\n  Exactly twenty-six chars\t", "Exactly twenty-six chars"},
		{"Supercalifragilisticexpialidocious words", "Supercalifragilisticexpia…"},
	} {
		got := uitext.TrayLabel(c.in)
		if got != c.want {
			t.Errorf("uitext.TrayLabel(%q) = %q, want %q", c.in, got, c.want)
		}
		if uitext.DisplayWidth(got) > uitext.TrayRecentWidth || strings.ContainsAny(got, "\n\t") {
			t.Errorf("uitext.TrayLabel(%q) = %q: %d units", c.in, got, uitext.DisplayWidth(got))
		}
	}
	if uitext.DisplayWidth("开始转录 fn") != 11 || uitext.DisplayWidth("e\u0301") != 1 {
		t.Error("uitext.DisplayWidth")
	}
	huge := strings.Repeat("长", 1000)
	if tip := trayTip(huge); len([]rune(tip)) != trayTipRunes+1 || trayTip(" short \n") != "short" {
		t.Errorf("tip %d", len([]rune(tip)))
	}
	// The items: the recent results, a click copies one.
	a, _, _ := newTestApp(t)
	var copied []string
	a.copyText = func(s string) error { copied = append(copied, s); return nil }
	long := "Please send the meeting notes to the whole team before Friday.\nThanks!"
	a.recent = []string{"hello there", long}
	found, title := 0, false
	for _, it := range a.trayItems() {
		switch it.Label {
		case "hello there", "Please send the meeting…":
			found++
			it.Click(it, nil)
			if it.Label != "hello there" && it.ToolTip != long {
				t.Errorf("tooltip %q", it.ToolTip)
			}
		case "Recent results":
			title = it.Disabled
		}
	}
	if found != 2 || !title || !slices.Equal(copied, []string{"hello there", long}) {
		t.Errorf("recent items: found %d, title %v, copied %q", found, title, copied)
	}
}
