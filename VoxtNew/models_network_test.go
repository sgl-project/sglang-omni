package main

import (
	"context"
	"slices"
	"strings"
	"testing"

	famsettings "github.com/hehehai/actnowkit/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/actnow"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/models"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/uitext"
)

// familyValue is the family's value of a shared key ("" unset).
func familyValue(t *testing.T, a *app, key string) string {
	t.Helper()
	s, _, err := famsettings.Read(a.fam.Dir())
	if err != nil {
		t.Fatal(err)
	}
	v, _ := s.Get(key)
	return v
}

// TestNetworkDialog: the deep link opens it; a change applies to the
// family while VoxtNew follows it, to VoxtNew alone after "Use a different
// setting for this app"; a custom proxy is checked.
func TestNetworkDialog(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.famStarted = true // familySettingsChanged writes the shared settings
	addPeer(t, a, dictaID, nil)
	a.familyFound(a.fam.Discover())
	a.openURL("voxtnew://models/network")
	if a.currentPage() != "models" || !a.netOpen || !has(tt, "How model downloads reach Hugging Face") {
		t.Fatalf("deep link: page %q open %v %q", a.currentPage(), a.netOpen, tt.Texts())
	}
	if !has(tt, "Applies to all ActNow apps") || !has(tt, "The proxy in the Mac's network settings, or the one the environment names.") {
		t.Fatalf("defaults: %q", tt.Texts())
	}

	// A custom proxy: checked, then the family's.
	click(t, tt, "Custom")
	a.netForm.proxyURL = "ftp://127.0.0.1:21"
	click(t, tt, "Save")
	if !a.netOpen || !has(tt, "Enter a proxy's address that starts with http://, https:// or socks5://, like http://127.0.0.1:7890.") {
		t.Fatalf("an ftp proxy was taken: %q", tt.Texts())
	}
	a.netForm.proxyURL = " socks5://127.0.0.1:1080 "
	click(t, tt, "hf-mirror.com only")
	click(t, tt, "Save")
	if a.netOpen || a.prefs.Network.Proxy != "socks5://127.0.0.1:1080" || a.prefs.Network.HFEndpoints != famsettings.EndpointsMirror {
		t.Fatalf("saved: open %v %+v", a.netOpen, a.prefs.Network)
	}
	if got := familyValue(t, a, actnow.KeyProxy); got != "socks5://127.0.0.1:1080" {
		t.Errorf("family's proxy %q", got)
	}
	if got := familyValue(t, a, actnow.KeyHFEndpoints); got != famsettings.EndpointsMirror {
		t.Errorf("family's source %q", got)
	}

	// For this app only: the family keeps its value.
	a.openNetwork()
	if a.netForm.proxyMode != proxyCustom || a.netForm.proxyURL != "socks5://127.0.0.1:1080" || a.netForm.source != 2 {
		t.Fatalf("form %+v", a.netForm)
	}
	click(t, tt, "Use a different setting for this app") // the proxy's
	if a.prefs.Family.Follows(actnow.KeyProxy) || !a.prefs.Family.Follows(actnow.KeyHFEndpoints) || !has(tt, "Only VoxtNew uses this setting.") {
		t.Fatalf("follow: %+v %q", a.prefs.Family.Follow, tt.Texts())
	}
	click(t, tt, "No proxy")
	click(t, tt, "Save")
	if a.prefs.Network.Proxy != famsettings.ProxyDirect || familyValue(t, a, actnow.KeyProxy) != "socks5://127.0.0.1:1080" {
		t.Errorf("own proxy %q, family's %q", a.prefs.Network.Proxy, familyValue(t, a, actnow.KeyProxy))
	}
	if got := a.fam.Effective(a.prefs, actnow.KeyProxy); got != famsettings.ProxyDirect {
		t.Errorf("effective %q", got)
	}

	// Back to the family's.
	a.openNetwork()
	click(t, tt, "Use the same setting as the other ActNow apps")
	if !a.prefs.Family.Follows(actnow.KeyProxy) || a.prefs.Network.Proxy != "socks5://127.0.0.1:1080" || a.netForm.proxyMode != proxyCustom {
		t.Errorf("following again: %+v %+v", a.prefs.Network, a.netForm)
	}
}

// TestNetworkUseTheirs: a proxy that differed from the family's at the
// first start shows "Use theirs" once.
func TestNetworkUseTheirs(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.famStarted = true
	addPeer(t, a, audioSmithID, nil)
	a.familyFound(a.fam.Discover())
	if _, err := famsettings.Set(context.Background(), a.fam.Dir(), audioSmithID, actnow.KeyProxy, "http://10.0.0.1:8080"); err != nil {
		t.Fatal(err)
	}
	a.fam.ReadShared()
	a.updateSettings(func(s *settings.Settings) {
		s.Network.Proxy = famsettings.ProxyDirect
		s.Family.Follow = map[string]bool{actnow.KeyProxy: false}
		s.Family.Conflicts = []string{actnow.KeyProxy}
	})
	a.navigate(modelsPath)
	click(t, tt, "Network…")
	if !has(tt, "Other ActNow apps use a different proxy.") || has(tt, "Other ActNow apps use a different download source.") {
		t.Fatalf("no notice: %q", tt.Texts())
	}
	click(t, tt, "Use theirs")
	if !a.prefs.Family.Follows(actnow.KeyProxy) || a.prefs.Network.Proxy != "http://10.0.0.1:8080" || len(a.prefs.Family.Conflicts) != 0 {
		t.Errorf("theirs: %+v", a.prefs)
	}
	if a.netForm.proxyMode != proxyCustom || a.netForm.proxyURL != "http://10.0.0.1:8080" || !has(tt, "Applies to all ActNow apps") {
		t.Errorf("form %+v", a.netForm)
	}

	// Shown once: closing the dialog forgets the notice.
	a.updateSettings(func(s *settings.Settings) {
		s.Family.Follow = map[string]bool{actnow.KeyHFEndpoints: false}
		s.Family.Conflicts = []string{actnow.KeyHFEndpoints}
	})
	a.openNetwork()
	if !has(tt, "Other ActNow apps use a different download source.") {
		t.Fatalf("no notice: %q", tt.Texts())
	}
	click(t, tt, "Cancel")
	if a.netOpen || slices.Contains(a.prefs.Family.Conflicts, actnow.KeyHFEndpoints) {
		t.Errorf("still noted: %v", a.prefs.Family.Conflicts)
	}
}

// TestFamilyLinesNeedAnotherApp: the shared settings speak of the family
// only when a release of another ActNow app is installed (a dev build of
// one does not count); alone, VoxtNew says nothing of it, though the data
// still follows the family's files. Model descriptions never say "another
// app": the status line names it ("Installed by Dicta").
func TestFamilyLinesNeedAnotherApp(t *testing.T) {
	a, _, tt := newTestApp(t)
	a.famStarted = true
	family := []string{"Applies to all ActNow apps", "Only VoxtNew uses this setting.", "Use the same setting as the other ActNow apps",
		"Use a different setting for this app"}
	check := func(when string, want bool) {
		t.Helper()
		a.navigate(modelsPath)
		a.openNetwork()
		frames(tt)
		if has(tt, family[0]) != want {
			t.Errorf("%s: %q shown %v", when, family[0], !want)
		}
		for _, s := range family[1:] {
			if !want && has(tt, s) {
				t.Errorf("%s: %q shown", when, s)
			}
		}
		click(t, tt, "Cancel")
		a.navigate("/settings/storage")
		frames(tt)
		if has(tt, "Shared by all ActNow apps") != want || has(tt, "Shared with other ActNow apps") != want || has(tt, "Used by VoxtNew") == want {
			t.Errorf("%s: storage %q", when, tt.Texts())
		}
	}
	check("alone", false)
	if got := familyValue(t, a, actnow.KeyProxy); got != "" {
		t.Errorf("family's proxy %q before any change", got)
	}
	a.updateSettings(func(s *settings.Settings) { s.Network.Proxy = famsettings.ProxyDirect })
	if got := familyValue(t, a, actnow.KeyProxy); got != famsettings.ProxyDirect {
		t.Errorf("alone, a change still goes to the family: %q", got)
	}

	addPeer(t, a, dictaID+".dev", nil)
	a.familyFound(a.fam.Discover())
	check("a dev build of Dicta", false)

	addPeer(t, a, dictaID, nil)
	a.familyFound(a.fam.Discover())
	check("Dicta installed", true)

	for _, id := range []string{models.LLMQwen3, models.LLMGemma4} {
		if about := uitext.ModelAbout(id); about == "" || strings.Contains(strings.ToLower(about), "another") {
			t.Errorf("%s: %q", id, about)
		}
	}
}
